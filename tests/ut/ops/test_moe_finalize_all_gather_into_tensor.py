# SPDX-License-Identifier: Apache-2.0
"""Actual parent/off/on finalize AST with CPU tensors and explicit HCCL mocks.

No vLLM, torch_npu or backend imports are performed. This tests rank ordering,
bytes and ownership; it cannot prove HCCL list/into API or graph compatibility.
"""

import ast
import builtins
import copy
import hashlib
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

if os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") != "0":
    raise RuntimeError("Set TORCH_DEVICE_BACKEND_AUTOLOAD=0 before importing CPU torch")

import torch

PACKAGE = Path(__file__).resolve().parents[4]
PREPARE = "vllm_ascend/ops/fused_moe/prepare_finalize.py"
COMM = "vllm_ascend/ops/fused_moe/moe_comm_method.py"
CONFIG = "vllm_ascend/ascend_config.py"
FLAG = "enable_moe_finalize_all_gather_into_tensor"
BASE_HASHES = {
    PREPARE: "af0e23410d6b4a8336c1606ec1eac3d24421be2429f47be2f6ddd050ef7cac2b",
    COMM: "6416a5807606c8858036f97f0a95bf58dbeb60408d9fd7562336bec42a7020e2",
    CONFIG: "7a765f889e6c24fe70175829d8adf3d1dac7dce716cae6ad7d40ee4c5e1d799e",
}
TP = 16
WIDTH = 6144


class StripAnnotations(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.returns = None
        node.decorator_list = []
        for arg in node.args.args + node.args.kwonlyargs:
            arg.annotation = None
        return self.generic_visit(node)


def compile_nodes(nodes, namespace):
    tree = StripAnnotations().visit(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))
    exec(compile(ast.fix_missing_locations(tree), "<c21-actual-source-ast>", "exec"), namespace)
    return namespace


def cls_node(side, relative, name):
    return next(
        n
        for n in ast.parse((PACKAGE / side / relative).read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == name
    )


def method_node(side, relative, cls, method):
    return next(n for n in cls_node(side, relative, cls).body if isinstance(n, ast.FunctionDef) and n.name == method)


def raw(tensor):
    return tensor.tensor if isinstance(tensor, MetadataTensor) else tensor


def bits(tensor):
    return raw(tensor).contiguous().view(torch.uint8)


class MetadataTensor:
    def __init__(self, tensor, *, device="npu", npu_format=2, stride=None, layout=None, contiguous=None):
        self.tensor = tensor
        self.device = SimpleNamespace(type=device, index=0)
        self.npu_format = npu_format
        self.layout = tensor.layout if layout is None else layout
        self._stride = stride
        self._contiguous = contiguous

    def __getattr__(self, name):
        return getattr(self.tensor, name)

    def __getitem__(self, index):
        return self.tensor[index]

    def stride(self):
        return self.tensor.stride() if self._stride is None else self._stride

    def is_contiguous(self):
        return self.tensor.is_contiguous() if self._contiguous is None else self._contiguous


class TorchFacade:
    def __init__(self):
        self.allocations = []
        self.splits = 0
        self.nn = SimpleNamespace(
            functional=SimpleNamespace(pad=lambda x, padding: torch.nn.functional.pad(raw(x), padding))
        )

    def __getattr__(self, name):
        return getattr(torch, name)

    def empty(self, shape, *, device, dtype):
        output = torch.empty(shape, dtype=dtype)
        self.allocations.append((output, device, dtype))
        return output

    def tensor_split(self, tensor, count, dim):
        self.splits += 1
        return torch.tensor_split(raw(tensor), count, dim=dim)


class MockDist:
    def __init__(self, payloads, group, rank=0):
        self.payloads = payloads
        self.group = group
        self.rank = rank
        self.world_size = TP
        self.calls = []

    def get_world_size(self, group):
        if group is not self.group:
            raise AssertionError("changed group")
        return self.world_size

    def _record(self, api, source, group, async_op):
        if group is not self.group or async_op is not False:
            raise AssertionError("changed group or synchronous behavior")
        if raw(source).data_ptr() != self.payloads[self.rank].data_ptr():
            raise AssertionError("changed rank input ownership")
        self.calls.append((api, group, async_op, source))

    def all_gather(self, outputs, source, group, async_op=False):
        self._record("list", source, group, async_op)
        if not isinstance(outputs, list) or len(outputs) != TP:
            raise AssertionError("parent API contract")
        for out, payload in zip(outputs, self.payloads, strict=True):
            out.copy_(payload)

    def all_gather_into_tensor(self, output, source, group, async_op=False):
        self._record("into", source, group, async_op)
        # CPU mock only: not a substitute for an HCCL implementation/oracle.
        output.copy_(torch.cat(self.payloads, dim=0))


def payloads(local_rows, round_id=0, width=WIDTH, dtype=torch.bfloat16):
    generator = torch.Generator().manual_seed(5100 + local_rows + round_id * 100)
    tensors = []
    for rank in range(TP):
        values = torch.randint(-32768, 32767, (local_rows, width), generator=generator, dtype=torch.int16)
        # Include +/- zero, subnormals, +/- inf, qNaN and sNaN bit patterns.
        values[:, :8] = torch.tensor([0, -32768, 1, -32767, 32640, -128, 32705, 32641], dtype=torch.int16)
        values[:, 8] = rank + round_id * TP
        tensor = values.view(torch.bfloat16)
        tensors.append(tensor if dtype == torch.bfloat16 else tensor.to(dtype))
    return tensors


def runtime(side, payload, *, rank=0, enabled=False, c16=False, omitted=False):
    group = object()
    dist = MockDist(payload, group, rank)
    facade = TorchFacade()
    context = SimpleNamespace()
    namespace = dict(
        torch=facade,
        nn=facade.nn,
        torch_npu=SimpleNamespace(get_npu_format=lambda t: t.npu_format),
        dist=dist,
        ABC=object,
        QuantType=SimpleNamespace(NONE=0),
        _EXTRA_CTX=context,
        MoEPrepareOutput=SimpleNamespace,
        get_tensor_model_parallel_world_size=lambda: TP,
        get_tensor_model_parallel_rank=lambda: rank,
    )
    helper = compile_nodes(
        [
            n
            for n in ast.parse((PACKAGE / "reference/native_local_pad_slice.py").read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.Assign))
        ],
        dict(torch=facade, torch_npu=namespace["torch_npu"]),
    )

    def scoped_import(name, *args, **kwargs):
        if name == "vllm_ascend.ops.fused_moe.native_local_pad_slice":
            return SimpleNamespace(maybe_moe_native_local_pad_slice=helper["maybe_moe_native_local_pad_slice"])
        raise AssertionError(f"unexpected import from compiled runtime: {name}")

    namespace["__builtins__"] = {**vars(builtins), "__import__": scoped_import}
    tree = ast.parse((PACKAGE / side / PREPARE).read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        or isinstance(n, ast.ClassDef)
        and n.name in ("PrepareAndFinalize", "PrepareAndFinalizeWithAll2All", "PrepareAndFinalizeWithMC2")
    ]
    compile_nodes(nodes, namespace)
    kwargs = dict(enable_native_local_pad_slice=c16)
    if side == "source" and not omitted:
        kwargs["enable_finalize_all_gather_into_tensor"] = enabled
    owner = namespace["PrepareAndFinalizeWithMC2"](
        SimpleNamespace(tp_group=SimpleNamespace(device_group=group)), **kwargs
    )
    owner.replace_allreduce = False
    owner.num_tokens = local_rows = payload[rank].shape[0]
    return SimpleNamespace(owner=owner, dist=dist, torch=facade, context=context, local_rows=local_rows)


def config_gate(flag=False, *, omitted=False, fused_mc2=0, **changes):
    hf = SimpleNamespace(model_type="glm_moe_dsa", hidden_size=WIDTH, n_routed_experts=256)
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=hf),
        kv_transfer_config=SimpleNamespace(kv_role="kv_consumer"),
        parallel_config=SimpleNamespace(tensor_parallel_size=TP, data_parallel_size=2, enable_expert_parallel=True),
        lora_config=None,
    )
    for name, value in changes.items():
        owner, field = name.split(".")
        setattr(dict(hf=hf, cfg=cfg, parallel=cfg.parallel_config, kv=cfg.kv_transfer_config)[owner], field, value)
    body = method_node("source", CONFIG, "AscendConfig", "__init__").body
    start = next(
        i
        for i, n in enumerate(body)
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == FLAG
    )
    end = next(
        i
        for i in range(start, len(body))
        if isinstance(body[i], ast.Assign)
        and isinstance(body[i].targets[0], ast.Attribute)
        and body[i].targets[0].attr == FLAG
    )
    holder = SimpleNamespace(enable_fused_mc2=fused_mc2)
    compile_nodes(
        body[start : end + 1], dict(self=holder, additional_config={} if omitted else {FLAG: flag}, vllm_config=cfg)
    )
    return getattr(holder, FLAG)


class TestFinalizeIntoTensor(unittest.TestCase):
    def assert_bits(self, output, expected):
        self.assertEqual(raw(output).shape, raw(expected).shape)
        self.assertEqual(raw(output).dtype, raw(expected).dtype)
        self.assertTrue(torch.equal(bits(output), bits(expected)))

    def test_parent_hash_and_unchanged_prepare_other_owners(self):
        for relative, digest in BASE_HASHES.items():
            self.assertEqual(hashlib.sha256((PACKAGE / "parent" / relative).read_bytes()).hexdigest(), digest)
        for cls in ("PrepareAndFinalize", "PrepareAndFinalizeWithAll2All", "PrepareAndFinalizeWithAllGather"):
            self.assertEqual(ast.dump(cls_node("parent", PREPARE, cls)), ast.dump(cls_node("source", PREPARE, cls)))
        for method in ("prepare", "_restore_tp_across_dp", "pad_and_split_input_ids"):
            self.assertEqual(
                ast.dump(method_node("parent", PREPARE, "PrepareAndFinalizeWithMC2", method)),
                ast.dump(method_node("source", PREPARE, "PrepareAndFinalizeWithMC2", method)),
            )
        for cls in ("AlltoAllCommImpl", "AllGatherCommImpl", "FusedMC2CommImpl"):
            self.assertEqual(ast.dump(cls_node("parent", COMM, cls)), ast.dump(cls_node("source", COMM, cls)))
        for prefix in ("torch_npu", "vllm", "vllm_ascend", "triton"):
            self.assertFalse(any(name == prefix or name.startswith(prefix + ".") for name in sys.modules))

    def test_actual_parent_off_on_all_ranks_bytes_layout_ownership(self):
        cases = 0
        for tokens in (0, 1, 2, 6, 12, 16, 17, 31, 32, 191, 192):
            local_rows = max(1, (tokens + TP - 1) // TP)
            shape = torch.Size((TP * local_rows, WIDTH))
            for round_id in range(3):
                shards = payloads(local_rows, round_id)
                snapshots = [bits(t).clone() for t in shards]
                expected = torch.cat(shards, dim=0)
                for rank in range(TP):
                    outputs = []
                    for side, enabled in (("parent", False), ("source", False), ("source", True)):
                        run = runtime(side, shards, rank=rank, enabled=enabled)
                        run.owner.num_tokens = tokens
                        output = run.owner.finalize(MetadataTensor(shards[rank]), bool(round_id % 2), shape)
                        outputs.append(output)
                        self.assert_bits(output, expected[:tokens])
                        self.assertEqual(output.stride(), (WIDTH, 1))
                        self.assertTrue(output.is_contiguous())
                        self.assertEqual(output.storage_offset(), 0)
                        allocation = run.torch.allocations[0][0]
                        self.assert_bits(allocation, expected)  # Includes unexposed padded tail.
                        self.assertEqual(allocation.shape, shape)
                        self.assertNotIn(
                            allocation.untyped_storage().data_ptr(), [t.untyped_storage().data_ptr() for t in shards]
                        )
                        self.assertEqual(run.dist.calls[0][0], "into" if enabled else "list")
                        self.assertEqual(len(run.dist.calls), 1)
                        self.assertEqual(run.torch.splits, 0 if enabled else 1)
                        self.assertEqual(len(run.torch.allocations), 1)
                        cases += 1
                    self.assertEqual(len({out.untyped_storage().data_ptr() for out in outputs}), 3)
                for shard, snapshot in zip(shards, snapshots, strict=True):
                    self.assertTrue(torch.equal(bits(shard), snapshot))
        print(f"actual parent/off/on: {cases} full finalize calls; 11 token shapes, 3 updates, 16 ranks")

    def test_on_off_on_cross_call_retained_views(self):
        shards = payloads(1)
        run = runtime("source", shards, enabled=True)
        run.owner.num_tokens = 12
        retained = []
        for update, enabled in enumerate((True, False, True)):
            shards = payloads(1, update)
            run.dist.payloads = shards
            run.owner._finalize_all_gather_into_tensor = enabled
            result = run.owner.finalize(MetadataTensor(shards[0]), False, torch.Size((TP, WIDTH)))
            retained.append((result, result[:3], bits(result).clone()))
            for previous, view, expected in retained:
                self.assertTrue(torch.equal(bits(previous), expected))
                self.assertTrue(torch.equal(bits(view), expected[:3]))
        self.assertEqual([call[0] for call in run.dist.calls], ["into", "list", "into"])
        self.assertEqual(len({result.untyped_storage().data_ptr() for result, _, _ in retained}), 3)

    def test_default_strict_bool_config_and_factory_isolation(self):
        self.assertFalse(config_gate(omitted=True))
        self.assertTrue(config_gate(True))
        for invalid in (1, 0, None, "true", [], {}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                config_gate(invalid)
            with self.assertRaises(ValueError):
                runtime("source", payloads(1), enabled=invalid)
        for change in (
            {"cfg.kv_transfer_config": None},
            {"kv.kv_role": "kv_producer"},
            {"hf.model_type": "other"},
            {"hf.hidden_size": 4096},
            {"hf.n_routed_experts": 128},
            {"parallel.tensor_parallel_size": 8},
            {"parallel.data_parallel_size": 1},
            {"parallel.enable_expert_parallel": False},
            {"cfg.lora_config": object()},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                config_gate(True, **change)
            self.assertFalse(config_gate(False, **change))
        with self.assertRaises(ValueError):
            config_gate(True, fused_mc2=1)
        for name in ("MC2CommImpl", "FusedMC2CommImpl", "AlltoAllCommImpl", "AllGatherCommImpl"):
            ns = compile_nodes(
                [method_node("source", COMM, name, "_get_prepare_finalize")],
                dict(
                    get_ascend_config=lambda: SimpleNamespace(enable_moe_native_local_pad_slice=True, **{FLAG: True}),
                    PrepareAndFinalizeWithMC2=lambda config, **kwargs: ("mc2", config, kwargs),
                    PrepareAndFinalizeWithAll2All=lambda config: ("all2all", config, {}),
                    PrepareAndFinalizeWithAllGather=lambda config: ("allgather", config, {}),
                ),
            )
            config = object()
            _, actual_config, kwargs = ns["_get_prepare_finalize"](SimpleNamespace(moe_config=config))
            self.assertIs(actual_config, config)
            self.assertEqual(
                kwargs,
                {"enable_native_local_pad_slice": True, "enable_finalize_all_gather_into_tensor": True}
                if name == "MC2CommImpl"
                else {},
            )
        run = runtime("source", payloads(1), omitted=True)
        self.assertIs(run.owner._finalize_all_gather_into_tensor, False)

    def test_unsupported_metadata_falls_back_to_actual_parent(self):
        cases = [
            ("device", dict(device="cpu")),
            ("format", dict(npu_format=29)),
            ("stride", dict(stride=(WIDTH + 1, 1))),
            ("contiguous", dict(contiguous=False)),
            ("layout", dict(layout=torch.sparse_coo)),
        ]
        for name, kwargs in cases:
            shards = payloads(1)
            outputs = []
            for side, enabled in (("parent", False), ("source", False), ("source", True)):
                run = runtime(side, shards, enabled=enabled)
                run.owner.num_tokens = 12
                outputs.append(run.owner.finalize(MetadataTensor(shards[0], **kwargs), True, torch.Size((TP, WIDTH))))
                self.assertEqual(run.dist.calls[0][0], "list", name)
            self.assert_bits(outputs[0], outputs[1])
            self.assert_bits(outputs[0], outputs[2])
        for kind in ("dtype", "width", "offset", "lora", "group_size", "invalid_tokens", "noninteger_tokens"):
            shards = payloads(
                1, width=WIDTH if kind != "width" else 512, dtype=torch.bfloat16 if kind != "dtype" else torch.float32
            )
            if kind == "offset":
                shards = [torch.cat((t, t), dim=0)[1:] for t in shards]
            run = runtime("source", shards, enabled=True)
            run.owner.num_tokens = 12
            if kind == "lora":
                run.owner.lora_context = object()
            if kind == "group_size":
                run.dist.world_size = 8
            if kind == "invalid_tokens":
                run.owner.num_tokens = TP + 1
            if kind == "noninteger_tokens":
                run.owner.num_tokens = True
            run.owner.finalize(MetadataTensor(shards[0]), False, torch.Size((TP, shards[0].shape[1])))
            self.assertEqual(run.dist.calls[0][0], "list", kind)
        # Invalid collective shapes are gate rejections only, not backend tests.
        run = runtime("source", payloads(1), enabled=True)
        run.owner.num_tokens = 12
        for shape in (None, torch.Size((TP,)), torch.Size((TP + 1, WIDTH)), torch.Size((TP, WIDTH + 1))):
            self.assertFalse(
                run.owner._can_finalize_all_gather_into_tensor(MetadataTensor(run.dist.payloads[0]), shape)
            )
        run.owner.tp_size = 8
        self.assertFalse(
            run.owner._can_finalize_all_gather_into_tensor(
                MetadataTensor(run.dist.payloads[0]), torch.Size((TP, WIDTH))
            )
        )

    def test_bypass_and_reduce_results_keep_parent_semantics(self):
        shards = payloads(1)
        wrapped = MetadataTensor(shards[0])
        for side, enabled in (("parent", False), ("source", False), ("source", True)):
            run = runtime(side, shards, enabled=enabled)
            run.owner.replace_allreduce = True
            for reduce_results in (False, True):
                self.assertIs(run.owner.finalize(wrapped, reduce_results, None), wrapped)
            self.assertEqual(run.dist.calls, [])
            self.assertEqual(run.torch.allocations, [])
            run.owner.replace_allreduce = False
            run.owner.tp_size = 1
            run.owner.num_tokens = 1
            self.assertIs(run.owner.finalize(wrapped, True, None), wrapped)
            self.assertEqual(run.dist.calls, [])

    def test_c16_prepare_composition_all_ranks(self):
        for tokens in (12, 16, 17, 192):
            padded = ((tokens + TP - 1) // TP) * TP
            hidden = payloads(max(1, tokens))[0]
            logits = torch.arange(tokens * 256, dtype=torch.float32).reshape(tokens, 256)
            hidden_before, logits_before = bits(hidden).clone(), bits(logits).clone()
            full_hidden = torch.nn.functional.pad(hidden, (0, 0, 0, padded - tokens))
            full_logits = torch.nn.functional.pad(logits, (0, 0, 0, padded - tokens))
            shards = list(torch.tensor_split(full_hidden, TP, dim=0))
            for c16 in (False, True):
                for enabled in (False, True):
                    for rank in range(TP):
                        run = runtime("source", shards, rank=rank, enabled=enabled, c16=c16)
                        run.context.mc2_mask = torch.arange(padded) < tokens
                        run.context.padded_num_tokens = padded
                        prepared = run.owner.prepare(MetadataTensor(hidden), MetadataTensor(logits))
                        self.assert_bits(prepared.hidden_states, shards[rank])
                        self.assert_bits(prepared.router_logits, torch.tensor_split(full_logits, TP, dim=0)[rank])
                        self.assertEqual(prepared.padded_hidden_states_shape, torch.Size((padded, WIDTH)))
                        self.assertIsNone(prepared.pertoken_scale)
                        # Routed compute/dispatch/combine are not mocked as real models:
                        # use distinct fresh rank payloads for finalize's actual contract.
                        routed = payloads(padded // TP)
                        run.dist.payloads = routed
                        result = run.owner.finalize(
                            MetadataTensor(routed[rank]), False, prepared.padded_hidden_states_shape
                        )
                        self.assert_bits(result, torch.cat(routed)[:tokens])
                        self.assertEqual(run.dist.calls[0][0], "into" if enabled else "list")
            self.assertTrue(torch.equal(bits(hidden), hidden_before))
            self.assertTrue(torch.equal(bits(logits), logits_before))

    def test_mixed_api_rank_mock_order_only(self):
        shards = payloads(1)
        for rank_zero_on in (True, False):
            results = []
            for rank in range(TP):
                enabled = rank_zero_on if rank == 0 else not rank_zero_on
                run = runtime("source", shards, rank=rank, enabled=enabled)
                run.owner.num_tokens = 12
                results.append(run.owner.finalize(MetadataTensor(shards[rank]), False, torch.Size((TP, WIDTH))))
                self.assertEqual(run.dist.calls[0][0], "into" if enabled else "list")
            for result in results:
                self.assert_bits(result, results[0])
        # No assertion here can establish cross-API HCCL compatibility.


if __name__ == "__main__":
    unittest.main(verbosity=2)
