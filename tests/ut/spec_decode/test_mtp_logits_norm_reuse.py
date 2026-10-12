# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the actual proposer methods without backend imports.

Run this file directly with TORCH_DEVICE_BACKEND_AUTOLOAD=0. Native NPU
RMSNorm batch-shape equivalence has a separate isolated device oracle.
"""

from __future__ import annotations

import ast
import copy
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch

REPO = Path(__file__).resolve().parents[3]


def load_runtime(namespace):
    source = REPO / "vllm_ascend/spec_decode/llm_base_proposer.py"
    tree = ast.parse(source.read_text())
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendSpecDecodeBaseProposer"
    )
    names = {
        "_can_reuse_mtp_logits_norm",
        "_compute_mtp_logits_from_normalized",
        "_run_merged_draft",
        "model_returns_tuple",
        "compute_draft_token_ids",
    }
    methods = [copy.deepcopy(node) for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in methods} == names
    for node in methods:
        node.decorator_list = []
    constants = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_GLM_MTP_NORM_HIDDEN_SIZE" for target in node.targets)
    ]
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=constants + methods, type_ignores=[])), str(source), "exec"),
        namespace,
    )
    return type("RuntimeProposer", (), {name: namespace[name] for name in names})


def config_value(value=..., sparse=True, role="kv_consumer"):
    source = REPO / "vllm_ascend/ascend_config.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendConfig")
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    start = next(
        i
        for i, node in enumerate(init.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "enable_mtp_logits_norm_reuse" for target in node.targets)
    )
    owner = SimpleNamespace()
    namespace = {
        "self": owner,
        "additional_config": {} if value is ... else {"enable_mtp_logits_norm_reuse": value},
        "use_sparse": sparse,
        "vllm_config": SimpleNamespace(kv_transfer_config=None if role is None else SimpleNamespace(kv_role=role)),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=init.body[start : start + 3], type_ignores=[])),
            str(source),
            "exec",
        ),
        namespace,
    )
    return owner.enable_mtp_logits_norm_reuse


class CountingHead:
    def __init__(self, width, bias):
        self.weight = torch.randn(width, dtype=torch.bfloat16)
        self.bias = torch.randn(width, dtype=torch.bfloat16) if bias else None
        self.head = torch.randn(19, width, dtype=torch.bfloat16)
        self.norm_shapes = []

    def __call__(self, x):
        self.norm_shapes.append(tuple(x.shape))
        out = torch.nn.functional.rms_norm(x, (x.shape[-1],), self.weight, 1e-5)
        return out.add_(self.bias) if self.bias is not None else out


class DraftModel:
    def __init__(self, width, bias):
        self.shared_head = CountingHead(width, bias)
        self.logits = []
        self.processor_inputs = []
        self.forward_inputs = []
        self.returned = []
        self.skip_topk = []
        self.compute_logits_calls = 0
        self.model = SimpleNamespace(
            num_mtp_layers=1,
            mtp_start_layer_idx=78,
            layers={"78": SimpleNamespace(shared_head=self.shared_head)},
            logits_processor=self.processor,
            set_skip_topk=self.skip_topk.append,
        )

    def processor(self, head, x):
        assert head is self.shared_head.head
        self.processor_inputs.append(x.clone())
        # Same head object and processor serve the original and reuse paths.
        logits = torch.nn.functional.linear(x, head).float()
        self.logits.append(logits.clone())
        return logits

    def __call__(self, input_ids, positions, inputs_embeds, hidden_states):
        assert inputs_embeds is None
        self.forward_inputs.append((input_ids.clone(), positions.clone(), hidden_states.clone()))
        pre_norm = hidden_states + ((input_ids + positions).to(torch.bfloat16) / 100).unsqueeze(1)
        normalized = self.shared_head(pre_norm)
        self.returned.append((pre_norm, normalized, pre_norm.clone(), normalized.clone()))
        return pre_norm, normalized

    def compute_logits(self, hidden_states):
        self.compute_logits_calls += 1
        return self.model.logits_processor(self.shared_head.head, self.shared_head(hidden_states))


def make_proposer(enabled=True, bias=True, rows=12, spec=5):
    torch.manual_seed(17021)
    config = SimpleNamespace(enable_mtp_logits_norm_reuse=enabled, enable_reduce_sample=False)
    context = SimpleNamespace(moe_layer_index=0, attn_metadata=None)
    extra = SimpleNamespace(flash_comm_v1_enabled=False, num_tokens=0, num_accept_tokens=0)
    switches = SimpleNamespace(lmhead_tp=False)
    namespace = {
        "torch": torch,
        "nn": torch.nn,
        "_EXTRA_CTX": extra,
        "get_ascend_config": lambda: config,
        "get_forward_context": lambda: context,
        "lmhead_tp_enable": lambda: switches.lmhead_tp,
        "greedy_sample": lambda logits: logits.argmax(-1),
    }
    proposer = load_runtime(namespace)()
    proposer.method = "mtp"
    proposer.num_speculative_tokens = spec
    proposer.parallel_drafting = False
    proposer.use_cuda_graph = False
    proposer.dcp_size = 1
    proposer.is_multimodal_model = False
    proposer.vllm_config = SimpleNamespace(
        lora_config=None,
        kv_transfer_config=SimpleNamespace(kv_role="kv_consumer"),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="glm_moe_dsa"), max_model_len=266240),
        scheduler_config=SimpleNamespace(max_num_seqs=32),
    )
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(lora_id_to_lora_request={}),
        uniform_decode_query_len=6,
    )
    proposer.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["DeepSeekMTPModel"],
            num_nextn_predict_layers=1,
        )
    )
    proposer.model = DraftModel(6144, bias)
    proposer.input_ids = torch.arange(rows, dtype=torch.int32)
    proposer.positions = torch.arange(rows, dtype=torch.int32) + 131071
    proposer.hidden_states = torch.randn(rows, 6144, dtype=torch.bfloat16)
    proposer.arange = torch.arange(192, dtype=torch.int32)
    proposer.device = torch.device("cpu")
    proposer.uses_mrope = False
    proposer.pass_hidden_states_to_model = True
    proposer.supports_mm_inputs = False
    proposer._share_mtp_indices = True
    proposer.use_compress = False
    proposer._get_positions = lambda count: proposer.positions[:count]
    proposer._set_positions = lambda count, value: proposer.positions[:count].copy_(value)
    proposer.maybe_pad_and_reduce = lambda hidden, positions: (hidden, positions)
    proposer.maybe_all_gather_and_unpad = lambda last, positions, hidden: (last, positions, hidden)
    return proposer, config, extra, switches


def run_proposer(proposer, indices=None):
    if indices is None:
        indices = [5, 11]
    return proposer._run_merged_draft(
        num_input_tokens=proposer.input_ids.numel(),
        batch_size=len(indices),
        token_indices_to_sample=torch.tensor(indices, dtype=torch.int32),
        target_positions=proposer.positions,
        inputs_embeds=None,
        multi_steps_attn_metadata=[{"step": i} for i in range(proposer.num_speculative_tokens)],
        num_tokens=proposer.input_ids.numel(),
    )


def equal_bytes(a, b):
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and torch.equal(
            a.contiguous().view(torch.uint8),
            b.contiguous().view(torch.uint8),
        )
    )


class TestMTPLogitsNormReuse(unittest.TestCase):
    def test_config(self):
        for value, expected in ((..., False), (False, False), (True, True)):
            self.assertIs(config_value(value), expected)
        for value in (None, 0, 1, "true", "false", [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "must be a bool"):
                config_value(value)
        for sparse, role in ((False, "kv_consumer"), (True, "kv_producer"), (True, "kv_both"), (True, None)):
            self.assertFalse(config_value(True, sparse, role))

    def test_five_step_norm_count_logits_recycle_and_dummy_rows(self):
        for bias in (False, True):
            for rows in (12, 16):
                for indices in ([5, 11], [11, 5], [5, 5]):
                    with self.subTest(bias=bias, rows=rows, indices=indices):
                        baseline, *_ = make_proposer(False, bias, rows)
                        reuse, *_ = make_proposer(True, bias, rows)
                        self.assertTrue(equal_bytes(run_proposer(baseline, indices), run_proposer(reuse, indices)))
                        self.assertEqual(baseline.model.shared_head.norm_shapes, [(rows, 6144), (2, 6144)] * 5)
                        self.assertEqual(reuse.model.shared_head.norm_shapes, [(rows, 6144)] * 5)
                        self.assertEqual(baseline.model.compute_logits_calls, 5)
                        self.assertEqual(reuse.model.compute_logits_calls, 0)
                        self.assertEqual(reuse.model.skip_topk, [False, True])
                        self.assertEqual(len(reuse.model.forward_inputs), 5)
                        for original, candidate in zip(baseline.model.forward_inputs, reuse.model.forward_inputs):
                            self.assertTrue(all(equal_bytes(a, b) for a, b in zip(original, candidate)))
                        for a, b in zip(baseline.model.logits, reuse.model.logits):
                            self.assertTrue(equal_bytes(a, b))
                            self.assertTrue(equal_bytes(a.log_softmax(-1), b.log_softmax(-1)))
                        for pre, normalized, frozen_pre, frozen_normalized in reuse.model.returned:
                            self.assertTrue(equal_bytes(pre, frozen_pre))
                            self.assertTrue(equal_bytes(normalized, frozen_normalized))

    def test_flag_off_never_inspects_model_contract(self):
        proposer, *_ = make_proposer(False)
        del proposer.model
        self.assertFalse(proposer._can_reuse_mtp_logits_norm(None, None))

    def test_all_runtime_gates(self):
        changes = {
            "method": lambda p, c, e, s: setattr(p, "method", "eagle3"),
            "single_token": lambda p, c, e, s: setattr(p, "num_speculative_tokens", 1),
            "parallel": lambda p, c, e, s: setattr(p, "parallel_drafting", True),
            "graph": lambda p, c, e, s: setattr(p, "use_cuda_graph", True),
            "dcp": lambda p, c, e, s: setattr(p, "dcp_size", 2),
            "multimodal": lambda p, c, e, s: setattr(p, "is_multimodal_model", True),
            "lmhead_tp": lambda p, c, e, s: setattr(s, "lmhead_tp", True),
            "reduce_sample": lambda p, c, e, s: setattr(c, "enable_reduce_sample", True),
            "flashcomm": lambda p, c, e, s: setattr(e, "flash_comm_v1_enabled", True),
            "lora_config": lambda p, c, e, s: setattr(p.vllm_config, "lora_config", object()),
            "lora_request": lambda p, c, e, s: p.runner.input_batch.lora_id_to_lora_request.update({1: object()}),
            "producer": lambda p, c, e, s: setattr(p.vllm_config.kv_transfer_config, "kv_role", "kv_producer"),
            "no_pd": lambda p, c, e, s: setattr(p.vllm_config, "kv_transfer_config", None),
            "non_glm": lambda p, c, e, s: setattr(
                p.vllm_config.model_config.hf_text_config, "model_type", "deepseek_v3"
            ),
            "non_tuple": lambda p, c, e, s: setattr(p.draft_model_config.hf_config, "architectures", []),
            "hf_multi_nextn": lambda p, c, e, s: setattr(p.draft_model_config.hf_config, "num_nextn_predict_layers", 2),
            "runtime_multi_nextn": lambda p, c, e, s: setattr(p.model.model, "num_mtp_layers", 2),
            "missing_predictor": lambda p, c, e, s: setattr(p.model, "model", None),
            "missing_runner": lambda p, c, e, s: setattr(p, "runner", None),
        }
        for name, change in changes.items():
            with self.subTest(gate=name):
                p, c, e, s = make_proposer()
                x = p.hidden_states
                self.assertTrue(p._can_reuse_mtp_logits_norm(x, x))
                change(p, c, e, s)
                self.assertFalse(p._can_reuse_mtp_logits_norm(x, x))
        p, *_ = make_proposer()
        x = p.hidden_states
        for logit, normalized in ((x.float(), x), (x, x.float()), (x, x[:1]), (x[:, :32], x[:, :32]), (x[0], x[0])):
            self.assertFalse(p._can_reuse_mtp_logits_norm(logit, normalized))

    def test_unsupported_flags_use_original_logits_in_all_steps(self):
        for gate in ("graph", "dcp", "reduce_sample", "flashcomm", "lora"):
            outputs = []
            for enabled in (False, True):
                p, c, e, _ = make_proposer(enabled)
                if gate == "graph":
                    p.use_cuda_graph = True
                elif gate == "dcp":
                    p.dcp_size = 2
                elif gate == "reduce_sample":
                    c.enable_reduce_sample = True
                elif gate == "flashcomm":
                    e.flash_comm_v1_enabled = True
                else:
                    p.vllm_config.lora_config = object()
                outputs.append(run_proposer(p))
                self.assertEqual(p.model.compute_logits_calls, 5)
                self.assertEqual(len(p.model.shared_head.norm_shapes), 10)
            self.assertTrue(equal_bytes(*outputs), gate)

    def test_single_token_falls_back(self):
        p, *_ = make_proposer(True, spec=1)
        self.assertEqual(tuple(run_proposer(p).shape), (2, 1))
        self.assertEqual(p.model.compute_logits_calls, 1)
        self.assertEqual(len(p.model.shared_head.norm_shapes), 2)

    def test_changing_inputs_and_flag_across_rounds_has_no_stale_reuse(self):
        p, config, *_ = make_proposer(True)
        for enabled in (True, False, True):
            config.enable_mtp_logits_norm_reuse = enabled
            p.input_ids.add_(1)
            p.positions.fill_(p.vllm_config.model_config.max_model_len - 2)
            p.hidden_states.add_(torch.tensor(0.125, dtype=torch.bfloat16))
            old_norm_count = len(p.model.shared_head.norm_shapes)
            run_proposer(p)
            self.assertEqual(len(p.model.shared_head.norm_shapes) - old_norm_count, 5 if enabled else 10)
            self.assertEqual(p.model.skip_topk[-2:], [False, True])
            self.assertTrue(torch.equal(p.model.forward_inputs[-1][1][:2], torch.zeros(2, dtype=torch.int32)))


if __name__ == "__main__":
    torch.set_num_threads(1)
    forbidden = ("torch_npu", "triton", "vllm_ascend")
    assert not [name for name in sys.modules if name.split(".")[0] in forbidden]
    unittest.main(verbosity=2)
