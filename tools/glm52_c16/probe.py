#!/usr/bin/env python3
"""C16 native shard byte/consumer/graph gates and paired timing; lease required."""

import argparse
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path

from common import (
    PREPARE_SHA256,
    ROW_CASES,
    allocation_check,
    baseline,
    byte_check,
    padded_cases,
    payload_inputs,
    save,
    sha,
    source_hashes,
)
from native_local_pad import NativeLocalPadSlice


class Probe:
    def __init__(self, torch, torch_npu):
        self.torch = torch
        self.npu = torch_npu
        self.candidate = NativeLocalPadSlice(torch, "npu", torch_npu.get_npu_format)

    def baseline(self, hidden, logits, padded, rank):
        return baseline(self.torch, hidden, logits, padded, rank)

    def describe(self, tensor):
        return dict(
            shape=list(tensor.shape),
            stride=list(tensor.stride()),
            dtype=str(tensor.dtype),
            offset=tensor.storage_offset(),
            pointer=tensor.data_ptr(),
            storage=tensor.untyped_storage().data_ptr(),
            npu_format=self.npu.get_npu_format(tensor),
            device=str(tensor.device),
            contiguous=tensor.is_contiguous(),
        )

    def check(self, actual, expected, label):
        byte_check(self.torch, actual, expected, label)

    def consumer(self, stage, prepared, bias):
        if stage == "prepare":
            return prepared
        weights, ids, _unused = self.torch.ops._C_ascend.moe_gating_top_k(
            prepared[1],
            k=8,
            k_group=1,
            group_count=1,
            group_select_mode=1,
            renorm=1,
            norm_type=1,
            out_flag=False,
            routed_scaling_factor=2.5,
            eps=1e-20,
            bias_opt=bias,
        )
        routed = prepared + (weights, ids.to(self.torch.int32))
        if stage == "native_routing":
            return routed
        if stage == "routing_plus_standalone_quant":
            quant, scale = self.npu.npu_dynamic_quant(prepared[0])
            return routed + (quant, scale)
        raise ValueError(stage)

    def checkpoint_gate(self, checkpoint, layer=78):
        from safetensors import safe_open

        index = json.loads(
            (checkpoint / "quant_model_weights.safetensors.index.json").read_text()
        )["weight_map"]
        tensors, provenance = {}, []
        for suffix in ("weight", "e_score_correction_bias"):
            name = f"model.layers.{layer}.mlp.gate.{suffix}"
            with safe_open(
                checkpoint / index[name], framework="pt", device="cpu"
            ) as handle:
                tensor = handle.get_tensor(name)
            tensors[suffix] = tensor
            provenance.append(
                dict(
                    name=name,
                    shard=index[name],
                    shape=list(tensor.shape),
                    dtype=str(tensor.dtype),
                    bits_sha=hashlib.sha256(
                        tensor.contiguous().view(self.torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                )
            )
        assert (
            tensors["weight"].shape == (256, 6144)
            and tensors["weight"].dtype == self.torch.bfloat16
        )
        assert tensors["e_score_correction_bias"].shape == (256,)
        return (
            tensors["weight"].to("npu:0"),
            tensors["e_score_correction_bias"].to("npu:0"),
            provenance,
        )

    def finite_inputs(self, rows, weight, seed, offset=False):
        extra = 5 if offset else 0
        hidden = self.torch.randn(
            (rows + extra, 6144),
            dtype=self.torch.bfloat16,
            generator=self.torch.Generator().manual_seed(seed),
        ).to("npu:0")
        # Real checkpoint weights, input-generation-only FP32 linear. This is
        # not a claim to reproduce production GateLinear GEMM lowering.
        logits = self.torch.nn.functional.linear(hidden.float(), weight.float())
        return hidden[extra:], logits[extra:]

    def functions(self, inputs, padded, rank, stage, bias):
        return {
            "baseline": lambda: self.consumer(
                stage, self.baseline(*inputs, padded, rank), bias
            ),
            "candidate": lambda: self.consumer(
                stage, self.candidate(*inputs, padded, 16, rank), bias
            ),
        }


class Captured:
    def __init__(self, torch, fn, unroll):
        self.torch = torch
        self.unroll = unroll
        started = time.perf_counter_ns()
        for _ in range(5):
            fn()
        torch.npu.synchronize()
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph):
            for _ in range(unroll):
                self.outputs = fn()
        self.pointers = [x.data_ptr() for x in self.outputs]
        torch.npu.synchronize()
        self.initialization_ms = (time.perf_counter_ns() - started) / 1e6

    def __call__(self):
        self.graph.replay()
        return self.outputs


def measure(torch, fn, repeats, unroll=1):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    start, end = (
        torch.npu.Event(enable_timing=True),
        torch.npu.Event(enable_timing=True),
    )
    wall = time.perf_counter_ns()
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.npu.synchronize()
    return dict(
        device_us=start.elapsed_time(end) * 1000 / (repeats * unroll),
        wall_us=(time.perf_counter_ns() - wall) / 1000 / (repeats * unroll),
        repeats=repeats,
        unroll=unroll,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-npu", action="store_true")
    parser.add_argument("--lease", required=True)
    parser.add_argument("--prepare-source", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("/public-flash/models/GLM-5.2-w8a8")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--graph-unroll", type=int, default=16)
    parser.add_argument("--timing-ranks", type=int, nargs="+", default=[0, 11, 12, 15])
    args = parser.parse_args()
    if not args.run_npu or not args.lease.strip():
        parser.error("No device work without --run-npu AND an actual root-issued lease")
    if args.rounds < 20 or args.repeats < 1 or args.graph_unroll < 1:
        parser.error("Require at least20 pairs, positive repeats and graph unroll")
    if not args.timing_ranks or any(not 0 <= rank < 16 for rank in args.timing_ranks):
        parser.error("Timing ranks must be within[0,15]")
    if sha(args.prepare_source) != PREPARE_SHA256:
        parser.error("Audited original prepare source has changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(dict(status="initializing", lease=args.lease)) + "\n")
    # Device imports are isolated behind argument/lease/source/output checks.
    import torch
    import torch_npu

    import vllm_ascend.vllm_ascend_C  # noqa: F401

    torch.npu.set_device(0)
    if torch.npu.device_count() != 1:
        raise RuntimeError("Require exactly one root-authorized visible logical NPU")
    free, total = torch.npu.mem_get_info()
    if free < 1024**3:
        raise RuntimeError(f"Need1GiB framework/graph headroom; free={free}")
    probe = Probe(torch, torch_npu)
    result = dict(
        status="running",
        lease=args.lease,
        prepare_sha256=sha(args.prepare_source),
        tools_sha256=source_hashes(Path(__file__).parent),
        free_bytes_before=free,
        total_bytes=total,
        correctness=[],
        graph_checks=[],
        paired=[],
        scope="native local shard plus real native routing and separate dynamic quant; EP32 pending",
        model_residency="root-issued idle resident-model window; not an empty device",
    )
    try:
        for rows in ROW_CASES:
            for padded in padded_cases(rows):
                for offset in (False, True):
                    raw = payload_inputs(torch, rows, offset)
                    # Preserve actual storage offsets on NPU by moving the full
                    # backing tensor and re-slicing; .to() of a view resets offset.
                    if offset:
                        raw_full = payload_inputs(torch, rows + 5)
                        inputs = tuple(x.to("npu:0")[5:] for x in raw_full)
                        raw = tuple(x[5:] for x in raw_full)
                    else:
                        inputs = tuple(x.to("npu:0") for x in raw)
                    metadata = [probe.describe(x) for x in inputs]
                    for x, width in zip(inputs, (6144, 256)):
                        assert (
                            x.stride() == (width, 1)
                            and torch_npu.get_npu_format(x) == 2
                        ), metadata
                    before = tuple(x.clone() for x in inputs)
                    for rank in range(16):
                        actual = probe.candidate(*inputs, padded, 16, rank)
                        expected = probe.baseline(*inputs, padded, rank)
                        if padded == rows:
                            assert actual is None
                            actual = expected
                        else:
                            assert actual is not None
                            allocation_check(inputs, actual)
                            assert all(torch_npu.get_npu_format(x) == 2 for x in actual)
                        probe.check(
                            actual,
                            expected,
                            ("device original", rows, padded, rank, offset),
                        )
                        probe.check(
                            actual,
                            baseline(torch, *raw, padded, rank),
                            ("CPU raw oracle", rows, padded, rank, offset),
                        )
                        result["correctness"].append(
                            dict(
                                rows=rows,
                                padded=padded,
                                rank=rank,
                                offset=offset,
                                eligible=padded > rows,
                                byte_equal=True,
                                input_metadata=metadata,
                            )
                        )
                    probe.check(inputs, before, "input unchanged")
                    save(args.output, result)
        weight, bias, provenance = probe.checkpoint_gate(args.checkpoint)
        result["checkpoint_gate"] = provenance
        # Cover all16 ranks for the actual N12/P16 shape, plus mixed boundaries.
        for rows, padded in (
            (0, 16),
            (2, 16),
            (6, 16),
            (12, 16),
            (16, 32),
            (17, 32),
            (192, 208),
        ):
            ranks = (
                range(16)
                if rows == 12
                else sorted({0, min(15, rows // (padded // 16)), 15})
            )
            for rank in ranks:
                inputs = probe.finite_inputs(rows, weight, 1000 + rows, offset=True)
                pointers = [x.data_ptr() for x in inputs]
                functions = probe.functions(
                    inputs, padded, rank, "routing_plus_standalone_quant", bias
                )
                captures = {
                    name: Captured(torch, fn, 1) for name, fn in functions.items()
                }
                for seed in (17, 29, 43):
                    fresh = probe.finite_inputs(rows, weight, seed)
                    for original, update in zip(inputs, fresh):
                        original.copy_(update)
                    expected = functions["baseline"]()
                    for name, graph in captures.items():
                        probe.check(
                            graph(),
                            expected,
                            ("graph consumer", rows, padded, rank, seed, name),
                        )
                        assert graph.pointers == [x.data_ptr() for x in graph.outputs]
                    assert pointers == [x.data_ptr() for x in inputs]
                result["graph_checks"].append(
                    dict(
                        rows=rows,
                        padded=padded,
                        rank=rank,
                        updates=3,
                        byte_equal=True,
                        input_output_addresses_fixed=True,
                        input_metadata=[probe.describe(x) for x in inputs],
                        capture_init_ms={
                            name: graph.initialization_ms
                            for name, graph in captures.items()
                        },
                    )
                )
                save(args.output, result)
                del captures, functions
                gc.collect()
        hidden, logits = probe.finite_inputs(12, weight, 99)
        result["physical_format_negative_fixture"] = []
        for index, tensor in enumerate((hidden, logits)):
            nz = torch_npu.npu_format_cast(tensor, 29)
            actual_format = torch_npu.get_npu_format(nz)
            record = dict(
                input_index=index, requested_format=29, actual_format=actual_format
            )
            if actual_format == 29:
                unsupported = (nz, logits) if index == 0 else (hidden, nz)
                assert probe.candidate(*unsupported, 16, 16, 0) is None
                record["status"] = "physical_nz_rejected"
            else:
                # Installed allow_internal_format=False returns ND despite a
                # request for NZ. Do not mutate backend options to force a test.
                assert actual_format == 2, record
                nd_inputs = (nz, logits) if index == 0 else (hidden, nz)
                nd_outputs = probe.candidate(*nd_inputs, 16, 16, 0)
                assert nd_outputs is not None
                probe.check(
                    nd_outputs,
                    probe.baseline(*nd_inputs, 16, 0),
                    "ND cast remains eligible",
                )
                record["status"] = "unavailable_backend_returned_nd"
                record["actual_nd_positive_eligible"] = True
            result["physical_format_negative_fixture"].append(record)
        save(args.output, result)
        # Graph timing first: eager-only improvement is insufficient.
        for mode in ("graph", "eager"):
            for rank in args.timing_ranks:
                inputs = probe.finite_inputs(12, weight, 99)
                for stage in (
                    "prepare",
                    "native_routing",
                    "routing_plus_standalone_quant",
                ):
                    functions = probe.functions(inputs, 16, rank, stage, bias)
                    run = (
                        {
                            name: Captured(torch, fn, args.graph_unroll)
                            for name, fn in functions.items()
                        }
                        if mode == "graph"
                        else functions
                    )
                    for iteration in range(args.rounds):
                        fresh = probe.finite_inputs(12, weight, iteration + 400)
                        for original, update in zip(inputs, fresh):
                            original.copy_(update)
                        expected = functions["baseline"]()
                        order = (
                            ("baseline", "candidate")
                            if iteration % 2 == 0
                            else ("candidate", "baseline")
                        )
                        record = dict(
                            rows=12,
                            padded=16,
                            rank=rank,
                            stage=stage,
                            mode=mode,
                            iteration=iteration,
                            order=order,
                            measurements={},
                        )
                        for name in order:
                            probe.check(
                                run[name](),
                                expected,
                                ("paired", rank, stage, mode, iteration, name),
                            )
                            record["measurements"][name] = measure(
                                torch,
                                run[name],
                                args.repeats,
                                args.graph_unroll if mode == "graph" else 1,
                            )
                        result["paired"].append(record)
                        save(args.output, result)
                    del run, functions
                    gc.collect()
        result["status"] = "single_device_oracles_and_pairs_complete_EP32_pending"
        result["max_allocated_bytes"] = torch.npu.max_memory_allocated()
        result["max_reserved_bytes"] = torch.npu.max_memory_reserved()
        result["free_bytes_after"] = torch.npu.mem_get_info()[0]
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        save(args.output, result)
    groups = {}
    for pair in result["paired"]:
        values = pair["measurements"]
        key = f"rank{pair['rank']}/{pair['stage']}/{pair['mode']}"
        groups.setdefault(key, []).append(
            100
            * (values["candidate"]["device_us"] / values["baseline"]["device_us"] - 1)
        )
    print(
        json.dumps(
            {
                key: dict(
                    median_percent=statistics.median(values),
                    wins=sum(x < 0 for x in values),
                    pairs=len(values),
                )
                for key, values in groups.items()
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
