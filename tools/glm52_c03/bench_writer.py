# SPDX-License-Identifier: Apache-2.0
"""Paired ACLGraph latency of actual two-scatter and fused LI writers."""

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import torch
import torch_npu

from vllm_ascend.utils import adapt_patch, register_ascend_customop


def percentile(values, pct):
    values = sorted(values)
    p = (len(values) - 1) * pct / 100
    low, high = int(p), min(int(p) + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (p - low)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    adapt_patch()
    adapt_patch(True)
    register_ascend_customop()
    torch_npu.npu.set_device(0)
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "c03_writer", root / "vllm_ascend/ops/triton/fused_indexer_cache_write.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = []
    for rows in (2, 12, 16, 192):
        capacity = 286976
        key = torch.randint(-128, 128, (rows, 128), dtype=torch.int8, device="npu")
        scale = torch.randn(rows, 1, dtype=torch.float16, device="npu")
        slots_cpu = torch.arange(rows, dtype=torch.int32) + 131071
        if rows == 2:
            slots_cpu[1] = 262143
        elif rows in (12, 16):
            slots_cpu[6:12] = torch.arange(6, dtype=torch.int32) + 262143
            if rows == 16:
                slots_cpu[12:] = -1
        slots = slots_cpu.to("npu")
        outputs = {}
        graphs = {}
        stream = torch_npu.npu.Stream()
        for tile in (0, 1, 2, 4, 8):
            key_cache = torch.zeros(capacity, 128, dtype=torch.int8, device="npu")
            scale_cache = torch.zeros(capacity, 1, dtype=torch.float16, device="npu")

            def run(tile=tile, key_cache=key_cache, scale_cache=scale_cache, slots=slots, key=key, scale=scale):
                if tile == 0:
                    torch_npu.npu_scatter_nd_update_(key_cache, slots.view(-1, 1), key)
                    torch_npu.npu_scatter_nd_update_(scale_cache, slots.view(-1, 1), scale)
                else:
                    assert module.fused_indexer_cache_write(key_cache, scale_cache, slots, key, scale, tile)

            with torch_npu.npu.stream(stream):
                for _ in range(5):
                    run()
            stream.synchronize()
            graph = torch_npu.npu.NPUGraph()
            with torch_npu.npu.graph(graph, stream=stream):
                run()
            graphs[tile] = graph
            outputs[tile] = (key_cache, scale_cache)
        for graph in graphs.values():
            for _ in range(10):
                graph.replay()
        torch_npu.npu.synchronize()
        for tile in (1, 2, 4, 8):
            for actual, reference in zip(outputs[tile], outputs[0]):
                torch.testing.assert_close(
                    actual.cpu().view(torch.uint8), reference.cpu().view(torch.uint8), rtol=0, atol=0
                )
        samples = {tile: [] for tile in graphs}
        paired = []
        for round_index in range(args.rounds):
            order = list(graphs) if round_index % 2 == 0 else list(reversed(graphs))
            pair = {}
            for tile in order:
                begin = torch_npu.npu.Event(enable_timing=True)
                end = torch_npu.npu.Event(enable_timing=True)
                begin.record()
                for _ in range(args.iterations):
                    graphs[tile].replay()
                end.record()
                end.synchronize()
                us = begin.elapsed_time(end) * 1000 / args.iterations
                samples[tile].append(us)
                pair[tile] = us
            paired.append(pair)
        summary = {
            str(tile): {
                "mean_us": statistics.fmean(values),
                "p50_us": percentile(values, 50),
                "p90_us": percentile(values, 90),
            }
            for tile, values in samples.items()
        }
        case = {
            "query_rows": rows,
            "valid_rows": 12 if rows == 16 else rows,
            "capacity_rows": capacity,
            "summary": summary,
            "paired_rounds_us": paired,
        }
        cases.append(case)
        print(json.dumps({k: v for k, v in case.items() if k != "paired_rounds_us"}), flush=True)
    result = {
        "scope": "actual writers only; native=tile0, fused=tile1/2/4/8; not full TPOT/ITL",
        "parameters": vars(args) | {"output": str(args.output)},
        "cases": cases,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
