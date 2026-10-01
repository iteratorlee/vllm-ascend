"""Paired NPU ACLGraph timings for the changed LI pre/post subpath.

The LI consumer is a weights sink. These times are not full decode steps or
TPOT; the deployment benchmark measures those separately.
"""

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import pytest
import torch
import torch_npu

from vllm_ascend.utils import adapt_patch, register_ascend_customop


def percentile(values, pct):
    values = sorted(values)
    index = (len(values) - 1) * pct / 100
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    # Standalone execution needs the registration normally done by conftest.
    adapt_patch()
    adapt_patch(True)
    register_ascend_customop()
    torch_npu.npu.set_device(0)
    root = Path(__file__).resolve().parents[2]
    test = root / "tests/e2e/nightly/test_sfa_indexer_projection_reuse.py"
    spec = importlib.util.spec_from_file_location("c02_npu_case", test)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = []
    with pytest.MonkeyPatch.context() as monkeypatch:
        for rows in (2, 12):
            impl, x, q, run = module.make_npu_case(monkeypatch, rows)
            graphs = {}
            outputs = {}
            calls = {}
            original = impl.wk_weights_proj
            counter = [0]

            def counted(value, counter=counter, original=original):
                counter[0] += 1
                return original(value)

            impl.wk_weights_proj = counted
            stream = torch_npu.npu.Stream()
            for enabled in (False, True):
                impl.reuse_indexer_projection = enabled
                with torch_npu.npu.stream(stream):
                    for _ in range(5):
                        run()
                stream.synchronize()
                graph = torch_npu.npu.NPUGraph()
                counter[0] = 0
                with torch_npu.npu.graph(graph, stream=stream):
                    outputs[enabled] = run()
                calls[str(enabled)] = counter[0]
                graphs[enabled] = graph
            for graph in graphs.values():
                for _ in range(10):
                    graph.replay()
            torch_npu.npu.synchronize()
            for got, want in zip(outputs[True], outputs[False]):
                torch.testing.assert_close(got, want, rtol=0, atol=0)
            samples = {False: [], True: []}
            paired = []
            for round_index in range(args.rounds):
                row = {}
                order = (False, True) if round_index % 2 == 0 else (True, False)
                for enabled in order:
                    begin = torch_npu.npu.Event(enable_timing=True)
                    end = torch_npu.npu.Event(enable_timing=True)
                    begin.record()
                    for _ in range(args.iterations):
                        graphs[enabled].replay()
                    end.record()
                    end.synchronize()
                    us = begin.elapsed_time(end) * 1000 / args.iterations
                    samples[enabled].append(us)
                    row[str(enabled)] = us
                paired.append(row)
            summary = {}
            for enabled in (False, True):
                values = samples[enabled]
                summary[str(enabled)] = {
                    "mean_us": statistics.fmean(values),
                    "p50_us": percentile(values, 50),
                    "p90_us": percentile(values, 90),
                }
            case = {
                "query_rows": rows,
                "capture_projection_calls": calls,
                "summary": summary,
                "paired_rounds_us": paired,
            }
            cases.append(case)
            print(json.dumps({k: v for k, v in case.items() if k != "paired_rounds_us"}), flush=True)
    result = {
        "scope": "actual SFA methods; native LI replaced by weights sink; not full TPOT",
        "parameters": {"rounds": args.rounds, "iterations": args.iterations},
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
