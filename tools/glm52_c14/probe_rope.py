#!/usr/bin/env python3
"""C14: exact live RoPE helper, byte oracle and eager/graph microbenchmark.

Only run on an explicitly leased, isolated NPU. No model/server is started.
The helper bodies are compiled verbatim from the supplied source AST; heavy
vLLM imports and unrelated globals are excluded from this isolated probe.
"""

import argparse
import ast
import gc
import hashlib
import json
import statistics
import time
from functools import partial
from pathlib import Path

import torch
import torch_npu


def load_helpers(path):
    source = path.read_text()
    names = {"_record_cos_and_sin_cache_interleaved", "get_cos_and_sin_mla"}
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    namespace = {"torch": torch, "_cos_cache": None, "_sin_cache": None}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, {
        "path": str(path),
        "sha256": hashlib.sha256(source.encode()).hexdigest(),
        "functions": {n.name: n.lineno for n in nodes},
    }


def snapshot():
    torch.npu.synchronize()
    free_bytes, total_bytes = torch.npu.mem_get_info()
    return {
        "allocated": torch.npu.memory_allocated(),
        "reserved": torch.npu.memory_reserved(),
        "max_allocated": torch.npu.max_memory_allocated(),
        "max_reserved": torch.npu.max_memory_reserved(),
        "hbm_free": free_bytes,
        "hbm_total": total_bytes,
    }


def describe(tensor):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "storage_offset": tensor.storage_offset(),
        "data_ptr": tensor.data_ptr(),
        "storage_ptr": tensor.untyped_storage().data_ptr(),
        "storage_bytes": tensor.untyped_storage().nbytes(),
        "contiguous": tensor.is_contiguous(),
    }


def positions_for(rows, cache_rows, step, dtype=torch.int64):
    anchors = [cache_rows - 1, 0, 131072, 131071, 131073, 1, 524287, 7, 0, cache_rows - 1, 65535, 999999]
    values = [
        anchors[(i + step) % len(anchors)] if i < 24 else (i * 49999 + step * 7919) % cache_rows for i in range(rows)
    ]
    return torch.tensor(values, dtype=dtype)


def oracle(raw_cpu, positions):
    selected = raw_cpu[positions.long()].view(-1, 2, 32).repeat(1, 1, 2)
    return selected[:, 0, :].reshape(-1, 1, 1, 64), selected[:, 1, :].reshape(-1, 1, 1, 64)


def check_bytes(actual, expected, label):
    for index, (got, want) in enumerate(zip(actual, expected)):
        cpu_bits = got.cpu().contiguous().view(torch.int16)
        assert cpu_bits.shape == want.shape, (label, cpu_bits.shape, want.shape)
        if not torch.equal(cpu_bits, want):
            mismatch = (cpu_bits != want).nonzero()
            raise AssertionError(f"{label}: output={index}, mismatches={len(mismatch)}, first={mismatch[:8].tolist()}")


def measure(fn, repeats, rounds):
    for _ in range(10):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(rounds):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        torch.npu.synchronize()
        before = time.perf_counter_ns()
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        torch.npu.synchronize()
        samples.append(
            {
                "device_us": start.elapsed_time(end) * 1000 / repeats,
                "wall_us": (time.perf_counter_ns() - before) / 1000 / repeats,
            }
        )
    return {
        "repeats": repeats,
        "rounds": rounds,
        "samples": samples,
        "device_us_median": statistics.median(x["device_us"] for x in samples),
        "wall_us_median": statistics.median(x["wall_us"] for x in samples),
    }


def make_variant(name, namespace, raw_npu):
    if name == "baseline" or name == "contiguous":
        return namespace["get_cos_and_sin_mla"]

    def finish(cos, sin, use_cache):
        cos, sin = cos[:, None, None, :], sin[:, None, None, :]
        if use_cache:
            rows = cos.shape[0]
            namespace["_cos_mla"][:rows].copy_(cos)
            namespace["_sin_mla"][:rows].copy_(sin)
            return namespace["_cos_mla"][:rows], namespace["_sin_mla"][:rows]
        return cos, sin

    if name == "index_select":

        def run(positions, use_cache=False):
            return finish(
                namespace["_cos_cache"].index_select(0, positions),
                namespace["_sin_cache"].index_select(0, positions),
                use_cache,
            )

        return run
    if name == "raw_gather_repeat":

        def run(positions, use_cache=False):
            rows = positions.shape[0]
            selected = raw_npu.index_select(0, positions).view(rows, 2, 32).repeat(1, 1, 2)
            return finish(selected[:, 0, :], selected[:, 1, :], use_cache)

        return run
    if name == "triton_stride":
        from stride_gather import gather

        def run(positions, use_cache=False):
            return finish(*gather(namespace["_cos_cache"], namespace["_sin_cache"], positions), use_cache)

        return run
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--cache-rows", type=int, default=1048576)
    parser.add_argument("--rows", default="0,1,2,12,192")
    parser.add_argument("--variants", default="baseline,contiguous,index_select,raw_gather_repeat,triton_stride")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    assert args.cache_rows == 1048576
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(args.device)
    torch.manual_seed(20261012)
    raw_cpu = torch.randint(-32768, 32768, (args.cache_rows, 64), dtype=torch.int16)
    # Include every BF16 encoding, including signed zero, NaNs, infinities,
    # and subnormals, without floating-point conversion.
    raw_cpu[:1024].copy_(torch.arange(65536, dtype=torch.int32).to(torch.int16).reshape(1024, 64))
    raw_npu = raw_cpu.view(torch.bfloat16).to(args.device)
    namespace, source = load_helpers(args.source)
    report = {
        "source": source,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device": args.device,
        "device_name": torch.npu.get_device_name(),
        "cache_rows": args.cache_rows,
        "raw": describe(raw_npu),
        "variants": [],
        "timing_note": (
            "NPU event duration around repeated eager launches includes queue starvation; graph reported separately."
        ),
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for name in args.variants.split(","):
        namespace["_cos_cache"] = namespace["_sin_cache"] = None
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()
        before = snapshot()
        begin = time.perf_counter_ns()
        namespace["_record_cos_and_sin_cache_interleaved"](raw_npu)
        if name == "contiguous":
            namespace["_cos_cache"] = namespace["_cos_cache"].contiguous()
            namespace["_sin_cache"] = namespace["_sin_cache"].contiguous()
        torch.npu.synchronize()
        init_wall_ms = (time.perf_counter_ns() - begin) / 1e6
        after = snapshot()
        entry = {
            "name": name,
            "init_wall_ms": init_wall_ms,
            "memory_before": before,
            "memory_after": after,
            "cos": describe(namespace["_cos_cache"]),
            "sin": describe(namespace["_sin_cache"]),
            "cases": [],
        }
        report["variants"].append(entry)
        namespace["_cos_mla"] = torch.empty((192, 1, 1, 64), dtype=torch.bfloat16, device=args.device)
        namespace["_sin_mla"] = torch.empty_like(namespace["_cos_mla"])
        run = make_variant(name, namespace, raw_npu)
        all_patterns = torch.arange(1024, dtype=torch.int64)
        check_bytes(run(all_patterns.to(args.device)), oracle(raw_cpu, all_patterns), name + "/all65536bits")
        entry["all_65536_bf16_encodings"] = "byte_equal"
        save()
        for rows in (int(x) for x in args.rows.split(",")):
            pos_cpu = positions_for(rows, args.cache_rows, 0)
            positions = pos_cpu.to(args.device)
            case = {"rows": rows, "positions": pos_cpu.tolist(), "correctness": []}
            entry["cases"].append(case)
            for dtype in (torch.int32, torch.int64):
                for step in range(5):
                    cpu = positions_for(rows, args.cache_rows, step, dtype)
                    device_positions = cpu.to(args.device)
                    for use_cache in (False, True):
                        check_bytes(
                            run(device_positions, use_cache),
                            oracle(raw_cpu, cpu),
                            f"{name}/{rows}/{dtype}/{step}/{use_cache}",
                        )
                case["correctness"].append(
                    {"positions_dtype": str(dtype), "updates": 5, "both_cache_modes": "byte_equal"}
                )
            for use_cache in (False, True):
                case[f"eager_cache_{use_cache}"] = measure(
                    partial(run, positions, use_cache), args.repeats, args.rounds
                )
            if rows:
                # Capture the helper itself, then replay after in-place position
                # updates. These are separate from the real runner-style test below.
                graph = torch.npu.NPUGraph()
                stream = torch.npu.Stream()
                with torch.npu.stream(stream):
                    run(positions, True)
                stream.synchronize()
                with torch.npu.graph(graph, stream=stream):
                    captured_output = run(positions, True)
                addresses = [x.data_ptr() for x in captured_output]
                for step in range(5):
                    cpu = positions_for(rows, args.cache_rows, step)
                    positions.copy_(cpu.to(args.device))
                    graph.replay()
                    torch.npu.synchronize()
                    check_bytes(captured_output, oracle(raw_cpu, cpu), f"{name}/{rows}/graph/{step}")
                    assert addresses == [x.data_ptr() for x in captured_output]
                case["graph_helper_cache_true"] = measure(graph.replay, args.repeats, args.rounds)
                case["graph_helper_addresses"] = addresses
                graph.reset()
                del graph, captured_output, stream
                torch.npu.synchronize()

                # Runtime topology: eager helper writes preallocated MLA buffers,
                # while the target graph consumes the same addresses on replay.
                buffers = run(positions, True)
                addresses = [x.data_ptr() for x in buffers]
                sinks = [torch.empty_like(x) for x in buffers]
                graph = torch.npu.NPUGraph()
                stream = torch.npu.Stream()
                with torch.npu.graph(graph, stream=stream):
                    sinks[0].copy_(buffers[0])
                    sinks[1].copy_(buffers[1])
                for step in range(5):
                    cpu = positions_for(rows, args.cache_rows, step + 1)
                    positions.copy_(cpu.to(args.device))
                    result = run(positions, True)
                    assert addresses == [x.data_ptr() for x in result]
                    graph.replay()
                    torch.npu.synchronize()
                    check_bytes(sinks, oracle(raw_cpu, cpu), f"{name}/{rows}/target_graph/{step}")
                case["target_graph_buffer_updates"] = {"updates": 5, "byte_equal": True, "addresses": addresses}
                graph.reset()
                del graph, stream, buffers, sinks, result
                gc.collect()
                torch.npu.empty_cache()
            save()
            print(
                json.dumps(
                    {
                        "variant": name,
                        "rows": rows,
                        "eager_us": case["eager_cache_True"]["device_us_median"],
                        "graph_us": case.get("graph_helper_cache_true", {}).get("device_us_median"),
                    }
                ),
                flush=True,
            )
        entry["memory_final"] = snapshot()
        namespace["_cos_cache"] = namespace["_sin_cache"] = None
        namespace["_cos_mla"] = namespace["_sin_mla"] = None
        del run
        gc.collect()
        torch.npu.empty_cache()
        save()
    print(str(args.output), flush=True)


if __name__ == "__main__":
    main()
