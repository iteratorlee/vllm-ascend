#!/usr/bin/env python3
"""Alternate warmed actual initializer modes and measure storage/allocator peaks."""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from probe_rope import describe, load_helpers, snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(0)
    source = torch.zeros(1048576, 64, dtype=torch.bfloat16, device="npu:0")
    namespace, provenance = load_helpers(args.source)
    samples = []
    for iteration in range(9):
        for enabled in (False, True) if iteration % 2 == 0 else (True, False):
            namespace["_cos_cache"] = namespace["_sin_cache"] = None
            gc.collect()
            torch.npu.empty_cache()
            torch.npu.reset_peak_memory_stats()
            before = snapshot()
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            begin = time.perf_counter_ns()
            start.record()
            namespace["_record_cos_and_sin_cache_interleaved"](source, make_contiguous=enabled)
            end.record()
            torch.npu.synchronize()
            wall_ms = (time.perf_counter_ns() - begin) / 1e6
            after = snapshot()
            # Diagnostic only; runtime initialization does not call empty_cache.
            torch.npu.empty_cache()
            after_trim = snapshot()
            if iteration:
                samples.append(
                    {
                        "iteration": iteration,
                        "make_contiguous": enabled,
                        "wall_ms": wall_ms,
                        "device_ms": start.elapsed_time(end),
                        "before": before,
                        "after": after,
                        "after_trim": after_trim,
                        "cos": describe(namespace["_cos_cache"]),
                        "sin": describe(namespace["_sin_cache"]),
                    }
                )
    args.output.write_text(
        json.dumps({"source": provenance, "warmup_iterations": 1, "samples": samples}, indent=2) + "\n"
    )
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
