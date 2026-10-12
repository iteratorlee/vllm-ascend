#!/usr/bin/env python3
"""Isolated owner profile; never mixes profiler startup with graph teardown."""

import argparse
from pathlib import Path

import torch
import torch_npu
from probe_rope import load_helpers, make_variant, positions_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(0)
    source = torch.zeros((1048576, 64), dtype=torch.bfloat16, device="npu:0")
    namespace, _ = load_helpers(args.source)
    namespace["_record_cos_and_sin_cache_interleaved"](source)
    if args.variant == "contiguous":
        namespace["_cos_cache"] = namespace["_cos_cache"].contiguous()
        namespace["_sin_cache"] = namespace["_sin_cache"].contiguous()
    namespace["_cos_mla"] = torch.empty((192, 1, 1, 64), dtype=torch.bfloat16, device="npu:0")
    namespace["_sin_mla"] = torch.empty_like(namespace["_cos_mla"])
    positions = positions_for(12, 1048576, 0).to("npu:0")
    run = make_variant(args.variant, namespace, source)
    for _ in range(10):
        run(positions, True)
    torch.npu.synchronize()
    args.output.mkdir(parents=True, exist_ok=True)
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
        record_shapes=True,
        with_stack=False,
        with_modules=False,
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.output)),
    ) as profiler:
        for _ in range(4):
            for cache in (False, True):
                with torch.profiler.record_function(f"C14/{args.variant}/get_cos_and_sin_mla/use_cache={cache}"):
                    run(positions, cache)
            torch.npu.synchronize()
            profiler.step()
    print("profile complete", args.output, flush=True)


if __name__ == "__main__":
    main()
