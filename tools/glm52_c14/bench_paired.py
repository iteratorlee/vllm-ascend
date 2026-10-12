#!/usr/bin/env python3
"""Alternating A/B pairs of the actual helper, using identical position updates."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from probe_rope import check_bytes, load_helpers, measure, oracle, positions_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(0)
    bits = torch.randint(
        -32768, 32768, (1048576, 64), dtype=torch.int16, generator=torch.Generator().manual_seed(20261012)
    )
    source = bits.view(torch.bfloat16).to("npu:0")
    functions = {}
    spaces = {}
    for name, enabled in (("baseline", False), ("contiguous", True)):
        namespace, provenance = load_helpers(args.source)
        namespace["_record_cos_and_sin_cache_interleaved"](source, make_contiguous=enabled)
        namespace["_cos_mla"] = torch.empty(192, 1, 1, 64, dtype=torch.bfloat16, device="npu:0")
        namespace["_sin_mla"] = torch.empty_like(namespace["_cos_mla"])
        spaces[name] = namespace
        functions[name] = namespace["get_cos_and_sin_mla"]
    positions = positions_for(12, 1048576, 0).to("npu:0")
    pairs = []
    for iteration in range(20):
        cpu_positions = positions_for(12, 1048576, iteration)
        positions.copy_(cpu_positions.to("npu:0"))
        expected = oracle(bits, cpu_positions)
        order = ("baseline", "contiguous") if iteration % 2 == 0 else ("contiguous", "baseline")
        for use_cache in (False, True) if iteration % 2 == 0 else (True, False):
            pair = {
                "iteration": iteration,
                "use_cache": use_cache,
                "order": list(order),
                "positions": cpu_positions.tolist(),
                "measurements": {},
            }
            for name in order:
                run = functions[name]
                check_bytes(run(positions, use_cache), expected, f"{name}/{iteration}/{use_cache}")
                pair["measurements"][name] = measure(partial(run, positions, use_cache), 100, 1)
            pairs.append(pair)
            print(
                json.dumps(
                    {
                        "iteration": iteration,
                        "use_cache": use_cache,
                        **{n: pair["measurements"][n]["device_us_median"] for n in order},
                    }
                ),
                flush=True,
            )
    args.output.write_text(
        json.dumps(
            {"source": provenance, "rows": 12, "rounds": 20, "calls_per_measurement": 100, "pairs": pairs}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
