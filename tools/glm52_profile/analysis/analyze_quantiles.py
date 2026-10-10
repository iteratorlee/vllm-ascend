#!/usr/bin/env python3
"""Kernel duration quantiles for the union of each role's two Top20 lists."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path


def quantile(values: list[int], probability: float) -> float:
    """Linear interpolation (R7), in microseconds, on sorted nanoseconds."""
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return (values[lower] + (values[upper] - values[lower]) * (position - lower)) / 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parents[1]
    parser.add_argument("--artifacts", type=Path, default=base / "artifacts")
    parser.add_argument("--summary", type=Path, default=Path(__file__).with_name("kernel-summary.json"))
    parser.add_argument("--output-dir", type=Path, default=base / "artifacts/validation")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    records: list[dict] = []
    for role in ("p", "d"):
        kernels = summary["roles"][role]
        targets = {
            (kernel["name"], kernel["type"]) for table in ("top20_raw", "top20_npu") for kernel in kernels[table]
        }
        samples: dict[tuple[str, str], list[int]] = defaultdict(list)
        files = sorted((args.artifacts / "trace" / role).glob("*/ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
        assert len(files) == 32
        for file in files:
            with file.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    key = (row["Name"], row["Type"])
                    if key in targets:
                        samples[key].append(int(Decimal(row["Duration(us)"].strip()) * 1000))
        for name, kind in sorted(targets):
            values = sorted(samples[(name, kind)])
            original = next(k for k in kernels["kernels"] if (k["name"], k["type"]) == (name, kind))
            assert len(values) == original["count"]
            assert abs(sum(values) / 1e6 - original["duration_ms"]) < 1e-9
            record = {"role": role, "name": name, "type": kind, "count": len(values)}
            record.update({f"p{percent}_us": quantile(values, percent / 100) for percent in (50, 75, 90, 99)})
            records.append(record)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "unit": "microseconds",
        "method": "R7 linear interpolation; all ranks and observed shapes pooled per Name+Type",
        "scope": "Union of Top20 raw and Top20 NPU; profiler duration is not server/client ITL",
        "records": records,
    }
    (args.output_dir / "kernel-duration-quantiles.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (args.output_dir / "kernel-duration-quantiles.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Verified and wrote duration quantiles for {len(records)} Name+Type groups")


if __name__ == "__main__":
    main()
