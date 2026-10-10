#!/usr/bin/env python3
"""Reproduce the 2026-10-10 P/D CSV analysis using the Python standard library.

This script does not combine clocks between ranks. Durations and timestamp
intervals are converted to integer nanoseconds before summation / union.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any

NPU_CORES = {"AI_CORE", "AI_VECTOR_CORE", "MIX_AIC", "MIX_AIV"}
RANK_RE = re.compile(r"dp(?P<dp>\d+)_pp(?P<pp>\d+)_tp(?P<tp>\d+)_dcp(?P<dcp>\d+)_ep(?P<ep>\d+)_rank(?P<rank>\d+)_")


def ns(value: str) -> int:
    value = value.strip()
    whole, _, fraction = value.partition(".")
    return int(whole) * 1000 + int((fraction + "000")[:3])


def clean_shape(value: str) -> str:
    return value.strip().strip('"')


def category(row: dict[str, str]) -> str:
    core, name = row["Accelerator Core"], row["Name"]
    if core == "COMMUNICATION":
        return "hccl_envelope" if name.startswith("hcom_") else "hccl_device_task"
    if core == "AI_CPU":
        if "init" in name.lower():
            return "aicpu_init"
        if row["Type"] == "StoreKvBlockMetadata":
            return "aicpu_kv_metadata"
        if any(x in name.lower() for x in ("allgather", "alltoall", "reduce_scatter", "hccl", "hcom")):
            return "aicpu_communication"
        return "aicpu_other"
    if core in NPU_CORES:
        return (
            "npu_moe_dispatch_combine"
            if row["Type"] in {"MoeDistributeDispatchV2", "MoeDistributeCombineV2"}
            else "npu_other"
        )
    return "other"


def union_length(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    intervals.sort()
    total = 0
    start, end = intervals[0]
    for a, b in intervals[1:]:
        if a > end:
            total += end - start
            start, end = a, b
        else:
            end = max(end, b)
    return total + end - start


def new_stats() -> dict[str, Any]:
    return {
        "count": 0,
        "duration_ns": 0,
        "wait_ns": 0,
        "min_ns": None,
        "max_ns": 0,
        "cores": collections.Counter(),
        "states": collections.Counter(),
        "state_duration_ns": collections.Counter(),
        "shapes": collections.Counter(),
        "output_shapes": collections.Counter(),
        "shape_duration_ns": collections.Counter(),
        "dtypes": collections.Counter(),
        "rank_counts": collections.Counter(),
        "rank_duration_ns": collections.Counter(),
    }


def update(stat: dict[str, Any], row: dict[str, str], dur: int, wait: int, rank_key: str) -> None:
    stat["count"] += 1
    stat["duration_ns"] += dur
    stat["wait_ns"] += wait
    stat["min_ns"] = dur if stat["min_ns"] is None else min(stat["min_ns"], dur)
    stat["max_ns"] = max(stat["max_ns"], dur)
    stat["cores"][row["Accelerator Core"]] += 1
    stat["states"][row["OP State"]] += 1
    stat["state_duration_ns"][row["OP State"]] += dur
    shape = clean_shape(row["Input Shapes"])
    stat["shapes"][shape] += 1
    stat["output_shapes"][clean_shape(row["Output Shapes"])] += 1
    stat["shape_duration_ns"][shape] += dur
    stat["dtypes"][row["Input Data Types"]] += 1
    stat["rank_counts"][rank_key] += 1
    stat["rank_duration_ns"][rank_key] += dur


def export_stat(name: str, typ: str, stat: dict[str, Any], total: int) -> dict[str, Any]:
    return {
        "name": name,
        "type": typ,
        "count": stat["count"],
        "duration_ms": stat["duration_ns"] / 1e6,
        "mean_us": stat["duration_ns"] / 1000 / stat["count"],
        "min_us": stat["min_ns"] / 1000,
        "max_us": stat["max_ns"] / 1000,
        "wait_ms": stat["wait_ns"] / 1e6,
        "raw_share_pct": 100 * stat["duration_ns"] / total,
        "cores": dict(stat["cores"]),
        "states": dict(stat["states"]),
        "state_duration_ms": {k: v / 1e6 for k, v in stat["state_duration_ns"].items()},
        "input_shapes": [
            {"shape": s, "count": n, "duration_ms": stat["shape_duration_ns"][s] / 1e6}
            for s, n in stat["shapes"].most_common(5)
        ],
        "input_shape_count": len(stat["shapes"]),
        "output_shapes": [{"shape": s, "count": n} for s, n in stat["output_shapes"].most_common(3)],
        "input_dtypes": dict(stat["dtypes"]),
        "rank_counts": dict(stat["rank_counts"]),
        "rank_duration_ms": {k: v / 1e6 for k, v in stat["rank_duration_ns"].items()},
    }


def analyze_role(root: Path, role: str, manifest: dict[str, dict[str, Any]]) -> dict[str, Any]:
    files = sorted((root / "trace" / role).glob("*/ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
    assert len(files) == 32, (role, len(files))
    kernels = collections.defaultdict(new_stats)
    cats = collections.defaultdict(lambda: {"count": 0, "duration_ns": 0, "wait_ns": 0})
    ranks = []
    hashes = []
    for f in files:
        metadata = {k: int(v) for k, v in RANK_RE.match(f.parent.parent.name).groupdict().items()}
        key = f"dp{metadata['dp']}/pp{metadata['pp']}/tp{metadata['tp']}/ep{metadata['ep']}"
        relative = f.relative_to(root).as_posix()
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        assert sha == manifest[relative]["sha256"], relative
        assert f.stat().st_size == manifest[relative]["bytes"], relative
        hashes.append({"path": relative, "bytes": f.stat().st_size, "sha256": sha})
        intervals = collections.defaultdict(list)
        local = collections.defaultdict(lambda: {"count": 0, "duration_ns": 0, "wait_ns": 0})
        concrete_signatures = collections.Counter()
        envelopes = []
        names = collections.Counter()
        shape_rows = 0
        rows = 0
        lo, hi = None, None
        with f.open(newline="") as handle:
            for row in csv.DictReader(handle):
                start, dur = ns(row["Start Time(us)"]), ns(row["Duration(us)"])
                wait = 0 if row["Wait Time(us)"].strip() in {"N/A", ""} else ns(row["Wait Time(us)"])
                assert dur >= 0 and wait >= 0
                end = start + dur
                lo = start if lo is None else min(lo, start)
                hi = end if hi is None else max(hi, end)
                cat = category(row)
                update(kernels[(row["Name"], row["Type"])], row, dur, wait, key)
                for target in (cats[cat], local[cat]):
                    target["count"] += 1
                    target["duration_ns"] += dur
                    target["wait_ns"] += wait
                intervals["all"].append((start, end))
                intervals[cat].append((start, end))
                if row["Accelerator Core"] in NPU_CORES:
                    intervals["npu"].append((start, end))
                if cat != "hccl_envelope":
                    intervals["without_hccl_envelope"].append((start, end))
                if cat == "hccl_envelope":
                    envelopes.append((row["Type"], start, dur))
                elif cat == "hccl_device_task":
                    concrete_signatures[(row["Type"], start, dur)] += 1
                names[row["Name"]] += 1
                shape_rows += clean_shape(row["Input Shapes"]) not in {"N/A", ""}
                rows += 1
        exact_match = sum(concrete_signatures[s] > 0 for s in envelopes)
        exact_match_ns = sum(s[2] for s in envelopes if concrete_signatures[s] > 0)
        ranks.append(
            {
                "key": key,
                **metadata,
                "source": relative,
                "rows": rows,
                "shape_rows": shape_rows,
                "span_ms": (hi - lo) / 1e6,
                "start_us": lo / 1000,
                "end_us": hi / 1000,
                "sum_duration_ms": sum(v["duration_ns"] for v in local.values()) / 1e6,
                "union_ms": {k: union_length(v) / 1e6 for k, v in intervals.items()},
                "categories": {
                    k: {"count": v["count"], "duration_ms": v["duration_ns"] / 1e6, "wait_ms": v["wait_ns"] / 1e6}
                    for k, v in local.items()
                },
                "envelope_exact_device_matches": exact_match,
                "envelope_exact_device_matches_duration_ms": exact_match_ns / 1e6,
                "selected_counts": {
                    k: v
                    for k, v in names.items()
                    if k
                    in {
                        "SparseFlashAttention",
                        "LightningIndexerQuant",
                        "_fused_indexer_cache_write",
                        "mla_preprocess_0",
                    }
                },
            }
        )
        print(role, key, rows, flush=True)
    total_ns = sum(v["duration_ns"] for v in kernels.values())
    all_kernels = [
        export_stat(name, typ, stat, total_ns)
        for (name, typ), stat in sorted(kernels.items(), key=lambda kv: (-kv[1]["duration_ns"], kv[0]))
    ]
    npu_ns = sum(v["duration_ns"] for k, v in cats.items() if k.startswith("npu_"))
    npu = [
        dict(v, npu_share_pct=100 * v["duration_ms"] * 1e6 / npu_ns)
        for v in all_kernels
        if set(v["cores"]) <= NPU_CORES
    ]
    categories = {
        k: {
            "count": v["count"],
            "duration_ms": v["duration_ns"] / 1e6,
            "wait_ms": v["wait_ns"] / 1e6,
            "raw_share_pct": 100 * v["duration_ns"] / total_ns,
        }
        for k, v in cats.items()
    }
    assert sum(v["count"] for v in all_kernels) == sum(v["rows"] for v in ranks)
    assert sum(v["count"] for v in categories.values()) == sum(v["rows"] for v in ranks)
    groups = {}
    for dp, pp in sorted({(r["dp"], r["pp"]) for r in ranks}):
        rs = [r for r in ranks if (r["dp"], r["pp"]) == (dp, pp)]
        stats = {}
        for metric in ("span_ms", "sum_duration_ms"):
            values = [r[metric] for r in rs]
            stats[metric] = {
                "min": min(values),
                "mean": statistics.mean(values),
                "max": max(values),
                "cv_pct": 100 * statistics.pstdev(values) / statistics.mean(values),
            }
        for metric in ("npu", "all", "hccl_device_task"):
            values = [r["union_ms"].get(metric, 0) for r in rs]
            stats[metric + "_union_ms"] = {
                "min": min(values),
                "mean": statistics.mean(values),
                "max": max(values),
                "cv_pct": 100 * statistics.pstdev(values) / statistics.mean(values),
            }
        values = [sum(v["duration_ms"] for k, v in r["categories"].items() if k.startswith("npu_")) for r in rs]
        stats["npu_sum_ms"] = {
            "min": min(values),
            "mean": statistics.mean(values),
            "max": max(values),
            "cv_pct": 100 * statistics.pstdev(values) / statistics.mean(values),
        }
        groups[f"dp{dp}/pp{pp}"] = stats
    return {
        "role": role,
        "files": hashes,
        "file_count": len(files),
        "rows": sum(r["rows"] for r in ranks),
        "duration_ms": total_ns / 1e6,
        "npu_duration_ms": npu_ns / 1e6,
        "shape_rows": sum(r["shape_rows"] for r in ranks),
        "name_type_groups": len(all_kernels),
        "categories": categories,
        "ranks": ranks,
        "groups": groups,
        "top20_raw": all_kernels[:20],
        "top20_npu": npu[:20],
        "kernels": all_kernels,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("kernel-summary.json"))
    args = parser.parse_args()
    manifest_rows = json.loads((args.artifacts / "analysis/input-manifest.json").read_text())
    manifest = {v["path"]: v for v in manifest_rows}
    data = {
        "schema": 1,
        "group_by": ["Name", "Type"],
        "duration_unit": "ms",
        "classification": (
            "See category() in analyze_kernels.py; NPU table excludes COMMUNICATION and AI_CPU; "
            "MoE dispatch/combine remains annotated as communication-bearing NPU work."
        ),
        "input_manifest_rows": len(manifest_rows),
        "roles": {role: analyze_role(args.artifacts, role, manifest) for role in ("p", "d")},
    }
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
