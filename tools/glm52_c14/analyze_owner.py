#!/usr/bin/env python3
"""Join NPU Slice -> torch_to_npu flow -> aten::index -> actual helper label."""

import argparse
import collections
import csv
import json
from pathlib import Path


def analyze(directory):
    kernel_path = next(directory.rglob("kernel_details.csv"))
    rows = list(csv.DictReader(kernel_path.open()))
    events = json.loads(kernel_path.with_name("trace_view.json").read_text())
    groups = collections.defaultdict(lambda: {"count": 0, "duration_us": 0.0, "shapes": []})
    for row in rows:
        group = groups[row["Name"]]
        group["count"] += 1
        group["duration_us"] += float(row["Duration(us)"])
        shape = {key: row[key] for key in ("Input Shapes", "Output Shapes", "Input Data Types", "Output Data Types")}
        if shape not in group["shapes"]:
            group["shapes"].append(shape)
    labels = [e for e in events if e.get("ph") == "X" and e.get("name", "").startswith("C14/")]
    slices = [e for e in events if e.get("name") == "aclnnIndex_SliceAiCore_Slice"]
    evidence = []
    for event in slices:
        ends = [
            e
            for e in events
            if e.get("ph") == "f"
            and e.get("cat") == "async_npu"
            and e.get("ts") == event["ts"]
            and e.get("pid") == event["pid"]
            and e.get("tid") == event["tid"]
        ]
        assert len(ends) == 1, ends
        starts = [
            e for e in events if e.get("ph") == "s" and e.get("cat") == "async_npu" and e.get("id") == ends[0]["id"]
        ]
        assert len(starts) == 1, starts
        start = starts[0]

        def encloses(e, start=start):
            return (
                e.get("pid") == start["pid"]
                and e.get("tid") == start["tid"]
                and (float(e["ts"]) <= float(start["ts"]) <= float(e["ts"]) + float(e.get("dur", 0)))
            )

        indices = [e for e in events if e.get("name") == "aten::index" and encloses(e)]
        owners = [e for e in labels if encloses(e)]
        assert len(indices) == len(owners) == 1, (indices, owners)
        launches = [
            e
            for e in events
            if e.get("name") == "Node@launch"
            and e.get("args", {}).get("connection_id") == event["args"]["connection_id"]
        ]
        evidence.append(
            {
                "device_slice": event,
                "launch_connection": launches,
                "flow_end": ends[0],
                "flow_start": start,
                "aten_index": indices[0],
                "helper_label": owners[0],
            }
        )
    return {
        "kernel_csv": str(kernel_path),
        "groups": dict(groups),
        "helper_calls": len(labels),
        "slice_count": len(slices),
        "slice_owner_evidence": evidence,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    data = {name: analyze(args.artifacts / ("owner-" + name)) for name in ("baseline", "contiguous")}
    assert data["baseline"]["slice_count"] == 8
    assert data["baseline"]["helper_calls"] == data["contiguous"]["helper_calls"] == 4
    assert data["contiguous"]["slice_count"] == 0
    output = args.artifacts / "owner-evidence.json"
    output.write_text(json.dumps(data, indent=2) + "\n")
    print("Verified 8 baseline Slice tasks, each linked to aten::index inside the actual helper; 0 contiguous Slice.")


if __name__ == "__main__":
    main()
