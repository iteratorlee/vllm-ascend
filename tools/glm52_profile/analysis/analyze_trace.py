#!/usr/bin/env python3
"""Validate graph/non-graph observations against the two actual D tp0 traces."""

import argparse
import collections
import json
from pathlib import Path

SELECTED = {
    "LightningIndexerQuant",
    "SparseFlashAttention",
    "_fused_indexer_cache_write",
    "mla_preprocess_0",
    "aclnnIndex_SliceAiCore_Slice",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("trace-evidence.json"))
    args = parser.parse_args()
    result = {"scope": "Only the two exported D tp0 representative traces; no cross-host clock merge.", "ranks": []}
    for f in sorted((args.artifacts / "trace/d").glob("*tp0_*/ASCEND_PROFILER_OUTPUT/trace_view.json")):
        data = json.loads(f.read_text())
        device = [e for e in data if e.get("ph") == "X" and "Task Type" in e.get("args", {})]
        launches = sorted((e for e in device if e.get("name") == "MODEL_EXECUTE"), key=lambda e: float(e["ts"]))
        selected = [e for e in device if e.get("name") in SELECTED]
        groups = collections.defaultdict(lambda: {"count": 0, "duration_us": 0.0})
        for e in selected:
            a = e["args"]
            key = (e["name"], a["Model Id"], a["Physic Stream Id"], a["Task Type"])
            groups[key]["count"] += 1
            groups[key]["duration_us"] += float(e["dur"])
        windows = []
        for i, launch in enumerate(launches):
            lo = float(launch["ts"])
            hi = float(launches[i + 1]["ts"]) if i + 1 < len(launches) else float("inf")
            non_graph = [e for e in selected if e["args"]["Model Id"] == 4294967295 and lo <= float(e["ts"]) < hi]
            windows.append(
                {
                    "index": i,
                    "model_execute_ts_us": launch["ts"],
                    "non_graph_selected_counts": dict(collections.Counter(e["name"] for e in non_graph)),
                    "non_graph_li_examples": [e for e in non_graph if e["name"] == "LightningIndexerQuant"],
                }
            )
        index_slices = sorted(
            (e for e in selected if e["name"] == "aclnnIndex_SliceAiCore_Slice"), key=lambda e: float(e["ts"])
        )
        first_slice = index_slices[0]
        # Preserve local event context, rather than invent a Python/module owner.
        context = [
            e
            for e in data
            if e.get("ph") == "X"
            and e.get("cat") == "cpu_op"
            and abs(float(e["ts"]) - float(first_slice["ts"])) < 2000
            and any(s in e.get("name", "") for s in ("index", "Index", "slice", "contiguous"))
        ]
        cpu = [e for e in data if e.get("cat") == "cpu_op"]
        rec = {
            "source": f.relative_to(args.artifacts).as_posix(),
            "event_count": len(data),
            "model_execute_count": len(launches),
            "model_execute_events": launches,
            "device_selected": [
                {"name": k[0], "model_id": k[1], "stream": k[2], "task_type": k[3], **v}
                for k, v in sorted(groups.items())
            ],
            "cpu_selected": {
                n: sum(e.get("name") == n for e in cpu)
                for n in (
                    "_C_ascend::npu_lightning_indexer_quant",
                    "aclnnLightningIndexerQuant",
                    "_fused_indexer_cache_write",
                )
            },
            "model_execute_windows": windows,
            "index_slice_total_count": len(index_slices),
            "index_slice_first_example": first_slice,
            "index_slice_local_cpu_context": context,
        }
        assert len(launches) == 17
        assert all(w["non_graph_selected_counts"]["LightningIndexerQuant"] == 5 for w in windows)
        assert all(w["non_graph_selected_counts"]["_fused_indexer_cache_write"] == 5 for w in windows)
        assert all(w["non_graph_selected_counts"]["SparseFlashAttention"] == 5 for w in windows)
        assert len(index_slices) == 204
        result["ranks"].append(rec)
        print(f.parent.parent.name, "17 graph launches; 5 non-graph LI/SFA/writer per launch; 204 Index Slice")
        del data, device, selected, cpu
    out = args.output
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("Wrote", out)


if __name__ == "__main__":
    main()
