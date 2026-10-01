"""Report measured A/B runs and audit original server token sequences."""

import argparse
import collections
import hashlib
import json
from pathlib import Path


def token_sequences(bench, directory):
    events = collections.defaultdict(list)
    for path in (bench / directory).glob("server-itl-*.jsonl"):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            events[event["request_id"]].append(event)
    result = {}
    for request_id, values in events.items():
        values.sort(key=lambda event: event["engine_timestamp_s"])
        result[request_id] = [token for event in values for token in event["token_ids"]]
    return result


def read_run(run):
    bench = run / "bench"
    result = json.loads((bench / "results.json").read_text())
    itl = json.loads((bench / "itl-results.json").read_text())
    for row in itl["histogram_reconciliation"]:
        assert row["count_matches"] and row["sum_matches"], row
    sequences = token_sequences(bench, "matched-server-events")
    hashes = collections.Counter()
    for row in itl["requests"]:
        sequence = sequences[row["server_request_id"]]
        assert len(sequence) == row["server_generation_tokens"] == 1024
        hashes[hashlib.sha256(json.dumps(sequence).encode()).hexdigest()] += 1
    warmup = token_sequences(bench, "matched-warmup-server-events")
    assert len(warmup) == 1 and len(next(iter(warmup.values()))) == 16
    summary = result["summary"]
    accepted = sum(summary["accepted_delta_per_node"])
    drafted = sum(summary["drafted_delta_per_node"])
    return {
        "run": run.name,
        "parameters": result["parameters"],
        "prompt_sha256": result["prompt_sha256"],
        "wall_s": result["wall_s"],
        "output_tokens_per_wall_s": len(result["requests"]) * 1024 / result["wall_s"],
        "mtp_accepted_fraction": accepted / drafted,
        "summary": summary,
        "server_itl_reconciliation": itl["histogram_reconciliation"],
        "unique_full_output_sequences": len(hashes),
        "warmup_token_ids": next(iter(warmup.values())),
    }


def compare(control, optimized):
    assert control["prompt_sha256"] == optimized["prompt_sha256"]
    for parameter in ("prompt_tokens", "output_tokens", "requests", "concurrency", "coverage_required"):
        assert control["parameters"][parameter] == optimized["parameters"][parameter], parameter
    changes = {}
    for group in ("all_requests", "peak_load_requests"):
        changes[group] = {}
        for percentile in (50, 75, 90, 99):
            key = f"tpot_p{percentile}_ms"
            baseline = control["summary"][group][key]
            changed = optimized["summary"][group][key]
            changes[group][key] = {
                "control": baseline,
                "optimized": changed,
                "reduction_pct": (baseline - changed) / baseline * 100,
            }
        for side in ("server_itl", "client_itl"):
            changes[group][side] = {}
            for percentile in (50, 75, 90, 99):
                key = f"p{percentile}_ms"
                baseline = control["summary"]["itl"][group][side][key]
                changed = optimized["summary"]["itl"][group][side][key]
                changes[group][side][key] = {
                    "control": baseline,
                    "optimized": changed,
                    "reduction_pct": (baseline - changed) / baseline * 100,
                }
    return {
        "run": optimized["run"],
        "warmup_16_tokens_equal": control["warmup_token_ids"] == optimized["warmup_token_ids"],
        "changes": changes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    control = read_run(args.control)
    optimized = [read_run(path) for path in args.optimized]
    result = {
        "control": control,
        "optimized": optimized,
        "comparisons": [compare(control, run) for run in optimized],
        "limits": "No synthetic token ITL or confidence claim; each round kept separate. "
        "Pooled ITL events are correlated across requests in one engine step.",
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {"control_wall_s": control["wall_s"], "optimized_wall_s": [run["wall_s"] for run in optimized]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
