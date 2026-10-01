"""Report measured A/B runs and audit original server token sequences."""

import argparse
import collections
import hashlib
import json
from pathlib import Path


def token_sequences(bench, directory):
    events = read_events(bench, directory)
    return {
        request_id: [token for event in values for token in event["token_ids"]] for request_id, values in events.items()
    }


def read_events(bench, directory):
    events = collections.defaultdict(list)
    for path in (bench / directory).glob("server-itl-*.jsonl"):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            events[event["request_id"]].append(event)
    for values in events.values():
        values.sort(key=lambda event: event["engine_timestamp_s"])
    return events


def percentile(values, percentage):
    values = sorted(values)
    position = (len(values) - 1) * percentage / 100
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def metric_total(path, name):
    return sum(
        float(line.rsplit(" ", 1)[1])
        for line in path.read_text().splitlines()
        if line.startswith(name + "{") or line.startswith(name + " ")
    )


def read_run(run):
    bench = run / "bench"
    result = json.loads((bench / "results.json").read_text())
    itl = json.loads((bench / "itl-results.json").read_text())
    for row in itl["histogram_reconciliation"]:
        assert row["count_matches"] and row["sum_matches"], row
    sequences = token_sequences(bench, "matched-server-events")
    events = read_events(bench, "matched-server-events")
    server_tpot = []
    hashes = collections.Counter()
    for row in itl["requests"]:
        sequence = sequences[row["server_request_id"]]
        assert len(sequence) == row["server_generation_tokens"] == 1024
        hashes[hashlib.sha256(json.dumps(sequence).encode()).hexdigest()] += 1
        outputs = [event for event in events[row["server_request_id"]] if event["num_new_tokens"] > 0]
        first, last = outputs[0]["engine_timestamp_s"], outputs[-1]["engine_timestamp_s"]
        server_tpot.append(
            {
                "index": row["index"],
                "decoder": row["decoder"],
                "server_request_id": row["server_request_id"],
                "first_engine_timestamp_s": first,
                "last_engine_timestamp_s": last,
                "completion_tokens": len(sequence),
                "tpot_ms": (last - first) * 1000 / (len(sequence) - 1),
                "peak_concurrency_coverage": row["peak_concurrency_coverage"],
            }
        )
    warmup = token_sequences(bench, "matched-warmup-server-events")
    assert len(warmup) == 1 and len(next(iter(warmup.values()))) == 16
    summary = result["summary"]
    groups = {
        "all_requests": server_tpot,
        "peak_load_requests": [
            row for row in server_tpot if row["peak_concurrency_coverage"] >= result["parameters"]["coverage_required"]
        ],
    }
    server_tpot_summary = {
        name: {
            "requests": len(rows),
            **{f"p{p}_ms": percentile([row["tpot_ms"] for row in rows], p) for p in (50, 75, 90, 99)},
        }
        for name, rows in groups.items()
    }
    mtp = []
    for decoder in (0, 1):
        before = bench / f"server-metrics/d{decoder}-before.prom"
        after = bench / f"server-metrics/d{decoder}-after.prom"
        delta = {}
        for key, metric in [
            ("accepted_tokens", "vllm:spec_decode_num_accepted_tokens_total"),
            ("drafted_tokens", "vllm:spec_decode_num_draft_tokens_total"),
            ("drafts", "vllm:spec_decode_num_drafts_total"),
        ]:
            delta[key] = metric_total(after, metric) - metric_total(before, metric)
        assert delta["drafted_tokens"] > 0
        delta.update({"decoder": decoder, "accepted_fraction": delta["accepted_tokens"] / delta["drafted_tokens"]})
        mtp.append(delta)
    accepted = sum(row["accepted_tokens"] for row in mtp)
    drafted = sum(row["drafted_tokens"] for row in mtp)
    return {
        "run": run.name,
        "parameters": result["parameters"],
        "prompt_sha256": result["prompt_sha256"],
        "wall_s": result["wall_s"],
        "output_tokens_per_wall_s": len(result["requests"]) * 1024 / result["wall_s"],
        "mtp_accepted_fraction": accepted / drafted,
        "mtp_formal_delta_per_node": mtp,
        "server_tpot_summary": server_tpot_summary,
        "server_tpot_requests": server_tpot,
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
        changes[group]["server_tpot"] = {}
        for percentile in (50, 75, 90, 99):
            key = f"p{percentile}_ms"
            baseline = control["server_tpot_summary"][group][key]
            changed = optimized["server_tpot_summary"][group][key]
            changes[group]["server_tpot"][key] = {
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
        "server_tpot_definition": "Derived from actual first/last positive-token engine events: "
        "(last_engine_timestamp-first_engine_timestamp)/(completion_tokens-1). "
        "Not a built-in vLLM TPOT metric; excludes final SSE/network completion.",
        "mtp_definition": "Formal after-before Prometheus counter deltas, excluding warmup; "
        "accepted draft tokens / drafted tokens; pooled ratio uses summed counters.",
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {"control_wall_s": control["wall_s"], "optimized_wall_s": [run["wall_s"] for run in optimized]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
