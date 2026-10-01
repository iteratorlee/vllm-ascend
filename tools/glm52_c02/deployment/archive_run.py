"""Freeze matched server records without modifying original recorder files."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-dir", type=Path, required=True)
    parser.add_argument("--server-dir", type=Path, required=True)
    args = parser.parse_args()
    results = json.loads((args.bench_dir / "results.json").read_text())
    groups = [("matched-server-events", results["requests"])]
    warmup = args.bench_dir / "warmup.json"
    if warmup.exists():
        groups.append(("matched-warmup-server-events", [json.loads(warmup.read_text())]))
    events = []
    for source in sorted(args.server_dir.glob("server-itl-*.jsonl")):
        events.append((source, source.read_text().splitlines()))
    for name, rows in groups:
        freeze_group(args.bench_dir, name, rows, events)


def freeze_group(bench_dir, name, rows, events):
    assert all(row["error"] is None for row in rows)
    ids = {rid for row in rows for rid in [row["request_id"], *row.get("server_response_ids", [])]}
    destination = bench_dir / name
    destination.mkdir(exist_ok=True)
    files = []
    tokens = 0
    for source, lines in events:
        selected = []
        for line in lines:
            event = json.loads(line)
            rid = event.get("request_id", "")
            if event.get("record_type") == "engine_output" and any(
                rid == candidate or rid.startswith(candidate + "-") for candidate in ids
            ):
                selected.append(line)
                tokens += event["num_new_tokens"]
        target = destination / source.name
        data = ("\n".join(selected) + "\n").encode() if selected else b""
        target.write_bytes(data)
        files.append(
            {
                "file": target.name,
                "records": len(selected),
                "sha256": hashlib.sha256(data).hexdigest(),
                "source": str(source),
            }
        )
    expected = sum(row["usage"]["completion_tokens"] for row in rows)
    assert tokens == expected, (tokens, expected)
    manifest = {
        "selection": "Formal successful request IDs only; original JSONL bytes retained per record",
        "completion_tokens": tokens,
        "files": files,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"group": name, "completion_tokens": tokens, "files": len(files)}))


if __name__ == "__main__":
    main()
