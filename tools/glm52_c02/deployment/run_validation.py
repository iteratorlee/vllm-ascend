"""Wait for D startup, then run the fixed probe and two saturation rounds."""

import datetime
import json
import subprocess
import time
import urllib.request
from pathlib import Path


def main():
    root = Path("/public-flash/liyan/glm52-vllm-ascend")
    experiment = root / "experiments/glm53-migration/C02"
    server = root / "runs/20261001-c02-reuse-b1/bench/server-events"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    urls = ["http://172.16.10.183:9900/health", "http://172.16.10.184:9900/health"]
    started = time.monotonic()
    checks = []
    while True:
        states = []
        for url in urls:
            try:
                with opener.open(url, timeout=3) as response:
                    states.append({"url": url, "status": response.status})
            except Exception as exc:
                states.append({"url": url, "error": str(exc)})
        row = {
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "elapsed_s": time.monotonic() - started,
            "states": states,
        }
        checks.append(row)
        (experiment / "validation/startup-health.json").write_text(json.dumps(checks, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        if all(state.get("status") == 200 for state in states):
            break
        if time.monotonic() - started > 7200:
            raise TimeoutError("D services did not become ready")
        time.sleep(45)
    for name, requests, concurrency in [
        ("20261001-c02-probe", 2, 2),
        ("20261001-c02-reuse-b1", 64, 16),
        ("20261001-c02-reuse-b2", 64, 16),
    ]:
        run = root / "runs" / name
        run.mkdir(exist_ok=True)
        config = run / "config"
        config.mkdir(exist_ok=True)
        command = [
            "python3",
            str(root / "bench/bench_saturation.py"),
            "--prompt-file",
            str(root / "bench/prompts/128k-context.txt"),
            "--requests",
            str(requests),
            "--concurrency",
            str(concurrency),
            "--collect-itl",
            "--server-event-dir",
            str(server),
            "--output-dir",
            str(run / "bench"),
        ]
        (config / "benchmark-command.json").write_text(json.dumps(command, indent=2) + "\n")
        print("START " + name, flush=True)
        with (run / "driver.log").open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        subprocess.run(
            [
                "python3",
                str(experiment / "scripts/archive_run.py"),
                "--bench-dir",
                str(run / "bench"),
                "--server-dir",
                str(server),
            ],
            check=True,
        )
        result = json.loads((run / "bench/results.json").read_text())
        print(json.dumps({"run": name, "wall_s": result["wall_s"], "summary": result["summary"]}), flush=True)


if __name__ == "__main__":
    main()
