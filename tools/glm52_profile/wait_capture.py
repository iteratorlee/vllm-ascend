"""Wait for the diagnostic deployment, then collect P/D profiling once."""

import datetime
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def main():
    root = Path("/public-flash/liyan/glm52-vllm-ascend")
    profile = root / "profiling/20261010-c03"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    urls = ["http://172.16.10.181:9081", "http://172.16.10.183:9900", "http://172.16.10.184:9900"]
    start = time.monotonic()
    checks = []
    while True:
        states = []
        for url in urls:
            try:
                with opener.open(url + "/health", timeout=3) as response:
                    states.append({"url": url, "status": response.status})
            except Exception as error:
                states.append({"url": url, "error": str(error)})
        row = {
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "elapsed_s": time.monotonic() - start,
            "states": states,
        }
        checks.append(row)
        temporary = profile / "validation/startup-health.tmp"
        temporary.write_text(json.dumps(checks, indent=2) + "\n")
        temporary.replace(profile / "validation/startup-health.json")
        print(json.dumps(row), flush=True)
        if all(state.get("status") == 200 for state in states):
            break
        if time.monotonic() - start > 10800:
            raise TimeoutError("Profiling deployment did not become ready")
        time.sleep(45)
    print("START P/D profiling", flush=True)
    with (profile / "validation/capture.log").open("w") as output:
        subprocess.run(
            [
                sys.executable,
                str(profile / "scripts/capture_pd.py"),
                "--root",
                str(root),
                "--output",
                str(profile / "workload"),
            ],
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print("P/D profiling workload completed", flush=True)


if __name__ == "__main__":
    main()
