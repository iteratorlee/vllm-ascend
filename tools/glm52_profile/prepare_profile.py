#!/usr/bin/env python3
"""Create private profiling launch templates without changing running services."""

import argparse
import hashlib
import json
import shlex
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--c03", choices=("on", "off"), required=True)
    args = parser.parse_args()
    scripts = args.output / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    for source in (args.root / "scripts").iterdir():
        if source.is_file():
            shutil.copy2(source, scripts / source.name)
    for name in ("decode-0.sh", "decode-1.sh", "config.sh", "start_role.sh", "launch_online_dp.py"):
        shutil.copy2(args.root / "experiments/glm53-migration/C03/scripts" / name, scripts / name)
    manifest = {"scope": "profiling only; not a formal C03 comparison", "c03": args.c03, "files": []}
    for name in ("prefill-0.sh", "prefill-1.sh", "decode-0.sh", "decode-1.sh"):
        path = scripts / name
        before = path.read_text()
        data = before
        if name.startswith("decode") and args.c03 == "off":
            old = '"enable_fused_indexer_cache_write": true'
            assert data.count(old) == 1
            data = data.replace(old, '"enable_fused_indexer_cache_write": false')
        side = "p" if name.startswith("prefill") else "d"
        profile = {
            "profiler": "torch",
            "torch_profiler_dir": str(args.output / "trace" / side),
            "torch_profiler_with_stack": False,
            "torch_profiler_with_memory": False,
            "ignore_frontend": True,
            "capture_torch_profiler": False,
            "max_iterations": 0,
        }
        needle = 'nohup vllm serve "$MODEL_PATH" \\\n'
        assert data.count(needle) == 1, name
        prefix = "PROFILER_CONFIG=" + shlex.quote(json.dumps(profile, separators=(",", ":"))) + "\n\n"
        data = data.replace(needle, prefix + needle + '    --profiler-config "$PROFILER_CONFIG" \\\n')
        path.write_text(data)
        manifest["files"].append(
            {
                "name": name,
                "source_sha256": hashlib.sha256(before.encode()).hexdigest(),
                "profile_sha256": hashlib.sha256(data.encode()).hexdigest(),
                "profiler_config": profile,
            }
        )
    (args.output / "config").mkdir(exist_ok=True)
    (args.output / "config/template-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
