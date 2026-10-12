#!/usr/bin/env python3
"""One C16 variant/mode/rank per subprocess after matching paired evidence."""

import argparse
import json
from pathlib import Path

from common import PREPARE_SHA256, save, sha, source_hashes
from probe import Captured, Probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-npu", action="store_true")
    parser.add_argument("--lease", required=True)
    parser.add_argument("--prepare-source", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("/public-flash/models/GLM-5.2-w8a8")
    )
    parser.add_argument("--validated-result", type=Path, required=True)
    parser.add_argument("--variant", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--mode", choices=("eager", "graph"), required=True)
    parser.add_argument("--rank", type=int, choices=range(16), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.run_npu or not args.lease.strip():
        parser.error("Explicit root-issued lease and --run-npu required")
    validated = json.loads(args.validated_result.read_text())
    if (
        validated.get("status")
        != "single_device_oracles_and_pairs_complete_EP32_pending"
    ):
        parser.error("Matching byte, native-consumer graph, and paired gates required")
    if validated.get("tools_sha256") != source_hashes(Path(__file__).parent):
        parser.error("Tools differ from validated result")
    if (
        sha(args.prepare_source) != PREPARE_SHA256
        or validated.get("prepare_sha256") != PREPARE_SHA256
    ):
        parser.error("Original prepare source differs")
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import torch_npu

    import vllm_ascend.vllm_ascend_C  # noqa: F401

    torch.npu.set_device(0)
    assert torch.npu.device_count() == 1
    assert torch.npu.mem_get_info()[0] >= 1024**3
    probe = Probe(torch, torch_npu)
    weight, bias, provenance = probe.checkpoint_gate(args.checkpoint)
    if provenance != validated.get("checkpoint_gate"):
        raise RuntimeError("Checkpoint gate provenance differs from paired evidence")
    inputs = probe.finite_inputs(12, weight, 7)
    functions = probe.functions(inputs, 16, args.rank, "native_routing", bias)
    fn = functions[args.variant]
    run = Captured(torch, fn, 1) if args.mode == "graph" else fn
    probe.check(run(), functions["baseline"](), "profile precheck")
    save(
        args.output / "scope.json",
        dict(
            lease=args.lease,
            tools_sha256=source_hashes(Path(__file__).parent),
            variant=args.variant,
            mode=args.mode,
            calls=20,
            rows=12,
            padded=16,
            rank=args.rank,
            checkpoint=provenance,
            validated_result=str(args.validated_result),
            validated_result_sha256=sha(args.validated_result),
            scope="prepare plus native routing; no dispatch",
            profiler_level="Level1",
            record_shapes=True,
        ),
    )
    torch.npu.synchronize()
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        record_shapes=True,
        with_stack=False,
        with_modules=False,
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.output)),
    ):
        for _ in range(20):
            run()
        torch.npu.synchronize()


if __name__ == "__main__":
    main()
