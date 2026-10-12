# C14: contiguous MLA RoPE caches

This experiment diagnoses the large `aclnnIndex_SliceAiCore_Slice` tasks
produced by indexing the strided cos/sin views of the current GLM-5.2 RoPE
cache. The runtime change only makes both views contiguous once, at cache
initialization. The existing repeat, dtype and gather/copy operations remain
unchanged.

`additional_config.enable_mla_rope_cache_contiguous` is a strict bool,
defaults to false, and is effective only for sparse attention models with
`kv_role="kv_consumer"`. Set it only on the current experimental D roles.
It does not require another optimization flag.

## Reproduce on an exclusively leased device

Run from the experiment directory or `tools/glm52_c14`. The examples use
`SOURCE` as a shell placeholder for the source file path, not as a new
runtime environment option. The saved live source is useful for reproducing
the original owner; the patched repository source is needed for initialization
and paired A/B tests that pass `make_contiguous`.

```sh
python3 probe_rope.py --source SOURCE --output results/probe.json
python3 bench_paired.py --source SOURCE --output results/paired.json
python3 bench_init.py --source SOURCE --output results/init.json
python3 profile_rope.py --source SOURCE --variant baseline --output results/owner-baseline
python3 profile_rope.py --source SOURCE --variant contiguous --output results/owner-contiguous
python3 analyze_owner.py --artifacts results
```

The main probe uses the real 1,048,576-row BF16 cache, rows 0/1/2/12/192,
all 65,536 BF16 encodings, int32/int64 repeated/unsorted/boundary positions,
and repeated updates. It verifies both capture of the helper itself and the
runtime topology of eager metadata updates followed by a target graph
reading the same preallocated buffer addresses. Graph timing is separate
from eager timing. Device event intervals around eager launches include
host submission gaps; wall-clock samples are also retained.

The paired test alternates A/B order over 20 rounds with 100 calls per
measurement. The init test alternates enabled/disabled initialization after
warmup, recording live allocation, reserved allocation, peak allocation and
physical free HBM before/after a diagnostic allocator trim. The runtime
does not call `empty_cache`.

The profiler is intentionally a separate process with explicit schedule,
Level1, shapes enabled and stack/module capture disabled. An earlier
combined graph/stack-profile process exited 139; its log is preserved and
is not counted as successful profiling.

## Verification and limitations

Targeted unit tests cover default/type/role gates, constructor wiring,
byte layout, ownership and initialization idempotence. The NPU test covers
the full cache and target graph buffer updates. The microbenchmarks are
performance evidence for this helper, not end-to-end throughput or ITL
estimates.

Both dense caches retain the same 256 MiB logical payload as the original
shared backing. Initialization temporarily needs about 256 MiB extra, and
the allocator can retain that reservation until reuse or trim. A fresh PD
A/B must verify startup success, KV capacity 286,976 tokens per instance,
actual concurrency 2+2, and output/acceptance correctness before this flag
is considered usable in the deployment.
