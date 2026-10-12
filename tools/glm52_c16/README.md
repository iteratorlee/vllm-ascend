# C16 native local MoE padding micro

Standalone tools only, based on clean C03
`8cd978de25eb5be3f7645ff9ad512fe0c74dd391`. No runtime patch, custom kernel,
new environment variable, or installation. C15 remains separate.

For positive padding, allocate only one TP16 shard. A fully valid shard uses
native `clone`, an entirely padded shard uses `new_zeros`, and a mixed shard
uses native `F.pad` on its valid local rows. Clone is required even when the
input view is already contiguous: the original positive-padding branch returns
fresh storage. No floating point arithmetic is used by the candidate.

The original full `F.pad` followed by `tensor_split` is the independent oracle.
The helper requires BF16 hidden `[N,6144]`, FP32 logits `[N,256]`, matching
devices, positive standard strides, contiguous ND format on NPU, TP16, and
P>N divisible by16. Unsupported metadata returns `None`. CPU dependency
injection is only for isolated testing. No full prepare integration is claimed.

## CPU gate

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OPENBLAS_NUM_THREADS=1 \
  python3 tools/glm52_c16/cpu_oracle.py \
  --prepare-source vllm_ascend/ops/fused_moe/prepare_finalize.py \
  --output ../cpu-oracle.json
```

Checks 704 N/P/rank/offset combinations, including signed zero, subnormal,
nonfinite and NaN payload bits. Rejects malformed shapes/layouts and tests
that four deliberately incorrect outputs fail. Requires the freshly audited
original prepare SHA and rejects imported NPU/Triton/Ascend modules.
The CPU-only autoload environment setting is not a runtime feature flag.

## Device gates

Run only after a specific root-issued device lease, no requests/profiling,
and at least1GiB free memory. An idle resident model is not an empty device.
Never retry a device runtime error without root review.

```sh
python3 tools/glm52_c16/probe.py --run-npu \
  --lease 'root-authorized host/container/logical0/window' \
  --prepare-source vllm_ascend/ops/fused_moe/prepare_finalize.py \
  --output ../paired.json --rounds 20 --repeats 50 --graph-unroll 16
```

One visible logical NPU is mandatory. The probe checks all raw bits against
both CPU and device original pad/split, distinct output storage, ND format,
input immutability, and physical NZ rejection. At N12/P16 every rank is checked
through actual native GLM gating and standalone native dynamic quant in graph
mode, with three input updates and fixed input/output addresses. Additional
shapes include N0, N2, N6, N16/P32, N17/P32 and N192/P208 boundaries.
Weights and correction bias come from checkpoint layer78. The input generator's
FP32 linear is outside timing and does not reproduce exact GateLinear lowering.

Timing runs graph before eager, with alternating AB/BA raw records and at least
20 pairs for ranks0,11,12,15. It measures prepare, prepare plus native routing,
and routing plus standalone dynamic quant. The standalone quant is **not**
MC2 dispatch internal quantization. CPU copies and synchronization for byte
checks occur outside hot timing and do not prove production stream ordering.

After matching byte/consumer/paired evidence, run each profiler combination in
a fresh subprocess. Use rank0 and rank12 to inspect clone and zero paths.

```sh
python3 tools/glm52_c16/profile_native_pad.py --run-npu \
  --lease 'root-authorized host/container/logical0/window' \
  --prepare-source vllm_ascend/ops/fused_moe/prepare_finalize.py \
  --validated-result ../paired.json --variant candidate --mode graph \
  --rank 0 --output ../profile/candidate-graph-rank0
```

All output paths are exclusive so failed evidence is retained. Source hashes
must match the paired evidence before profiling. Profile expected task removal
rather than assuming clone/zero task counts from API names.

## Future runtime contract

A positive micro result only justifies an isolated default-off implementation.
That implementation must preserve mask source/slicing, input IDs, original N,
full padded `[P,6144]` shape, original prepare tuple, native routing/dispatch,
finalize/all-gather, and shared expert ownership. N=P retains the original
view/alias path. Physical layout and graph/stream contracts need actual native
consumers and eventual EP32 integration; a single-device probe is insufficient.
Final acceptance uses server/client ITL p50/p75/p90/p99 and separate correctness,
failure, and MTP acceptance checks. No E2E gain is claimed by these tools.
