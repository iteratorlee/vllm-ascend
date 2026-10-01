# D indexer projection reuse

Enable with `--additional-config '{"enable_indexer_projection_reuse": true}'`
alongside the deployment's other additional configuration. The default is
`false`. SFA checks both sparse-model configuration and `kv_consumer`; shared
top-k layers do not introduce an extra projection.

One local tensor carries the fused `wk_weights_proj` output from indexer pre
processing to post processing in the same forward. It is computed after the
existing sequence-parallel gather. No projection is cached across layers,
MTP iterations, requests, or graph replays. Existing preprocessing and cache
write operations retain their numerical behavior.

## Validation

- `pytest tests/ut/attention/test_sfa_indexer_projection_reuse.py -q`
- On an Ascend NPU:
  `pytest tests/e2e/nightly/test_sfa_indexer_projection_reuse.py -q`
- `python tools/glm52_c02/bench_projection.py --output /tmp/projection.json`

The NPU test and microbenchmark use the actual SFA pre/post methods with a
weights sink in place of the native lightning-indexer consumer. They check
key, quantization scale and weight equivalence, changing-input ACLGraph replay,
and projection call counts. Microbenchmark timings cover this isolated subpath;
they do not measure full decode-step latency or TPOT. The production GLM-5.2
PD workload is validated separately in the deployment experiment records.
