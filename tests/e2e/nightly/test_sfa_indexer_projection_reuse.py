"""NPU byte-equivalence and changing-input ACLGraph replay for D LI reuse.

The native lightning-indexer consumer is replaced by a weights sink so this
test isolates the modified projection/pre/post path. Full PD inference is
validated separately by the deployment benchmark. No speedup is asserted.
"""

from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.attention import sfa_v1
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def make_npu_case(monkeypatch, rows):
    init_device_properties_triton()
    torch.manual_seed(1024)
    device = "npu:0"
    impl = object.__new__(sfa_v1.AscendSFAImpl)
    impl.has_indexer = True
    impl.skip_topk = False
    impl.reuse_indexer_projection = False
    impl.head_dim = 128
    impl.n_head = 32
    impl.qk_rope_head_dim = 64
    impl.is_rope_neox_style = False
    impl.enable_sparse_li_c8 = True
    impl.c8_k_cache_dtype = torch.int8
    impl.c8_k_scale_cache_dtype = torch.float16
    impl.use_torch_npu_lightning_indexer = True
    impl.layer_name = "test.indexer"
    wk = torch.nn.Linear(6144, 160, bias=False, device=device, dtype=torch.bfloat16)
    wq = torch.nn.Linear(2048, 4096, bias=False, device=device, dtype=torch.bfloat16)
    impl.wk_weights_proj = lambda x: (wk(x), None)
    impl.wq_b = lambda x: (wq(x), None)
    impl.k_norm = torch.nn.LayerNorm(128, device=device, dtype=torch.bfloat16)
    # Both alternatives consume identical matrices. The checkpoint's Hadamard
    # values are unrelated to whether an unchanged projection is evaluated once.
    hadamard = torch.eye(128, device=device, dtype=torch.bfloat16)
    monkeypatch.setattr(sfa_v1.AscendSFAImpl, "k_hadamard", hadamard, raising=False)
    monkeypatch.setattr(sfa_v1.AscendSFAImpl, "q_hadamard", hadamard, raising=False)
    monkeypatch.setattr(
        sfa_v1.DeviceOperator,
        "indexer_select_post_process",
        lambda impl, q, q_scale, q_shape, weights, *args: weights,
    )
    x = torch.randn(rows, 6144, device=device, dtype=torch.bfloat16)
    q = torch.randn(rows, 2048, device=device, dtype=torch.bfloat16)
    cos = torch.ones(rows, 64, device=device, dtype=torch.bfloat16)
    sin = torch.zeros_like(cos)
    lengths = torch.tensor([rows], device=device, dtype=torch.int32)
    key_lengths = torch.tensor([131072], device=device, dtype=torch.int32)

    def run():
        projection = impl._project_indexer_for_reuse(x)
        key, scale = impl.indexer_select_pre_process(x, cos, sin, projected_kw=projection)
        weights = impl.indexer_select_post_process(
            x,
            q,
            (),
            SimpleNamespace(),
            cos,
            sin,
            lengths,
            key_lengths,
            projected_kw=projection,
        )
        return key, scale, weights

    return impl, x, q, run


@pytest.mark.parametrize("rows", [1, 2, 6, 12])
@torch.inference_mode()
def test_npu_projection_reuse_eager_and_graph(monkeypatch, rows):
    if not torch_npu.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    torch_npu.npu.set_device(0)
    impl, x, q, run = make_npu_case(monkeypatch, rows)
    expected = run()
    impl.reuse_indexer_projection = True
    actual = run()
    torch_npu.npu.synchronize()
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    graphs = []
    outputs = []
    stream = torch_npu.npu.Stream()
    for enabled in (False, True):
        impl.reuse_indexer_projection = enabled
        with torch_npu.npu.stream(stream):
            for _ in range(3):
                run()
        stream.synchronize()
        graph = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(graph, stream=stream):
            out = run()
        graphs.append(graph)
        outputs.append(out)
    # Update the fixed graph input buffers. Replays must use the new input,
    # including successive MTP-like iterations, never a prior projection.
    for step in range(5):
        x.copy_(torch.randn_like(x) + step)
        q.copy_(torch.randn_like(q))
        fresh = run()
        for graph in graphs:
            graph.replay()
        torch_npu.npu.synchronize()
        for output in outputs:
            for got, want in zip(output, fresh):
                torch.testing.assert_close(got, want, rtol=0, atol=0)
