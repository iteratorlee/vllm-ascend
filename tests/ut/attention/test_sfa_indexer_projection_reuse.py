"""Check per-forward projection reuse without replacing the SFA methods."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import VllmConfig

from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.attention import sfa_v1


def make_impl(monkeypatch):
    impl = object.__new__(sfa_v1.AscendSFAImpl)
    impl.has_indexer = True
    impl.skip_topk = False
    impl.reuse_indexer_projection = True
    impl.head_dim = 4
    impl.n_head = 2
    impl.qk_rope_head_dim = 2
    impl.is_rope_neox_style = False
    impl.enable_sparse_li_c8 = False
    impl.use_torch_npu_lightning_indexer = False
    impl.layer_name = "test.indexer"
    torch.manual_seed(17)
    wk = torch.nn.Linear(8, 6, bias=False)
    wq = torch.nn.Linear(3, 8, bias=False)
    impl.wk_weights_proj = MagicMock(side_effect=lambda x: (wk(x), None))
    impl.wq_b = lambda x: (wq(x), None)
    impl.k_norm = torch.nn.LayerNorm(4)
    monkeypatch.setattr(sfa_v1, "HAS_TRITON", False)
    monkeypatch.setattr(sfa_v1.torch_npu, "npu_rotary_mul", lambda x, cos, sin: x)
    monkeypatch.setattr(
        sfa_v1.DeviceOperator,
        "indexer_select_post_process",
        lambda impl, q, q_scale, q_shape, weights, *args: weights,
    )
    return impl


def run_parts(impl, x, q):
    cos = torch.ones(x.shape[0], 2)
    sin = torch.zeros_like(cos)
    projection = impl._project_indexer_for_reuse(x)
    key, scale = impl.indexer_select_pre_process(x, cos, sin, projected_kw=projection)
    weights = impl.indexer_select_post_process(
        x,
        q,
        (),
        SimpleNamespace(),
        cos,
        sin,
        torch.tensor([x.shape[0]], dtype=torch.int32),
        torch.tensor([131072], dtype=torch.int32),
        projected_kw=projection,
    )
    return key, scale, weights


@pytest.mark.parametrize("rows", [0, 1, 2, 6, 12])
def test_projection_reuse_preserves_key_weights_and_input(monkeypatch, rows):
    impl = make_impl(monkeypatch)
    # A strided input also checks that reuse introduces no contiguous/cast copy.
    x = torch.randn(8, rows).t()
    q = torch.randn(rows, 3)
    before = x.clone()
    impl.reuse_indexer_projection = False
    expected = run_parts(impl, x, q)
    assert impl.wk_weights_proj.call_count == 2
    impl.wk_weights_proj.reset_mock()
    impl.reuse_indexer_projection = True
    actual = run_parts(impl, x, q)
    assert impl.wk_weights_proj.call_count == 1
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    assert actual[1] is expected[1] is None
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)


def test_projection_is_fresh_for_every_mtp_iteration(monkeypatch):
    impl = make_impl(monkeypatch)
    for step in range(5):
        x = torch.randn(2, 8) + step
        q = torch.randn(2, 3)
        impl.reuse_indexer_projection = False
        expected = run_parts(impl, x, q)
        impl.reuse_indexer_projection = True
        actual = run_parts(impl, x, q)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    assert impl.wk_weights_proj.call_count == 15


@pytest.mark.parametrize("has_indexer,skip_topk", [(False, True), (True, True)])
def test_shared_topk_path_does_not_preproject(monkeypatch, has_indexer, skip_topk):
    impl = make_impl(monkeypatch)
    impl.has_indexer = has_indexer
    impl.skip_topk = skip_topk
    assert impl._project_indexer_for_reuse(torch.randn(2, 8)) is None
    impl.wk_weights_proj.assert_not_called()


@pytest.mark.parametrize("value,expected", [(None, False), (False, False), (True, True)])
def test_config_default_and_boolean_option(monkeypatch, value, expected):
    monkeypatch.setattr("vllm_ascend.utils.model_uses_sfa_sparse", lambda config: True)
    monkeypatch.setattr("vllm_ascend.platform.NPUPlatform.check_and_update_config", MagicMock())
    config = VllmConfig()
    config.additional_config = {} if value is None else {"enable_indexer_projection_reuse": value}
    assert AscendConfig(config).enable_indexer_projection_reuse is expected


@pytest.mark.parametrize("value", ["true", 1])
def test_config_rejects_non_boolean_option(monkeypatch, value):
    monkeypatch.setattr("vllm_ascend.utils.model_uses_sfa_sparse", lambda config: True)
    monkeypatch.setattr("vllm_ascend.platform.NPUPlatform.check_and_update_config", MagicMock())
    config = VllmConfig()
    config.additional_config = {"enable_indexer_projection_reuse": value}
    with pytest.raises(ValueError, match="enable_indexer_projection_reuse must be a bool"):
        AscendConfig(config)
