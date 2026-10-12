# SPDX-License-Identifier: Apache-2.0
"""C14 config gates, ownership, exact layout and initialization wiring."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import VllmConfig

from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.ops import rotary_embedding as rope


@pytest.mark.parametrize("value,expected", [(None, False), (False, False), (True, True)])
def test_config_default_and_bool(monkeypatch, value, expected):
    monkeypatch.setattr("vllm_ascend.utils.model_uses_sfa_sparse", lambda model: True)
    monkeypatch.setattr("vllm_ascend.platform.NPUPlatform.check_and_update_config", MagicMock())
    additional = {} if value is None else {"enable_mla_rope_cache_contiguous": value}
    config = VllmConfig(additional_config=additional)
    config.kv_transfer_config = SimpleNamespace(kv_role="kv_consumer")
    assert AscendConfig(config).enable_mla_rope_cache_contiguous is expected


@pytest.mark.parametrize("value", ["true", "false", 0, 1, None, [], {}])
def test_config_rejects_non_bool(monkeypatch, value):
    monkeypatch.setattr("vllm_ascend.utils.model_uses_sfa_sparse", lambda model: True)
    monkeypatch.setattr("vllm_ascend.platform.NPUPlatform.check_and_update_config", MagicMock())
    with pytest.raises(ValueError, match="enable_mla_rope_cache_contiguous must be a bool"):
        AscendConfig(VllmConfig(additional_config={"enable_mla_rope_cache_contiguous": value}))


@pytest.mark.parametrize(
    "sparse,role", [(False, "kv_consumer"), (True, "kv_producer"), (True, "kv_both"), (True, None)]
)
def test_config_scope(monkeypatch, sparse, role):
    monkeypatch.setattr("vllm_ascend.utils.model_uses_sfa_sparse", lambda model: sparse)
    monkeypatch.setattr("vllm_ascend.platform.NPUPlatform.check_and_update_config", MagicMock())
    config = VllmConfig(additional_config={"enable_mla_rope_cache_contiguous": True})
    config.kv_transfer_config = None if role is None else SimpleNamespace(kv_role=role)
    assert AscendConfig(config).enable_mla_rope_cache_contiguous is False


@pytest.mark.parametrize("enabled", [False, True])
def test_exact_all_bf16_bits_and_storage_ownership(monkeypatch, enabled):
    monkeypatch.setattr(rope, "_cos_cache", None)
    monkeypatch.setattr(rope, "_sin_cache", None)
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).reshape(1024, 64)
    cache = bits.view(torch.bfloat16)
    rope._record_cos_and_sin_cache_interleaved(cache, make_contiguous=enabled)
    expected = bits.view(1024, 2, 32).repeat(1, 1, 2)
    assert torch.equal(rope._cos_cache.view(torch.int16), expected[:, 0])
    assert torch.equal(rope._sin_cache.view(torch.int16), expected[:, 1])
    assert rope._cos_cache.is_contiguous() is enabled
    assert rope._sin_cache.is_contiguous() is enabled
    pointers = {
        t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in (rope._cos_cache, rope._sin_cache)
    }
    assert len(pointers) == (2 if enabled else 1)
    assert sum(pointers.values()) == cache.numel() * cache.element_size() * 2
    if enabled:
        assert rope._cos_cache._base is None
        assert rope._sin_cache._base is None
    old = rope._cos_cache, rope._sin_cache
    rope._record_cos_and_sin_cache_interleaved(torch.zeros_like(cache), make_contiguous=enabled)
    assert rope._cos_cache is old[0] and rope._sin_cache is old[1]


@pytest.mark.parametrize("enabled", [False, True])
def test_constructor_passes_config_once(monkeypatch, enabled):
    monkeypatch.setattr(rope, "get_current_vllm_config", lambda: SimpleNamespace(speculative_config=None))
    monkeypatch.setattr(rope, "get_ascend_config", lambda: SimpleNamespace(enable_mla_rope_cache_contiguous=enabled))
    monkeypatch.setattr(rope.RotaryEmbedding, "__init__", lambda *args, **kwargs: None)
    monkeypatch.setattr(rope, "_record_cos_sin_cache", MagicMock())
    record = MagicMock()
    monkeypatch.setattr(rope, "_record_cos_and_sin_cache_interleaved", record)
    embedding = rope.AscendRotaryEmbedding.__new__(rope.AscendRotaryEmbedding)
    embedding.cos_sin_cache = torch.zeros(16, 64, dtype=torch.bfloat16)
    rope.AscendRotaryEmbedding.__init__(embedding, 64, 64, 16, 10000, False, torch.bfloat16)
    record.assert_called_once_with(embedding.cos_sin_cache, make_contiguous=enabled)
