# SPDX-License-Identifier: Apache-2.0
"""Config default/type gates and unsupported-device no-mutation fallback."""

from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import VllmConfig

from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.ops.triton.fused_indexer_cache_write import fused_indexer_cache_write


@pytest.mark.parametrize("value,expected", [(None, False), (False, False), (True, True)])
def test_config_boolean_and_default(monkeypatch, value, expected):
    monkeypatch.setattr("vllm_ascend.utils.model_uses_sfa_sparse", lambda model: True)
    monkeypatch.setattr("vllm_ascend.platform.NPUPlatform.check_and_update_config", MagicMock())
    additional = {} if value is None else {"enable_fused_indexer_cache_write": value}
    assert AscendConfig(VllmConfig(additional_config=additional)).enable_fused_indexer_cache_write is expected


@pytest.mark.parametrize("value", ["true", 1])
def test_config_rejects_non_boolean(monkeypatch, value):
    monkeypatch.setattr("vllm_ascend.utils.model_uses_sfa_sparse", lambda model: True)
    monkeypatch.setattr("vllm_ascend.platform.NPUPlatform.check_and_update_config", MagicMock())
    with pytest.raises(ValueError, match="must be a bool"):
        AscendConfig(VllmConfig(additional_config={"enable_fused_indexer_cache_write": value}))


def test_non_sparse_model_disables_option(monkeypatch):
    monkeypatch.setattr("vllm_ascend.utils.model_uses_sfa_sparse", lambda model: False)
    monkeypatch.setattr("vllm_ascend.platform.NPUPlatform.check_and_update_config", MagicMock())
    assert not AscendConfig(
        VllmConfig(additional_config={"enable_fused_indexer_cache_write": True})
    ).enable_fused_indexer_cache_write


@pytest.mark.parametrize("rows", [0, 2])
def test_cpu_fallback_has_no_mutation(rows):
    key_cache = torch.ones(16, 128, dtype=torch.int8)
    scale_cache = torch.ones(16, 1, dtype=torch.float16)
    slots = torch.arange(rows, dtype=torch.int32)
    key = torch.zeros(rows, 128, dtype=torch.int8)
    scale = torch.zeros(rows, 1, dtype=torch.float16)
    assert not fused_indexer_cache_write(key_cache, scale_cache, slots, key, scale)
    assert torch.equal(key_cache, torch.ones_like(key_cache))
    assert torch.equal(scale_cache, torch.ones_like(scale_cache))
