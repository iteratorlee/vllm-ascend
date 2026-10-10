# SPDX-License-Identifier: Apache-2.0
"""Entire dirty LI caches versus native scatter; dynamic-slot ACLGraph replay."""

import pytest
import torch
import torch_npu

from vllm_ascend.ops.triton.fused_indexer_cache_write import fused_indexer_cache_write


def native_write(key_cache, scale_cache, slots, key, scale):
    # This CANN build rejects noncontiguous in-place output views (workspace
    # error). Run its scatter on dense logical rows, then copy to the oracle
    # view; the fused side keeps its real independent strides and gaps.
    dense_key = key_cache.contiguous()
    dense_scale = scale_cache.contiguous()
    indices = slots.contiguous().view(-1, 1)
    torch_npu.npu_scatter_nd_update_(dense_key, indices, key.contiguous())
    torch_npu.npu_scatter_nd_update_(dense_scale, indices, scale.contiguous())
    if dense_key.data_ptr() != key_cache.data_ptr():
        key_cache.copy_(dense_key)
    if dense_scale.data_ptr() != scale_cache.data_ptr():
        scale_cache.copy_(dense_scale)


def assert_bytes(left, right):
    torch.testing.assert_close(left.cpu().view(torch.uint8), right.cpu().view(torch.uint8), rtol=0, atol=0)


def make_case(rows, strided, slot_dtype):
    capacity = 1024
    source_width = 256 if strided else 128
    key_source = torch.randint(-128, 128, (rows, source_width), dtype=torch.int8, device="npu")
    key = key_source[:, :128] if strided else key_source
    # All half payloads: signed zero, denorm, NaN/Inf, not just random floats.
    bits = torch.randint(-32768, 32768, (rows, 2 if strided else 1), dtype=torch.int16)
    scale_source = bits.view(torch.float16).to("npu")
    scale = scale_source[:, :1]
    key_base = torch.randint(-128, 128, (capacity, source_width), dtype=torch.int8, device="npu")
    scale_base = (
        torch.randint(-32768, 32768, (capacity, 2 if strided else 1), dtype=torch.int16).view(torch.float16).to("npu")
    )
    slots_base = torch.empty(rows * (2 if strided else 1), dtype=slot_dtype, device="npu")
    slots = slots_base[::2] if strided else slots_base
    key_view = key_base[:, :128] if strided else key_base
    scale_view = scale_base[:, :1]
    return key_base, scale_base, key_view, scale_view, slots, key, scale


@pytest.mark.parametrize("rows", [0, 1, 2, 6, 12, 16, 192])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_entire_cache_bytes_eager_and_changing_input_graph(rows, strided, slot_dtype):
    if not torch_npu.npu.is_available():
        pytest.skip("requires Ascend NPU")
    torch_npu.npu.set_device(0)
    key_base, scale_base, key_cache, scale_cache, slots, key, scale = make_case(rows, strided, slot_dtype)
    reference_key = key_base.clone()
    reference_scale = scale_base.clone()
    ref_key_view = reference_key[:, :128] if strided else reference_key
    ref_scale_view = reference_scale[:, :1]
    slot_values = torch.arange(rows, dtype=slot_dtype) + 127
    if rows >= 6:
        slot_values[-4:] = torch.tensor([1023, -1, 1024, -2], dtype=slot_dtype)
    slots.copy_(slot_values)
    native_write(ref_key_view, ref_scale_view, slots, key, scale)
    assert fused_indexer_cache_write(key_cache, scale_cache, slots, key, scale)
    torch_npu.npu.synchronize()
    assert_bytes(key_base, reference_key)
    assert_bytes(scale_base, reference_scale)
    if rows == 0:
        return
    stream = torch_npu.npu.Stream()
    for _ in range(3):
        with torch_npu.npu.stream(stream):
            fused_indexer_cache_write(key_cache, scale_cache, slots, key, scale)
    stream.synchronize()
    graph = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(graph, stream=stream):
        fused_indexer_cache_write(key_cache, scale_cache, slots, key, scale)
    for step in range(5):
        slot_values = torch.arange(rows, dtype=slot_dtype) + 127 + step * 128
        if rows >= 6:
            if step == 1:
                slot_values[-4:] = torch.tensor([-1, 1023, -2, torch.iinfo(slot_dtype).max], dtype=slot_dtype)
            elif step == 2:
                slot_values.fill_(-1)
            else:
                slot_values[-4:] = torch.tensor([1023, -1, 1024, torch.iinfo(slot_dtype).min], dtype=slot_dtype)
        slots.copy_(slot_values)
        key.copy_(torch.randint_like(key, -128, 128))
        bits = torch.randint(-32768, 32768, scale.shape, dtype=torch.int16)
        scale.copy_(bits.view(torch.float16).to("npu"))
        native_write(ref_key_view, ref_scale_view, slots, key, scale)
        graph.replay()
        torch_npu.npu.synchronize()
        assert_bytes(key_base, reference_key)
        assert_bytes(scale_base, reference_scale)


@torch.inference_mode()
def test_reject_row_capacity_mismatch_before_write():
    torch_npu.npu.set_device(0)
    key_cache = torch.zeros(16, 128, dtype=torch.int8, device="npu")
    scale_cache = torch.zeros(16, 1, dtype=torch.float16, device="npu")
    slots = torch.zeros(2, dtype=torch.int32, device="npu")
    key = torch.zeros(2, 128, dtype=torch.int8, device="npu")
    scale = torch.zeros(1, 1, dtype=torch.float16, device="npu")
    with pytest.raises(ValueError, match="rows must match"):
        fused_indexer_cache_write(key_cache, scale_cache, slots, key, scale)
    with pytest.raises(ValueError, match="capacities must match"):
        fused_indexer_cache_write(key_cache, scale_cache[:8], slots, key, scale.expand(2, 1))
    assert key_cache.count_nonzero().cpu().item() == 0
    assert scale_cache.count_nonzero().cpu().item() == 0


@torch.inference_mode()
def test_column_strided_payload_falls_back_without_writes():
    torch_npu.npu.set_device(0)
    key_base = torch.randint(-128, 128, (16, 256), dtype=torch.int8, device="npu")
    before = key_base.clone()
    scale_cache = torch.zeros(16, 1, dtype=torch.float16, device="npu")
    slots = torch.tensor([2], dtype=torch.int32, device="npu")
    key = torch.ones(1, 256, dtype=torch.int8, device="npu")[:, ::2]
    scale = torch.ones(1, 1, dtype=torch.float16, device="npu")
    assert not fused_indexer_cache_write(key_base[:, ::2], scale_cache, slots, key, scale)
    assert_bytes(key_base, before)
    assert scale_cache.count_nonzero().cpu().item() == 0


@pytest.mark.parametrize("overlap", ["key", "scale"])
@torch.inference_mode()
def test_overlapping_destination_rows_fall_back_without_writes(overlap):
    torch_npu.npu.set_device(0)
    key_base = torch.zeros(16, 128, dtype=torch.int8, device="npu")
    scale_base = torch.zeros(16, 1, dtype=torch.float16, device="npu")
    key_cache = key_base.as_strided((16, 128), (1, 1)) if overlap == "key" else key_base
    scale_cache = scale_base[:1].expand(16, 1) if overlap == "scale" else scale_base
    slots = torch.tensor([2, 3], dtype=torch.int32, device="npu")
    key = torch.ones(2, 128, dtype=torch.int8, device="npu")
    scale = torch.ones(2, 1, dtype=torch.float16, device="npu")
    assert not fused_indexer_cache_write(key_cache, scale_cache, slots, key, scale)
    assert key_base.count_nonzero().cpu().item() == 0
    assert scale_base.count_nonzero().cpu().item() == 0
