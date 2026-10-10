# SPDX-License-Identifier: Apache-2.0
"""One-launch, byte-preserving INT8 LI key / FP16 scale cache update."""

import torch
from vllm.triton_utils import tl, triton

INDEXER_KEY_WIDTH = 128
INDEXER_SCALE_WIDTH = 1


@triton.jit
def _fused_indexer_cache_write(
    key_cache,
    scale_cache,
    slots,
    key,
    scale,
    NUM_TOKENS: tl.constexpr,
    CACHE_ROWS: tl.constexpr,
    KEY_WIDTH: tl.constexpr,
    KEY_CACHE_ROW_STRIDE: tl.constexpr,
    KEY_CACHE_COL_STRIDE: tl.constexpr,
    SCALE_CACHE_ROW_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    KEY_ROW_STRIDE: tl.constexpr,
    KEY_COL_STRIDE: tl.constexpr,
    SCALE_ROW_STRIDE: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    columns = tl.arange(0, KEY_WIDTH)
    program = tl.program_id(0).to(tl.int64)
    for offset in tl.static_range(ROWS_PER_PROGRAM):
        row = program * ROWS_PER_PROGRAM + offset
        exists = row < NUM_TOKENS
        slot = tl.load(slots + row * SLOT_STRIDE, mask=exists, other=-1).to(tl.int64)
        valid = exists & (slot >= 0) & (slot < CACHE_ROWS)
        safe_row = tl.where(exists, row, 0)
        safe_slot = tl.where(valid, slot, 0)
        key_values = tl.load(key + safe_row * KEY_ROW_STRIDE + columns * KEY_COL_STRIDE, mask=valid, other=0)
        scale_value = tl.load(scale + safe_row * SCALE_ROW_STRIDE, mask=valid, other=0)
        tl.store(key_cache + safe_slot * KEY_CACHE_ROW_STRIDE + columns * KEY_CACHE_COL_STRIDE, key_values, mask=valid)
        tl.store(scale_cache + safe_slot * SCALE_CACHE_ROW_STRIDE, scale_value, mask=valid)


def fused_indexer_cache_write(
    key_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    slots: torch.Tensor,
    key: torch.Tensor,
    scale: torch.Tensor,
    rows_per_program: int = 1,
) -> bool:
    """Return False without writes for unsupported dtype/device/layout.

    Inputs/caches use the existing flat token-row view. Valid slots must be
    unique (scheduler contract). Negative/PAD/out-of-range slots are skipped;
    there is no host read of slot values or group metadata. Only the current
    128-column INT8 LI key and single FP16 scale layout is supported.
    """
    tensors = (key_cache, scale_cache, slots, key, scale)
    if any(tensor.device.type != "npu" for tensor in tensors):
        return False
    if any(tensor.device != key_cache.device for tensor in tensors):
        return False
    if key.dtype != torch.int8 or key_cache.dtype != torch.int8:
        return False
    if scale.dtype != torch.float16 or scale_cache.dtype != torch.float16:
        return False
    if slots.dtype not in (torch.int32, torch.int64):
        return False
    if any(tensor.ndim != 2 for tensor in (key_cache, scale_cache, key, scale)) or slots.ndim != 1:
        return False
    if (
        key.shape[1] != INDEXER_KEY_WIDTH
        or key_cache.shape[1] != INDEXER_KEY_WIDTH
        or scale.shape[1] != INDEXER_SCALE_WIDTH
        or scale_cache.shape[1] != INDEXER_SCALE_WIDTH
    ):
        return False
    # Ascend Triton INT8 column-strided loads/stores are not byte-preserving
    # in the installed backend. Row padding is supported; payload columns
    # must be contiguous, matching the current LI allocation and quant output.
    if key.stride(1) != 1 or key_cache.stride(1) != 1:
        return False
    # Distinct logical slots must also address distinct destination rows.
    if key_cache.stride(0) < INDEXER_KEY_WIDTH or scale_cache.stride(0) < INDEXER_SCALE_WIDTH:
        return False
    if key.shape[0] != scale.shape[0] or key.shape[0] != slots.numel():
        raise ValueError("LI key, scale and slot rows must match.")
    if key_cache.shape[0] != scale_cache.shape[0]:
        raise ValueError("LI key and scale cache row capacities must match.")
    if rows_per_program not in (1, 2, 4, 8):
        raise ValueError("rows_per_program must be 1, 2, 4 or 8.")
    if slots.numel() == 0:
        return True
    _fused_indexer_cache_write[(triton.cdiv(slots.numel(), rows_per_program),)](
        key_cache,
        scale_cache,
        slots,
        key,
        scale,
        NUM_TOKENS=slots.numel(),
        CACHE_ROWS=key_cache.shape[0],
        KEY_WIDTH=INDEXER_KEY_WIDTH,
        KEY_CACHE_ROW_STRIDE=key_cache.stride(0),
        KEY_CACHE_COL_STRIDE=key_cache.stride(1),
        SCALE_CACHE_ROW_STRIDE=scale_cache.stride(0),
        SLOT_STRIDE=slots.stride(0),
        KEY_ROW_STRIDE=key.stride(0),
        KEY_COL_STRIDE=key.stride(1),
        SCALE_ROW_STRIDE=scale.stride(0),
        ROWS_PER_PROGRAM=rows_per_program,
        num_warps=4,
    )
    return True
