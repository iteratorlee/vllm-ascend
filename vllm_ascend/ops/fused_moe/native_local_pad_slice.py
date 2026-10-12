# SPDX-License-Identifier: Apache-2.0
"""Native allocation of one TP shard of positively padded MC2 inputs."""

import torch
import torch_npu

TP_SIZE = 16
HIDDEN_WIDTH = 6144
EXPERT_COUNT = 256
ND_FORMAT = 2


def is_native_local_pad_slice_supported(
    hidden: torch.Tensor,
    logits: torch.Tensor,
    padded: int,
    tp_size: int,
    rank: int,
) -> bool:
    if (
        type(padded) is not int
        or type(tp_size) is not int
        or type(rank) is not int
        or tp_size != TP_SIZE
        or not 0 <= rank < tp_size
        or padded <= 0
        or padded % tp_size
        or hidden.ndim != 2
        or logits.ndim != 2
    ):
        return False
    rows = hidden.shape[0]
    if padded <= rows or hidden.device != logits.device:
        return False
    for tensor, width, dtype in (
        (hidden, HIDDEN_WIDTH, torch.bfloat16),
        (logits, EXPERT_COUNT, torch.float32),
    ):
        if (
            tensor.device.type != "npu"
            or tensor.shape != (rows, width)
            or tensor.dtype != dtype
            or tensor.stride() != (width, 1)
            or not tensor.is_contiguous()
            or torch_npu.get_npu_format(tensor) != ND_FORMAT
        ):
            return False
    return True


def maybe_moe_native_local_pad_slice(
    hidden: torch.Tensor,
    logits: torch.Tensor,
    padded: int,
    tp_size: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Match full native pad/split while retaining fresh output ownership.

    Unsupported metadata returns None, so the caller retains its original path.
    Shape/rank decisions use host metadata; no tensor values are read on CPU.
    """
    if not is_native_local_pad_slice_supported(hidden, logits, padded, tp_size, rank):
        return None
    rows = hidden.shape[0]
    local_rows = padded // tp_size
    first = rank * local_rows
    valid = min(local_rows, max(0, rows - first))

    def shard(tensor: torch.Tensor) -> torch.Tensor:
        if valid == 0:
            return tensor.new_zeros((local_rows, tensor.shape[1]))
        selected = tensor.narrow(0, first, valid)
        if valid == local_rows:
            # contiguous() may return an alias; the padded branch must be fresh.
            return selected.clone(memory_format=torch.contiguous_format)
        return torch.nn.functional.pad(selected, (0, 0, 0, local_rows - valid))

    return shard(hidden), shard(logits)
