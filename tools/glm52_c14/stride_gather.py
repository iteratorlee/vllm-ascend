"""Experimental integer-bit stride gather, not a runtime patch."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather(cos, sin, positions, out_cos, out_sin, STRIDE: tl.constexpr, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, WIDTH)
    position = tl.load(positions + row).to(tl.int64)
    c = tl.load(cos + position * STRIDE + column)
    s = tl.load(sin + position * STRIDE + column)
    tl.store(out_cos + row * WIDTH + column, c)
    tl.store(out_sin + row * WIDTH + column, s)


def gather(cos, sin, positions):
    rows = positions.numel()
    out_cos = torch.empty((rows, cos.shape[1]), dtype=cos.dtype, device=cos.device)
    out_sin = torch.empty_like(out_cos)
    if rows:
        _gather[(rows,)](
            cos.view(torch.int16),
            sin.view(torch.int16),
            positions,
            out_cos.view(torch.int16),
            out_sin.view(torch.int16),
            STRIDE=cos.stride(0),
            WIDTH=cos.shape[1],
            num_warps=4,
        )
    return out_cos, out_sin
