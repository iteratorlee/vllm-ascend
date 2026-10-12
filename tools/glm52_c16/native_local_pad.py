"""Isolated C16 prototype: native allocation of one padded TP shard.

No installation, environment changes, custom operators, or runtime patching.
Backend objects are explicit so CPU tests never need a device backend import.
"""

from dataclasses import dataclass
from typing import Any, Callable


TP_SIZE = 16
HIDDEN_WIDTH = 6144
EXPERT_COUNT = 256
ND_FORMAT = 2


@dataclass(frozen=True)
class NativeLocalPadSlice:
    torch: Any
    device_type: str
    get_format: Callable | None = None

    def __call__(self, hidden, logits, padded, tp_size, rank):
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
            return None
        rows = hidden.shape[0]
        if padded <= rows or hidden.device != logits.device:
            return None
        for tensor, width, dtype in (
            (hidden, HIDDEN_WIDTH, self.torch.bfloat16),
            (logits, EXPERT_COUNT, self.torch.float32),
        ):
            if (
                tensor.device.type != self.device_type
                or tensor.shape != (rows, width)
                or tensor.dtype != dtype
                or tensor.stride() != (width, 1)
                or not tensor.is_contiguous()
            ):
                return None
            if self.device_type == "npu" and (
                self.get_format is None or self.get_format(tensor) != ND_FORMAT
            ):
                return None
        local_rows = padded // tp_size
        first = rank * local_rows
        valid = min(local_rows, max(0, rows - first))

        def shard(tensor):
            if valid == 0:
                return tensor.new_zeros((local_rows, tensor.shape[1]))
            selected = tensor.narrow(0, first, valid)
            if valid == local_rows:
                # contiguous() may return an alias; clone must allocate here.
                return selected.clone(memory_format=self.torch.contiguous_format)
            return self.torch.nn.functional.pad(selected, (0, 0, 0, local_rows - valid))

        return shard(hidden), shard(logits)
