"""Backend-neutral independent oracle, payloads, and evidence helpers."""

import hashlib
import json
from pathlib import Path


BASE_COMMIT = "8cd978de25eb5be3f7645ff9ad512fe0c74dd391"
PREPARE_SHA256 = "c467d0ed0a0d41525aa90008aafbfd26bdbb718b0e29db7de2bed9abe09cabdd"
ROW_CASES = (0, 1, 2, 6, 12, 15, 16, 17, 31, 191, 192)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(directory):
    return {path.name: sha(path) for path in sorted(Path(directory).glob("*.py"))}


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def baseline(torch, hidden, logits, padded, rank):
    # Independent original full pad -> split. It shares no address mapping code
    # or output allocation with the candidate.
    return tuple(
        torch.tensor_split(
            torch.nn.functional.pad(x, (0, 0, 0, padded - x.shape[0])), 16, dim=0
        )[rank]
        for x in (hidden, logits)
    )


def payload_inputs(torch, rows, offset=False):
    outputs = []
    # Signed integer literals preserve sign bits and NaN payloads without any
    # floating point conversion (including signaling NaNs and subnormals).
    cases = (
        (
            6144,
            torch.int16,
            torch.bfloat16,
            (0, -32768, 1, -32767, 32639, -129, 32640, -128, 32705, 32641),
        ),
        (
            256,
            torch.int32,
            torch.float32,
            (
                0,
                -2147483648,
                1,
                -2147483647,
                2139095039,
                -8388609,
                2139095040,
                -8388608,
                2143289361,
                2139095041,
            ),
        ),
    )
    extra = 5 if offset else 0
    for width, integer, floating, patterns in cases:
        count = (rows + extra) * width
        generator = torch.Generator().manual_seed(168 + rows + width)
        limits = torch.iinfo(integer)
        bits = torch.randint(
            limits.min, limits.max, (count,), dtype=integer, generator=generator
        )
        if count:
            bits[: min(count, len(patterns))] = torch.tensor(
                patterns[:count], dtype=integer
            )
            # Put special payloads in every retained row, including offset data.
            matrix = bits.reshape(rows + extra, width)
            matrix[:, : len(patterns)] = torch.tensor(patterns, dtype=integer)
        value = bits.view(floating).reshape(rows + extra, width)[extra:]
        # Explicit standard strides also for N0; no accidental zero-stride case.
        assert value.stride() == (width, 1)
        outputs.append(value)
    return tuple(outputs)


def byte_check(torch, actual, expected, label):
    assert len(actual) == len(expected), (label, "output count")
    for index, (got, want) in enumerate(zip(actual, expected)):
        assert got.shape == want.shape and got.dtype == want.dtype, (
            label,
            index,
            "metadata",
        )
        got_bits = got.detach().cpu().contiguous().view(torch.uint8)
        want_bits = want.detach().cpu().contiguous().view(torch.uint8)
        if not torch.equal(got_bits, want_bits):
            mismatch = (got_bits != want_bits).nonzero()
            raise AssertionError(
                (
                    label,
                    index,
                    "mismatched bytes",
                    len(mismatch),
                    mismatch[:16].tolist(),
                )
            )


def allocation_check(inputs, outputs):
    for output in outputs:
        assert output.storage_offset() == 0 and output.is_contiguous()
        assert all(
            output.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
            for x in inputs
        )
    assert (
        outputs[0].untyped_storage().data_ptr()
        != outputs[1].untyped_storage().data_ptr()
    )


def padded_cases(rows):
    return sorted({max(16, (rows + 15) // 16 * 16), max(32, (rows + 31) // 16 * 16)})
