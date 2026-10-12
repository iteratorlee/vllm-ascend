#!/usr/bin/env python3
"""CPU only: invoke with TORCH_DEVICE_BACKEND_AUTOLOAD=0."""

import argparse
import ast
import sys
from pathlib import Path

from common import (
    PREPARE_SHA256,
    ROW_CASES,
    allocation_check,
    baseline,
    byte_check,
    padded_cases,
    payload_inputs,
    save,
    sha,
    source_hashes,
)
from native_local_pad import NativeLocalPadSlice


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert sha(args.prepare_source) == PREPARE_SHA256, (
        "Baseline prepare differs from current audited file"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write('{"status":"initializing"}\n')
    import torch

    forbidden = [
        name
        for name in sys.modules
        if name.split(".")[0] in {"torch_npu", "triton", "vllm_ascend"}
    ]
    assert not forbidden, forbidden
    torch.set_num_threads(1)
    candidate = NativeLocalPadSlice(torch, "cpu")
    result = dict(
        status="running",
        prepare_sha256=sha(args.prepare_source),
        tools_sha256=source_hashes(Path(__file__).parent),
        cases=[],
        mutants=[],
    )
    try:
        for rows in ROW_CASES:
            for padded in padded_cases(rows):
                for offset in (False, True):
                    inputs = payload_inputs(torch, rows, offset)
                    before = tuple(x.clone() for x in inputs)
                    for rank in range(16):
                        expected = baseline(torch, *inputs, padded, rank)
                        actual = candidate(*inputs, padded, 16, rank)
                        if padded == rows:
                            assert actual is None
                        else:
                            assert actual is not None
                            byte_check(
                                torch, actual, expected, (rows, padded, rank, offset)
                            )
                            allocation_check(inputs, actual)
                            # Mutating candidate output cannot change caller inputs.
                            actual[0].zero_()
                            byte_check(
                                torch, inputs, before, "input mutation isolation"
                            )
                        result["cases"].append(
                            dict(
                                rows=rows,
                                padded=padded,
                                rank=rank,
                                offset=offset,
                                eligible=padded > rows,
                                byte_equal=True,
                            )
                        )
                    byte_check(torch, inputs, before, "input readonly")
        inputs = payload_inputs(torch, 12)
        rejected = [
            (*inputs, 16, 8, 0),
            (*inputs, 16, 16, -1),
            (*inputs, 16, 16, 16),
            (*inputs, 15, 16, 0),
            (*inputs, True, 16, 0),
            (*inputs, 16, 16, True),
            (inputs[0].float(), inputs[1], 16, 16, 0),
            (inputs[0][:, ::2], inputs[1], 16, 16, 0),
            (inputs[0], inputs[1].T.contiguous().T, 16, 16, 0),
            (inputs[0], inputs[1][:11], 16, 16, 0),
            (inputs[0].reshape(-1), inputs[1], 16, 16, 0),
        ]
        for case in rejected:
            assert candidate(*case) is None
        empty = tuple(
            torch.empty(0, dtype=x.dtype).as_strided((0, x.shape[1]), (0, 0))
            for x in inputs
        )
        assert candidate(*empty, 16, 16, 0) is None
        result["rejected_layout_shape_dtype_rank_cases"] = len(rejected) + 1
        expected = baseline(torch, *inputs, 16, 0)
        mutants = {
            "wrong_rank": baseline(torch, *inputs, 16, 1),
            "numeric_changes_signed_zero": tuple(x + 0 for x in expected),
            "padding_nonzero": tuple(
                torch.ones_like(x) for x in baseline(torch, *inputs, 16, 15)
            ),
        }
        for name, actual in mutants.items():
            wanted = (
                baseline(torch, *inputs, 16, 15)
                if name == "padding_nonzero"
                else expected
            )
            try:
                byte_check(torch, actual, wanted, name)
            except AssertionError:
                result["mutants"].append(name)
            else:
                raise AssertionError((name, "oracle failed to reject mutant"))
        try:
            allocation_check(inputs, (inputs[0][:1], inputs[1][:1]))
        except AssertionError:
            result["mutants"].append("input_alias")
        else:
            raise AssertionError("alias mutant escaped")
        # Check audited prepare structure instead of pretending the micro helper
        # exercises full prepare/context/finalize or dispatch.
        tree = ast.parse(args.prepare_source.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "PrepareAndFinalizeWithMC2"
        )
        prepare = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "prepare"
        )
        result["original_prepare_ast_sha256"] = (
            __import__("hashlib").sha256(ast.dump(prepare).encode()).hexdigest()
        )
        result["backend_modules_loaded"] = forbidden
        result["scope"] = (
            "standalone native shard helper; no production prepare integration"
        )
        result["status"] = "cpu_raw_bits_alias_and_negative_gates_passed"
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        save(args.output, result)
    print(
        f"PASS {len(result['cases'])} row/padded/rank/offset cases, {len(result['mutants'])} rejected mutants; no backend imports"
    )


if __name__ == "__main__":
    main()
