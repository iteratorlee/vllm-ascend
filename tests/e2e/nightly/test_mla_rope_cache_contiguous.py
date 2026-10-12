# SPDX-License-Identifier: Apache-2.0
"""Full-sized RoPE tables: arbitrary BF16 bits and target graph buffer updates."""

import pytest
import torch
import torch_npu

from vllm_ascend.ops import rotary_embedding as rope


@pytest.fixture(scope="module")
def full_rope_cache():
    if not torch_npu.npu.is_available():
        pytest.skip("requires Ascend NPU")
    torch.npu.set_device(0)
    generator = torch.Generator().manual_seed(20261012)
    bits = torch.randint(-32768, 32768, (1048576, 64), dtype=torch.int16, generator=generator)
    bits[:1024] = torch.arange(65536, dtype=torch.int32).to(torch.int16).reshape(1024, 64)
    return bits, bits.view(torch.bfloat16).to("npu")


def assert_bytes(actual, expected):
    for got, want in zip(actual, expected):
        assert torch.equal(got.cpu().contiguous().view(torch.int16), want)


def expected_rows(bits, positions):
    selected = bits[positions.long()].view(-1, 2, 32).repeat(1, 1, 2)
    return selected[:, 0, None, None, :], selected[:, 1, None, None, :]


@pytest.mark.parametrize("rows", [0, 1, 2, 12, 192])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_bytes_and_target_graph(full_rope_cache, monkeypatch, rows, position_dtype):
    bits, source = full_rope_cache
    monkeypatch.setattr(rope, "_cos_cache", None)
    monkeypatch.setattr(rope, "_sin_cache", None)
    monkeypatch.setattr(rope, "_cos_mla", torch.empty(192, 1, 1, 64, dtype=torch.bfloat16, device="npu"))
    monkeypatch.setattr(rope, "_sin_mla", torch.empty_like(rope._cos_mla))
    rope._record_cos_and_sin_cache_interleaved(source, make_contiguous=True)
    all_bits = torch.arange(1024)
    assert_bytes(rope.get_cos_and_sin_mla(all_bits.to("npu")), expected_rows(bits, all_bits))
    anchors = torch.tensor(
        [1048575, 0, 131072, 131071, 131073, 0, 999999, 1, 65535, 1048575, 7, 0], dtype=position_dtype
    )
    positions = torch.empty(rows, dtype=position_dtype, device="npu")
    graph = None
    sinks = None
    for step in range(5):
        cpu_positions = anchors.roll(step).repeat((rows + 11) // 12)[:rows]
        positions.copy_(cpu_positions.to("npu"))
        expected = expected_rows(bits, cpu_positions)
        assert_bytes(rope.get_cos_and_sin_mla(positions), expected)
        buffers = rope.get_cos_and_sin_mla(positions, use_cache=True)
        assert_bytes(buffers, expected)
        assert buffers[0].data_ptr() == rope._cos_mla[:rows].data_ptr()
        assert buffers[1].data_ptr() == rope._sin_mla[:rows].data_ptr()
        if rows and graph is None:
            sinks = [torch.empty_like(t) for t in buffers]
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                sinks[0].copy_(buffers[0])
                sinks[1].copy_(buffers[1])
        if graph is not None:
            graph.replay()
            torch.npu.synchronize()
            assert_bytes(sinks, expected)
    if graph is not None:
        graph.reset()
