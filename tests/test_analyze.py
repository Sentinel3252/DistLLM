import json

import pytest

from distllm.analyze import analyze, metrics, ring_bytes, union


def test_overlap_uses_union_not_sum_of_concurrent_kernels():
    got = metrics([(0, 5), (2, 7), (10, 12)], [(3, 9), (4, 8)])
    assert got["compute_us"] == 9
    assert got["communication_us"] == 6
    assert got["overlap_us"] == 4
    assert got["communication_hidden_fraction"] == 4 / 6
    assert union([(0, 2), (2, 3)]) == [(0, 3)]


def test_empty_and_disjoint_intervals():
    assert metrics([], [])["communication_hidden_fraction"] is None
    assert metrics([(0, 1)], [(2, 3)])["overlap_us"] == 0


@pytest.mark.parametrize("op,expected", [("all_reduce", 150), ("all_gather", 300), ("reduce_scatter", 75)])
def test_ring_model(op, expected):
    assert ring_bytes(op, 100, 4) == expected
    assert ring_bytes(op, 100, 1) == 0


def test_trace_parser_separates_devices_and_ignores_host_events(tmp_path):
    events = [
        dict(cat="kernel", ph="X", name="gemm", ts=0, dur=5, pid=11, args={"device": 0}),
        dict(cat="kernel", ph="X", name="NcClAllReduce", ts=3, dur=6, pid=11, args={"device": 0}),
        dict(cat="kernel", ph="X", name="gemm", ts=0, dur=2, pid=12, args={"device": 1}),
        dict(cat="kernel", ph="X", name="gemm", ts=0, dur=4, pid=13),
        dict(cat="cpu_op", ph="X", name="ncclAllReduce", ts=0, dur=100, pid=11),
        dict(cat="kernel", ph="B", name="gemm", ts=0, pid=11),
    ]
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
    result = analyze(path)
    assert set(result) == {"0", "1", "13"}
    assert result["0"]["overlap_us"] == 2
    assert result["0"]["communication_us"] == 6
    assert result["1"]["communication_us"] == 0
    assert result["13"]["compute_us"] == 4


def test_cpu_only_trace_cannot_claim_gpu_overlap(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": [dict(cat="cpu_op", ph="X")]}))
    with pytest.raises(ValueError, match="No GPU kernel"):
        analyze(path)
