import pytest

from distllm.analyze import metrics, ring_bytes, union


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
