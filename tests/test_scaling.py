import copy

import pytest

from distllm.scaling import compare


def test_scaling_uses_fixed_workload_and_rejects_profile_timings():
    reference = dict(timing_kind="benchmark", world_size=1, mode="reference", backend="nccl",
                     torch="2.14", dtype="float32", model={"width": 16}, sequence=8,
                     batch=8, pattern="equal", device_name="test", steps=[dict(rank_max_seconds=4)])
    candidate = copy.deepcopy(reference)
    candidate.update(world_size=2, mode="ddp", steps=[dict(rank_max_seconds=2)])
    result = compare(reference, candidate)
    assert result["speedup"] == 2 and result["strong_scaling_efficiency"] == 1
    candidate["timing_kind"] = "verification_or_profile"
    with pytest.raises(ValueError, match="not benchmark"):
        compare(reference, candidate)
    candidate["timing_kind"] = "benchmark"
    candidate["batch"] = 16
    with pytest.raises(ValueError, match="batch"):
        compare(reference, candidate)
    reference["pattern"] = "empty-rank"
    with pytest.raises(ValueError, match="workload"):
        compare(reference, candidate)


@pytest.fixture
def comparable_runs():
    reference = dict(timing_kind="benchmark", world_size=1, mode="reference", backend="gloo",
                     torch="2.14", dtype="float32", model={"width": 16}, sequence=8,
                     batch=8, pattern="equal", device_name="CPU", steps=[dict(rank_max_seconds=4)])
    candidate = copy.deepcopy(reference)
    candidate.update(world_size=2, mode="ddp", steps=[dict(rank_max_seconds=2)])
    return reference, candidate


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("samples", [[], [{}], [None]] + [
    [{"rank_max_seconds": value}] for value in (0, -1, float("nan"), float("inf"), "2", True)
])
def test_invalid_timings_are_rejected_on_both_sides(comparable_runs, index, samples):
    comparable_runs[index]["steps"] = samples
    with pytest.raises(ValueError, match="steps"):
        compare(*comparable_runs)


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_invalid_world_size(comparable_runs, size):
    comparable_runs[1]["world_size"] = size
    with pytest.raises(ValueError, match="world_size"):
        compare(*comparable_runs)


def test_missing_field_names_the_problem(comparable_runs):
    del comparable_runs[1]["model"]
    with pytest.raises(ValueError, match="candidate: missing model"):
        compare(*comparable_runs)
