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
