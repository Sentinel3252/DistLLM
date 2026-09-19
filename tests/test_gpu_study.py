import torch

from distllm.gpu_study import quality_study


def test_quality_study_matches_full_batch_and_exposes_bias():
    result = quality_study()
    sweep = result["gradient_skew_sweep"]
    assert sweep[0]["mean_of_means_gradient_relative_l2"] < 1e-12
    assert sweep[-1]["mean_of_means_gradient_relative_l2"] > 0.1
    assert max(row["token_correct_gradient_relative_l2"] for row in sweep) < 1e-12
    trajectory = result["optimizer_trajectory"]
    assert trajectory["token_correct_parameter_relative_l2"] < 1e-12
    assert trajectory["mean_of_means_parameter_relative_l2"] > 1e-3
    assert torch.isfinite(torch.tensor(trajectory["mean_of_means_token_objective"]))
