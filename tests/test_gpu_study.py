import json
import sys

import pytest
import torch

from distllm import gpu_study
from distllm.gpu_study import _gradient, quality_study


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


def test_gradient_comparison_retains_sub_float32_differences():
    model = torch.nn.Linear(1, 1, bias=False).double()
    model.weight.grad = torch.tensor([[1.0]], dtype=torch.float64)
    reference = _gradient(model).clone()
    model.weight.grad.add_(1e-10)
    actual = _gradient(model)
    assert actual.dtype == torch.float64
    assert (actual - reference).norm() > 0
    assert torch.equal(actual.float(), reference.float())


@pytest.mark.parametrize("arguments,message", [
    (["--repeats", "0"], "repeats"),
    (["--repeats", "-1"], "repeats"),
    (["--warmup", "-1"], "warmup"),
])
def test_bad_cli_arguments_fail_before_studies(arguments, message, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["distllm-gpu-study", *arguments])
    monkeypatch.setattr(gpu_study, "quality_study", lambda: pytest.fail("study started"))
    with pytest.raises(SystemExit) as error:
        gpu_study.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_quality_only_accepts_minimal_timing_arguments(tmp_path, monkeypatch):
    output = tmp_path / "study.json"
    monkeypatch.setattr(sys, "argv", ["study", "--quality-only", "--repeats", "1",
                                     "--warmup", "0", "--output", str(output)])
    monkeypatch.setattr(gpu_study, "quality_study", lambda: {"verified": True})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA checked"))
    gpu_study.main()
    assert json.loads(output.read_text()) == {"quality": {"verified": True}}


def test_missing_cuda_fails_before_quality_study(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["study"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(gpu_study, "quality_study", lambda: pytest.fail("study started"))
    with pytest.raises(SystemExit):
        gpu_study.main()
    assert "quality-only" in capsys.readouterr().err
