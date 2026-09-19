import copy

import torch

from distllm.loss import token_count
from distllm.model import Config, Transformer
from distllm.reproduce import reproduce
from distllm.train import backward_window


def test_mean_of_means_changes_gradient_and_fix_restores_it():
    result = reproduce()
    assert result["wrong_gradient_relative_l2"] > 0.01
    assert result["fixed_gradient_relative_l2"] < 1e-10
    assert abs(result["reference_loss"] - result["fixed_loss"]) < 1e-12


def test_empty_window_skips_adam_state_and_weight_decay():
    model = Transformer(Config(layers=1)).double()
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.2)
    x = torch.randint(32, (2, 5))
    _, valid = backward_window(model, x, x, 1, "reference")
    assert valid
    optimizer.step()
    before = copy.deepcopy(model.state_dict())
    state_before = copy.deepcopy(optimizer.state_dict())
    loss, valid = backward_window(model, x, torch.full_like(x, -100), 1, "reference")
    assert loss == 0 and not valid
    if valid:
        optimizer.step()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    for index, state in optimizer.state_dict()["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, state_before["state"][index][name], rtol=0, atol=0)


def test_shifted_count_ignores_first_label():
    assert token_count(torch.tensor([[12, -100, 3], [7, 2, -100]])).item() == 2
