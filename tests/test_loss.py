import copy

import torch

from distllm.loss import loss_sum, normalize_gradients, token_count
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


def test_loss_sum_matches_supervised_next_token_log_probabilities():
    logits = torch.tensor([[[1., 2., 3.], [3., 1., 2.], [2., 3., 1.], [1., 1., 1.]]],
                          dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[2, 0, -100, 1]])
    log_probs = logits.log_softmax(-1)
    expected = -log_probs[0, 0, 0] - log_probs[0, 2, 1]
    torch.testing.assert_close(loss_sum(logits, labels), expected)
    assert token_count(labels).item() == 2


def test_ignored_loss_has_finite_zero_gradients():
    logits = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    loss = loss_sum(logits, torch.full((2, 3), -100, dtype=torch.long))
    assert torch.isfinite(loss) and loss.item() == 0
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), rtol=0, atol=0)


def test_normalization_compensates_ddp_and_skips_unused_parameters():
    used = torch.nn.Parameter(torch.tensor([2., 4.], dtype=torch.float64))
    unused = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    used.grad = torch.tensor([4., 8.], dtype=torch.float64)
    assert normalize_gradients(iter([used, unused]), torch.tensor(8), dp_size=4)
    torch.testing.assert_close(used.grad, torch.tensor([2., 4.], dtype=torch.float64))
    assert unused.grad is None
    before = used.grad.clone()
    assert not normalize_gradients([used], torch.tensor(0), dp_size=4)
    torch.testing.assert_close(used.grad, before, rtol=0, atol=0)
