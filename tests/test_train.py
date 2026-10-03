from unittest.mock import Mock

import pytest
import torch

from distllm.train import backward_window


@pytest.mark.parametrize("tokens,labels,message", [
    (torch.zeros(2, 4), torch.zeros(3, 4), "identical"),
    (torch.zeros(2, 4), torch.zeros(2, 5), "identical"),
    (torch.zeros(4), torch.zeros(4), "identical"),
    (torch.zeros(0, 4), torch.zeros(0, 4), "local batch"),
    (torch.zeros(2, 1), torch.zeros(2, 1), "two tokens"),
])
def test_invalid_inputs_fail_before_mutating_model_or_communicating(tokens, labels, message):
    model, comm = Mock(), Mock()
    with pytest.raises(ValueError, match=message):
        backward_window(model, tokens, labels, 2, "ddp", comm)
    model.zero_grad.assert_not_called()
    comm.all_reduce.assert_not_called()


@pytest.mark.parametrize("mode,message", [("typo", "unsupported"), ("ddp", "requires"), ("pp", "requires")])
def test_invalid_mode_or_missing_comm(mode, message):
    model = Mock()
    inputs = torch.zeros(2, 4, dtype=torch.long)
    with pytest.raises(ValueError, match=message):
        backward_window(model, inputs, inputs, 2, mode)
    model.zero_grad.assert_not_called()


def test_nonpositive_microbatch_is_rejected():
    inputs = torch.zeros(2, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="microbatch"):
        backward_window(Mock(), inputs, inputs, 0, "reference")
