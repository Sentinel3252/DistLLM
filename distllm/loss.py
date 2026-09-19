"""One objective: sum of supervised next-token losses / global token count."""

import torch
from torch.nn import functional as F


def token_count(labels):
    return labels[..., 1:].ne(-100).sum()


def loss_sum(logits, labels):
    # Sum remains differentiable and finite even for an entirely ignored batch.
    return F.cross_entropy(
        logits[..., :-1, :].reshape(-1, logits.size(-1)),
        labels[..., 1:].reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )


@torch.no_grad()
def normalize_gradients(parameters, global_tokens, dp_size=1):
    """Undo DDP's averaging, then normalize once before clipping/optimizer.step.

    TP/PP ranks contain copies/partitions of the same examples, so they must not
    contribute duplicate counts. Return False to skip an empty optimizer window
    (including Adam momentum and decoupled weight decay).
    """
    count = int(global_tokens.item())
    if count == 0:
        return False
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(dp_size / count)
    return True

