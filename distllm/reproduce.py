"""Reproduce #1728's mean-of-microbatch-means mechanism without its ML stack."""

import argparse
import copy
import json
from pathlib import Path

import torch

from .loss import loss_sum, token_count
from .model import Config, Transformer
from .train import backward_window


def reproduce():
    torch.set_num_threads(1)
    torch.manual_seed(23)
    reference = Transformer(Config(layers=1)).double()
    wrong, fixed = copy.deepcopy(reference), copy.deepcopy(reference)
    x = torch.randint(32, (4, 7))
    labels = x.clone()
    labels[0, 2:] = -100
    labels[1, 3:] = -100
    labels[2, 5:] = -100
    expected, _ = backward_window(reference, x, labels, 4, "reference")
    got, _ = backward_window(fixed, x, labels, 1, "reference")
    old_loss = 0.0
    for a, b in zip(x.split(1), labels.split(1)):
        loss = loss_sum(wrong(a), b) / token_count(b) / 4
        old_loss += loss.detach().item()
        loss.backward()
    reference_gradient = torch.cat([p.grad.flatten() for p in reference.parameters()])
    wrong_gradient = torch.cat([p.grad.flatten() for p in wrong.parameters()])
    fixed_gradient = torch.cat([p.grad.flatten() for p in fixed.parameters()])
    denominator = reference_gradient.norm()
    return dict(torch=torch.__version__, dtype="float64", device="cpu", seed=23,
                supervised_tokens_per_microbatch=[token_count(y).item() for y in labels.split(1)],
                reference_loss=expected.item(), mean_of_means_loss=old_loss, fixed_loss=got.item(),
                wrong_gradient_relative_l2=((wrong_gradient - reference_gradient).norm() / denominator).item(),
                fixed_gradient_relative_l2=((fixed_gradient - reference_gradient).norm() / denominator).item(),
                fixed_gradient_max_abs=(fixed_gradient - reference_gradient).abs().max().item())


def main():
    """Run the deterministic normalization reproduction."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", default="results/reproduction.json")
    args = parser.parse_args()
    result = reproduce()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
