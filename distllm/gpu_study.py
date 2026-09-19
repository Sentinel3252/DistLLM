"""Single-GPU study of token-correct accumulation quality and overhead."""

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch

from .loss import loss_sum, token_count
from .model import Config, Transformer
from .train import backward_window


def _gradient(model):
    return torch.cat([parameter.grad.float().flatten() for parameter in model.parameters()])


def _wrong_backward(model, tokens, labels, microbatch):
    """Reproduce mean-of-microbatch-means used by the faulty objective."""
    model.zero_grad(set_to_none=True)
    total = torch.zeros((), device=tokens.device, dtype=next(model.parameters()).dtype)
    chunks = 0
    for x, y in zip(tokens.split(microbatch), labels.split(microbatch)):
        count = token_count(y)
        if count:
            loss = loss_sum(model(x), y) / count
            total += loss.detach()
            loss.backward()
            chunks += 1
    if chunks:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(chunks)
    return total / max(chunks, 1)


def quality_study():
    """Measure gradient and optimizer-trajectory error as length skew grows."""
    torch.manual_seed(123)
    config = Config(vocab=64, width=32, layers=2)
    initial = Transformer(config).double()
    tokens = torch.randint(config.vocab, (8, 32))
    rows = []
    for minimum in (31, 24, 16, 8, 2):
        labels = tokens.clone()
        lengths = torch.linspace(minimum, 31, 8).round().long()
        for index, length in enumerate(lengths):
            labels[index, length + 1 :] = -100
        reference, fixed, wrong = (copy.deepcopy(initial) for _ in range(3))
        backward_window(reference, tokens, labels, 8, "reference")
        backward_window(fixed, tokens, labels, 1, "reference")
        _wrong_backward(wrong, tokens, labels, 1)
        target = _gradient(reference)
        denominator = target.norm()
        rows.append(
            {
                "minimum_supervised_tokens": minimum,
                "maximum_supervised_tokens": 31,
                "token_imbalance_ratio": 31 / minimum,
                "mean_of_means_gradient_relative_l2": float(
                    (_gradient(wrong) - target).norm() / denominator
                ),
                "token_correct_gradient_relative_l2": float(
                    (_gradient(fixed) - target).norm() / denominator
                ),
            }
        )

    labels = tokens.clone()
    lengths = torch.tensor([2, 3, 5, 8, 13, 19, 25, 31])
    for index, length in enumerate(lengths):
        labels[index, length + 1 :] = -100
    reference, fixed, wrong = (copy.deepcopy(initial) for _ in range(3))
    optimizers = [torch.optim.AdamW(model.parameters(), lr=2e-3) for model in (reference, fixed, wrong)]
    for _ in range(50):
        backward_window(reference, tokens, labels, 8, "reference")
        backward_window(fixed, tokens, labels, 1, "reference")
        _wrong_backward(wrong, tokens, labels, 1)
        for optimizer in optimizers:
            optimizer.step()
    reference_parameters = torch.cat([p.detach().flatten() for p in reference.parameters()])
    fixed_parameters = torch.cat([p.detach().flatten() for p in fixed.parameters()])
    wrong_parameters = torch.cat([p.detach().flatten() for p in wrong.parameters()])
    with torch.no_grad():
        reference_objective = loss_sum(reference(tokens), labels) / token_count(labels)
        fixed_objective = loss_sum(fixed(tokens), labels) / token_count(labels)
        wrong_objective = loss_sum(wrong(tokens), labels) / token_count(labels)
    trajectory = {
        "steps": 50,
        "supervised_tokens_per_sample": lengths.tolist(),
        "token_correct_parameter_relative_l2": float(
            (fixed_parameters - reference_parameters).norm() / reference_parameters.norm()
        ),
        "mean_of_means_parameter_relative_l2": float(
            (wrong_parameters - reference_parameters).norm() / reference_parameters.norm()
        ),
        "reference_token_objective": float(reference_objective),
        "token_correct_token_objective": float(fixed_objective),
        "mean_of_means_token_objective": float(wrong_objective),
    }
    return {"gradient_skew_sweep": rows, "optimizer_trajectory": trajectory}


def _labels(tokens):
    labels = tokens.clone()
    sequence = tokens.size(1)
    lengths = torch.linspace(max(2, sequence // 16), sequence - 1, tokens.size(0)).long()
    for index, length in enumerate(lengths):
        labels[index, length + 1 :] = -100
    return labels


def _timed(model, tokens, labels, microbatch, method, repeats, warmup):
    elapsed = []
    torch.cuda.reset_peak_memory_stats()
    for iteration in range(warmup + repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        if method == "token_correct":
            backward_window(model, tokens, labels, microbatch, "reference")
        else:
            _wrong_backward(model, tokens, labels, microbatch)
        torch.cuda.synchronize()
        if iteration >= warmup:
            elapsed.append(time.perf_counter() - start)
    elapsed.sort()
    median = statistics.median(elapsed)
    supervised = int(token_count(labels))
    return {
        "method": method,
        "microbatch": microbatch,
        "median_step_ms": median * 1000,
        "p95_step_ms": elapsed[min(len(elapsed) - 1, int(len(elapsed) * 0.95))] * 1000,
        "supervised_tokens_per_second": supervised / median,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "samples_per_second": tokens.size(0) / median,
    }


def performance_study(repeats, warmup):
    """Sweep accumulation granularity and quantify correction overhead."""
    device = torch.device("cuda")
    torch.manual_seed(321)
    torch.cuda.manual_seed_all(321)
    config = Config(vocab=4096, width=512, layers=6)
    model = Transformer(config).to(device=device, dtype=torch.bfloat16)
    tokens = torch.randint(config.vocab, (16, 256), device=device)
    labels = _labels(tokens)
    rows = []
    for microbatch in (1, 2, 4, 8, 16):
        rows.append(_timed(model, tokens, labels, microbatch, "token_correct", repeats, warmup))
    # Paired comparison at a useful middle point. Alternate order to limit drift.
    paired = []
    for method in ("mean_of_means", "token_correct", "token_correct", "mean_of_means"):
        paired.append(_timed(model, tokens, labels, 4, method, repeats, warmup))
    correct = statistics.median(row["median_step_ms"] for row in paired if row["method"] == "token_correct")
    wrong = statistics.median(row["median_step_ms"] for row in paired if row["method"] == "mean_of_means")
    return {
        "device_name": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": "bfloat16",
        "model": {**config.__dict__, "parameters": sum(p.numel() for p in model.parameters())},
        "batch": 16,
        "sequence": 256,
        "supervised_tokens": int(token_count(labels)),
        "repeats": repeats,
        "warmup": warmup,
        "microbatch_sweep": rows,
        "paired_mb4": paired,
        "token_correct_time_overhead_percent": (correct / wrong - 1) * 100,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", default="results/gpu-study.json")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--quality-only", action="store_true")
    args = parser.parse_args()
    result = {"quality": quality_study()}
    if not args.quality_only:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required unless --quality-only is set")
        result["performance"] = performance_study(args.repeats, args.warmup)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
