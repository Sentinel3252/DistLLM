"""One optimizer window per call. All ranks must execute the same window count."""

from contextlib import nullcontext

import torch
import torch.distributed as dist

from .loss import loss_sum, normalize_gradients, token_count


def backward_window(model, tokens, labels, microbatch, mode, comm=None):
    if tokens.ndim != 2 or labels.ndim != 2 or tokens.shape != labels.shape:
        raise ValueError("tokens and labels must have identical [batch, sequence] shapes")
    if microbatch < 1 or tokens.size(0) == 0:
        raise ValueError("microbatch and local batch must be positive")
    if tokens.size(1) < 2:
        raise ValueError("sequence must contain at least two tokens for causal loss")
    if mode not in {"reference", "ddp", "tp", "pp"}:
        raise ValueError(f"unsupported training mode: {mode}")
    if mode in {"ddp", "pp"} and comm is None:
        raise ValueError(f"{mode} requires a communication object")
    model.zero_grad(set_to_none=True)
    count = token_count(labels)
    count_work = comm.all_reduce(count, "dp_token_count") if mode == "ddp" else None
    chunks = list(zip(tokens.split(microbatch), labels.split(microbatch)))
    total = torch.zeros((), dtype=next(model.parameters()).dtype, device=tokens.device)
    if mode == "pp":
        _pipeline_backward(model, chunks, total, comm)
    else:
        for index, (x, y) in enumerate(chunks):
            context = model.no_sync() if mode == "ddp" and index < len(chunks) - 1 else nullcontext()
            # DDP no_sync must enclose both forward and backward.
            with context, torch.profiler.record_function("microbatch"):
                loss = loss_sum(model(x), y)
                total += loss.detach()
                loss.backward()
    if count_work is not None:
        count_work.wait()
        comm.all_reduce(total, "dp_loss_sum").wait()
    valid = normalize_gradients(model.parameters(), count, comm.size if mode == "ddp" else 1)
    return total / count.clamp_min(1), valid


def _pipeline_backward(stage, chunks, total, comm):
    rank, size = dist.get_rank(), comm.size
    saved = []
    parameter = next(stage.parameters())
    for x, y in chunks:
        if rank > 0:
            # Every non-first stage begins with a Transformer block.
            width = stage[0].norm1.normalized_shape[0]
            x = torch.empty((*x.shape, width), dtype=parameter.dtype, device=x.device)
            comm.p2p(x, rank - 1, send=False)
            x.requires_grad_()
        output = stage(x)
        if rank < size - 1:
            comm.p2p(output.detach().contiguous(), rank + 1, send=True)
        saved.append((x, output, y))
    # GPipe: drain all forwards before reverse-order backwards.
    for x, output, y in reversed(saved):
        if rank == size - 1:
            loss = loss_sum(output, y)
            total += loss.detach()
            loss.backward()
        else:
            gradient = torch.empty_like(output)
            comm.p2p(gradient, rank + 1, send=False)
            output.backward(gradient)
        if rank > 0:
            comm.p2p(x.grad.contiguous(), rank - 1, send=True)
    dist.broadcast(total, src=size - 1, group=comm.group)
