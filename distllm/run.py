"""torchrun entry point; correctness uses a separate unpartitioned reference."""

import argparse
import copy
import json
import os
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .comm import Comm, ddp_hook
from .model import Config, Transformer, pipeline_stage, tensor_parallel
from .train import backward_window


def fixture(batch, sequence, vocab, device, seed=77, pattern="uneven"):
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(vocab, (batch, sequence), generator=generator).to(device)
    labels = tokens.clone()
    if pattern == "empty":
        labels.fill_(-100)
    elif pattern == "uneven":
        for i in range(batch):
            labels[i, :1 + (i * 3) % sequence] = -100
        labels[0].fill_(-100)
    return tokens, labels


def compare(model, reference, mode, rank, size, gradients):
    errors = []
    ref = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        name = name.removeprefix("module.")
        if mode == "pp" and name.startswith("block_"):
            index, suffix = name.split(".", 1)
            name = "blocks." + index.removeprefix("block_") + "." + suffix
        expected = ref[name].grad if gradients else ref[name].detach()
        actual = parameter.grad if gradients else parameter.detach()
        if mode == "tp":
            if ".mlp.up.weight" in name:
                expected = expected.chunk(size, dim=0)[rank]
            elif ".mlp.down.weight" in name:
                expected = expected.chunk(size, dim=1)[rank]
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7, msg=lambda msg: name + ": " + msg)
        errors.append((actual - expected).abs().max().item())
    return max(errors)


def experiment(args):
    rank, size = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank))) if args.backend == "nccl" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.manual_seed(11)
    config = Config(width=args.width, layers=args.layers)
    dtype = torch.float64 if args.verify else torch.float32
    original = Transformer(config).to(device=device, dtype=dtype)
    reference = copy.deepcopy(original) if args.verify else None
    comm = Comm(device, trace=args.trace)
    if args.mode == "ddp":
        model = DDP(original, device_ids=[device.index] if device.type == "cuda" else None)
        model.register_comm_hook(comm, ddp_hook)
    elif args.mode == "tp":
        model = tensor_parallel(original, comm, args.overlap)
    elif args.mode == "pp":
        model = pipeline_stage(original, rank, size)
    else:
        if size != 1:
            raise ValueError("reference mode requires world_size=1")
        model = original
    del original
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
    ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001, weight_decay=0.1) if reference else None
    tokens, labels = fixture(args.batch, args.sequence, config.vocab, device, pattern=args.pattern)
    if args.pattern == "empty-rank":
        labels[::size] = -100
    if args.mode == "ddp":
        if args.batch < size:
            raise ValueError("batch must contain at least one example per DDP rank")
        local_x, local_y = tokens[rank::size], labels[rank::size]
    else:
        local_x, local_y = tokens, labels
    results = []
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profiler = torch.profiler.profile(activities=activities) if args.profile else None
    if profiler:
        profiler.start()
    for step in range(args.warmup + args.steps):
        dist.barrier()
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.profiler.record_function("training_step"):
            loss, valid = backward_window(model, local_x, local_y, args.microbatch, args.mode, comm)
            grad_error = weight_error = None
            # Verification is deliberately excluded from performance claims.
            if reference:
                expected_loss, expected_valid = backward_window(reference, tokens, labels, args.batch, "reference")
                torch.testing.assert_close(loss, expected_loss, rtol=2e-6, atol=2e-8)
                assert valid == expected_valid
                grad_error = compare(model, reference, args.mode, rank, size, True)
            if valid:
                optimizer.step()
                if reference:
                    ref_optimizer.step()
            if reference:
                weight_error = compare(model, reference, args.mode, rank, size, False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = torch.tensor(time.perf_counter() - start, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        if step >= args.warmup:
            results.append(dict(step=step - args.warmup, loss=loss.item(), updated=valid,
                                rank_max_seconds=elapsed.item(), grad_max_abs_error=grad_error,
                                weight_max_abs_error=weight_error))
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    if profiler:
        profiler.stop()
        profiler.export_chrome_trace(str(destination / f"profile.rank{rank}.json"))
    if args.trace:
        comm.save(destination / f"comm.rank{rank}.json")
    report = dict(mode=args.mode, backend=args.backend, world_size=size, rank=rank,
                  torch=torch.__version__, device=str(device), dtype=str(dtype),
                  model=asdict(config), sequence=args.sequence,
                  device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                  batch=args.batch, microbatch=args.microbatch, pattern=args.pattern,
                  overlap=args.overlap, verified=args.verify,
                  timing_kind="verification_or_profile" if args.verify or args.trace or args.profile else "benchmark",
                  steps=results)
    (destination / f"result.rank{rank}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if rank == 0:
        print(json.dumps(report, indent=2), flush=True)


def parser():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--mode", choices=["reference", "ddp", "tp", "pp"], default="ddp")
    p.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--sequence", type=int, default=7)
    p.add_argument("--microbatch", type=int, default=2)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--pattern", choices=["uneven", "equal", "empty", "empty-rank"], default="uneven")
    p.add_argument("--overlap", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--output", default="results/run")
    return p


def validate_args(args):
    if args.width % 4 or args.sequence < 2 or min(args.steps, args.batch, args.layers, args.width, args.microbatch) < 1 or args.warmup < 0:
        raise ValueError("width must be divisible by 4; sequence >= 2; positive steps/batch/layers required")


def main():
    args = parser().parse_args()
    validate_args(args)
    if args.backend == "nccl":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device_id = int(os.environ["LOCAL_RANK"]) if args.backend == "nccl" else None
    dist.init_process_group(
        args.backend,
        timeout=timedelta(seconds=90),
        device_id=torch.device("cuda", device_id) if device_id is not None else None,
    )
    try:
        experiment(args)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
