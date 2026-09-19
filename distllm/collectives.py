"""AllReduce / AllGather / ReduceScatter correctness and latency microbenchmark."""

import argparse
import json
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from .analyze import ring_bytes


def benchmark(device, elements=4096, repeats=10):
    rank, size = dist.get_rank(), dist.get_world_size()
    reports = []
    for op in ("all_reduce", "all_gather", "reduce_scatter"):
        count = elements * size if op == "reduce_scatter" else elements
        source = torch.full((count,), rank + 1.0, device=device)
        output = torch.empty(elements * size if op == "all_gather" else elements, device=device)
        samples = []
        for iteration in range(repeats + 2):
            value = source.clone()
            dist.barrier()
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            if op == "all_reduce":
                work = dist.all_reduce(value, async_op=True)
            elif op == "all_gather":
                work = dist.all_gather_single(output, value, async_op=True)
            else:
                work = dist.reduce_scatter_single(output, value, async_op=True)
            work.wait()
            if device.type == "cuda":
                torch.cuda.synchronize()
            duration = torch.tensor(time.perf_counter() - start, device=device)
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
            actual = value if op == "all_reduce" else output
            if op == "all_gather":
                expected = torch.arange(1, size + 1, device=device).repeat_interleave(elements).float()
            else:
                expected = torch.full_like(actual, size * (size + 1) / 2)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if iteration >= 2:
                samples.append(duration.item())
        payload = source.numel() * source.element_size()
        reports.append(dict(op=op, payload_bytes=payload, ranks=size,
                            ideal_ring_tx_bytes=ring_bytes(op, payload, size),
                            rank_max_median_seconds=statistics.median(samples),
                            samples_seconds=samples, verified=True))
    return reports


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    parser.add_argument("--elements", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", default="results/collectives.json")
    args = parser.parse_args()
    if min(args.elements, args.repeats) < 1:
        raise ValueError("elements and repeats must be positive")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"])) if args.backend == "nccl" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(args.backend, timeout=timedelta(seconds=90))
    try:
        result = dict(backend=args.backend, torch=torch.__version__, operations=benchmark(device, args.elements, args.repeats))
        if dist.get_rank() == 0:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
