"""Launch a local CPU/Gloo experiment using an explicitly configured TCPStore."""

from datetime import timedelta

import torch.distributed as dist
import torch.multiprocessing as mp

from .run import experiment, parser, validate_args


def worker(rank, size, port, args):
    store = dist.TCPStore("127.0.0.1", port, is_master=False, use_libuv=False)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=size,
                            timeout=timedelta(seconds=90))
    try:
        experiment(args)
    finally:
        dist.destroy_process_group()


def main():
    p = parser()
    p.description = __doc__
    p.add_argument("--nproc-per-node", type=int, default=2)
    args = p.parse_args()
    validate_args(args)
    size = args.nproc_per_node
    if size < 1:
        p.error("--nproc-per-node must be positive")
    if args.backend != "gloo":
        p.error("the local CPU launcher requires --backend gloo")
    if args.mode == "reference" and size != 1:
        p.error("reference mode requires --nproc-per-node=1")
    if args.mode == "ddp" and args.batch < size:
        p.error("batch must contain at least one example per DDP rank")
    if args.mode == "pp" and args.layers % size:
        p.error("layers must be divisible by the number of pipeline stages")
    if args.mode == "tp" and args.width * 4 % size:
        p.error("MLP hidden width must be divisible by the number of TP ranks")
    store = dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False, use_libuv=False)
    mp.spawn(worker, args=(size, store.port, args), nprocs=size, join=True)


if __name__ == "__main__":
    main()
