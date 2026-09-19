import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distllm.run import experiment, parser
from distllm.collectives import benchmark


def worker(rank, size, port, output):
    torch.set_num_threads(1)
    store = dist.TCPStore("127.0.0.1", port, is_master=False, use_libuv=False)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=size,
                            timeout=timedelta(seconds=60))
    try:
        collective_results = benchmark(torch.device("cpu"), elements=16, repeats=2)
        if rank == 0:
            (Path(output) / "collectives.json").write_text(json.dumps(collective_results, indent=2))
        for mode in ("ddp", "tp", "pp"):
            for pattern, microbatch in (("uneven", 1), ("uneven", 3), ("equal", 2), ("empty", 3), ("empty-rank", 2)):
                args = parser().parse_args([
                    "--mode", mode, "--verify", "--trace", "--steps", "2",
                    "--batch", "10", "--pattern", pattern, "--microbatch", str(microbatch),
                    "--output", str(Path(output) / f"{mode}-{pattern}-mb{microbatch}"),
                ])
                experiment(args)
        args = parser().parse_args([
            "--mode", "tp", "--verify", "--no-overlap", "--steps", "2",
            "--output", str(Path(output) / "tp-serial"),
        ])
        experiment(args)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("size", [2, 4])
def test_real_processes_match_reference(size, tmp_path):
    store = dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False, use_libuv=False)
    mp.spawn(worker, args=(size, store.port, str(tmp_path)), nprocs=size, join=True)
    reports = list(tmp_path.glob("*/result.rank*.json"))
    assert len(reports) == 16 * size
    evidence = []
    for report in reports:
        record = json.loads(report.read_text())
        assert record["verified"]
        assert all(step["grad_max_abs_error"] < 2e-7 for step in record["steps"])
        evidence.append(record)
    destination = Path("results")
    destination.mkdir(exist_ok=True)
    (destination / f"verified-gloo-{size}ranks.json").write_text(json.dumps(evidence, indent=2))
    (destination / f"collectives-gloo-{size}ranks.json").write_text((tmp_path / "collectives.json").read_text())
