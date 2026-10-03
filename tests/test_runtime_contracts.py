import copy
import json
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from distllm.comm import Comm, ddp_hook
from distllm.loss import loss_sum, token_count
from distllm.model import Config, Transformer, pipeline_stage, tensor_parallel
from distllm.run import compare, fixture
from distllm.train import backward_window


def runtime_worker(rank, port, output):
    torch.set_num_threads(1)
    store = dist.TCPStore("127.0.0.1", port, is_master=False, use_libuv=False)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        for tracing in (False, True):
            comm = Comm(torch.device("cpu"), trace=tracing)
            value = torch.tensor([rank + 1.0])
            comm.all_reduce(value, "contract_sum").wait()
            torch.testing.assert_close(value, torch.tensor([3.0]), rtol=0, atol=0)
            value = torch.tensor([7.0]) if rank == 0 else torch.zeros(1)
            comm.p2p(value, 1 - rank, send=rank == 0)
            torch.testing.assert_close(value, torch.tensor([7.0]), rtol=0, atol=0)
            comm.save(Path(output) / f"comm-{tracing}.rank{rank}.json")

        for mode in ("ddp", "tp", "pp"):
            torch.manual_seed(101)
            original = Transformer(Config(layers=2)).float()
            reference = copy.deepcopy(original)
            comm = Comm(torch.device("cpu"))
            if mode == "ddp":
                model = DDP(original)
                model.register_comm_hook(comm, ddp_hook)
            elif mode == "tp":
                model = tensor_parallel(original, comm)
            else:
                model = pipeline_stage(original, rank, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
            ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001, weight_decay=0.1)
            errors = []
            for step, pattern in enumerate(("uneven", "uneven", "empty")):
                tokens, labels = fixture(10, 7, 32, torch.device("cpu"), seed=77 + step,
                                         pattern=pattern)
                local_x, local_y = (tokens[rank::2], labels[rank::2]) if mode == "ddp" else (tokens, labels)
                before = copy.deepcopy(model.state_dict())
                state_before = copy.deepcopy(optimizer.state_dict())
                loss, valid = backward_window(model, local_x, local_y, 3, mode, comm)
                # The oracle differentiates one full-batch objective directly.
                reference.zero_grad(set_to_none=True)
                count = token_count(labels)
                expected = loss_sum(reference(tokens), labels) / count.clamp_min(1)
                expected.backward()
                torch.testing.assert_close(loss, expected.detach(), rtol=2e-6, atol=2e-8)
                assert valid == (count.item() > 0)
                grad_error = compare(model, reference, mode, rank, 2, True)
                if valid:
                    optimizer.step()
                    ref_optimizer.step()
                else:
                    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
                    torch.testing.assert_close(optimizer.state_dict(), state_before, rtol=0, atol=0)
                weight_error = compare(model, reference, mode, rank, 2, False)
                errors.append(dict(gradient=grad_error, weight=weight_error, updated=valid))
            (Path(output) / f"fp32-{mode}.rank{rank}.json").write_text(
                json.dumps(errors), encoding="utf-8"
            )
    finally:
        dist.destroy_process_group()


def test_fp32_oracles_and_communication_records(tmp_path):
    store = dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False, use_libuv=False)
    mp.spawn(runtime_worker, args=(store.port, str(tmp_path)), nprocs=2, join=True)
    reports = list(tmp_path.glob("fp32-*.json"))
    assert len(reports) == 6
    for path in reports:
        steps = json.loads(path.read_text(encoding="utf-8"))
        assert [step["updated"] for step in steps] == [True, True, False]
    for rank in (0, 1):
        assert json.loads((tmp_path / f"comm-False.rank{rank}.json").read_text()) == []
        records = json.loads((tmp_path / f"comm-True.rank{rank}.json").read_text())
        assert len(records) == 2
        reduction, boundary = records
        assert reduction["op"] == "all_reduce" and reduction["label"] == "contract_sum"
        assert reduction["rank"] == rank and reduction["group_ranks"] == [0, 1]
        assert reduction["payload_bytes"] == 4
        assert reduction["host_submit_us"] >= 0 and reduction["host_wait_us"] >= 0
        assert "cuda_interval_ms" not in reduction
        assert reduction["caller_stream"] is None and reduction["submission_stream"] is None
        assert boundary["op"] == ("send" if rank == 0 else "recv")
        assert boundary["peer"] == 1 - rank and boundary["payload_bytes"] == 4
        assert boundary["host_call_us"] >= 0
