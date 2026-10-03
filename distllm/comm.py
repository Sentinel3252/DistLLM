"""Explicit collectives; CPU submission timings are never called GPU latency."""

import json
import time
from pathlib import Path

import torch
import torch.distributed as dist


class Pending:
    def __init__(self, work, row, event=None, start=None):
        self.work, self.row = work, row
        self.event, self.start = event, start

    def wait(self):
        before = time.perf_counter_ns()
        if self.event is None:
            self.work.wait()
        else:
            torch.cuda.current_stream().wait_event(self.event)
        self.row["host_wait_us"] = (time.perf_counter_ns() - before) / 1000


class Comm:
    def __init__(self, device, group=None, trace=False):
        self.group = group
        self.size = dist.get_world_size(group)
        self.device = device
        self.trace = trace
        self.rows = []
        self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

    def all_reduce(self, tensor, label):
        row = self.record("all_reduce", tensor, label)
        before = time.perf_counter_ns()
        start = end = None
        if self.stream is None:
            work = dist.all_reduce(tensor, group=self.group, async_op=True)
        else:
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                start = torch.cuda.Event(enable_timing=True) if self.trace else None
                end = torch.cuda.Event(enable_timing=self.trace)
                if start is not None:
                    start.record()
                work = dist.all_reduce(tensor, group=self.group, async_op=True)
                # NCCL wait inserts a dependency on this stream; it does not
                # synchronize the host. The caller waits only at consumption.
                work.wait()
                end.record()
                tensor.record_stream(self.stream)
        row["host_submit_us"] = (time.perf_counter_ns() - before) / 1000
        pending = Pending(work, row, end, start)
        if self.trace:
            self.rows.append(pending)
        return pending

    def record(self, op, tensor, label):
        return {
            "rank": dist.get_rank(), "group_ranks": dist.get_process_group_ranks(self.group or dist.group.WORLD),
            "op": op, "label": label, "payload_bytes": tensor.numel() * tensor.element_size(),
            "caller_stream": torch.cuda.current_stream().cuda_stream if self.stream else None,
            "submission_stream": self.stream.cuda_stream if self.stream else None,
        }

    def p2p(self, tensor, peer, send):
        before = time.perf_counter_ns()
        (dist.send if send else dist.recv)(tensor, peer, group=self.group)
        if self.trace:
            row = self.record("send" if send else "recv", tensor, "pipeline_boundary")
            row["submission_stream"] = row["caller_stream"]
            row.update(peer=peer, host_call_us=(time.perf_counter_ns() - before) / 1000)
            self.rows.append(row)

    def save(self, path):
        records = []
        for entry in self.rows:
            if isinstance(entry, dict):
                records.append(entry)
                continue
            entry.work.wait()
            if entry.event is not None:
                entry.event.synchronize()
                entry.row["cuda_interval_ms"] = entry.start.elapsed_time(entry.event)
            records.append(entry.row)
        Path(path).write_text(json.dumps(records, indent=2), encoding="utf-8")


def ddp_hook(comm, bucket):
    # Match native DDP's averaged-gradient contract. Its future integrates with
    # the reducer; no additional gradient buffers or hooks on parameters.
    tensor = bucket.buffer()
    row = comm.record("all_reduce", tensor, "ddp_gradient_bucket")
    before = time.perf_counter_ns()
    work = dist.all_reduce(tensor, group=comm.group, async_op=True)
    row["host_submit_us"] = (time.perf_counter_ns() - before) / 1000
    # DDP launches through its current stream, not Comm's explicit TP stream.
    row["submission_stream"] = row["caller_stream"]
    if comm.trace:
        comm.rows.append(row)

    def finish(future):
        return future.value()[0].div_(comm.size)

    return work.get_future().then(finish)
