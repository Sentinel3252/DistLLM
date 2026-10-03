"""Analyze GPU kernel intervals from PyTorch Chrome traces, per device/process."""

import argparse
import json
from pathlib import Path


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("negative interval")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def metrics(compute, communication):
    compute, communication = union(compute), union(communication)
    overlap = sum(max(0, min(b, d) - max(a, c)) for a, b in compute for c, d in communication)
    compute_time = sum(b - a for a, b in compute)
    communication_time = sum(b - a for a, b in communication)
    busy = sum(b - a for a, b in union(compute + communication))
    return dict(compute_us=compute_time, communication_us=communication_time, overlap_us=overlap,
                communication_hidden_fraction=overlap / communication_time if communication_time else None,
                communication_busy_fraction=communication_time / busy if busy else None)


def ring_bytes(op, payload_bytes, ranks):
    """Per-rank transmitted bytes in an ideal ring, excluding protocol overhead.

    payload_bytes = local input size (AllGather), full input size (other ops).
    This is a model, not measured NCCL traffic or an algorithm assumption.
    """
    if ranks < 1 or payload_bytes < 0:
        raise ValueError("positive ranks and nonnegative payload required")
    factors = {"all_reduce": 2 * (ranks - 1) / ranks,
               "reduce_scatter": (ranks - 1) / ranks, "all_gather": ranks - 1}
    return payload_bytes * factors[op]


def analyze(path):
    groups = {}
    for event in json.loads(Path(path).read_text(encoding="utf-8"))["traceEvents"]:
        if event.get("cat") != "kernel" or event.get("ph") != "X":
            continue
        key = str(event.get("args", {}).get("device", event.get("pid")))
        compute, communication = groups.setdefault(key, ([], []))
        target = communication if "nccl" in event["name"].lower() else compute
        target.append((event["ts"], event["ts"] + event["dur"]))
    if not groups:
        raise ValueError("No GPU kernel events; CPU submission traces cannot measure GPU overlap")
    return {device: metrics(*intervals) for device, intervals in groups.items()}


def main():
    """Run the trace analyzer CLI."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("trace")
    print(json.dumps(analyze(parser.parse_args().trace), indent=2))


if __name__ == "__main__":
    main()

