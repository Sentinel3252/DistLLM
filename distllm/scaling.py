"""Compare unprofiled strong-scaling runs with a matching one-rank baseline."""

import argparse
import json
import statistics
from pathlib import Path


def compare(reference, candidate):
    for record in (reference, candidate):
        if record["timing_kind"] != "benchmark":
            raise ValueError("verification/profile timings are not benchmark results")
    if reference["world_size"] != 1 or reference["mode"] != "reference":
        raise ValueError("baseline must be a one-rank reference run")
    if reference["pattern"] == "empty-rank":
        raise ValueError("empty-rank changes the workload with world size")
    for field in ("backend", "torch", "dtype", "model", "sequence", "batch", "pattern", "device_name"):
        if reference[field] != candidate[field]:
            raise ValueError(f"non-comparable {field}")
    baseline = statistics.median(step["rank_max_seconds"] for step in reference["steps"])
    duration = statistics.median(step["rank_max_seconds"] for step in candidate["steps"])
    speedup = baseline / duration
    return dict(median_step_seconds=duration, speedup=speedup,
                strong_scaling_efficiency=speedup / candidate["world_size"],
                input_tokens_per_second=candidate["batch"] * candidate["sequence"] / duration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()
    print(json.dumps(compare(json.loads(Path(args.reference).read_text()),
                             json.loads(Path(args.candidate).read_text())), indent=2))
