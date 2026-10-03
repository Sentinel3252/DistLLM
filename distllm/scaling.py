"""Compare unprofiled strong-scaling runs with a matching one-rank baseline."""

import argparse
import json
import math
import statistics
from pathlib import Path


def _validate(record, label):
    required = ("timing_kind", "world_size", "mode", "backend", "torch", "dtype", "model",
                "sequence", "batch", "pattern", "device_name", "steps")
    for field in required:
        if field not in record:
            raise ValueError(f"{label}: missing {field}")
    size = record["world_size"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError(f"{label}: world_size must be a positive integer")
    if not isinstance(record["steps"], list) or not record["steps"]:
        raise ValueError(f"{label}: steps must be a nonempty list")
    for index, step in enumerate(record["steps"]):
        duration = step.get("rank_max_seconds") if isinstance(step, dict) else None
        if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or duration <= 0):
            raise ValueError(f"{label}: steps[{index}].rank_max_seconds must be finite and positive")


def compare(reference, candidate):
    _validate(reference, "reference")
    _validate(candidate, "candidate")
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


def main():
    """Run the strong-scaling comparison CLI."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()
    print(json.dumps(compare(json.loads(Path(args.reference).read_text()),
                             json.loads(Path(args.candidate).read_text())), indent=2))


if __name__ == "__main__":
    main()
