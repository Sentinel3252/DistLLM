# Roadmap

The items below describe possible next steps. They are not tied to a release schedule.

## Near term

- Validate NCCL correctness and CUDA stream lifetimes on 2- and 4-GPU hosts.
- Publish Nsight Systems traces to check whether TP communication overlaps with computation.
- Add mixed-precision correctness tests with explicit tolerances.
- Exercise source and wheel builds in release automation.

## Later

- Define a composable process-mesh abstraction for combined data and model parallelism.
- Add deterministic checkpoint/resume coverage at optimizer-window boundaries.
- Add a small packed-sequence data example with supervised-token counting and normalization checks.
- Compare token normalization with selected production training frameworks.

## Out of scope for now

- A full model zoo or dataset pipeline.
- Cluster scheduling and fault tolerance.
- Claims about end-model quality without a controlled training study.

For each proposal, describe the implementation and how its results will be checked against an independent reference.
