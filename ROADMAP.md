# Roadmap

DistLLM grows through small, verifiable increments. The following items are candidates rather than release promises.

## Near term

- Validate NCCL correctness and CUDA stream lifetimes on 2- and 4-GPU hosts.
- Publish Nsight Systems traces that demonstrate or falsify TP communication overlap.
- Add mixed-precision correctness tests with explicit tolerances.
- Exercise source and wheel builds in release automation.

## Later

- Define a composable process-mesh abstraction for combined data and model parallelism.
- Add deterministic checkpoint/resume coverage at optimizer-window boundaries.
- Integrate a small packed-sequence data path without obscuring the objective contract.
- Compare the normalization contract with selected production training frameworks.

## Out of scope for now

- A full model zoo or dataset pipeline.
- Cluster scheduling and fault tolerance.
- Claims about end-model quality without a controlled training study.

Proposals should explain how they preserve auditability and how correctness will be checked against an independent oracle.
