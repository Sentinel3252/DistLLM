# Design

DistLLM uses a small model so that individual gradients and communication operations are easy to inspect. This document explains how loss normalization and the three parallel modes work.

## Loss and gradient normalization

For data-parallel rank `r` and micro-batch `m`, let `S[r,m]` be the sum of supervised next-token losses and `N[r,m]` the number of labels not equal to `-100` after the causal shift. The desired objective is:

```text
L = sum(S[r,m]) / sum(N[r,m])
```

Each micro-batch calls cross entropy with `reduction="sum"`. Gradients accumulate without dividing by the number of micro-batches. After all micro-batches in an optimizer window, `normalize_gradients` divides the accumulated gradients by the global supervised-token count, compensating for DDP's gradient averaging when needed.

An all-empty window returns `False`, and callers skip `optimizer.step()`. This also prevents Adam momentum and decoupled weight decay from changing parameters when there is no supervised signal.

## Data parallelism

DDP averages gradient buckets across ranks. DistLLM asynchronously sums the token counts for the window and scales the final averaged gradients by:

```text
data_parallel_world_size / global_supervised_tokens
```

All non-final micro-batches run forward and backward inside `DDP.no_sync()`. The final backward triggers bucket reduction.

## Tensor parallelism

Only each block's MLP is sharded:

- The up projection is column-parallel (output features).
- The down projection is row-parallel (input features).
- Row-parallel outputs are summed with AllReduce.
- Backward input gradients are summed with AllReduce.

When overlap is enabled, the input-gradient AllReduce is submitted on a dedicated CUDA stream while the weight-gradient GEMM runs on the caller stream. A CUDA event makes the caller stream wait before using the input gradient. A GPU timeline is needed to check whether the kernels actually overlap and whether this reduces runtime.

## Pipeline parallelism

Transformer blocks are divided evenly across ranks. The first stage owns the embedding, the last owns normalization and the LM head, and adjacent stages exchange activations and gradients with Send/Recv. The schedule fills all micro-batch forwards, then drains backwards in reverse order.

## Communication records

`Comm` records communication calls made directly by DistLLM; framework-internal collectives may be absent from the trace. Submission and wait durations are measured on the CPU. CUDA event intervals can include time spent waiting on dependencies, so they measure more than NCCL kernel execution.

## Constraints

- All ranks execute the same number of optimizer windows.
- A DDP rank may have a different local sample or token count, but must have at least one local sample.
- TP and PP ranks operate on replicated samples, so their token counts are not summed across ranks.
- PP requires the layer count to be divisible by the number of stages.
- TP requires the MLP hidden width to be divisible by the number of ranks.

## Scope

The implementation covers a small Transformer and a limited set of training configurations. Each parallel mode is checked numerically against the full-batch model; broader model, optimizer, and cluster support is outside the current scope.
