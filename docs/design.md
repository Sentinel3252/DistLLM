# Design

DistLLM keeps the model deliberately small so that every gradient and communication boundary can be inspected. This document describes the contracts behind the implementation.

## Objective contract

For data-parallel rank `r` and micro-batch `m`, let `S[r,m]` be the sum of supervised next-token losses and `N[r,m]` the number of labels not equal to `-100` after the causal shift. The desired objective is:

```text
L = sum(S[r,m]) / sum(N[r,m])
```

Each micro-batch therefore calls cross entropy with `reduction="sum"`. Gradients accumulate without dividing by the number of micro-batches. Once the complete optimizer window has been processed, `normalize_gradients` applies the single global denominator.

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

When overlap is enabled, the input-gradient AllReduce is submitted on a dedicated CUDA stream while the weight-gradient GEMM runs on the caller stream. A CUDA event establishes the dependency before the input gradient is consumed. Actual kernel overlap must be verified with a GPU timeline; asynchronous submission alone is not evidence of speedup.

## Pipeline parallelism

Transformer blocks are divided evenly across ranks. The first stage owns the embedding, the last owns normalization and the LM head, and adjacent stages exchange activations and gradients with Send/Recv. The schedule fills all micro-batch forwards, then drains backwards in reverse order.

## Communication records

`Comm` records operations explicitly issued by this project. It does not intercept every framework-internal collective. Host submission and wait durations are CPU measurements; CUDA event intervals include dependency effects and are not presented as pure NCCL kernel time.

## Invariants

- All ranks execute the same number of optimizer windows.
- A DDP rank may have a different local sample or token count, but must have at least one local sample.
- TP and PP ranks operate on replicated samples, so their token counts are not summed across ranks.
- PP requires the layer count to be divisible by the number of stages.
- TP requires the MLP hidden width to be divisible by the number of ranks.

## Non-goals

The code is not a drop-in replacement for a production training stack. It favors transparent invariants and numerical verification over broad model, optimizer, or cluster support.
