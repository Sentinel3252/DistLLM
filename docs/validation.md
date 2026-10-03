# Validation and benchmarks

This document describes the correctness tests, saved benchmark results, and remaining multi-GPU checks.

## Correctness tests

Every distributed configuration is compared with an unpartitioned full-batch Transformer using the same initialization and examples. Tests compare:

1. token-normalized loss;
2. every parameter gradient;
3. every parameter after an AdamW update; and
4. whether an optimizer step is correctly skipped for an empty window.

The numerical suite uses FP64, a causal Transformer without dropout, and deterministic synthetic labels. It covers unequal token counts, micro-batches of different sizes, tail micro-batches, an empty rank, an entirely empty window, and synchronous/asynchronous TP execution.

## Distributed matrix

The checked-in Gloo results contain 32 configurations across 2 and 4 processes and 96 per-rank reports:

- DDP, TP, and PP;
- micro-batch sizes 1 and 3 with uneven labels;
- equal token counts;
- all-empty windows;
- a rank with no supervised labels; and
- a synchronous TP control.

Maximum observed FP64 errors against the full-batch oracle:

| Measurement | Maximum absolute error |
|---|---:|
| Gradient | `1.11e-16` |
| Weight after update | `9.58e-16` |

These maxima come from the two per-rank result files linked below. The Gloo tests and historical JUnit report used PyTorch `2.11.0+cu128`, the RTX 4090 benchmarks used `2.11.0+cu130`, and the saved normalization reproduction used `2.14.0+cpu`. Each set of results applies to its own environment. The current package dependency is PyTorch 2.14.

The current test suite also checks CPU FP32 DDP/TP/PP against a direct full-batch objective for two nonempty windows followed by an empty window. It uses per-parameter tolerances of `rtol=2e-5` and `atol=2e-7`. These tests cover neither mixed precision nor NCCL.

Result files:

- [Validation summary](../results/validation.json)
- [JUnit report](../results/pytest.xml)
- [Two-process runs](../results/verified-gloo-2ranks.json)
- [Four-process runs](../results/verified-gloo-4ranks.json)
- [Two-process collectives](../results/collectives-gloo-2ranks.json)
- [Four-process collectives](../results/collectives-gloo-4ranks.json)

## Normalization reproduction

`python -m distllm.reproduce` uses seed 23, CPU/FP64, and four micro-batches with `[1, 2, 4, 6]` supervised tokens.

| Metric | Mean of micro-batch means | Token-correct |
|---|---:|---:|
| Loss | 3.7084317621 | 3.7033826063 |
| Gradient relative L2 error | 0.7922870098 | 1.5052e-16 |

The full-batch reference loss is `3.7033826063`. This example isolates the normalization error; it does not measure the effect on model quality. See the [raw result](../results/reproduction.json).

In a separate skew sweep, the supervised-token max/min ratios were 1.0×, 1.29×, 1.94×, 3.88×, and 15.5×. Mean-of-means gradient relative L2 errors were 0, 0.081, 0.221, 0.454, and 1.007. The token-correct path matched the full-batch oracle throughout.

After 50 AdamW steps on a fixed synthetic workload, mean-of-means parameters differed from the oracle by 13.51% in relative L2 norm; token-correct parameters differed by `1.35e-16`.

## RTX 4090 study

The single-GPU benchmark uses a 23.08M-parameter Transformer, BF16, batch 16, sequence length 256, and 2,161 supervised tokens. Each point uses 5 warmups and 20 timed iterations.

The single-GPU timer measures forward, backward, and gradient normalization, excluding the optimizer update. The saved `median_step_ms` field therefore reports the duration of an accumulation window. The distributed runner includes AdamW updates in its timing, so its step times cannot be compared directly with these values.

| Micro-batch | Median step | P95 | Supervised tokens/s | Peak allocated |
|---:|---:|---:|---:|---:|
| 1 | 34.87 ms | 35.20 ms | 61,975 | 137.4 MiB |
| 2 | 17.56 ms | 17.64 ms | 123,065 | 167.7 MiB |
| 4 | 8.94 ms | 9.40 ms | 241,805 | 233.3 MiB |
| 8 | 6.59 ms | 6.68 ms | 327,790 | 352.5 MiB |
| 16 | 6.09 ms | 6.13 ms | 354,630 | 549.2 MiB |

Increasing the micro-batch size from 1 to 16 raised throughput by 5.72× and peak allocated memory from 137.4 to 549.2 MiB. In an ABBA comparison at micro-batch 4, token normalization took 0.67% less time than mean-of-means, a difference within measurement noise, with no increase in peak memory. These measurements apply to the small model above and do not establish large-model MFU. See the [raw result](../results/gpu-study-rtx4090.json).

## Multi-GPU status

NCCL execution, CUDA stream/event handling, and Nsight collection are implemented. The maintainers have yet to validate them on a multi-GPU host, so scaling and communication overlap remain unverified.

On a Linux host with at least four supported NVIDIA GPUs:

```bash
bash scripts/validate_gpu.sh
bash scripts/benchmark_gpu.sh
python -m distllm.scaling \
  results/bench-reference-1/result.rank0.json \
  results/bench-ddp-2-mb4/result.rank0.json
```

`scaling.py` rejects profiled/verification timings and mismatched workloads. `analyze.py` computes interval unions per GPU, so concurrently running kernels are not double-counted.

## Reproducing CPU validation

```bash
python -m distllm.reproduce
python -m distllm.gpu_study --quality-only
pytest -q
```

Tests use operating-system-assigned loopback ports. On Windows they use a TCPStore because a FileStore may not handle non-ASCII workspace paths reliably.
