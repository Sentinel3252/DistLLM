# Validation and benchmarks

This document separates verified evidence from execution paths that still require suitable hardware.

## Correctness strategy

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
| Gradient | `9.72e-17` |
| Weight after update | `7.22e-16` |

Raw evidence:

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

The full-batch reference loss is `3.7033826063`. This is a mechanism reproduction, not a claim about universal model-quality impact. See the [raw result](../results/reproduction.json).

In a separate skew sweep, the supervised-token max/min ratios were 1.0×, 1.29×, 1.94×, 3.88×, and 15.5×. Mean-of-means gradient relative L2 errors were 0, 0.081, 0.221, 0.454, and 1.007. The token-correct path matched the full-batch oracle throughout.

After 50 AdamW steps on a fixed synthetic workload, mean-of-means parameters differed from the oracle by 13.51% in relative L2 norm; token-correct parameters differed by `1.35e-16`.

## RTX 4090 study

The single-GPU benchmark uses a 23.08M-parameter Transformer, BF16, batch 16, sequence length 256, and 2,161 supervised tokens. Each point uses 5 warmups and 20 timed iterations.

| Micro-batch | Median step | P95 | Supervised tokens/s | Peak allocated |
|---:|---:|---:|---:|---:|
| 1 | 34.87 ms | 35.20 ms | 61,975 | 137.4 MiB |
| 2 | 17.56 ms | 17.64 ms | 123,065 | 167.7 MiB |
| 4 | 8.94 ms | 9.40 ms | 241,805 | 233.3 MiB |
| 8 | 6.59 ms | 6.68 ms | 327,790 | 352.5 MiB |
| 16 | 6.09 ms | 6.13 ms | 354,630 | 549.2 MiB |

Throughput increased 5.72× from micro-batch 1 to 16, exposing the utilization/memory tradeoff. An ABBA comparison at micro-batch 4 measured token-correct step time at -0.67% relative to mean-of-means (noise scale) with no peak-memory regression. This is a small-model step benchmark, not a large-model MFU claim. See the [raw result](../results/gpu-study-rtx4090.json).

## Multi-GPU status

NCCL execution, CUDA stream/event handling, and Nsight collection are implemented but have not been validated by the maintainers on a multi-GPU host. Do not interpret the checked-in single-GPU data as scaling or communication-overlap evidence.

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
