<p align="center">
  <img src="assets/logo.png" width="160" alt="DistLLM logo">
</p>

# DistLLM

**Correct gradients for variable-length language-model training—across gradient accumulation, DDP, tensor parallelism, and pipeline parallelism.**

[![CI](https://github.com/Sentinel3252/DistLLM/actions/workflows/ci.yml/badge.svg)](https://github.com/Sentinel3252/DistLLM/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

[中文文档](README.zh-CN.md) · [Design](docs/design.md) · [Validation](docs/validation.md) · [Contributing](CONTRIBUTING.md)

DistLLM is a compact PyTorch reference implementation for token-correct causal language-model training. It demonstrates how to preserve the global supervised-token objective when batches have different sequence lengths and work is split across micro-batches or distributed ranks.

The project includes runnable DDP, MLP tensor-parallel, and GPipe-style pipeline-parallel paths; communication tracing; reproducible correctness checks; and GPU profiling scripts. It is intentionally small enough to audit, modify, and use as a systems-learning or regression-testing foundation.

## Why DistLLM?

With variable-length labels, averaging each micro-batch's mean loss optimizes the wrong objective:

```text
mean(S_i / N_i)  !=  sum(S_i) / sum(N_i)
```

Here, `S_i` is the summed next-token loss and `N_i` is the number of supervised tokens. DistLLM accumulates loss sums, aggregates token counts at the correct parallel boundary, and normalizes gradients exactly once before the optimizer step.

This matters whenever padding, masking, packing, or per-rank data skew causes supervised-token counts to differ.

## Highlights

- One token-normalization contract shared by reference, DDP, tensor-parallel, and pipeline-parallel execution.
- Native PyTorch Distributed primitives: AllReduce, AllGather, ReduceScatter, and Send/Recv.
- DDP `no_sync` accumulation with correct compensation for DDP's averaged gradients.
- Column/row MLP tensor parallelism with optional CUDA stream/event overlap.
- GPipe fill/drain scheduling across Transformer blocks.
- Full-batch numerical oracle with per-parameter gradient and optimizer-update checks.
- Machine-readable results and scripts for Gloo, NCCL, CUDA, and Nsight Systems.

## Quick start

Requirements: Python 3.11+ and PyTorch 2.14. A CPU-only installation is enough for the reproduction and Gloo tests.

```bash
git clone https://github.com/Sentinel3252/DistLLM.git
cd DistLLM
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Reproduce the normalization error and verify the fix:

```bash
distllm-reproduce
pytest -q
```

Run a two-process DDP correctness check on CPU:

```bash
torchrun --standalone --nproc-per-node=2 -m distllm.run \
  --mode ddp --backend gloo --verify --trace \
  --batch 10 --microbatch 3 --output results/ddp-demo
```

Try tensor or pipeline parallelism by changing `--mode` to `tp` or `pp`. For NCCL, use `--backend nccl` on a Linux host with one visible GPU per process.

## Parallel modes

| Mode | Partitioning | Communication | Current validation |
|---|---|---|---|
| DDP | Samples across ranks | Gradient and token-count AllReduce | 2/4-process Gloo |
| TP | MLP hidden dimension | Input-gradient and row-output AllReduce | 2/4-process Gloo |
| PP | Transformer blocks | Activation/gradient Send/Recv | 2/4-process Gloo |
| Reference | None | None | Full-batch numerical oracle |

The modes are independent examples, not a combined 3D-parallel runtime. Attention and embeddings remain replicated in TP mode.

## Verified results

The checked-in validation corpus covers 32 distributed configurations and 96 per-rank reports. In FP64 CPU tests, the maximum absolute errors against a full-batch oracle were:

| Measurement | Maximum error |
|---|---:|
| Per-parameter gradient | `9.72e-17` |
| Weight after AdamW step | `7.22e-16` |

On a fixed synthetic workload with a 15.5× supervised-token imbalance, mean-of-means accumulation produced a gradient relative L2 error of `1.007`; the token-correct path matched the oracle to floating-point precision. See [Validation](docs/validation.md) for methodology, scope, raw result links, and the RTX 4090 study.

## Project layout

```text
distllm/             Core model, training, communication, and analysis modules
tests/               Unit and real multi-process numerical tests
scripts/             GPU validation, benchmarking, and Nsight workflows
docs/                Design, validation methodology, and result interpretation
research/            Pinned upstream provenance for the motivating case study
results/             Checked-in reproducibility evidence
```

## Scope

DistLLM focuses on objective correctness and explicit communication boundaries. It does not currently provide checkpointing, AMP/gradient scaling, ZeRO, MoE, production data loading, or a combined DDP × TP × PP topology. These omissions are deliberate; see the [roadmap](ROADMAP.md) for suitable next steps.

## Community

Bug reports and focused feature proposals are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. For usage questions, use [GitHub Discussions](https://github.com/Sentinel3252/DistLLM/discussions); for security issues, follow [SECURITY.md](SECURITY.md).

## License

DistLLM is licensed under the [Apache License 2.0](LICENSE). The pinned upstream source under `research/` retains its original license and attribution.
