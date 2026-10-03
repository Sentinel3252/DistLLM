<p align="center">
  <img src="assets/logo.png" width="160" alt="DistLLM logo">
</p>

# DistLLM

Gradient normalization for variable-length language-model training with gradient accumulation, DDP, tensor parallelism, and pipeline parallelism.

[![CI](https://github.com/Sentinel3252/DistLLM/actions/workflows/ci.yml/badge.svg)](https://github.com/Sentinel3252/DistLLM/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

[中文文档](README.zh-CN.md) · [Design](docs/design.md) · [Validation](docs/validation.md) · [Contributing](CONTRIBUTING.md)

DistLLM is a PyTorch reference implementation for causal language-model training. It focuses on gradient normalization when micro-batches or data-parallel ranks have different numbers of supervised tokens.

The repository includes DDP, MLP tensor-parallel, and GPipe-style pipeline-parallel examples, along with communication tracing, numerical checks, and GPU profiling scripts. The model is small enough to inspect individual gradients and communication operations.

## Why token counts matter

When supervised-token counts differ, averaging the micro-batch mean losses differs from averaging over all supervised tokens:

```text
mean(S_i / N_i)  !=  sum(S_i) / sum(N_i)
```

Here, `S_i` is the summed next-token loss and `N_i` is the number of supervised tokens. DistLLM backpropagates each loss sum, aggregates token counts according to the parallel mode, and normalizes the accumulated gradients once before the optimizer step.

This matters whenever padding, masking, packing, or per-rank data skew causes supervised-token counts to differ.

## Features

- The same token-normalization rule in reference, DDP, tensor-parallel, and pipeline-parallel execution.
- Native PyTorch Distributed primitives: AllReduce, AllGather, ReduceScatter, and Send/Recv.
- DDP `no_sync` accumulation with correct compensation for DDP's averaged gradients.
- Column/row MLP tensor parallelism with optional CUDA stream/event overlap.
- GPipe fill/drain scheduling across Transformer blocks.
- Full-batch reference checks for each parameter's gradient and weight after an optimizer update.
- Machine-readable results and scripts for Gloo, NCCL, CUDA, and Nsight Systems.

## Quick start

Requirements: Python 3.11+ and PyTorch 2.14. A CPU-only installation is enough for the reproduction and Gloo tests.

```bash
git clone https://github.com/Sentinel3252/DistLLM.git
cd DistLLM
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

PowerShell on Windows:

```powershell
git clone https://github.com/Sentinel3252/DistLLM.git
Set-Location DistLLM
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
distllm-reproduce --output results/reproduction-local.json
python -m distllm.launch_cpu --nproc-per-node=2 --mode ddp --verify --batch 10 --microbatch 3 --output results/ddp-demo
```

The local CPU launcher uses an OS-assigned loopback port and disables libuv to support Windows wheels built without it. It accepts the same training flags as `distllm.run`; use `torchrun` for NCCL and multi-host launches.

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

The saved test results cover 32 distributed configurations and 96 per-rank reports. In FP64 CPU tests, the maximum absolute errors against the full-batch reference were:

| Measurement | Maximum error |
|---|---:|
| Per-parameter gradient | `1.11e-16` |
| Weight after AdamW step | `9.58e-16` |

These Gloo tests ran on PyTorch 2.11. The current package requires 2.14, so the saved results apply to the earlier version.

On a fixed synthetic workload where the largest supervised-token count was 15.5× the smallest, mean-of-means accumulation produced a gradient relative L2 error of `1.007`. Token normalization matched the full-batch reference to floating-point precision. See [Validation](docs/validation.md) for the test setup, raw results, and RTX 4090 benchmarks.

## Project layout

```text
distllm/             Core model, training, communication, and analysis modules
tests/               Unit and real multi-process numerical tests
scripts/             GPU validation, benchmarking, and Nsight workflows
docs/                Design, validation methodology, and result interpretation
research/            Pinned upstream provenance for the motivating case study
results/             Saved test and benchmark results
```

## Scope

DistLLM focuses on gradient normalization and communication in parallel training. Checkpointing, AMP/gradient scaling, ZeRO, MoE, production data loading, and combined DDP × TP × PP execution are not implemented. See the [roadmap](ROADMAP.md) for planned work.

## Community

Bug reports and focused feature proposals are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. For usage questions, use [GitHub Discussions](https://github.com/Sentinel3252/DistLLM/discussions); for security issues, follow [SECURITY.md](SECURITY.md).

## License

DistLLM is licensed under the [Apache License 2.0](LICENSE). The pinned upstream source under `research/` retains its original license and attribution.
