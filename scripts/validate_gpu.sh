#!/usr/bin/env bash
# Run on Linux with at least four NVIDIA GPUs and the CUDA PyTorch 2.14 wheel.
set -euo pipefail
cd "$(dirname "$0")/.."
for ranks in 2 4; do
  for mode in ddp tp pp; do
    for pattern in uneven equal empty empty-rank; do
      torchrun --standalone --nproc-per-node="$ranks" -m distllm.run \
        --backend nccl --mode "$mode" --pattern "$pattern" --verify \
        --batch 10 --microbatch 3 --steps 3 --trace \
        --output "results/nccl-${ranks}-${mode}-${pattern}"
    done
  done
  torchrun --standalone --nproc-per-node="$ranks" -m distllm.collectives \
    --backend nccl --elements 1048576 --output "results/collectives-nccl-${ranks}.json"
done
