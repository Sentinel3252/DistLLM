#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m distllm.gpu_study --repeats 20 --warmup 5 \
  --output results/gpu-study-rtx4090.json
