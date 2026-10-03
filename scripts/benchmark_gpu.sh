#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Fixed global workload for strong scaling. Verification/profile runs are separate.
common=(--backend nccl --batch 32 --sequence 512 --width 512 --layers 8 --warmup 10 --steps 30)
torchrun --standalone --nproc-per-node=1 -m distllm.run "${common[@]}" \
  --mode reference --microbatch 4 --output results/bench-reference-1
for ranks in 2 4; do
  for mode in ddp tp pp; do
    for microbatch in 1 2 4 8; do
      torchrun --standalone --nproc-per-node="$ranks" -m distllm.run "${common[@]}" \
        --mode "$mode" --microbatch "$microbatch" --output "results/bench-${mode}-${ranks}-mb${microbatch}"
    done
  done
  # ABBA order reduces ordering/temperature bias in overlap comparisons.
  for trial in 0 1 2 3; do
    overlap=--overlap
    if [[ "$trial" == 0 || "$trial" == 3 ]]; then overlap=--no-overlap; fi
    torchrun --standalone --nproc-per-node="$ranks" -m distllm.run "${common[@]}" \
      --mode tp --microbatch 4 "$overlap" --output "results/abba-tp-${ranks}-${trial}"
  done
done
command -v nsys >/dev/null
mkdir -p results/nsight
# Nsight traces CUDA/NCCL kernels in the torchrun child processes.
nsys profile --trace=cuda,nvtx,osrt --sample=none --output=results/nsight/tp-overlap \
  torchrun --standalone --nproc-per-node=2 -m distllm.run "${common[@]}" \
  --mode tp --microbatch 4 --steps 5 --output results/nsight/run
nsys stats --report cuda_gpu_kern_sum results/nsight/tp-overlap.nsys-rep > results/nsight/kernels.txt
# Separate PyTorch trace for machine-readable interval-union overlap analysis.
torchrun --standalone --nproc-per-node=2 -m distllm.run "${common[@]}" \
  --mode tp --microbatch 4 --steps 5 --profile --output results/profile-tp
python -m distllm.analyze results/profile-tp/profile.rank0.json > results/profile-tp/overlap.rank0.json
