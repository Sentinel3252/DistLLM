<p align="center">
  <img src="assets/logo.png" width="160" alt="DistLLM logo">
</p>

# DistLLM

变长语言模型训练中的梯度归一化，涵盖梯度累积、DDP、张量并行和流水线并行。

[English](README.md) · [设计文档](docs/design.md) · [验证报告](docs/validation.md) · [贡献指南](CONTRIBUTING.md)

DistLLM 是一个基于 PyTorch 的分布式训练参考实现，主要关注一个问题：不同 micro-batch 或数据并行 rank 的有效监督 token 数不同时，直接平均各自的 mean loss，得到的结果与所有有效 token 的平均 loss 不同。

```text
mean(S_i / N_i)  !=  sum(S_i) / sum(N_i)
```

实现中，每个 micro-batch 用 loss sum 反向传播，再根据并行方式汇总 token 数，在 optimizer step 前统一归一化梯度。仓库包含 DDP、MLP 张量并行和 GPipe 流水线并行的运行示例，也提供通信追踪和完整 batch 的数值对照。

## 特性

- DDP `no_sync` 梯度累积与全局 token 计数。
- MLP Column/Row Tensor Parallel，以及可选 CUDA stream/event 重叠。
- 按 Transformer block 切分的 GPipe fill/drain 执行。
- AllReduce、AllGather、ReduceScatter、Send/Recv 的显式通信路径。
- 逐参数梯度、AdamW 更新后权重和空窗口行为验证。
- Gloo/NCCL、单 GPU 性能和 Nsight Systems 采集脚本。

## 快速开始

需要 Python 3.11+ 与 PyTorch 2.14。复现和 Gloo 测试不需要 GPU。

```bash
git clone https://github.com/Sentinel3252/DistLLM.git
cd DistLLM
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"

distllm-reproduce
pytest -q
```

Windows PowerShell 安装与 CPU 验证：

```powershell
git clone https://github.com/Sentinel3252/DistLLM.git
Set-Location DistLLM
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
distllm-reproduce --output results/reproduction-local.json
python -m distllm.launch_cpu --nproc-per-node=2 --mode ddp --verify --batch 10 --microbatch 3 --output results/ddp-demo
```

本机 CPU 启动器使用系统分配的回环端口，并禁用 libuv，以兼容不带 libuv 的 Windows wheel。它接受与 `distllm.run` 相同的训练参数；NCCL 和跨主机运行使用 `torchrun`。

CPU 上运行两个 DDP 进程（Bash）：

```bash
torchrun --standalone --nproc-per-node=2 -m distllm.run \
  --mode ddp --backend gloo --verify --trace \
  --batch 10 --microbatch 3 --output results/ddp-demo
```

将 `--mode` 改为 `tp` 或 `pp` 可运行张量并行或流水线并行。Linux 多 GPU 环境下使用 `--backend nccl`。

## 已验证结果

仓库保存了 32 组分布式配置的测试结果，共 96 份逐 rank 报告。在 CPU/FP64 测试中，与完整 batch 的参考结果相比，逐参数梯度最大绝对误差为 `1.11e-16`，AdamW 更新后权重最大绝对误差为 `9.58e-16`。这些 Gloo 测试使用 PyTorch 2.11，当前安装依赖已更新为 2.14，因此这批结果只对应当时的版本。

在一组固定的合成数据上，监督 token 数的最大值与最小值相差 15.5 倍时，mean-of-means 方法的梯度相对 L2 误差为 `1.007`；按 token 数归一化的结果与参考结果的差异仍在浮点精度范围内。测试方法和原始数据见[验证报告](docs/validation.md)。

## 项目边界

DistLLM 主要用于研究梯度归一化和并行训练中的通信过程。目前尚未实现 checkpoint、AMP、ZeRO、MoE、生产数据管道，以及 DDP × TP × PP 组合并行。后续计划见 [ROADMAP.md](ROADMAP.md)。

## 参与项目

欢迎提交可复现的 bug、文档改进和范围明确的功能。提交 PR 前请阅读[贡献指南](CONTRIBUTING.md)；安全问题请按 [SECURITY.md](SECURITY.md) 私下报告。

项目采用 [Apache License 2.0](LICENSE)。`research/` 中固定的上游源码保留其原始许可证与归属。
