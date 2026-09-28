<p align="center">
  <img src="assets/logo.png" width="160" alt="DistLLM logo">
</p>

# DistLLM

**让变长语言模型训练在梯度累积、DDP、张量并行和流水线并行下保持正确的全局 token 目标。**

[English](README.md) · [设计文档](docs/design.md) · [验证报告](docs/validation.md) · [贡献指南](CONTRIBUTING.md)

DistLLM 是一个小而完整的 PyTorch 分布式训练参考实现。它解决这样一个常见但容易忽略的问题：不同 micro-batch 或数据并行 rank 的有效监督 token 数不同时，直接平均各自的 mean loss 并不等价于全局 token mean。

```text
mean(S_i / N_i)  !=  sum(S_i) / sum(N_i)
```

项目统一使用 loss sum 反向传播，在正确的并行边界汇总 token 数，并在 optimizer step 前只归一化一次梯度。代码覆盖 DDP、MLP Tensor Parallel、GPipe Pipeline Parallel、通信追踪以及完整 batch 数值对照。

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
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"

distllm-reproduce
pytest -q
```

CPU 上运行两个 DDP 进程：

```bash
torchrun --standalone --nproc-per-node=2 -m distllm.run \
  --mode ddp --backend gloo --verify --trace \
  --batch 10 --microbatch 3 --output results/ddp-demo
```

将 `--mode` 改为 `tp` 或 `pp` 可运行张量并行或流水线并行。Linux 多 GPU 环境下使用 `--backend nccl`。

## 已验证结果

仓库内保存了 32 组分布式配置、96 份逐 rank 报告。CPU/FP64 对照完整 batch oracle 时，逐参数梯度最大绝对误差为 `9.72e-17`，AdamW 更新后权重最大绝对误差为 `7.22e-16`。

在固定合成负载中，当监督 token 不均衡达到 15.5×，错误的 mean-of-means 路径产生 `1.007` 的梯度相对 L2 偏差；正确路径仍在浮点精度内匹配 oracle。完整方法、边界说明和数据见[验证报告](docs/validation.md)。

## 项目边界

DistLLM 是聚焦“目标函数正确性”和“显式通信边界”的参考实现，不是完整的大模型训练框架。目前不提供 checkpoint、AMP、ZeRO、MoE、生产数据管道，也没有组合 DDP × TP × PP 三维拓扑。后续方向见 [ROADMAP.md](ROADMAP.md)。

## 参与项目

欢迎提交可复现的 bug、文档改进和范围明确的功能。提交 PR 前请阅读[贡献指南](CONTRIBUTING.md)；安全问题请按 [SECURITY.md](SECURITY.md) 私下报告。

项目采用 [Apache License 2.0](LICENSE)。`research/` 中固定的上游源码保留其原始许可证与归属。
