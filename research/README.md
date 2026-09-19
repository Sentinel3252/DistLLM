# 选题与源码证据

核查日期：2026-09-19。简历第二项要求的栈是 PyTorch Distributed、NCCL、DDP/TP/PP、CUDA Streams、Nsight Systems；本项目逐项映射到 README 中的实际实现和验证状态。

## 候选筛选

| GitHub 问题 | 核查结果 | 选择 |
|---|---|---|
| [PyTorch #163859](https://github.com/pytorch/pytorch/issues/163859) GPipe 梯度不一致 | 已 Closed；报告中的 reference 还额外除以 micro-batch 数 | 排除，不能当作未解决问题 |
| [PyTorch #157606](https://github.com/pytorch/pytorch/issues/157606) TP/SP 缺少 backward collective | Open，但必须区分真实梯度缺失与 autograd 不需要输入梯度；当前证据不足以确认 bug | 排除，不凭 issue 标题修复 |
| [Megatron-LM #7213](https://github.com/NVIDIA/Megatron-LM/issues/7213) MoE 累积 token 分母 | Open，已有相关 PR #7214，且需要 MoE 栈 | 排除，避免扩张技术范围 |
| [open-instruct #1728](https://github.com/allenai/open-instruct/issues/1728) 不等 token 的梯度累积 | Open，当前源码仍存在普通路径的 mean-of-means 机制 | 选定其中 token 归一化子问题 |

GitHub REST `GET /repos/allenai/open-instruct/issues/1728` 返回 `state=open`，`updated_at=2026-07-07T07:49:28Z`（本机 PowerShell 显示为 UTC+8 15:49:28）。状态是核查时快照，不保证未来仍 Open。

## 固定代码版本

`git ls-remote https://github.com/allenai/open-instruct.git HEAD`：

```
b9269782a3d2c81f2e43ce14ba5290417923f4c0
```

保存的 `finetune.upstream.py` 来自该 commit 的 `open_instruct/finetune.py`：

```
SHA256 f3ee4de900ea48cd157c5e7b26dccbc72e622136fd0493043ccca8b1a04b2527
```

版权及 Apache 2.0 许可证见 `LICENSE.open-instruct`；原始快照未编辑。

可核对的当前源码位置：

- `finetune.upstream.py:828`：进入 `accelerator.accumulate(model)`。
- `finetune.upstream.py:832`：普通 causal LM forward。
- `finetune.upstream.py:835`：直接使用 `outputs.loss`。
- `finetune.upstream.py:838`：SP 分支单独执行 token 加权逻辑；其范围不是完整累积窗口。
- `finetune.upstream.py:856`：`accelerator.backward(loss)`。
- `finetune.upstream.py:858`：随后 clip / optimizer step。

公式上，多个非空 micro-batch 的 `mean(S_i/N_i)` 与 `sum(S_i)/sum(N_i)` 只有在计数相同或恰好抵消时才等价。复现器独立构造 causal Transformer 和标签，不加载 open-instruct 的 Accelerate/DeepSpeed 环境。故证据支持“复现并解决这一归一化机制”，不支持“运行并修复整个 Tulu 训练栈”或归因原报告全部模型质量差异。

## 本项目根因修复

生产路径统一在 `distllm/loss.py` 中累积 `reduction=sum`，在 DDP averaging 之后按 `DP_size/global_tokens` 修正梯度；TP/PP 不重复计算 token。所有并行路径使用同一个损失函数和归一化函数，错误 mean-of-means 仅留在复现器中用于反例。

采用 FP64、小模型、无 dropout 的完整 batch reference，先对齐 loss 和逐参数梯度，再比较 AdamW 更新后的权重。不能通过仅比较 gradient norm 排除方向错误，因此测试逐元素比较全部参数。

## 当前依赖实现与文档

本机安装的是 PyTorch `2.14.0+cpu`，源码 commit `08187d9e0fba026dc8217405802ab5381dc88d90`。已检查安装包的实际代码：

- `torch/distributed/algorithms/ddp_comm_hooks/default_hooks.py`：bucket reduction 的平均契约与 Future 接口。
- `torch/distributed/distributed_c10d.py`：`all_reduce`、`send/recv`、`all_gather_single`、`reduce_scatter_single`。
- `torch/cuda/streams.py`：`wait_stream`、`wait_event`、event record 的异步依赖语义。
- `torch/distributed/rendezvous.py`：Windows FileStore 的路径转换。

2.14 已将 `reduce_scatter_tensor` 标记 deprecated，微基准使用 `reduce_scatter_single`；AllGather 同样使用当前 `all_gather_single` API。项目不提供旧版本兼容分支。

官方文档：

- [PyTorch 2.14 Distributed](https://docs.pytorch.org/docs/2.14/distributed.html)
- [PyTorch 2.14 CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

## 已知验证边界

只有 CPU/Gloo 实测，不存在本项目的多 GPU 性能数据。FP64 CPU 数值误差不代表 BF16/FP16 或 NCCL 的误差上限。NCCL、CUDA allocator 的跨 stream 生命周期和实际 overlap 必须用随附 GPU 脚本进一步验收；不能从 `async_op=True` 推断实际加速。
