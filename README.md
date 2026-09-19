# DistLLM — Token 正确的分布式 Transformer 训练

解决 SFT 变长监督标签下的梯度累积偏差：不同 micro-batch / DDP rank 的有效 token 数不同时，平均各自的 mean loss 不等于全局 token mean。实现覆盖 PyTorch Distributed、DDP、MLP Column/Row TP、GPipe PP、NCCL 后端、CUDA Stream/Event 和 Nsight Systems 验收流程。

**当前交付：CPU/Gloo 的真实 2/4 进程数值验证与 RTX 4090 单卡 CUDA 性能实验已完成。** 当前机器只有一张 GPU，因此不虚构多卡 NCCL scaling、跨卡 overlap 或 Nsight timeline；多卡执行路径和采集脚本保留为后续验收项，也不声称上游已合并修复。

## 问题来源与定位

2026-09-19 检索并核实 [allenai/open-instruct #1728](https://github.com/allenai/open-instruct/issues/1728) 为 Open。检查最新 HEAD `b9269782a3d2c81f2e43ce14ba5290417923f4c0` 的 `open_instruct/finetune.py`，普通训练路径依旧直接使用 `outputs.loss` 后调用 `accelerator.backward(loss)`，没有对整个更新窗口的监督 token 总量归一化。

本项目是针对该问题中 **mean-of-microbatch-means 机制** 的独立实现和验证平台；不是 open-instruct 的原地补丁，不覆盖报告中的历史学习率迁移、DeepSpeed ZeRO 或 MoE 辅助损失。没有运行 Tulu 8B 训练，也没有向上游提交 PR。详见 [研究记录](research/README.md)。

## 技术栈与实际实现

| 技术 | 代码与作用 | 验证状态 |
|---|---|---|
| PyTorch Distributed / DDP | 原生 DDP、`no_sync` 梯度累积、异步 bucket AllReduce | 2/4 进程 Gloo 已测 |
| Tensor Parallel | MLP up 按输出维切分、down 按输入维切分；输入梯度和行并行输出 AllReduce | 同步/异步路径已测 |
| Pipeline Parallel | 按 Transformer block 分阶段；GPipe fill/drain、activation/gradient Send/Recv | 2/4 stage 已测 |
| AllGather / ReduceScatter | 独立通信正确性微基准及 Ring 传输量模型 | Gloo 已测；不冒充 TP 训练路径 |
| NCCL | `--backend nccl`、每 rank 绑定 LOCAL_RANK GPU | 多 GPU 待测 |
| CUDA Streams / Events | 独立通信 stream；TP dX AllReduce 与 dW GEMM 重叠；消费前 event wait | 实现已完成，GPU 待测 |
| Nsight Systems | CUDA kernel/stream 采集脚本、统计导出 | 脚本已提供，timeline 待采集 |

三个并行模式分别运行；未实现 DDP×TP×PP 混合拓扑。TP 只切分 MLP，attention/embedding 保持复制。模型采用 causal attention、LayerNorm、GELU，无 dropout，便于逐参数数值对齐。

## 正确性原则

设第 r 个数据并行 rank、第 m 个 micro-batch 的监督 token 损失总和为 S_rm，有效 token 数为 N_rm。目标：

```
L = sum(S_rm) / sum(N_rm)
```

旧方法平均 `S_rm / N_rm`，会对短样本赋予过高权重。新路径仅保留一套处理：

1. 对 next-token labels（`labels[..., 1:]`）计数，忽略 `-100`。
2. 每个 micro-batch 使用 `cross_entropy(reduction="sum")` 并累积梯度。
3. DDP 默认平均梯度，因此在最终 backward 后乘 `DP_size / global_tokens`；TP/PP 使用 `1 / tokens`，避免重复计算复制样本的计数。
4. 归一化后执行 optimizer step。若接入 clipping，应放在归一化之后；当前 runner 不执行 clipping。
5. 整个窗口 token 数为零时，所有 rank 跳过 optimizer step，避免 AdamW 动量/weight decay 改动权重。

不额外除以 micro-batch 数，不要求尾部窗口满长。不支持不同 rank 执行不同数量的 **optimizer windows**；数据装载层应保证各 rank 同步结束。支持每个窗口各 rank 不同的样本数、不同的 micro-batch 数和不同监督 token 数。

## 本机复现

只依赖 PyTorch，测试另需 pytest；Python >= 3.11。最新回归环境为 Python 3.12、PyTorch 2.11.0、pytest 9.1.1；GPU 性能环境为 PyTorch 2.11.0+cu130、RTX 4090。

Windows PowerShell（本项目已建好 `.venv`）：

```powershell
.venv\Scripts\python.exe -m distllm.reproduce
.venv\Scripts\python.exe -m pytest -q
```

在受限临时目录环境中，为 pytest 指定一个**新建的专用空目录**作为 `--basetemp`。pytest 会清理该目录，不要指向已有工作目录。

Linux：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
python -m distllm.reproduce
python -m distllm.gpu_study --quality-only
pytest -q
torchrun --standalone --nproc-per-node=2 -m distllm.run \
  --mode ddp --backend gloo --verify --trace --batch 10 --microbatch 3
```

测试启动器使用操作系统分配的回环 TCP 端口，不依赖固定端口。Windows 的 C++ FileStore 无法打开本机中文工作区路径，因此测试直接使用 TCPStore，不改动系统网络配置。

## 正确性与优化轨迹结果

`python -m distllm.reproduce` 固定 seed=23，单层 Transformer、CPU/FP64、每个 micro-batch 的有效监督 token 为 `[1, 2, 4, 6]`：

| 指标 | mean-of-means | 修复后 |
|---|---:|---:|
| 相对完整 batch 的梯度 L2 偏差 | 0.7922870098 | 1.5052e-16 |
| loss | 3.7084317621 | 3.7033826063 |

完整 batch reference loss 为 `3.7033826063`。这是固定合成负载下的机制复现，不是模型质量提升或普遍误差比例。

分布式矩阵：2/4 ranks ×（DDP/TP/PP × 5 个配置 + TP 同步对照）= **32 组配置**，每组 2 步 AdamW 更新，保存 **96 份 rank 结果**。覆盖不均衡 token、micro-batch=1/3、等 token、全空窗口、单个数据 rank 无监督 token、尾部不足及同步/异步 TP。逐 rank 对照完整 batch 的 loss、逐参数梯度和更新后权重；FP64 最大梯度绝对误差 `9.72e-17`，最大权重绝对误差 `7.22e-16`。

进一步扫描监督 token 最大/最小比 1.0×、1.29×、1.94×、3.88×、15.5×：mean-of-means 的梯度相对 L2 偏差分别为 0、0.081、0.221、0.454、1.007，token-correct 路径在 FP64 下均与完整 batch 一致。固定变长窗口训练 50 个 AdamW step 后，错误路径相对 reference 的参数 L2 偏离 **13.51%**，全局 token objective 为 1.1020；修复路径参数偏差 `1.35e-16`，objective 0.6283，与 reference 一致。这是固定合成任务上的机制与轨迹实验，不外推为真实数据集的模型质量收益。

原始证据：

- [验收摘要（12 项测试通过）](results/validation.json)
- [JUnit 测试报告](results/pytest.xml)
- [归一化复现](results/reproduction.json)
- [2-rank 结果](results/verified-gloo-2ranks.json)
- [4-rank 结果](results/verified-gloo-4ranks.json)
- [2-rank collectives](results/collectives-gloo-2ranks.json)
- [4-rank collectives](results/collectives-gloo-4ranks.json)
- [RTX 4090 完整实验](results/gpu-study-rtx4090.json)

## RTX 4090 单卡性能实验

`gpu_study.py` 使用 23.08M 参数 Transformer（6 层、width=512、vocab=4096）、BF16、batch=16、sequence=256，在相同 2,161 个监督 token 上扫描 accumulation micro-batch。每点预热 5 次、测量 20 次，端到端包含 forward、backward、zero-grad 和最终梯度归一化：

| micro-batch | step 中位数 | P95 | 监督 token/s | peak allocated |
|---:|---:|---:|---:|---:|
| 1 | 34.87 ms | 35.20 ms | 61,975 | 137.4 MiB |
| 2 | 17.56 ms | 17.64 ms | 123,065 | 167.7 MiB |
| 4 | 8.94 ms | 9.40 ms | 241,805 | 233.3 MiB |
| 8 | 6.59 ms | 6.68 ms | 327,790 | 352.5 MiB |
| 16 | 6.09 ms | 6.13 ms | 354,630 | 549.2 MiB |

从 micro-batch 1 增至 16，吞吐提升 **5.72×**，显式展示 kernel launch/小矩阵利用率与 activation memory 的权衡。micro-batch=4 的 ABBA 配对实验中，token-correct 相对错误 mean-of-means 的 step time 为 **-0.67%**（测量噪声量级），peak allocated 同为 234.3 MiB，说明修复没有可测的吞吐或显存回退。这里报告的是本项目小模型训练 step，不是大模型 MFU。

复现：

```bash
python -m distllm.gpu_study --repeats 20 --warmup 5 \
  --output results/gpu-study-rtx4090.json
```

## 多 GPU 验收与性能剖析（待多卡机器）

Linux 4 GPU 环境安装匹配驱动的 PyTorch 2.14 CUDA wheel，以及 Nsight Systems。先运行正确性，再运行性能：

```bash
bash scripts/validate_gpu.sh
bash scripts/benchmark_gpu.sh
python -m distllm.scaling \
  results/bench-reference-1/result.rank0.json \
  results/bench-ddp-2-mb4/result.rank0.json
```

多卡验证使用 FP64，基准使用 FP32。由于当前只有一张 RTX 4090，这部分脚本未执行，不能把单卡数据解释为 scaling 或通信重叠收益。基准固定 global batch=32，扫描 micro-batch=1/2/4/8，使用 rank-max step time 的中位数，单 GPU reference 作为 strong-scaling 基准。TP 重叠对照按 ABBA 顺序运行。`scaling.py` 拒绝把验证/追踪计时当作 benchmark，也拒绝不同模型、设备型号或工作负载的对比。

`Comm` 记录的是本项目显式调用（DDP bucket / TP / token count / loss / PP P2P），不声称拦截框架内部所有 collective。JSON 中 `caller_stream` 和 `submission_stream` 不是 NCCL 的内部执行 stream；后者应在 Nsight 中确认。CPU `host_submit_us` / `host_wait_us` 不代表 GPU 执行时长。CUDA event 区间标为 `cuda_interval_ms`，包含通信依赖开销，不能当作纯 NCCL kernel 时间。

`analyze.py` 从 PyTorch GPU kernel trace 分设备计算时间区间并集与交集：

```
communication_hidden_fraction = |compute ∩ NCCL| / |NCCL|
communication_busy_fraction   = |NCCL| / |compute ∪ NCCL|
```

多个并发 kernel 不重复计时；没有 GPU kernel 的 CPU trace 会直接拒绝分析。这里的 compute 包含非 NCCL GPU kernels；并集指标不是端到端 speedup。Nsight 用来确认 dX 通信与 dW GEMM 的真实并发及 stream 依赖。当前未报告 pipeline bubble 百分比，也未用 CPU submit timing 推导它。

## 文件结构
 
```
distllm/loss.py         token objective 与最终梯度缩放
distllm/model.py        Transformer、MLP TP、PP 分段
distllm/train.py        DDP/TP 累积与 GPipe 执行
distllm/comm.py         异步通信、CUDA stream/event、trace
distllm/run.py          训练、单进程 reference 对照、结果保存
distllm/reproduce.py    错误归一化机制复现
distllm/gpu_study.py    token skew、优化轨迹与 4090 性能实验
distllm/collectives.py  三种 collective 数值验证与计时
distllm/analyze.py      GPU 区间并集、重叠及 Ring 模型
distllm/scaling.py      强扩展效率计算与可比性校验
tests/                 数值回归与指标语义测试
scripts/               Linux 多 GPU 验收与 Nsight 采集
research/              固定版本源码、许可证、定位记录
results/               实际运行产生的证据
```

实现没有 checkpoint、AMP、ZeRO、MoE 或模型下载依赖；这些不是本项目解决该归一化问题的必要组成部分。
