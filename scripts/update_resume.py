"""Update the DistLLM paragraphs in the existing resume without restyling it."""

import shutil
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
REPLACEMENTS = {
    "DistLLM｜Transformer 多 GPU 并行与 NCCL 通信-计算重叠":
        "DistLLM｜Token-Correct 分布式 Transformer 训练与并行执行",
    "PyTorch Distributed · NCCL · DDP/TP/PP · CUDA Streams · Nsight Systems｜【项目时间】":
        "PyTorch Distributed · DDP/TP/PP · NCCL/Gloo · CUDA Streams/Events｜【项目时间】",
    "• 多 GPU 执行：实现最小 Transformer 分布式训练骨架，覆盖 DDP、Column/Row Tensor Parallel 与 micro-batch Pipeline Parallel；将 AllGather/ReduceScatter/AllReduce/SendRecv 映射到具体激活/梯度路径，并与单卡 reference 对齐 forward/backward 数值。":
        "• 并行训练实现：实现 DDP no_sync 梯度累积、Column/Row Tensor Parallel 与 GPipe Pipeline Parallel，封装 AllReduce/AllGather/ReduceScatter/SendRecv；针对变长 SFT 的 mean-of-means 偏差，以 loss sum / global supervised tokens 统一归一化，并正确处理空窗口与尾部 micro-batch。",
    "• 通信建模与剖析：实现 per-rank collective tracer，记录 process group、collective、message bytes、latency 与 stream；结合 Ring collective 通信量模型拆分 compute / communication / pipeline bubble，扫描 micro-batch 与并行配置定位扩展效率拐点。":
        "• 分布式正确性：构建单卡完整 batch reference，完成 2/4 进程 × DDP/TP/PP 的 32 组配置、96 份 rank 结果验证；覆盖 token 不均衡、单 rank 无标签、全空窗口和同步/异步 TP，FP64 梯度/权重最大误差分别为 9.72e-17 / 7.22e-16，12 项测试通过。",
    "• 通信-计算重叠：使用 async collective、CUDA stream/event 将梯度或激活通信与 backward/GEMM 重叠，并以 Nsight Systems 验证 timeline；在 2/4 GPU microbenchmark 中仅报告可复现的 scaling efficiency、communication ratio 与 overlap ratio。":
        "• 误差与性能评测：监督 token 不均衡达 15.5× 时，错误归一化梯度相对偏差升至 1.007、50-step 参数轨迹偏离 13.51%，修复后保持 1.35e-16；RTX 4090 上 23.08M BF16 模型扫描 micro-batch 1→16，吞吐提升 5.72× 至 354.6K token/s，配对实验额外开销 -0.67%、无显存回退。",
}


def paragraph_text(paragraph):
    return "".join(node.text or "" for node in paragraph.iter(W + "t"))


def main():
    path = Path(__file__).resolve().parents[2] / "3.docx"
    with zipfile.ZipFile(path) as source:
        xml = source.read("word/document.xml")
        root = ET.fromstring(xml)
        replaced = set()
        for paragraph in root.iter(W + "p"):
            old = paragraph_text(paragraph)
            if old not in REPLACEMENTS:
                continue
            texts = list(paragraph.iter(W + "t"))
            texts[0].text = REPLACEMENTS[old]
            for node in texts[1:]:
                node.text = ""
            replaced.add(old)
        missing = set(REPLACEMENTS) - replaced
        if missing:
            raise RuntimeError(f"resume paragraphs not found: {sorted(missing)}")
        updated_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        with zipfile.ZipFile(temporary_path, "w") as destination:
            for item in source.infolist():
                data = updated_xml if item.filename == "word/document.xml" else source.read(item.filename)
                destination.writestr(item, data)
    shutil.move(temporary_path, path)


if __name__ == "__main__":
    main()
