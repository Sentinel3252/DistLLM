"""Small causal Transformer; only the MLP is tensor-sharded."""

import copy
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab: int = 32
    width: int = 16
    heads: int = 4
    layers: int = 4
    expansion: int = 4

    def __post_init__(self):
        for name in ("vocab", "width", "heads", "layers", "expansion"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")


class ColumnLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, comm, overlap):
        ctx.save_for_backward(x, weight)
        ctx.comm, ctx.overlap = comm, overlap
        return F.linear(x, weight)

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        grad = grad.contiguous()
        dx = grad @ weight
        pending = ctx.comm.all_reduce(dx, "tp_input_gradient")
        if not ctx.overlap:
            pending.wait()
        with torch.profiler.record_function("tp_weight_gradient"):
            dw = grad.flatten(0, -2).T @ x.flatten(0, -2)
        if ctx.overlap:
            pending.wait()
        return dx, dw, None, None


class RowReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, comm):
        result = x.clone()
        comm.all_reduce(result, "tp_row_output").wait()
        return result

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up = nn.Linear(config.width, config.width * config.expansion, bias=False)
        self.down = nn.Linear(config.width * config.expansion, config.width, bias=False)
        self.comm = None
        self.overlap = False

    def forward(self, x):
        if self.comm is None:
            return self.down(F.gelu(self.up(x)))
        x = ColumnLinear.apply(x, self.up.weight, self.comm, self.overlap)
        return RowReduce.apply(self.down(F.gelu(x)), self.comm)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width)
        self.norm2 = nn.LayerNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.proj = nn.Linear(config.width, config.width, bias=False)
        self.mlp = MLP(config)
        self.heads = config.heads

    def forward(self, x):
        batch, seq, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(batch, seq, 3, self.heads, width // self.heads).unbind(2)
        attention = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True,
        ).transpose(1, 2).reshape(batch, seq, width)
        x = x + self.proj(attention)
        return x + self.mlp(self.norm2(x))


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab, config.width)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.width)
        self.head = nn.Linear(config.width, config.vocab, bias=False)

    def forward(self, tokens):
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def tensor_parallel(model, comm, overlap=True):
    model = copy.deepcopy(model)
    rank = dist.get_rank(comm.group)
    for block in model.blocks:
        mlp = block.mlp
        if mlp.up.weight.size(0) % comm.size:
            raise ValueError("MLP hidden width must be divisible by TP size")
        mlp.up.weight = nn.Parameter(mlp.up.weight.chunk(comm.size, dim=0)[rank].clone())
        mlp.down.weight = nn.Parameter(mlp.down.weight.chunk(comm.size, dim=1)[rank].clone())
        mlp.comm, mlp.overlap = comm, overlap
    return model


def pipeline_stage(model, rank, size):
    if len(model.blocks) % size:
        raise ValueError("Transformer layers must be divisible by PP size")
    count = len(model.blocks) // size
    parts = {}
    if rank == 0:
        parts["embed"] = copy.deepcopy(model.embed)
    for i in range(rank * count, (rank + 1) * count):
        parts[f"block_{i}"] = copy.deepcopy(model.blocks[i])
    if rank == size - 1:
        parts["norm"] = copy.deepcopy(model.norm)
        parts["head"] = copy.deepcopy(model.head)
    return nn.Sequential(OrderedDict(parts))
