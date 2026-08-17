#!/usr/bin/env python3
"""Small CPU-only DeepSeek V4 Flash inference model for TP communication study.

The model intentionally keeps exactly one zero-based layer 21 block. It keeps
the original high-level topology while shrinking widths and expert counts. All
weights are deterministic random FP32 values; no checkpoint is loaded.

Run TP2 inference:
    torchrun --standalone --nproc-per-node=2 debug/small_model.py

Export rank-local structural ONNX models:
    python debug/small_model.py --export-onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SmallModelConfig:
    layer_id: int = 21
    vocab_size: int = 4096
    dim: int = 256
    moe_inter_dim: int = 128
    n_heads: int = 8
    n_routed_experts: int = 16
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    score_func: str = "sqrtsoftplus"
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    q_lora_rank: int = 128
    head_dim: int = 128
    rope_head_dim: int = 64
    norm_eps: float = 1e-6
    o_groups: int = 4
    o_lora_rank: int = 128
    window_size: int = 128
    compress_ratio: int = 128
    compress_rope_theta: float = 160000.0
    original_seq_len: int = 65536
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    max_batch_size: int = 1
    max_seq_len: int = 256
    seed: int = 20260817

    def validate(self, world_size: int) -> None:
        if self.layer_id != 21:
            raise ValueError("This debug model must contain only zero-based layer_id=21")
        if self.compress_ratio != 128:
            raise ValueError("DeepSeek V4 Flash layer_id=21 must use compress_ratio=128")
        if self.n_shared_experts != 1:
            raise ValueError("The model topology requires exactly one shared expert")
        for name, value in (
            ("vocab_size", self.vocab_size),
            ("n_heads", self.n_heads),
            ("o_groups", self.o_groups),
            ("n_routed_experts", self.n_routed_experts),
        ):
            if value % world_size:
                raise ValueError(f"{name}={value} must be divisible by world_size={world_size}")
        if self.n_heads % self.o_groups:
            raise ValueError("n_heads must be divisible by o_groups")
        if self.head_dim % 2 or self.rope_head_dim % 2:
            raise ValueError("head_dim and rope_head_dim must be even")
        if self.rope_head_dim > self.head_dim:
            raise ValueError("rope_head_dim cannot exceed head_dim")
        if self.n_activated_experts > self.n_routed_experts:
            raise ValueError("n_activated_experts cannot exceed n_routed_experts")
        if self.max_seq_len < max(self.window_size, self.compress_ratio):
            raise ValueError("max_seq_len must cover the attention window and compression ratio")


def _seed_for(name: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def random_tensor(
    name: str,
    shape: tuple[int, ...],
    base_seed: int,
    std: float = 0.02,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_seed_for(name, base_seed))
    return torch.randn(shape, generator=generator, dtype=torch.float32) * std


def frozen_parameter(value: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(value, requires_grad=False)


class CollectiveTracer:
    def __init__(self, rank: int, enabled: bool) -> None:
        self.rank = rank
        self.enabled = enabled
        self.records: list[dict[str, Any]] = []

    def measure(
        self,
        name: str,
        operation: str,
        value: torch.Tensor,
        collective: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        started = time.perf_counter()
        output = collective()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.records.append(
            {
                "index": len(self.records),
                "rank": self.rank,
                "name": name,
                "operation": operation,
                "input_shape": list(value.shape),
                "output_shape": list(output.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
                "payload_bytes": value.numel() * value.element_size(),
                "elapsed_ms": elapsed_ms,
            }
        )
        if self.enabled:
            print(
                f"[rank {self.rank}] {operation:<10} {name:<28} "
                f"shape={tuple(value.shape)} bytes={value.numel() * value.element_size()} "
                f"time_ms={elapsed_ms:.3f}",
                flush=True,
            )
        return output


@dataclass
class ParallelContext:
    world_size: int
    rank: int
    tracer: CollectiveTracer | None = None

    def all_reduce(self, value: torch.Tensor, name: str) -> torch.Tensor:
        if self.world_size == 1:
            return value
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized for TP collectives")

        def run() -> torch.Tensor:
            output = value.contiguous()
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
            return output

        if self.tracer is None:
            return run()
        return self.tracer.measure(name, "all_reduce", value, run)

    def all_gather(self, value: torch.Tensor, name: str) -> torch.Tensor:
        if self.world_size == 1:
            return value
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized for TP collectives")

        def run() -> torch.Tensor:
            parts = [torch.empty_like(value) for _ in range(self.world_size)]
            dist.all_gather(parts, value.contiguous())
            return torch.cat(parts, dim=-1)

        if self.tracer is None:
            return run()
        return self.tracer.measure(name, "all_gather", value, run)


class ReplicatedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        name: str,
        config: SmallModelConfig,
    ) -> None:
        super().__init__()
        weight = random_tensor(name, (out_features, in_features), config.seed)
        self.weight = frozen_parameter(weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, self.weight)


class ColumnParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        name: str,
        config: SmallModelConfig,
        parallel: ParallelContext,
    ) -> None:
        super().__init__()
        if out_features % parallel.world_size:
            raise ValueError("Column-parallel output dimension must be divisible by world size")
        full_weight = random_tensor(name, (out_features, in_features), config.seed)
        self.weight = frozen_parameter(
            full_weight.chunk(parallel.world_size, dim=0)[parallel.rank].contiguous()
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, self.weight)


class RowParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        name: str,
        collective_name: str,
        config: SmallModelConfig,
        parallel: ParallelContext,
    ) -> None:
        super().__init__()
        if in_features % parallel.world_size:
            raise ValueError("Row-parallel input dimension must be divisible by world size")
        full_weight = random_tensor(name, (out_features, in_features), config.seed)
        self.weight = frozen_parameter(
            full_weight.chunk(parallel.world_size, dim=1)[parallel.rank].contiguous()
        )
        self.collective_name = collective_name
        self.parallel = parallel

    def forward(self, local_value: torch.Tensor) -> torch.Tensor:
        partial = F.linear(local_value, self.weight)
        return self.parallel.all_reduce(partial, self.collective_name)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, epsilon: float) -> None:
        super().__init__()
        self.weight = frozen_parameter(torch.ones(dim, dtype=torch.float32))
        self.epsilon = epsilon

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        variance = value.float().square().mean(dim=-1, keepdim=True)
        normalized = value.float() * torch.rsqrt(variance + self.epsilon)
        return normalized * self.weight


def precompute_rope_angles(config: SmallModelConfig) -> torch.Tensor:
    dim = config.rope_head_dim

    def correction_dim(rotations: int) -> float:
        numerator = math.log(config.original_seq_len / (rotations * 2 * math.pi))
        return dim * numerator / (2 * math.log(config.compress_rope_theta))

    low = max(math.floor(correction_dim(config.beta_fast)), 0)
    high = min(math.ceil(correction_dim(config.beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = (torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)
    smooth = 1 - ramp.clamp(0, 1)
    frequencies = 1.0 / (
        config.compress_rope_theta
        ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )
    frequencies = frequencies / config.rope_factor * (1 - smooth) + frequencies * smooth
    positions = torch.arange(config.max_seq_len, dtype=torch.float32)
    return torch.outer(positions, frequencies)


def apply_rope(
    value: torch.Tensor,
    angles: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    rope_dim = angles.size(-1) * 2
    prefix = value[..., :-rope_dim]
    rotary = value[..., -rope_dim:]
    even = rotary[..., 0::2]
    odd = rotary[..., 1::2]
    angle_shape = (1, angles.size(0), *([1] * (even.ndim - 3)), angles.size(1))
    cosine = angles.cos().view(angle_shape)
    sine = angles.sin().view(angle_shape)
    if inverse:
        sine = -sine
    rotated = torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine),
        dim=-1,
    ).flatten(-2)
    return torch.cat((prefix, rotated), dim=-1)


def sparse_attention(
    query: torch.Tensor,
    kv_bank: torch.Tensor,
    indices: torch.Tensor,
    attention_sink: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    batch_size, sequence_length, _, head_dim = query.shape
    safe_indices = indices.clamp_min(0)
    expanded_bank = kv_bank.unsqueeze(1).expand(
        batch_size, sequence_length, kv_bank.size(1), head_dim
    )
    gathered = torch.gather(
        expanded_bank,
        2,
        safe_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim),
    )
    scores = torch.einsum("bshd,bskd->bshk", query, gathered) * scale
    scores = scores.masked_fill(indices.unsqueeze(2) < 0, float("-inf"))
    sink_scores = attention_sink.view(1, 1, -1, 1).expand(
        batch_size, sequence_length, -1, -1
    )
    probabilities = torch.cat((scores, sink_scores), dim=-1).softmax(dim=-1)[..., :-1]
    return torch.einsum("bshk,bskd->bshd", probabilities, gathered)


class ParallelEmbedding(nn.Module):
    def __init__(self, config: SmallModelConfig, parallel: ParallelContext) -> None:
        super().__init__()
        local_vocab = config.vocab_size // parallel.world_size
        full_weight = random_tensor(
            "embed.weight", (config.vocab_size, config.dim), config.seed
        )
        self.weight = frozen_parameter(
            full_weight.chunk(parallel.world_size, dim=0)[parallel.rank].contiguous()
        )
        self.start = parallel.rank * local_vocab
        self.end = self.start + local_vocab
        self.parallel = parallel

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        outside = (input_ids < self.start) | (input_ids >= self.end)
        local_ids = (input_ids - self.start).masked_fill(outside, 0)
        local = F.embedding(local_ids, self.weight).masked_fill(outside.unsqueeze(-1), 0)
        return self.parallel.all_reduce(local, "embedding_vocab_parallel")


class KVCompressor(nn.Module):
    def __init__(self, config: SmallModelConfig) -> None:
        super().__init__()
        self.config = config
        self.wkv = ReplicatedLinear(
            config.dim, config.head_dim, "block21.attn.compressor.wkv.weight", config
        )
        self.wgate = ReplicatedLinear(
            config.dim, config.head_dim, "block21.attn.compressor.wgate.weight", config
        )
        self.norm = RMSNorm(config.head_dim, config.norm_eps)
        self.ape = frozen_parameter(
            random_tensor(
                "block21.attn.compressor.ape",
                (config.compress_ratio, config.head_dim),
                config.seed,
            )
        )
        max_compressed = math.ceil(config.max_seq_len / config.compress_ratio)
        self.register_buffer(
            "kv_state",
            torch.zeros(
                config.max_batch_size, config.compress_ratio, config.head_dim
            ),
            persistent=False,
        )
        self.register_buffer(
            "score_state",
            torch.zeros(
                config.max_batch_size, config.compress_ratio, config.head_dim
            ),
            persistent=False,
        )
        self.register_buffer(
            "compressed_cache",
            torch.zeros(config.max_batch_size, max_compressed, config.head_dim),
            persistent=False,
        )
        self.compressed_count = 0

    def reset_cache(self) -> None:
        self.kv_state.zero_()
        self.score_state.zero_()
        self.compressed_cache.zero_()
        self.compressed_count = 0

    def forward(
        self,
        value: torch.Tensor,
        start_pos: int,
        rope_angles: torch.Tensor,
    ) -> None:
        batch_size, sequence_length, _ = value.shape
        projected_kv = self.wkv(value.float())
        projected_scores = self.wgate(value.float())
        ratio = self.config.compress_ratio
        for offset in range(sequence_length):
            absolute_position = start_pos + offset
            state_position = absolute_position % ratio
            self.kv_state[:batch_size, state_position] = projected_kv[:, offset]
            self.score_state[:batch_size, state_position] = (
                projected_scores[:, offset] + self.ape[state_position]
            )
            if state_position != ratio - 1:
                continue
            weights = self.score_state[:batch_size].softmax(dim=1)
            compressed = (self.kv_state[:batch_size] * weights).sum(dim=1)
            compressed = self.norm(compressed)
            group_start = absolute_position + 1 - ratio
            compressed = apply_rope(
                compressed.unsqueeze(1), rope_angles[group_start : group_start + 1]
            ).squeeze(1)
            cache_position = absolute_position // ratio
            self.compressed_cache[:batch_size, cache_position] = compressed
            self.compressed_count = max(self.compressed_count, cache_position + 1)


class SparseMLAAttention(nn.Module):
    def __init__(self, config: SmallModelConfig, parallel: ParallelContext) -> None:
        super().__init__()
        self.config = config
        self.parallel = parallel
        self.local_heads = config.n_heads // parallel.world_size
        self.local_groups = config.o_groups // parallel.world_size
        self.heads_per_group = config.n_heads // config.o_groups
        self.wq_a = ReplicatedLinear(
            config.dim, config.q_lora_rank, "block21.attn.wq_a.weight", config
        )
        self.q_norm = RMSNorm(config.q_lora_rank, config.norm_eps)
        self.wq_b = ColumnParallelLinear(
            config.q_lora_rank,
            config.n_heads * config.head_dim,
            "block21.attn.wq_b.weight",
            config,
            parallel,
        )
        self.wkv = ReplicatedLinear(
            config.dim, config.head_dim, "block21.attn.wkv.weight", config
        )
        self.kv_norm = RMSNorm(config.head_dim, config.norm_eps)
        full_wo_a = random_tensor(
            "block21.attn.wo_a.weight",
            (
                config.o_groups,
                config.o_lora_rank,
                self.heads_per_group * config.head_dim,
            ),
            config.seed,
        )
        self.wo_a = frozen_parameter(
            full_wo_a.chunk(parallel.world_size, dim=0)[parallel.rank].contiguous()
        )
        self.wo_b = RowParallelLinear(
            config.o_groups * config.o_lora_rank,
            config.dim,
            "block21.attn.wo_b.weight",
            "block21_attention_output",
            config,
            parallel,
        )
        full_sink = random_tensor(
            "block21.attn.attn_sink", (config.n_heads,), config.seed
        )
        self.attention_sink = frozen_parameter(
            full_sink.chunk(parallel.world_size, dim=0)[parallel.rank].contiguous()
        )
        self.compressor = KVCompressor(config)
        self.register_buffer("rope_angles", precompute_rope_angles(config), persistent=False)
        self.register_buffer(
            "raw_kv_cache",
            torch.zeros(config.max_batch_size, config.max_seq_len, config.head_dim),
            persistent=False,
        )

    def reset_cache(self) -> None:
        self.raw_kv_cache.zero_()
        self.compressor.reset_cache()

    def _build_indices(
        self,
        batch_size: int,
        start_pos: int,
        sequence_length: int,
        end_pos: int,
    ) -> torch.Tensor:
        max_raw = min(self.config.window_size, end_pos)
        max_compressed = self.compressor.compressed_count
        width = max_raw + max_compressed
        indices = torch.full(
            (batch_size, sequence_length, width), -1, dtype=torch.long
        )
        for local_position in range(sequence_length):
            absolute_position = start_pos + local_position
            raw_start = max(0, absolute_position - self.config.window_size + 1)
            raw_indices = torch.arange(raw_start, absolute_position + 1)
            completed_groups = (absolute_position + 1) // self.config.compress_ratio
            compressed_indices = end_pos + torch.arange(completed_groups)
            selected = torch.cat((raw_indices, compressed_indices))
            indices[:, local_position, : selected.numel()] = selected
        return indices

    def forward(self, value: torch.Tensor, start_pos: int) -> torch.Tensor:
        batch_size, sequence_length, _ = value.shape
        end_pos = start_pos + sequence_length
        if end_pos > self.config.max_seq_len:
            raise ValueError(f"end position {end_pos} exceeds max_seq_len")
        angles = self.rope_angles[start_pos:end_pos]

        query = self.q_norm(self.wq_a(value))
        query = self.wq_b(query).unflatten(
            -1, (self.local_heads, self.config.head_dim)
        )
        query = query * torch.rsqrt(
            query.square().mean(dim=-1, keepdim=True) + self.config.norm_eps
        )
        query = apply_rope(query, angles)

        kv = self.kv_norm(self.wkv(value))
        kv = apply_rope(kv, angles)
        self.raw_kv_cache[:batch_size, start_pos:end_pos] = kv
        self.compressor(value, start_pos, self.rope_angles)

        raw_bank = self.raw_kv_cache[:batch_size, :end_pos]
        compressed_bank = self.compressor.compressed_cache[
            :batch_size, : self.compressor.compressed_count
        ]
        kv_bank = torch.cat((raw_bank, compressed_bank), dim=1)
        indices = self._build_indices(
            batch_size, start_pos, sequence_length, end_pos
        )
        output = sparse_attention(
            query,
            kv_bank,
            indices,
            self.attention_sink,
            self.config.head_dim**-0.5,
        )
        output = apply_rope(output, angles, inverse=True)
        output = output.reshape(
            batch_size,
            sequence_length,
            self.local_groups,
            self.heads_per_group * self.config.head_dim,
        )
        output = torch.einsum("bsgd,grd->bsgr", output, self.wo_a)
        return self.wo_b(output.flatten(2))


class Expert(nn.Module):
    def __init__(self, expert_name: str, config: SmallModelConfig) -> None:
        super().__init__()
        self.config = config
        self.w1 = ReplicatedLinear(
            config.dim,
            config.moe_inter_dim,
            f"{expert_name}.w1.weight",
            config,
        )
        self.w2 = ReplicatedLinear(
            config.moe_inter_dim,
            config.dim,
            f"{expert_name}.w2.weight",
            config,
        )
        self.w3 = ReplicatedLinear(
            config.dim,
            config.moe_inter_dim,
            f"{expert_name}.w3.weight",
            config,
        )

    def forward(
        self,
        value: torch.Tensor,
        routing_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gate = self.w1(value).clamp(max=self.config.swiglu_limit)
        up = self.w3(value).clamp(
            min=-self.config.swiglu_limit, max=self.config.swiglu_limit
        )
        activated = F.silu(gate) * up
        if routing_weight is not None:
            activated = activated * routing_weight
        return self.w2(activated)


class ScoreBasedMoE(nn.Module):
    def __init__(self, config: SmallModelConfig, parallel: ParallelContext) -> None:
        super().__init__()
        self.config = config
        self.parallel = parallel
        self.gate_weight = frozen_parameter(
            random_tensor(
                "block21.moe.gate.weight",
                (config.n_routed_experts, config.dim),
                config.seed,
            )
        )
        self.gate_bias = frozen_parameter(
            random_tensor(
                "block21.moe.gate.bias",
                (config.n_routed_experts,),
                config.seed,
                std=0.005,
            )
        )
        local_experts = config.n_routed_experts // parallel.world_size
        self.expert_start = parallel.rank * local_experts
        self.expert_end = self.expert_start + local_experts
        self.experts = nn.ModuleDict(
            {
                str(expert_id): Expert(
                    f"block21.moe.experts.{expert_id}", config
                )
                for expert_id in range(self.expert_start, self.expert_end)
            }
        )
        self.shared_expert = Expert("block21.moe.shared_expert", config)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        shape = value.shape
        tokens = value.reshape(-1, self.config.dim)
        scores = F.softplus(F.linear(tokens.float(), self.gate_weight)).sqrt()
        biased_scores = scores + self.gate_bias
        indices = biased_scores.topk(self.config.n_activated_experts, dim=-1).indices
        weights = scores.gather(1, indices)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.config.route_scale

        local_output = torch.zeros_like(tokens)
        for expert_id in range(self.expert_start, self.expert_end):
            token_index, top_index = torch.where(indices == expert_id)
            if token_index.numel() == 0:
                continue
            routed = self.experts[str(expert_id)](
                tokens[token_index], weights[token_index, top_index, None]
            )
            local_output.index_add_(0, token_index, routed)
        routed_output = self.parallel.all_reduce(
            local_output, "block21_routed_experts"
        )
        output = routed_output + self.shared_expert(tokens)
        return output.reshape(shape)


def hc_pre(
    value: torch.Tensor,
    function_weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    config: SmallModelConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    original_shape = value.shape
    flattened = value.flatten(2).float()
    inverse_rms = torch.rsqrt(
        flattened.square().mean(dim=-1, keepdim=True) + config.norm_eps
    )
    mixes = F.linear(flattened, function_weight) * inverse_rms
    hc_mult = config.hc_mult
    pre_values = mixes[..., :hc_mult]
    post_values = mixes[..., hc_mult : 2 * hc_mult]
    combination_values = mixes[..., 2 * hc_mult :].unflatten(
        -1, (hc_mult, hc_mult)
    )
    pre_base = base[:hc_mult]
    post_base = base[hc_mult : 2 * hc_mult]
    combination_base = base[2 * hc_mult :].unflatten(
        -1, (hc_mult, hc_mult)
    )
    pre_weights = torch.sigmoid(pre_values * scale[0] + pre_base) + config.hc_eps
    post_weights = 2 * torch.sigmoid(
        post_values * scale[1] + post_base
    )
    combination = (
        combination_values * scale[2] + combination_base
    ).softmax(dim=-1) + config.hc_eps
    combination = combination / (
        combination.sum(dim=-2, keepdim=True) + config.hc_eps
    )
    for _ in range(config.hc_sinkhorn_iters - 1):
        combination = combination / (
            combination.sum(dim=-1, keepdim=True) + config.hc_eps
        )
        combination = combination / (
            combination.sum(dim=-2, keepdim=True) + config.hc_eps
        )
    reduced = torch.sum(
        pre_weights.unsqueeze(-1) * flattened.view(original_shape), dim=2
    )
    return reduced, post_weights, combination


def hc_post(
    value: torch.Tensor,
    residual: torch.Tensor,
    post_weights: torch.Tensor,
    combination: torch.Tensor,
) -> torch.Tensor:
    transformed = post_weights.unsqueeze(-1) * value.unsqueeze(-2)
    residual_mix = torch.sum(
        combination.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
    )
    return transformed + residual_mix


class TransformerBlock21(nn.Module):
    def __init__(self, config: SmallModelConfig, parallel: ParallelContext) -> None:
        super().__init__()
        self.layer_id = config.layer_id
        self.config = config
        self.attention = SparseMLAAttention(config, parallel)
        self.moe = ScoreBasedMoE(config, parallel)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        mix_hc = (2 + config.hc_mult) * config.hc_mult
        hc_dim = config.hc_mult * config.dim
        self.hc_attention_function = frozen_parameter(
            random_tensor(
                "block21.hc_attention_function",
                (mix_hc, hc_dim),
                config.seed,
            )
        )
        self.hc_ffn_function = frozen_parameter(
            random_tensor(
                "block21.hc_ffn_function", (mix_hc, hc_dim), config.seed
            )
        )
        self.hc_attention_base = frozen_parameter(
            random_tensor(
                "block21.hc_attention_base", (mix_hc,), config.seed, std=0.005
            )
        )
        self.hc_ffn_base = frozen_parameter(
            random_tensor(
                "block21.hc_ffn_base", (mix_hc,), config.seed, std=0.005
            )
        )
        self.hc_attention_scale = frozen_parameter(
            random_tensor(
                "block21.hc_attention_scale", (3,), config.seed, std=0.05
            )
        )
        self.hc_ffn_scale = frozen_parameter(
            random_tensor(
                "block21.hc_ffn_scale", (3,), config.seed, std=0.05
            )
        )

    def reset_cache(self) -> None:
        self.attention.reset_cache()

    def forward(self, value: torch.Tensor, start_pos: int) -> torch.Tensor:
        residual = value
        reduced, post_weights, combination = hc_pre(
            value,
            self.hc_attention_function,
            self.hc_attention_scale,
            self.hc_attention_base,
            self.config,
        )
        attention_output = self.attention(
            self.attention_norm(reduced), start_pos
        )
        value = hc_post(attention_output, residual, post_weights, combination)

        residual = value
        reduced, post_weights, combination = hc_pre(
            value,
            self.hc_ffn_function,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.config,
        )
        moe_output = self.moe(self.ffn_norm(reduced))
        return hc_post(moe_output, residual, post_weights, combination)


class ParallelLMHead(nn.Module):
    def __init__(self, config: SmallModelConfig, parallel: ParallelContext) -> None:
        super().__init__()
        full_weight = random_tensor(
            "lm_head.weight", (config.vocab_size, config.dim), config.seed
        )
        self.weight = frozen_parameter(
            full_weight.chunk(parallel.world_size, dim=0)[parallel.rank].contiguous()
        )
        self.parallel = parallel

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        local_logits = F.linear(value[:, -1].float(), self.weight)
        return self.parallel.all_gather(local_logits, "lm_head_vocab_parallel")


class SmallDeepSeekV4Flash(nn.Module):
    def __init__(self, config: SmallModelConfig, parallel: ParallelContext) -> None:
        super().__init__()
        config.validate(parallel.world_size)
        self.config = config
        self.parallel = parallel
        self.embedding = ParallelEmbedding(config, parallel)
        self.block = TransformerBlock21(config, parallel)
        self.final_norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = ParallelLMHead(config, parallel)
        hc_dim = config.hc_mult * config.dim
        self.hc_head_function = frozen_parameter(
            random_tensor(
                "hc_head.function", (config.hc_mult, hc_dim), config.seed
            )
        )
        self.hc_head_base = frozen_parameter(
            random_tensor(
                "hc_head.base", (config.hc_mult,), config.seed, std=0.005
            )
        )
        self.hc_head_scale = frozen_parameter(
            random_tensor("hc_head.scale", (1,), config.seed, std=0.05)
        )

    def reset_cache(self) -> None:
        self.block.reset_cache()

    @property
    def compressed_kv_count(self) -> int:
        return self.block.attention.compressor.compressed_count

    def _hc_head(self, value: torch.Tensor) -> torch.Tensor:
        original_shape = value.shape
        flattened = value.flatten(2).float()
        inverse_rms = torch.rsqrt(
            flattened.square().mean(dim=-1, keepdim=True) + self.config.norm_eps
        )
        mixes = F.linear(flattened, self.hc_head_function) * inverse_rms
        weights = torch.sigmoid(
            mixes * self.hc_head_scale + self.hc_head_base
        ) + self.config.hc_eps
        return torch.sum(
            weights.unsqueeze(-1) * flattened.view(original_shape), dim=2
        )

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.size(0) > self.config.max_batch_size:
            raise ValueError("batch size exceeds max_batch_size")
        hidden = self.embedding(input_ids)
        hidden = hidden.unsqueeze(2).repeat(1, 1, self.config.hc_mult, 1)
        hidden = self.block(hidden, start_pos)
        hidden = self.final_norm(self._hc_head(hidden))
        return self.lm_head(hidden)


@dataclass
class InferenceStep:
    input_ids: torch.Tensor
    start_pos: int
    logits: torch.Tensor


@torch.inference_mode()
def run_inference(
    model: SmallDeepSeekV4Flash,
    input_ids: torch.Tensor,
    mode: str,
    decode_steps: int,
) -> list[InferenceStep]:
    model.reset_cache()
    steps: list[InferenceStep] = []
    logits = model(input_ids, 0)
    steps.append(InferenceStep(input_ids.clone(), 0, logits.clone()))
    if mode == "prefill-decode":
        start_pos = input_ids.size(1)
        for _ in range(decode_steps):
            next_token = logits.argmax(dim=-1, keepdim=True)
            logits = model(next_token, start_pos)
            steps.append(InferenceStep(next_token.clone(), start_pos, logits.clone()))
            start_pos += 1
    return steps


def compare_with_tp1(
    config: SmallModelConfig,
    distributed_steps: list[InferenceStep],
) -> float:
    reference = SmallDeepSeekV4Flash(
        config, ParallelContext(world_size=1, rank=0)
    )
    reference.reset_cache()
    maximum_error = 0.0
    for step in distributed_steps:
        reference_logits = reference(step.input_ids, step.start_pos)
        error = (reference_logits - step.logits).abs().max().item()
        maximum_error = max(maximum_error, error)
    return maximum_error


def make_input(config: SmallModelConfig, sequence_length: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed + 1)
    return torch.randint(
        0,
        config.vocab_size,
        (config.max_batch_size, sequence_length),
        generator=generator,
    )


def collect_rank_records(
    rank: int,
    world_size: int,
    records: list[dict[str, Any]],
) -> list[list[dict[str, Any]]] | None:
    if world_size == 1:
        return [records]
    gathered: list[list[dict[str, Any]] | None] | None = (
        [None for _ in range(world_size)] if rank == 0 else None
    )
    dist.gather_object(records, gathered, dst=0)
    if rank == 0:
        return [item for item in gathered if item is not None]
    return None


def validate_rank_outputs(
    final_logits: torch.Tensor,
    rank: int,
    world_size: int,
) -> float:
    if world_size == 1:
        return 0.0
    gathered = [torch.empty_like(final_logits) for _ in range(world_size)]
    dist.all_gather(gathered, final_logits)
    local_error = max(
        (candidate - gathered[0]).abs().max().item() for candidate in gathered
    )
    status = [local_error if rank == 0 else None]
    dist.broadcast_object_list(status, src=0)
    return float(status[0])


def write_communication_report(
    output_dir: Path,
    config: SmallModelConfig,
    args: argparse.Namespace,
    rank_records: list[list[dict[str, Any]]],
    output_shape: tuple[int, ...],
    compressed_kv_count: int,
    rank_error: float,
    reference_error: float | None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "communications.json"
    report = {
        "model": "DeepSeek-V4-Flash-small-layer21",
        "config": asdict(config),
        "run": {
            "mode": args.mode,
            "sequence_length": args.seq_len,
            "decode_steps": args.decode_steps,
            "world_size": len(rank_records),
            "output_shape": list(output_shape),
            "compressed_kv_count": compressed_kv_count,
            "rank_max_abs_error": rank_error,
            "tp1_max_abs_error": reference_error,
        },
        "ranks": [
            {"rank": rank, "collectives": records}
            for rank, records in enumerate(rank_records)
        ],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report_path


def export_rank_onnx(
    model: SmallDeepSeekV4Flash,
    config: SmallModelConfig,
    rank: int,
    phase: str,
    sequence_length: int,
    history_length: int,
    output_path: Path,
) -> dict[str, Any]:
    try:
        import onnx
        from onnx import TensorProto, checker, helper, numpy_helper
    except ImportError as error:
        raise RuntimeError("ONNX export requires the onnx package") from error

    batch_size = 1
    end_position = history_length + sequence_length
    compressed_entries = end_position // config.compress_ratio
    local_vocab = config.vocab_size // 2
    local_heads = config.n_heads // 2
    state_dict = model.state_dict()
    initializers = []
    parameter_names: dict[str, str] = {}
    for state_name, tensor in state_dict.items():
        initializer_name = f"weights.{state_name}"
        parameter_names[state_name] = initializer_name
        initializers.append(
            numpy_helper.from_array(
                tensor.detach().cpu().contiguous().numpy(), name=initializer_name
            )
        )

    graph_inputs = [
        helper.make_tensor_value_info(
            "input_ids", TensorProto.INT64, [batch_size, sequence_length]
        )
    ]
    cache_inputs: list[str] = []
    if phase == "decode":
        cache_specs = (
            ("raw_kv_cache_in", [batch_size, history_length, config.head_dim]),
            (
                "compressed_kv_cache_in",
                [batch_size, history_length // config.compress_ratio, config.head_dim],
            ),
            (
                "compressor_kv_state_in",
                [batch_size, config.compress_ratio, config.head_dim],
            ),
            (
                "compressor_score_state_in",
                [batch_size, config.compress_ratio, config.head_dim],
            ),
        )
        for name, shape in cache_specs:
            graph_inputs.append(
                helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
            )
            cache_inputs.append(name)

    value_info: list[Any] = []

    def value(name: str, shape: list[int], dtype: int = TensorProto.FLOAT) -> str:
        value_info.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    hidden_shape = [batch_size, sequence_length, config.dim]
    hc_shape = [batch_size, sequence_length, config.hc_mult, config.dim]
    post_shape = [batch_size, sequence_length, config.hc_mult]
    combination_shape = [
        batch_size,
        sequence_length,
        config.hc_mult,
        config.hc_mult,
    ]
    routed_rows = batch_size * sequence_length
    nodes = []

    embedding_local = value("embedding.local", hidden_shape)
    nodes.append(
        helper.make_node(
            "VocabParallelEmbedding",
            ["input_ids", parameter_names["embedding.weight"]],
            [embedding_local],
            name="embedding.local_lookup",
            domain="ai.deepseek",
            rank=rank,
            vocab_start=rank * local_vocab,
            vocab_end=(rank + 1) * local_vocab,
        )
    )
    embedding_global = value("embedding.global", hidden_shape)
    nodes.append(
        helper.make_node(
            "AllReduce",
            [embedding_local],
            [embedding_global],
            name="embedding.all_reduce",
            domain="ai.deepseek.distributed",
            group_size=2,
            rank=rank,
            reduction="sum",
        )
    )
    hc_input = value("hc.expanded", hc_shape)
    nodes.append(
        helper.make_node(
            "HCExpand",
            [embedding_global],
            [hc_input],
            name="hc.expand",
            domain="ai.deepseek",
            hc_mult=config.hc_mult,
        )
    )

    attention_input = value("block21.attention.hc_pre.output", hidden_shape)
    attention_post = value("block21.attention.hc_pre.post", post_shape)
    attention_combination = value(
        "block21.attention.hc_pre.combination", combination_shape
    )
    nodes.append(
        helper.make_node(
            "HCPre",
            [
                hc_input,
                parameter_names["block.hc_attention_function"],
                parameter_names["block.hc_attention_scale"],
                parameter_names["block.hc_attention_base"],
            ],
            [attention_input, attention_post, attention_combination],
            name="block21.attention.hc_pre",
            domain="ai.deepseek",
            hc_mult=config.hc_mult,
            sinkhorn_iters=config.hc_sinkhorn_iters,
        )
    )
    attention_normalized = value("block21.attention.norm.output", hidden_shape)
    nodes.append(
        helper.make_node(
            "RMSNorm",
            [attention_input, parameter_names["block.attention_norm.weight"]],
            [attention_normalized],
            name="block21.attention.norm",
            domain="ai.deepseek",
            epsilon=config.norm_eps,
        )
    )

    attention_parameter_inputs = [
        parameter_names[name]
        for name in sorted(parameter_names)
        if name.startswith("block.attention.")
    ]
    attention_partial = value("block21.attention.local_output", hidden_shape)
    raw_cache_out = value(
        "raw_kv_cache_out", [batch_size, end_position, config.head_dim]
    )
    compressed_cache_out = value(
        "compressed_kv_cache_out",
        [batch_size, compressed_entries, config.head_dim],
    )
    compressor_kv_state_out = value(
        "compressor_kv_state_out",
        [batch_size, config.compress_ratio, config.head_dim],
    )
    compressor_score_state_out = value(
        "compressor_score_state_out",
        [batch_size, config.compress_ratio, config.head_dim],
    )
    nodes.append(
        helper.make_node(
            "SparseMLAWithKVCache",
            [attention_normalized, *cache_inputs, *attention_parameter_inputs],
            [
                attention_partial,
                raw_cache_out,
                compressed_cache_out,
                compressor_kv_state_out,
                compressor_score_state_out,
            ],
            name="block21.attention.sparse_mla",
            domain="ai.deepseek",
            rank=rank,
            world_size=2,
            start_pos=history_length,
            local_heads=local_heads,
            head_dim=config.head_dim,
            rope_head_dim=config.rope_head_dim,
            window_size=config.window_size,
            compress_ratio=config.compress_ratio,
        )
    )
    attention_output = value("block21.attention.global_output", hidden_shape)
    nodes.append(
        helper.make_node(
            "AllReduce",
            [attention_partial],
            [attention_output],
            name="block21.attention.output_all_reduce",
            domain="ai.deepseek.distributed",
            group_size=2,
            rank=rank,
            reduction="sum",
        )
    )
    attention_hc_output = value("block21.attention.hc_post.output", hc_shape)
    nodes.append(
        helper.make_node(
            "HCPost",
            [attention_output, hc_input, attention_post, attention_combination],
            [attention_hc_output],
            name="block21.attention.hc_post",
            domain="ai.deepseek",
        )
    )

    ffn_input = value("block21.ffn.hc_pre.output", hidden_shape)
    ffn_post = value("block21.ffn.hc_pre.post", post_shape)
    ffn_combination = value("block21.ffn.hc_pre.combination", combination_shape)
    nodes.append(
        helper.make_node(
            "HCPre",
            [
                attention_hc_output,
                parameter_names["block.hc_ffn_function"],
                parameter_names["block.hc_ffn_scale"],
                parameter_names["block.hc_ffn_base"],
            ],
            [ffn_input, ffn_post, ffn_combination],
            name="block21.ffn.hc_pre",
            domain="ai.deepseek",
            hc_mult=config.hc_mult,
            sinkhorn_iters=config.hc_sinkhorn_iters,
        )
    )
    ffn_normalized = value("block21.ffn.norm.output", hidden_shape)
    nodes.append(
        helper.make_node(
            "RMSNorm",
            [ffn_input, parameter_names["block.ffn_norm.weight"]],
            [ffn_normalized],
            name="block21.ffn.norm",
            domain="ai.deepseek",
            epsilon=config.norm_eps,
        )
    )
    routing_weights = value(
        "block21.moe.routing_weights",
        [routed_rows, config.n_activated_experts],
    )
    routing_indices = value(
        "block21.moe.routing_indices",
        [routed_rows, config.n_activated_experts],
        TensorProto.INT64,
    )
    nodes.append(
        helper.make_node(
            "ScoreBasedRouter",
            [
                ffn_normalized,
                parameter_names["block.moe.gate_weight"],
                parameter_names["block.moe.gate_bias"],
            ],
            [routing_weights, routing_indices],
            name="block21.moe.router",
            domain="ai.deepseek",
            top_k=config.n_activated_experts,
            score_function=config.score_func,
            route_scale=config.route_scale,
        )
    )
    local_expert_parameters = [
        parameter_names[name]
        for name in sorted(parameter_names)
        if name.startswith("block.moe.experts.")
    ]
    local_routed_output = value("block21.moe.local_routed_output", hidden_shape)
    local_expert_count = config.n_routed_experts // 2
    nodes.append(
        helper.make_node(
            "LocalRoutedExperts",
            [
                ffn_normalized,
                routing_weights,
                routing_indices,
                *local_expert_parameters,
            ],
            [local_routed_output],
            name="block21.moe.local_experts",
            domain="ai.deepseek",
            expert_start=rank * local_expert_count,
            expert_end=(rank + 1) * local_expert_count,
            intermediate_size=config.moe_inter_dim,
        )
    )
    routed_output = value("block21.moe.routed_output", hidden_shape)
    nodes.append(
        helper.make_node(
            "AllReduce",
            [local_routed_output],
            [routed_output],
            name="block21.moe.expert_all_reduce",
            domain="ai.deepseek.distributed",
            group_size=2,
            rank=rank,
            reduction="sum",
        )
    )
    shared_parameters = [
        parameter_names[name]
        for name in sorted(parameter_names)
        if name.startswith("block.moe.shared_expert.")
    ]
    shared_output = value("block21.moe.shared_output", hidden_shape)
    nodes.append(
        helper.make_node(
            "SharedExpert",
            [ffn_normalized, *shared_parameters],
            [shared_output],
            name="block21.moe.shared_expert",
            domain="ai.deepseek",
            intermediate_size=config.moe_inter_dim,
        )
    )
    moe_output = value("block21.moe.output", hidden_shape)
    nodes.append(
        helper.make_node(
            "Add",
            [routed_output, shared_output],
            [moe_output],
            name="block21.moe.add_shared",
        )
    )
    block_output = value("block21.output", hc_shape)
    nodes.append(
        helper.make_node(
            "HCPost",
            [moe_output, attention_hc_output, ffn_post, ffn_combination],
            [block_output],
            name="block21.ffn.hc_post",
            domain="ai.deepseek",
        )
    )

    head_hidden = value("hc_head.output", hidden_shape)
    nodes.append(
        helper.make_node(
            "HCHead",
            [
                block_output,
                parameter_names["hc_head_function"],
                parameter_names["hc_head_scale"],
                parameter_names["hc_head_base"],
            ],
            [head_hidden],
            name="hc_head",
            domain="ai.deepseek",
            hc_mult=config.hc_mult,
        )
    )
    normalized_hidden = value("final_norm.output", hidden_shape)
    nodes.append(
        helper.make_node(
            "RMSNorm",
            [head_hidden, parameter_names["final_norm.weight"]],
            [normalized_hidden],
            name="final_norm",
            domain="ai.deepseek",
            epsilon=config.norm_eps,
        )
    )
    local_logits = value("lm_head.local_logits", [batch_size, local_vocab])
    nodes.append(
        helper.make_node(
            "LocalLMHead",
            [normalized_hidden, parameter_names["lm_head.weight"]],
            [local_logits],
            name="lm_head.local",
            domain="ai.deepseek",
            use_last_token=1,
        )
    )
    nodes.append(
        helper.make_node(
            "AllGather",
            [local_logits],
            ["logits"],
            name="lm_head.all_gather",
            domain="ai.deepseek.distributed",
            axis=-1,
            group_size=2,
            rank=rank,
        )
    )

    graph_outputs = [
        helper.make_tensor_value_info(
            "logits", TensorProto.FLOAT, [batch_size, config.vocab_size]
        ),
        helper.make_tensor_value_info(
            raw_cache_out,
            TensorProto.FLOAT,
            [batch_size, end_position, config.head_dim],
        ),
        helper.make_tensor_value_info(
            compressed_cache_out,
            TensorProto.FLOAT,
            [batch_size, compressed_entries, config.head_dim],
        ),
        helper.make_tensor_value_info(
            compressor_kv_state_out,
            TensorProto.FLOAT,
            [batch_size, config.compress_ratio, config.head_dim],
        ),
        helper.make_tensor_value_info(
            compressor_score_state_out,
            TensorProto.FLOAT,
            [batch_size, config.compress_ratio, config.head_dim],
        ),
    ]
    graph_output_names = {item.name for item in graph_outputs}
    value_info = [item for item in value_info if item.name not in graph_output_names]
    graph = helper.make_graph(
        nodes,
        f"DeepSeek-V4-Flash-small-rank{rank}-{phase}",
        graph_inputs,
        graph_outputs,
        initializer=initializers,
        value_info=value_info,
    )
    onnx_model = helper.make_model(
        graph,
        producer_name="deepseek-v4-small-cpu-debug",
        producer_version="1.0",
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid("ai.deepseek", 1),
            helper.make_opsetid("ai.deepseek.distributed", 1),
        ],
    )
    onnx_model.metadata_props.extend(
        [
            onnx.StringStringEntryProto(key="rank", value=str(rank)),
            onnx.StringStringEntryProto(key="world_size", value="2"),
            onnx.StringStringEntryProto(key="phase", value=phase),
            onnx.StringStringEntryProto(key="layer_id", value=str(config.layer_id)),
            onnx.StringStringEntryProto(
                key="weights", value="deterministic random FP32 initializers"
            ),
        ]
    )
    checker.check_model(onnx_model, full_check=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(onnx_model, output_path)
    checker.check_model(onnx.load(output_path), full_check=True)
    return {
        "path": str(output_path),
        "rank": rank,
        "phase": phase,
        "sequence_length": sequence_length,
        "history_length": history_length,
        "node_count": len(nodes),
        "initializer_count": len(initializers),
        "initializer_bytes": sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values()),
        "collectives": [
            {"name": "embedding.all_reduce", "operation": "AllReduce"},
            {
                "name": "block21.attention.output_all_reduce",
                "operation": "AllReduce",
            },
            {
                "name": "block21.moe.expert_all_reduce",
                "operation": "AllReduce",
            },
            {"name": "lm_head.all_gather", "operation": "AllGather"},
        ],
    }


def export_onnx_models(args: argparse.Namespace) -> None:
    config = SmallModelConfig()
    config.validate(world_size=2)
    if args.onnx_prefill_seq_len <= 0:
        raise ValueError("onnx-prefill-seq-len must be positive")
    if args.onnx_decode_history < config.compress_ratio:
        raise ValueError(
            "onnx-decode-history must be at least one compression interval"
        )
    if args.onnx_prefill_seq_len > config.max_seq_len:
        raise ValueError("ONNX prefill length exceeds max_seq_len")
    if args.onnx_decode_history + 1 > config.max_seq_len:
        raise ValueError("ONNX decode history exceeds max_seq_len")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    exports: list[dict[str, Any]] = []
    for rank in range(2):
        model = SmallDeepSeekV4Flash(
            config, ParallelContext(world_size=2, rank=rank)
        ).eval()
        exports.append(
            export_rank_onnx(
                model,
                config,
                rank,
                "prefill",
                args.onnx_prefill_seq_len,
                0,
                args.output_dir / f"rank{rank}_prefill.onnx",
            )
        )
        exports.append(
            export_rank_onnx(
                model,
                config,
                rank,
                "decode",
                1,
                args.onnx_decode_history,
                args.output_dir / f"rank{rank}_decode.onnx",
            )
        )
    manifest_path = args.output_dir / "onnx_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "model": "DeepSeek-V4-Flash-small-layer21",
                "world_size": 2,
                "config": asdict(config),
                "exports": exports,
                "runtime_note": (
                    "Custom ai.deepseek and ai.deepseek.distributed kernels are "
                    "required for ONNX Runtime execution."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "onnx_models": [item["path"] for item in exports],
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).resolve().parent / "output"
    parser = argparse.ArgumentParser(
        description="Run one DeepSeek V4 Flash layer-21 block with CPU TP2/Gloo"
    )
    parser.add_argument(
        "--mode", choices=("prefill", "prefill-decode"), default="prefill"
    )
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--decode-steps", type=int, default=2)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--trace-comm", action="store_true")
    parser.add_argument(
        "--check-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--expect-compressed-kv", type=int)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--onnx-prefill-seq-len", type=int, default=128)
    parser.add_argument("--onnx-decode-history", type=int, default=128)
    return parser.parse_args()


def run_distributed(args: argparse.Namespace) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size != 2:
        raise RuntimeError(
            "This communication study requires exactly two processes; run with "
            "torchrun --standalone --nproc-per-node=2"
        )
    dist.init_process_group("gloo")
    torch.set_num_threads(args.num_threads)
    config = SmallModelConfig()
    config.validate(world_size)
    if args.seq_len <= 0:
        raise ValueError("seq-len must be positive")
    total_length = args.seq_len + (
        args.decode_steps if args.mode == "prefill-decode" else 0
    )
    if total_length > config.max_seq_len:
        raise ValueError(
            f"requested length {total_length} exceeds max_seq_len={config.max_seq_len}"
        )

    tracer = CollectiveTracer(rank, args.trace_comm)
    parallel = ParallelContext(world_size=world_size, rank=rank, tracer=tracer)
    model = SmallDeepSeekV4Flash(config, parallel).eval()
    input_ids = make_input(config, args.seq_len)
    started = time.perf_counter()
    steps = run_inference(model, input_ids, args.mode, args.decode_steps)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    final_logits = steps[-1].logits

    expected_collectives = 4 * len(steps)
    if len(tracer.records) != expected_collectives:
        raise AssertionError(
            f"expected {expected_collectives} model collectives, got {len(tracer.records)}"
        )
    if not torch.isfinite(final_logits).all():
        raise AssertionError("model output contains NaN or Inf")
    if args.expect_compressed_kv is not None:
        if model.compressed_kv_count != args.expect_compressed_kv:
            raise AssertionError(
                f"expected {args.expect_compressed_kv} compressed KV entries, "
                f"got {model.compressed_kv_count}"
            )

    rank_error = validate_rank_outputs(final_logits, rank, world_size)
    if rank_error != 0.0:
        raise AssertionError(f"rank outputs differ: max_abs_error={rank_error}")

    reference_error: float | None = None
    reference_status: list[dict[str, Any] | None] = [None]
    dist.barrier()
    if rank == 0 and args.check_reference:
        try:
            reference_error = compare_with_tp1(config, steps)
            tolerance = 2e-5
            if reference_error > tolerance:
                reference_status[0] = {
                    "ok": False,
                    "message": (
                        f"TP2 differs from TP1: max_abs_error={reference_error:.8g}, "
                        f"tolerance={tolerance}"
                    ),
                }
            else:
                reference_status[0] = {"ok": True, "error": reference_error}
        except Exception as error:
            reference_status[0] = {"ok": False, "message": repr(error)}
    elif rank == 0:
        reference_status[0] = {"ok": True, "error": None}
    dist.broadcast_object_list(reference_status, src=0)
    if not reference_status[0]["ok"]:
        raise AssertionError(reference_status[0]["message"])
    if args.check_reference:
        reference_error = float(reference_status[0]["error"])

    rank_records = collect_rank_records(rank, world_size, tracer.records)
    if rank == 0:
        report_path = write_communication_report(
            args.output_dir,
            config,
            args,
            rank_records,
            tuple(final_logits.shape),
            model.compressed_kv_count,
            rank_error,
            reference_error,
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(
            json.dumps(
                {
                    "status": "ok",
                    "block_count": 1,
                    "layer_id": config.layer_id,
                    "world_size": world_size,
                    "mode": args.mode,
                    "steps": len(steps),
                    "output_shape": list(final_logits.shape),
                    "compressed_kv_count": model.compressed_kv_count,
                    "collectives_per_step": 4,
                    "local_parameter_count_rank0": parameter_count,
                    "tp1_max_abs_error": reference_error,
                    "elapsed_ms": elapsed_ms,
                    "communication_report": str(report_path),
                },
                indent=2,
            ),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    if args.export_onnx:
        export_onnx_models(args)
    else:
        run_distributed(args)


if __name__ == "__main__":
    main()