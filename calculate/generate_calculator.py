#!/usr/bin/env python3
"""Generate a formula-driven DeepSeek V4 Flash architecture calculator.

The workbook is intended for architecture exploration, not measured performance.
All major results are Excel formulas with cached baseline values so the file is
both editable in Excel and auditable from source control.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xlsxwriter
from xlsxwriter.utility import xl_col_to_name


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "inference" / "config.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "deepseek_v4_flash_calculator.xlsx"
MAX_LAYER_ROWS = 64


@dataclass(frozen=True)
class Inputs:
    tp: int
    prefill_batch: int
    prefill_sequence: int
    decode_batch: int
    decode_tokens: int
    decode_context: int
    max_context: int
    hidden: int
    vocab: int
    heads: int
    head_dim: int
    rope_dim: int
    q_rank: int
    o_groups: int
    o_rank: int
    routed_experts: int
    activated_experts: int
    shared_experts: int
    expert_inter: int
    window: int
    short_ratio: int
    long_ratio: int
    index_heads: int
    index_dim: int
    index_topk: int
    hc_slots: int
    hc_iters: int
    hash_layers: int
    bf16_bytes: float = 2
    fp32_bytes: float = 4
    fp8_bytes: float = 1
    fp4_bytes: float = 0.5
    scale_bytes: float = 1
    int32_bytes: float = 4
    int64_bytes: float = 8
    fp8_block: int = 128
    fp4_block: int = 32
    kernel_min_heads: int = 16
    peak_tflops: float = 1000
    hbm_gbps: float = 3000
    interconnect_gbps: float = 900
    prefill_target_ms: float = 1000
    decode_target_ms: float = 10


@dataclass(frozen=True)
class LayerCounts:
    total: int
    window: int
    short: int
    long: int
    hash: int


@dataclass
class Item:
    category: str
    name: str
    layer_scope: str
    readable_formula: str
    global_flops_formula: str
    global_flops: float
    rank_flops_formula: str
    rank_flops: float
    read_formula: str
    read_bytes: float
    write_formula: str
    write_bytes: float
    network_formula: str
    network_bytes: float
    accounting: str
    notes: str


def ceil_div(value: float, divisor: float) -> int:
    return math.ceil(value / divisor)


def fp8_matrix_bytes(in_features: int, out_features: int, p: Inputs) -> float:
    return (
        in_features * out_features * p.fp8_bytes
        + ceil_div(in_features, p.fp8_block)
        * ceil_div(out_features, p.fp8_block)
        * p.scale_bytes
    )


def xl_fp8(in_features: str, out_features: str) -> str:
    return (
        f"(({in_features})*({out_features})*FP8Bytes+"
        f"ROUNDUP(({in_features})/FP8Block,0)*"
        f"ROUNDUP(({out_features})/FP8Block,0)*ScaleBytes)"
    )


def expert_bytes(p: Inputs) -> float:
    logical = 3 * p.hidden * p.expert_inter
    scales = 3 * p.hidden * p.expert_inter / p.fp4_block
    return logical * p.fp4_bytes + scales * p.scale_bytes


def xl_expert_bytes() -> str:
    return (
        "(3*HiddenSize*ExpertInter*FP4Bytes+"
        "3*HiddenSize*ExpertInter/FP4Block*ScaleBytes)"
    )


def sum_floor(sequence: int, ratio: int) -> int:
    quotient, remainder = divmod(sequence, ratio)
    return ratio * quotient * (quotient - 1) // 2 + quotient * (remainder + 1)


def sum_floor_capped(sequence: int, ratio: int, cap: int) -> int:
    if sequence < cap * ratio:
        return sum_floor(sequence, ratio)
    return ratio * cap * (cap - 1) // 2 + (sequence - cap * ratio + 1) * cap


def raw_window_pairs(sequence: int, window: int) -> int:
    if sequence <= window:
        return sequence * (sequence + 1) // 2
    return window * (window + 1) // 2 + (sequence - window) * window


def load_inputs(config_path: Path) -> tuple[Inputs, list[int]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ratios = list(config["compress_ratios"][: config["n_layers"]])
    return (
        Inputs(
            tp=1,
            prefill_batch=1,
            prefill_sequence=8192,
            decode_batch=1,
            decode_tokens=1,
            decode_context=1 << 20,
            max_context=1 << 20,
            hidden=config["dim"],
            vocab=config["vocab_size"],
            heads=config["n_heads"],
            head_dim=config["head_dim"],
            rope_dim=config["rope_head_dim"],
            q_rank=config["q_lora_rank"],
            o_groups=config["o_groups"],
            o_rank=config["o_lora_rank"],
            routed_experts=config["n_routed_experts"],
            activated_experts=config["n_activated_experts"],
            shared_experts=config["n_shared_experts"],
            expert_inter=config["moe_inter_dim"],
            window=config["window_size"],
            short_ratio=4,
            long_ratio=128,
            index_heads=config["index_n_heads"],
            index_dim=config["index_head_dim"],
            index_topk=config["index_topk"],
            hc_slots=config["hc_mult"],
            hc_iters=config["hc_sinkhorn_iters"],
            hash_layers=config["n_hash_layers"],
        ),
        ratios,
    )


def count_layers(ratios: list[int], p: Inputs) -> LayerCounts:
    return LayerCounts(
        total=len(ratios),
        window=sum(value == 0 for value in ratios),
        short=sum(value == p.short_ratio for value in ratios),
        long=sum(value == p.long_ratio for value in ratios),
        hash=min(p.hash_layers, len(ratios)),
    )


def weight_values(p: Inputs, layers: LayerCounts) -> dict[str, float]:
    local_heads = p.heads // p.tp
    local_groups = p.o_groups // p.tp
    local_index_heads = p.index_heads // p.tp

    core_global = (
        fp8_matrix_bytes(p.hidden, p.q_rank, p)
        + fp8_matrix_bytes(p.q_rank, p.heads * p.head_dim, p)
        + fp8_matrix_bytes(p.hidden, p.head_dim, p)
        + p.heads * p.head_dim * p.o_rank * p.bf16_bytes
        + fp8_matrix_bytes(p.o_groups * p.o_rank, p.hidden, p)
    )
    core_rank = (
        fp8_matrix_bytes(p.hidden, p.q_rank, p)
        + fp8_matrix_bytes(p.q_rank, local_heads * p.head_dim, p)
        + fp8_matrix_bytes(p.hidden, p.head_dim, p)
        + local_heads * p.head_dim * p.o_rank * p.bf16_bytes
        + fp8_matrix_bytes(local_groups * p.o_rank, p.hidden, p)
    )

    short_main_compressor = (
        2 * p.hidden * (2 * p.head_dim) * p.fp32_bytes
        + p.short_ratio * 2 * p.head_dim * p.fp32_bytes
        + p.head_dim * p.fp32_bytes
    )
    long_main_compressor = (
        2 * p.hidden * p.head_dim * p.fp32_bytes
        + p.long_ratio * p.head_dim * p.fp32_bytes
        + p.head_dim * p.fp32_bytes
    )
    index_core_global = (
        fp8_matrix_bytes(p.q_rank, p.index_heads * p.index_dim, p)
        + p.hidden * p.index_heads * p.bf16_bytes
    )
    index_core_rank = (
        fp8_matrix_bytes(p.q_rank, local_index_heads * p.index_dim, p)
        + p.hidden * local_index_heads * p.bf16_bytes
    )
    index_compressor = (
        2 * p.hidden * (2 * p.index_dim) * p.fp32_bytes
        + p.short_ratio * 2 * p.index_dim * p.fp32_bytes
        + p.index_dim * p.fp32_bytes
    )

    one_expert = expert_bytes(p)
    shared_one = (
        2 * fp8_matrix_bytes(p.hidden, p.expert_inter, p)
        + fp8_matrix_bytes(p.expert_inter, p.hidden, p)
    )
    active_hash = layers.hash
    score_layers = layers.total - active_hash
    router = (
        layers.total * p.routed_experts * p.hidden * p.bf16_bytes
        + score_layers * p.routed_experts * p.fp32_bytes
        + active_hash
        * p.vocab
        * p.activated_experts
        * p.int32_bytes
    )

    mix_hc = (2 + p.hc_slots) * p.hc_slots
    hc_block = 2 * (
        mix_hc * p.hc_slots * p.hidden * p.fp32_bytes
        + mix_hc * p.fp32_bytes
        + 3 * p.fp32_bytes
    )
    norms_per_block = (
        2 * p.hidden + p.q_rank + p.head_dim + local_heads
    ) * p.fp32_bytes
    tail = (
        p.hc_slots * p.hc_slots * p.hidden * p.fp32_bytes
        + (p.hc_slots + 1) * p.fp32_bytes
        + p.hidden * p.fp32_bytes
    )

    return {
        "embedding": p.vocab / p.tp * p.hidden * p.bf16_bytes,
        "lm_head": p.vocab / p.tp * p.hidden * p.fp32_bytes,
        "core_global_layer": core_global,
        "core_rank_layer": core_rank,
        "short_main_compressor": short_main_compressor,
        "long_main_compressor": long_main_compressor,
        "index_core_global": index_core_global,
        "index_core_rank": index_core_rank,
        "index_compressor": index_compressor,
        "expert_one": one_expert,
        "routed_rank": layers.total * (p.routed_experts / p.tp) * one_expert,
        "shared_rank": layers.total * p.shared_experts * shared_one,
        "router_rank": router,
        "hc_rank": layers.total * hc_block,
        "norms_rank": layers.total * norms_per_block,
        "tail_rank": tail,
    }


def scenario_helpers(p: Inputs, mode: str) -> dict[str, float]:
    if mode == "prefill":
        sequence = p.prefill_sequence
        rows = p.prefill_batch * sequence
        return {
            "rows": rows,
            "raw_pairs": raw_window_pairs(sequence, p.window),
            "short_pairs": sum_floor_capped(sequence, p.short_ratio, p.index_topk),
            "long_pairs": sum_floor(sequence, p.long_ratio),
            "index_pairs": sum_floor(sequence, p.short_ratio),
            "expert_probability": 1
            - (1 - p.activated_experts / p.routed_experts) ** rows,
        }
    rows = p.decode_batch * p.decode_tokens
    return {
        "rows": rows,
        "raw_pairs": p.decode_tokens * min(p.window, p.decode_context),
        "short_pairs": p.decode_tokens
        * min(p.index_topk, p.decode_context // p.short_ratio),
        "long_pairs": p.decode_tokens * (p.decode_context // p.long_ratio),
        "index_pairs": p.decode_tokens * (p.decode_context // p.short_ratio),
        "expert_probability": 1
        - (1 - p.activated_experts / p.routed_experts) ** rows,
    }


def scenario_items(
    p: Inputs,
    layers: LayerCounts,
    mode: str,
) -> tuple[list[Item], dict[str, float]]:
    prefix = "PF" if mode == "prefill" else "DC"
    batch_name = "PrefillBatch" if mode == "prefill" else "DecodeBatch"
    rows_name = f"{prefix}_Rows"
    helpers = scenario_helpers(p, mode)
    weights = weight_values(p, layers)
    local_heads = p.heads / p.tp
    kernel_heads = max(local_heads, p.kernel_min_heads)
    local_groups = p.o_groups / p.tp
    local_index_heads = p.index_heads / p.tp
    rows = helpers["rows"]
    pair_total = (
        layers.window * helpers["raw_pairs"]
        + layers.short * (helpers["raw_pairs"] + helpers["short_pairs"])
        + layers.long * (helpers["raw_pairs"] + helpers["long_pairs"])
    )
    xl_pair_total = (
        f"(WindowLayers*{prefix}_RawPairs+ShortLayers*"
        f"({prefix}_RawPairs+{prefix}_ShortPairs)+LongLayers*"
        f"({prefix}_RawPairs+{prefix}_LongPairs))"
    )

    core_global_per_row = 2 * (
        p.hidden * p.q_rank
        + p.q_rank * p.heads * p.head_dim
        + p.hidden * p.head_dim
        + p.heads * p.head_dim * p.o_rank
        + p.o_groups * p.o_rank * p.hidden
    )
    core_rank_per_row = 2 * (
        p.hidden * p.q_rank
        + p.q_rank * local_heads * p.head_dim
        + p.hidden * p.head_dim
        + local_heads * p.head_dim * p.o_rank
        + local_groups * p.o_rank * p.hidden
    )
    core_weight_xl = (
        f"({xl_fp8('HiddenSize', 'QLoraRank')}+"
        f"{xl_fp8('QLoraRank', 'LocalHeads*HeadDim')}+"
        f"{xl_fp8('HiddenSize', 'HeadDim')}+"
        "LocalHeads*HeadDim*OLoraRank*BF16Bytes+"
        f"{xl_fp8('LocalOGroups*OLoraRank', 'HiddenSize')})"
    )
    core_activation_read = rows * layers.total * (
        2 * p.hidden * p.bf16_bytes
        + p.q_rank * p.bf16_bytes
        + kernel_heads * p.head_dim * p.bf16_bytes
        + local_groups * p.o_rank * p.bf16_bytes
    )
    core_activation_write = rows * layers.total * (
        p.q_rank
        + kernel_heads * p.head_dim
        + p.head_dim
        + local_groups * p.o_rank
        + p.hidden
    ) * p.bf16_bytes

    items: list[Item] = []
    items.append(
        Item(
            "Attention",
            "Q/K/O projections",
            "All active layers",
            "2*rows*layers*(Wq_a+Wq_b+Wkv+Wo_a+Wo_b)",
            (
                f"={rows_name}*TotalLayers*2*(HiddenSize*QLoraRank+"
                "QLoraRank*NumHeads*HeadDim+HiddenSize*HeadDim+"
                "NumHeads*HeadDim*OLoraRank+OGroups*OLoraRank*HiddenSize)"
            ),
            rows * layers.total * core_global_per_row,
            (
                f"={rows_name}*TotalLayers*2*(HiddenSize*QLoraRank+"
                "QLoraRank*LocalHeads*HeadDim+HiddenSize*HeadDim+"
                "LocalHeads*HeadDim*OLoraRank+LocalOGroups*OLoraRank*HiddenSize)"
            ),
            rows * layers.total * core_rank_per_row,
            (
                f"=TotalLayers*{core_weight_xl}+{rows_name}*TotalLayers*"
                "(2*HiddenSize*BF16Bytes+QLoraRank*BF16Bytes+"
                "KernelLocalHeads*HeadDim*BF16Bytes+LocalOGroups*OLoraRank*BF16Bytes)"
            ),
            layers.total * weights["core_rank_layer"] + core_activation_read,
            (
                f"={rows_name}*TotalLayers*(QLoraRank+KernelLocalHeads*HeadDim+"
                "HeadDim+LocalOGroups*OLoraRank+HiddenSize)*BF16Bytes"
            ),
            core_activation_write,
            "=0",
            0,
            "Major",
            "Wq_a and Wkv are replicated; Wq_b/Wo_a/Wo_b are TP-sharded.",
        )
    )

    sparse_global = 4 * p.prefill_batch * p.heads * p.head_dim * pair_total if mode == "prefill" else 4 * p.decode_batch * p.heads * p.head_dim * pair_total
    sparse_rank = sparse_global * kernel_heads / p.heads
    sparse_read = (
        rows * layers.total * kernel_heads * p.head_dim * p.bf16_bytes
        + (p.prefill_batch if mode == "prefill" else p.decode_batch)
        * pair_total
        * p.head_dim
        * p.bf16_bytes
    )
    sparse_write = rows * layers.total * kernel_heads * p.head_dim * p.bf16_bytes
    items.append(
        Item(
            "Attention",
            "Sparse attention QK + AV",
            "Window/short/long layer mix",
            "4*batch*heads*head_dim*sum(candidate pairs)",
            f"=4*{batch_name}*NumHeads*HeadDim*{xl_pair_total}",
            sparse_global,
            f"=4*{batch_name}*KernelLocalHeads*HeadDim*{xl_pair_total}",
            sparse_rank,
            (
                f"={rows_name}*TotalLayers*KernelLocalHeads*HeadDim*BF16Bytes+"
                f"{batch_name}*{xl_pair_total}*HeadDim*BF16Bytes"
            ),
            sparse_read,
            f"={rows_name}*TotalLayers*KernelLocalHeads*HeadDim*BF16Bytes",
            sparse_write,
            "=0",
            0,
            "Major",
            "KV is shared across heads; per-rank kernel heads include minimum-head padding.",
        )
    )

    index_projection_global = (
        rows
        * layers.short
        * 2
        * (p.q_rank * p.index_heads * p.index_dim + p.hidden * p.index_heads)
    )
    index_projection_rank = index_projection_global / p.tp
    index_core_weight_xl = (
        f"({xl_fp8('QLoraRank', 'LocalIndexHeads*IndexHeadDim')}+"
        "HiddenSize*LocalIndexHeads*BF16Bytes)"
    )
    index_projection_read = (
        layers.short * weights["index_core_rank"]
        + rows
        * layers.short
        * (p.q_rank + p.hidden)
        * p.bf16_bytes
    )
    index_projection_write = (
        rows
        * layers.short
        * local_index_heads
        * (p.index_dim + 1)
        * p.bf16_bytes
    )
    items.append(
        Item(
            "Attention",
            "Ratio-4 Indexer projections",
            "Short-compression layers",
            "2*rows*short_layers*(q_rank*index_heads*index_dim+hidden*index_heads)",
            (
                f"=2*{rows_name}*ShortLayers*(QLoraRank*IndexHeads*IndexHeadDim+"
                "HiddenSize*IndexHeads)"
            ),
            index_projection_global,
            (
                f"=2*{rows_name}*ShortLayers*(QLoraRank*LocalIndexHeads*IndexHeadDim+"
                "HiddenSize*LocalIndexHeads)"
            ),
            index_projection_rank,
            (
                f"=ShortLayers*{index_core_weight_xl}+{rows_name}*ShortLayers*"
                "(QLoraRank+HiddenSize)*BF16Bytes"
            ),
            index_projection_read,
            (
                f"={rows_name}*ShortLayers*LocalIndexHeads*"
                "(IndexHeadDim+1)*BF16Bytes"
            ),
            index_projection_write,
            "=0",
            0,
            "Major",
            "Indexer Q and token-weight projections; only ratio-4 layers use it.",
        )
    )

    index_score_global = (
        2
        * (p.prefill_batch if mode == "prefill" else p.decode_batch)
        * layers.short
        * p.index_heads
        * p.index_dim
        * helpers["index_pairs"]
    )
    index_score_rank = index_score_global / p.tp
    index_score_read = (
        (p.prefill_batch if mode == "prefill" else p.decode_batch)
        * layers.short
        * helpers["index_pairs"]
        * (
            p.index_dim * p.bf16_bytes
            + local_index_heads * p.bf16_bytes
        )
    )
    index_score_write = (
        (p.prefill_batch if mode == "prefill" else p.decode_batch)
        * layers.short
        * helpers["index_pairs"]
        * (local_index_heads * p.bf16_bytes + p.fp32_bytes)
    )
    items.append(
        Item(
            "Attention",
            "Ratio-4 Indexer score scan",
            "Short-compression layers",
            "2*batch*short_layers*index_heads*index_dim*causal compressed pairs",
            (
                f"=2*{batch_name}*ShortLayers*IndexHeads*IndexHeadDim*"
                f"{prefix}_IndexPairs"
            ),
            index_score_global,
            (
                f"=2*{batch_name}*ShortLayers*LocalIndexHeads*IndexHeadDim*"
                f"{prefix}_IndexPairs"
            ),
            index_score_rank,
            (
                f"={batch_name}*ShortLayers*{prefix}_IndexPairs*"
                "(IndexHeadDim*BF16Bytes+LocalIndexHeads*BF16Bytes)"
            ),
            index_score_read,
            (
                f"={batch_name}*ShortLayers*{prefix}_IndexPairs*"
                "(LocalIndexHeads*BF16Bytes+FP32Bytes)"
            ),
            index_score_write,
            "=0",
            0,
            "Major",
            "Scans all completed ratio-4 index entries before Top-K selection.",
        )
    )

    main_comp_global = rows * (
        layers.short * 8 * p.hidden * p.head_dim
        + layers.long * 4 * p.hidden * p.head_dim
    )
    main_comp_weight_xl = (
        "ShortLayers*(2*HiddenSize*(2*HeadDim)*FP32Bytes+"
        "ShortRatio*2*HeadDim*FP32Bytes+HeadDim*FP32Bytes)+"
        "LongLayers*(2*HiddenSize*HeadDim*FP32Bytes+"
        "LongRatio*HeadDim*FP32Bytes+HeadDim*FP32Bytes)"
    )
    main_comp_read = (
        layers.short * weights["short_main_compressor"]
        + layers.long * weights["long_main_compressor"]
        + rows
        * (layers.short + layers.long)
        * p.hidden
        * p.fp32_bytes
    )
    main_comp_write = rows * (
        layers.short * 4 * p.head_dim + layers.long * 2 * p.head_dim
    ) * p.fp32_bytes
    items.append(
        Item(
            "Attention",
            "Main KV compressor projections",
            "All compressed-KV layers",
            "rows*(short_layers*8*hidden*head_dim+long_layers*4*hidden*head_dim)",
            (
                f"={rows_name}*(ShortLayers*8*HiddenSize*HeadDim+"
                "LongLayers*4*HiddenSize*HeadDim)"
            ),
            main_comp_global,
            (
                f"={rows_name}*(ShortLayers*8*HiddenSize*HeadDim+"
                "LongLayers*4*HiddenSize*HeadDim)"
            ),
            main_comp_global,
            f"={main_comp_weight_xl}+{rows_name}*(ShortLayers+LongLayers)*HiddenSize*FP32Bytes",
            main_comp_read,
            (
                f"={rows_name}*(ShortLayers*4*HeadDim+LongLayers*2*HeadDim)*"
                "FP32Bytes"
            ),
            main_comp_write,
            "=0",
            0,
            "Major",
            "Replicated on every TP rank; ratio-4 uses overlapping 2x projections.",
        )
    )

    index_comp_global = rows * layers.short * 8 * p.hidden * p.index_dim
    index_comp_weight_xl = (
        "ShortLayers*(2*HiddenSize*(2*IndexHeadDim)*FP32Bytes+"
        "ShortRatio*2*IndexHeadDim*FP32Bytes+IndexHeadDim*FP32Bytes)"
    )
    index_comp_read = (
        layers.short * weights["index_compressor"]
        + rows * layers.short * p.hidden * p.fp32_bytes
    )
    index_comp_write = (
        rows * layers.short * 4 * p.index_dim * p.fp32_bytes
    )
    items.append(
        Item(
            "Attention",
            "Indexer compressor projections",
            "Short-compression layers",
            "rows*short_layers*8*hidden*index_dim",
            f"={rows_name}*ShortLayers*8*HiddenSize*IndexHeadDim",
            index_comp_global,
            f"={rows_name}*ShortLayers*8*HiddenSize*IndexHeadDim",
            index_comp_global,
            f"={index_comp_weight_xl}+{rows_name}*ShortLayers*HiddenSize*FP32Bytes",
            index_comp_read,
            f"={rows_name}*ShortLayers*4*IndexHeadDim*FP32Bytes",
            index_comp_write,
            "=0",
            0,
            "Major",
            "Replicated auxiliary compressor used to build the Indexer cache.",
        )
    )

    router_global = rows * layers.total * 2 * p.hidden * p.routed_experts
    router_read = (
        layers.total * p.routed_experts * p.hidden * p.bf16_bytes
        + (layers.total - layers.hash) * p.routed_experts * p.fp32_bytes
        + rows * layers.total * p.hidden * p.bf16_bytes
        + rows * layers.hash * p.activated_experts * p.int32_bytes
    )
    router_write = rows * layers.total * p.routed_experts * p.fp32_bytes
    items.append(
        Item(
            "MoE",
            "Router score projection",
            "All active layers",
            "2*rows*layers*hidden*routed_experts",
            f"=2*{rows_name}*TotalLayers*HiddenSize*RoutedExperts",
            router_global,
            f"=2*{rows_name}*TotalLayers*HiddenSize*RoutedExperts",
            router_global,
            (
                f"=TotalLayers*RoutedExperts*HiddenSize*BF16Bytes+"
                "(TotalLayers-ActiveHashLayers)*RoutedExperts*FP32Bytes+"
                f"{rows_name}*TotalLayers*HiddenSize*BF16Bytes+"
                f"{rows_name}*ActiveHashLayers*ActivatedExperts*INT32Bytes"
            ),
            router_read,
            f"={rows_name}*TotalLayers*RoutedExperts*FP32Bytes",
            router_write,
            "=0",
            0,
            "Major",
            "The source computes gate scores even in hash-routed layers.",
        )
    )

    one_expert_flops = 6 * p.hidden * p.expert_inter
    routed_global = rows * layers.total * p.activated_experts * one_expert_flops
    routed_rank = routed_global / p.tp
    routed_param_read = (
        layers.total
        * (p.routed_experts / p.tp)
        * helpers["expert_probability"]
        * weights["expert_one"]
    )
    routed_activation_read = (
        rows
        * layers.total
        * (p.activated_experts / p.tp)
        * p.hidden
        * p.bf16_bytes
    )
    routed_write = routed_activation_read
    items.append(
        Item(
            "MoE",
            "Top-K routed experts",
            "All active layers",
            "rows*layers*top_k*(W1+W3+W2), MAC=2 FLOPs",
            (
                f"={rows_name}*TotalLayers*ActivatedExperts*6*"
                "HiddenSize*ExpertInter"
            ),
            routed_global,
            (
                f"={rows_name}*TotalLayers*(ActivatedExperts/TPSize)*6*"
                "HiddenSize*ExpertInter"
            ),
            routed_rank,
            (
                f"=TotalLayers*LocalExperts*{prefix}_ExpertActiveProbability*"
                f"{xl_expert_bytes()}+{rows_name}*TotalLayers*"
                "(ActivatedExperts/TPSize)*HiddenSize*BF16Bytes"
            ),
            routed_param_read + routed_activation_read,
            (
                f"={rows_name}*TotalLayers*(ActivatedExperts/TPSize)*"
                "HiddenSize*BF16Bytes"
            ),
            routed_write,
            "=0",
            0,
            "Major",
            "Balanced expected assignments per rank; parameter read uses expert activation probability.",
        )
    )

    shared_global = (
        rows
        * layers.total
        * p.shared_experts
        * one_expert_flops
    )
    shared_weight_per_layer = (
        2 * fp8_matrix_bytes(p.hidden, p.expert_inter, p)
        + fp8_matrix_bytes(p.expert_inter, p.hidden, p)
    )
    shared_read = (
        layers.total * p.shared_experts * shared_weight_per_layer
        + rows
        * layers.total
        * p.shared_experts
        * p.hidden
        * p.bf16_bytes
    )
    shared_write = (
        rows
        * layers.total
        * p.shared_experts
        * p.hidden
        * p.bf16_bytes
    )
    shared_weight_xl = (
        f"(2*{xl_fp8('HiddenSize', 'ExpertInter')}+"
        f"{xl_fp8('ExpertInter', 'HiddenSize')})"
    )
    items.append(
        Item(
            "MoE",
            "Shared expert",
            "All active layers",
            "rows*layers*shared_experts*(W1+W3+W2), MAC=2 FLOPs",
            (
                f"={rows_name}*TotalLayers*SharedExperts*6*"
                "HiddenSize*ExpertInter"
            ),
            shared_global,
            (
                f"={rows_name}*TotalLayers*SharedExperts*6*"
                "HiddenSize*ExpertInter"
            ),
            shared_global,
            (
                f"=TotalLayers*SharedExperts*{shared_weight_xl}+"
                f"{rows_name}*TotalLayers*SharedExperts*HiddenSize*BF16Bytes"
            ),
            shared_read,
            (
                f"={rows_name}*TotalLayers*SharedExperts*HiddenSize*BF16Bytes"
            ),
            shared_write,
            "=0",
            0,
            "Major",
            "Shared experts are replicated and computed independently on every TP rank.",
        )
    )

    mix_hc = (2 + p.hc_slots) * p.hc_slots
    hc_projection = 2 * rows * p.hc_slots * p.hidden * mix_hc
    hc_per_block = 2 * (
        hc_projection
        + rows * (2 * p.hc_slots * p.hidden + 1)
        + rows * p.hidden * (2 * p.hc_slots - 1)
        + rows * p.hc_slots * p.hidden * (2 * p.hc_slots + 1)
    )
    hc_global = layers.total * hc_per_block
    hc_read = layers.total * weights["hc_rank"] / max(layers.total, 1) + rows * layers.total * 4 * p.hc_slots * p.hidden * p.bf16_bytes
    hc_write = rows * layers.total * 4 * p.hc_slots * p.hidden * p.bf16_bytes
    items.append(
        Item(
            "Other",
            "Hyper-Connections and block norms",
            "All active layers",
            "Two HC pre/post paths per block; Sinkhorn special math excluded",
            (
                f"=TotalLayers*2*(2*{rows_name}*HCSlots*HiddenSize*"
                "((2+HCSlots)*HCSlots)+"
                f"{rows_name}*(2*HCSlots*HiddenSize+1)+"
                f"{rows_name}*HiddenSize*(2*HCSlots-1)+"
                f"{rows_name}*HCSlots*HiddenSize*(2*HCSlots+1))"
            ),
            hc_global,
            (
                f"=TotalLayers*2*(2*{rows_name}*HCSlots*HiddenSize*"
                "((2+HCSlots)*HCSlots)+"
                f"{rows_name}*(2*HCSlots*HiddenSize+1)+"
                f"{rows_name}*HiddenSize*(2*HCSlots-1)+"
                f"{rows_name}*HCSlots*HiddenSize*(2*HCSlots+1))"
            ),
            hc_global,
            (
                "=TotalLayers*2*((2+HCSlots)*HCSlots*HCSlots*HiddenSize*FP32Bytes+"
                "((2+HCSlots)*HCSlots+3)*FP32Bytes)+"
                f"{rows_name}*TotalLayers*4*HCSlots*HiddenSize*BF16Bytes"
            ),
            hc_read,
            f"={rows_name}*TotalLayers*4*HCSlots*HiddenSize*BF16Bytes",
            hc_write,
            "=0",
            0,
            "Auxiliary",
            "Approximate HC arithmetic; sigmoid/exp are tracked qualitatively.",
        )
    )

    lm_global = 2 * (p.prefill_batch if mode == "prefill" else rows) * p.hidden * p.vocab
    lm_rank = lm_global / p.tp
    lm_read = weights["lm_head"] + (p.prefill_batch if mode == "prefill" else rows) * p.hidden * p.fp32_bytes
    lm_write = (p.prefill_batch if mode == "prefill" else rows) * (p.vocab / p.tp) * p.fp32_bytes
    items.append(
        Item(
            "Other",
            "LM head",
            "Tail",
            "2*batch_or_decode_rows*hidden*vocab; prefill uses last token only",
            (
                "=2*PrefillBatch*HiddenSize*VocabSize"
                if mode == "prefill"
                else f"=2*{rows_name}*HiddenSize*VocabSize"
            ),
            lm_global,
            (
                "=2*PrefillBatch*HiddenSize*LocalVocab"
                if mode == "prefill"
                else f"=2*{rows_name}*HiddenSize*LocalVocab"
            ),
            lm_rank,
            (
                "=LocalVocab*HiddenSize*FP32Bytes+PrefillBatch*HiddenSize*FP32Bytes"
                if mode == "prefill"
                else f"=LocalVocab*HiddenSize*FP32Bytes+{rows_name}*HiddenSize*FP32Bytes"
            ),
            lm_read,
            (
                "=PrefillBatch*LocalVocab*FP32Bytes"
                if mode == "prefill"
                else f"={rows_name}*LocalVocab*FP32Bytes"
            ),
            lm_write,
            "=0",
            0,
            "Major",
            "The inference path computes logits only from the final hidden token.",
        )
    )

    payload_rows = rows
    allreduce_factor = 4 * (p.tp - 1) / p.tp
    allgather_factor = 2 * (p.tp - 1)
    embedding_network = (
        allreduce_factor
        * payload_rows
        * p.hidden
        * p.bf16_bytes
    )
    attention_network = (
        allreduce_factor
        * payload_rows
        * p.hidden
        * p.fp32_bytes
        * layers.total
    )
    moe_network = attention_network
    lm_network = (
        allgather_factor
        * (p.prefill_batch if mode == "prefill" else rows)
        * (p.vocab / p.tp)
        * p.fp32_bytes
    )
    items.append(
        Item(
            "Communication",
            "TP collectives",
            "Embedding + each block + LM head",
            "Ring AllReduce/AllGather send+receive bytes per rank",
            "=0",
            0,
            "=0",
            0,
            "=0",
            0,
            "=0",
            0,
            (
                f"=4*(TPSize-1)/TPSize*{rows_name}*HiddenSize*BF16Bytes+"
                f"2*4*(TPSize-1)/TPSize*{rows_name}*HiddenSize*FP32Bytes*TotalLayers+"
                f"2*(TPSize-1)*{('PrefillBatch' if mode == 'prefill' else rows_name)}*"
                "LocalVocab*FP32Bytes"
            ),
            embedding_network + attention_network + moe_network + lm_network,
            "Auxiliary",
            "Zero at TP=1. This is transfer volume, not HBM traffic.",
        )
    )
    return items, helpers


def cache_values(p: Inputs, layers: LayerCounts, context: int, batch: int) -> dict[str, float]:
    raw_slots = min(context, p.window)
    main = batch * p.head_dim * p.bf16_bytes * (
        layers.window * raw_slots
        + layers.short * (raw_slots + context // p.short_ratio)
        + layers.long * (raw_slots + context // p.long_ratio)
    )
    indexer = (
        batch
        * layers.short
        * (context // p.short_ratio)
        * p.index_dim
        * p.bf16_bytes
    )
    main_state = batch * p.fp32_bytes * (
        layers.short * 2 * (2 * p.short_ratio) * (2 * p.head_dim)
        + layers.long * 2 * p.long_ratio * p.head_dim
    )
    index_state = (
        batch
        * layers.short
        * 2
        * (2 * p.short_ratio)
        * (2 * p.index_dim)
        * p.fp32_bytes
    )
    return {
        "main": main,
        "indexer": indexer,
        "states": main_state + index_state,
        "total": main + indexer + main_state + index_state,
    }


class CalculatorWriter:
    def __init__(self, output: Path, p: Inputs, ratios: list[int]) -> None:
        self.output = output
        self.p = p
        self.ratios = ratios
        self.layers = count_layers(ratios, p)
        self.workbook = xlsxwriter.Workbook(output)
        self.workbook.set_calc_mode("auto")
        self.workbook.set_properties(
            {
                "title": "DeepSeek V4 Flash Architecture Calculator",
                "subject": "Formula-driven TP, FLOPs, HBM, and memory analysis",
                "author": "DeepSeek V4 Flash repository calculator",
            }
        )
        self.formats = self._formats()
        self.parameter_rows: dict[str, int] = {}
        self.scenario_ranges: dict[str, tuple[int, int]] = {}
        self.scenario_summaries: dict[str, dict[str, float]] = {}
        self.memory_rows: dict[str, int] = {}

    def _formats(self) -> dict[str, Any]:
        wb = self.workbook
        return {
            "title": wb.add_format(
                {"bold": True, "font_size": 18, "font_color": "#17324D"}
            ),
            "section": wb.add_format(
                {
                    "bold": True,
                    "font_color": "#FFFFFF",
                    "bg_color": "#21618C",
                    "border": 1,
                }
            ),
            "header": wb.add_format(
                {
                    "bold": True,
                    "font_color": "#FFFFFF",
                    "bg_color": "#34495E",
                    "border": 1,
                    "text_wrap": True,
                    "align": "center",
                    "valign": "vcenter",
                }
            ),
            "input": wb.add_format(
                {"bg_color": "#D9EAF7", "border": 1, "num_format": "0.########"}
            ),
            "derived": wb.add_format(
                {"bg_color": "#E8F5E9", "border": 1, "num_format": "0.########"}
            ),
            "text": wb.add_format({"border": 1, "text_wrap": True, "valign": "top"}),
            "number": wb.add_format({"border": 1, "num_format": "#,##0.000"}),
            "integer": wb.add_format({"border": 1, "num_format": "#,##0"}),
            "scientific": wb.add_format({"border": 1, "num_format": "0.000E+00"}),
            "bytes": wb.add_format({"border": 1, "num_format": "#,##0"}),
            "ratio": wb.add_format({"border": 1, "num_format": "0.000"}),
            "percent": wb.add_format({"border": 1, "num_format": "0.0%"}),
            "ok": wb.add_format(
                {"bg_color": "#C6EFCE", "font_color": "#006100", "border": 1}
            ),
            "note": wb.add_format(
                {"font_color": "#555555", "italic": True, "text_wrap": True}
            ),
            "total": wb.add_format(
                {
                    "bold": True,
                    "bg_color": "#D6EAF8",
                    "border": 1,
                    "num_format": "#,##0.000",
                }
            ),
        }

    def _define_cell_name(self, name: str, sheet: str, row: int, col: int) -> None:
        column = xl_col_to_name(col)
        self.workbook.define_name(name, f"='{sheet}'!${column}${row + 1}")

    def write_parameters(self) -> None:
        ws = self.workbook.add_worksheet("Parameters")
        ws.hide_gridlines(2)
        ws.freeze_panes(3, 0)
        ws.write(0, 0, "DeepSeek V4 Flash - Editable Architecture Parameters", self.formats["title"])
        ws.write(1, 0, "Blue cells are editable inputs. Green cells are derived Excel formulas.", self.formats["note"])
        headers = ["Category", "Parameter", "Excel name", "Value", "Unit", "Description", "Validation"]
        for col, header in enumerate(headers):
            ws.write(2, col, header, self.formats["header"])

        records: list[tuple[str, str, str, float, str, str]] = [
            ("Parallel", "Tensor parallel size", "TPSize", self.p.tp, "ranks", "Number of tensor-parallel ranks."),
            ("Scenario", "Prefill batch size", "PrefillBatch", self.p.prefill_batch, "sequences", "Editable prefill batch size."),
            ("Scenario", "Prefill sequence length", "PrefillSequence", self.p.prefill_sequence, "tokens", "Prompt tokens processed in one prefill."),
            ("Scenario", "Decode batch size", "DecodeBatch", self.p.decode_batch, "sequences", "Editable decode batch size."),
            ("Scenario", "Decode tokens per step", "DecodeTokens", self.p.decode_tokens, "tokens", "Usually 1; formulas model each token at DecodeContext."),
            ("Scenario", "Decode context", "DecodeContext", self.p.decode_context, "tokens", "Effective context visible during decode."),
            ("Scenario", "Maximum allocated context", "MaxContext", self.p.max_context, "tokens", "KV cache preallocation capacity."),
            ("Model", "Hidden size", "HiddenSize", self.p.hidden, "elements", "Transformer residual width."),
            ("Model", "Vocabulary size", "VocabSize", self.p.vocab, "tokens", "Embedding and LM-head vocabulary."),
            ("Attention", "Query heads", "NumHeads", self.p.heads, "heads", "Global Q-head count."),
            ("Attention", "Head dimension", "HeadDim", self.p.head_dim, "elements", "Shared latent KV width and per-Q-head width."),
            ("Attention", "RoPE dimension", "RopeDim", self.p.rope_dim, "elements", "Rotary dimensions inside HeadDim."),
            ("Attention", "Q LoRA rank", "QLoraRank", self.p.q_rank, "elements", "Low-rank intermediate width for Q."),
            ("Attention", "Output groups", "OGroups", self.p.o_groups, "groups", "Grouped low-rank output projection groups."),
            ("Attention", "Output LoRA rank", "OLoraRank", self.p.o_rank, "elements", "Per-group output low-rank width."),
            ("MoE", "Routed experts", "RoutedExperts", self.p.routed_experts, "experts", "Global routed-expert count per block."),
            ("MoE", "Activated experts", "ActivatedExperts", self.p.activated_experts, "experts/token", "Top-K routed experts per token."),
            ("MoE", "Shared experts", "SharedExperts", self.p.shared_experts, "experts", "Replicated shared experts per block."),
            ("MoE", "Expert intermediate size", "ExpertInter", self.p.expert_inter, "elements", "SwiGLU expert intermediate width."),
            ("Cache", "Raw sliding window", "WindowSize", self.p.window, "tokens", "Raw KV positions retained per layer."),
            ("Cache", "Short compression ratio", "ShortRatio", self.p.short_ratio, "tokens/entry", "Default ratio-4 mode."),
            ("Cache", "Long compression ratio", "LongRatio", self.p.long_ratio, "tokens/entry", "Default ratio-128 mode."),
            ("Indexer", "Indexer heads", "IndexHeads", self.p.index_heads, "heads", "Global ratio-4 indexer heads."),
            ("Indexer", "Indexer head dimension", "IndexHeadDim", self.p.index_dim, "elements", "Indexer Q/K dimension."),
            ("Indexer", "Indexer Top-K", "IndexTopK", self.p.index_topk, "entries", "Compressed entries retained by ratio-4 attention."),
            ("HC", "HC slots", "HCSlots", self.p.hc_slots, "streams", "Hyper-Connection residual streams."),
            ("HC", "Sinkhorn iterations", "HCIters", self.p.hc_iters, "iterations", "HC combination normalization iterations."),
            ("Routing", "Hash-routed layers", "HashLayers", self.p.hash_layers, "layers", "First active layers using token-ID routing."),
            ("DType", "BF16 bytes", "BF16Bytes", self.p.bf16_bytes, "bytes", "Storage bytes per BF16 element."),
            ("DType", "FP32 bytes", "FP32Bytes", self.p.fp32_bytes, "bytes", "Storage bytes per FP32 element."),
            ("DType", "FP8 bytes", "FP8Bytes", self.p.fp8_bytes, "bytes", "Storage bytes per FP8 element."),
            ("DType", "FP4 bytes", "FP4Bytes", self.p.fp4_bytes, "bytes", "Logical packed bytes per FP4 element."),
            ("DType", "Scale bytes", "ScaleBytes", self.p.scale_bytes, "bytes", "E8M0 scale storage."),
            ("DType", "INT32 bytes", "INT32Bytes", self.p.int32_bytes, "bytes", "Storage bytes per INT32 element."),
            ("DType", "INT64 bytes", "INT64Bytes", self.p.int64_bytes, "bytes", "Storage bytes per INT64 element."),
            ("Quantization", "FP8 scale block", "FP8Block", self.p.fp8_block, "elements", "2D FP8 weight scale block."),
            ("Quantization", "FP4 scale block", "FP4Block", self.p.fp4_block, "elements", "FP4 expert scale block along K."),
            ("Kernel", "Minimum sparse-attention heads", "KernelMinHeads", self.p.kernel_min_heads, "heads", "Kernel pads local heads below this value."),
            ("Hardware", "Peak compute", "PeakTFLOPs", self.p.peak_tflops, "TFLOP/s", "Editable illustrative hardware peak."),
            ("Hardware", "HBM bandwidth", "HBMBandwidthGBps", self.p.hbm_gbps, "GB/s", "Editable sustained/target HBM bandwidth."),
            ("Hardware", "Interconnect bandwidth", "InterconnectGBps", self.p.interconnect_gbps, "GB/s", "Editable effective bidirectional bandwidth."),
            ("Hardware", "Prefill target latency", "PrefillTargetMs", self.p.prefill_target_ms, "ms", "Used to derive required throughput."),
            ("Hardware", "Decode target latency", "DecodeTargetMs", self.p.decode_target_ms, "ms", "Used to derive required throughput."),
        ]
        row = 3
        for category, label, name, value, unit, description in records:
            ws.write(row, 0, category, self.formats["text"])
            ws.write(row, 1, label, self.formats["text"])
            ws.write(row, 2, name, self.formats["text"])
            ws.write_number(row, 3, value, self.formats["input"])
            ws.write(row, 4, unit, self.formats["text"])
            ws.write(row, 5, description, self.formats["text"])
            self.parameter_rows[name] = row
            self._define_cell_name(name, "Parameters", row, 3)
            row += 1

        derived = [
            ("Derived", "Active layers", "TotalLayers", "=SUM(Layer_Config!$B$4:$B$67)", self.layers.total, "layers", "Count of active Layer_Config rows."),
            ("Derived", "Window-only layers", "WindowLayers", '=COUNTIFS(Layer_Config!$B$4:$B$67,1,Layer_Config!$C$4:$C$67,"window")', self.layers.window, "layers", "Active window-only rows."),
            ("Derived", "Short-compression layers", "ShortLayers", '=COUNTIFS(Layer_Config!$B$4:$B$67,1,Layer_Config!$C$4:$C$67,"short")', self.layers.short, "layers", "Active ratio-4/short rows."),
            ("Derived", "Long-compression layers", "LongLayers", '=COUNTIFS(Layer_Config!$B$4:$B$67,1,Layer_Config!$C$4:$C$67,"long")', self.layers.long, "layers", "Active ratio-128/long rows."),
            ("Derived", "Active hash layers", "ActiveHashLayers", '=COUNTIFS(Layer_Config!$B$4:$B$67,1,Layer_Config!$A$4:$A$67,"<"&HashLayers)', self.layers.hash, "layers", "Active layer IDs below HashLayers."),
            ("Derived", "Local Q heads", "LocalHeads", "=NumHeads/TPSize", self.p.heads / self.p.tp, "heads/rank", "Q heads assigned to one rank."),
            ("Derived", "Kernel local heads", "KernelLocalHeads", "=MAX(LocalHeads,KernelMinHeads)", max(self.p.heads / self.p.tp, self.p.kernel_min_heads), "heads/rank", "Physical sparse-kernel heads after padding."),
            ("Derived", "Local index heads", "LocalIndexHeads", "=IndexHeads/TPSize", self.p.index_heads / self.p.tp, "heads/rank", "Indexer heads assigned to one rank."),
            ("Derived", "Local experts", "LocalExperts", "=RoutedExperts/TPSize", self.p.routed_experts / self.p.tp, "experts/rank", "Routed experts resident on one rank."),
            ("Derived", "Local output groups", "LocalOGroups", "=OGroups/TPSize", self.p.o_groups / self.p.tp, "groups/rank", "Output groups assigned to one rank."),
            ("Derived", "Local vocabulary", "LocalVocab", "=VocabSize/TPSize", self.p.vocab / self.p.tp, "tokens/rank", "Vocabulary rows assigned to one rank."),
        ]
        for category, label, name, formula, value, unit, description in derived:
            ws.write(row, 0, category, self.formats["text"])
            ws.write(row, 1, label, self.formats["text"])
            ws.write(row, 2, name, self.formats["text"])
            ws.write_formula(row, 3, formula, self.formats["derived"], value)
            ws.write(row, 4, unit, self.formats["text"])
            ws.write(row, 5, description, self.formats["text"])
            self.parameter_rows[name] = row
            self._define_cell_name(name, "Parameters", row, 3)
            row += 1

        checks = [
            ("Q heads divisible by TP", "=IF(MOD(NumHeads,TPSize)=0,\"OK\",\"ERROR\")", self.p.heads % self.p.tp == 0),
            ("Experts divisible by TP", "=IF(MOD(RoutedExperts,TPSize)=0,\"OK\",\"ERROR\")", self.p.routed_experts % self.p.tp == 0),
            ("Output groups divisible by TP", "=IF(MOD(OGroups,TPSize)=0,\"OK\",\"ERROR\")", self.p.o_groups % self.p.tp == 0),
            ("Indexer heads divisible by TP", "=IF(MOD(IndexHeads,TPSize)=0,\"OK\",\"ERROR\")", self.p.index_heads % self.p.tp == 0),
            ("Top-K is valid", "=IF(ActivatedExperts<=RoutedExperts,\"OK\",\"ERROR\")", self.p.activated_experts <= self.p.routed_experts),
            ("RoPE fits in head", "=IF(RopeDim<=HeadDim,\"OK\",\"ERROR\")", self.p.rope_dim <= self.p.head_dim),
        ]
        row += 1
        ws.write(row, 0, "Architecture validation", self.formats["section"])
        ws.merge_range(row, 0, row, 6, "Architecture validation", self.formats["section"])
        row += 1
        for label, formula, valid in checks:
            ws.write(row, 0, label, self.formats["text"])
            ws.write_formula(row, 6, formula, self.formats["ok"], "OK" if valid else "ERROR")
            row += 1
        ws.conditional_format(3, 6, row, 6, {"type": "text", "criteria": "containing", "value": "ERROR", "format": self.workbook.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})})
        ws.autofilter(2, 0, 2 + len(records) + len(derived), 6)
        ws.set_column("A:A", 16)
        ws.set_column("B:B", 29)
        ws.set_column("C:C", 25)
        ws.set_column("D:D", 18)
        ws.set_column("E:E", 16)
        ws.set_column("F:F", 58)
        ws.set_column("G:G", 16)

    def write_layer_config(self) -> None:
        ws = self.workbook.add_worksheet("Layer_Config")
        ws.hide_gridlines(2)
        ws.freeze_panes(3, 0)
        ws.write(0, 0, "Per-Layer Attention and Routing Configuration", self.formats["title"])
        ws.write(1, 0, "Edit Active and Mode. Compression ratio and routing behavior update automatically.", self.formats["note"])
        headers = ["Layer ID", "Active", "Mode", "Compression ratio", "Attention behavior", "Routing behavior"]
        for col, header in enumerate(headers):
            ws.write(2, col, header, self.formats["header"])
        for index in range(MAX_LAYER_ROWS):
            row = 3 + index
            active = 1 if index < len(self.ratios) else 0
            ratio = self.ratios[index] if index < len(self.ratios) else 0
            mode = "window" if ratio == 0 else "short" if ratio == self.p.short_ratio else "long"
            ws.write_number(row, 0, index, self.formats["integer"])
            ws.write_number(row, 1, active, self.formats["input"])
            ws.write(row, 2, mode, self.formats["input"])
            ratio_formula = f'=IF(C{row + 1}="window",0,IF(C{row + 1}="short",ShortRatio,LongRatio))'
            ws.write_formula(row, 3, ratio_formula, self.formats["derived"], ratio if active else 0)
            behavior_formula = f'=IF(B{row + 1}=0,"inactive",IF(C{row + 1}="window","raw sliding window",IF(C{row + 1}="short","compressed KV + Indexer Top-K","deterministic compressed KV")))'
            behavior = "inactive" if not active else "raw sliding window" if mode == "window" else "compressed KV + Indexer Top-K" if mode == "short" else "deterministic compressed KV"
            ws.write_formula(row, 4, behavior_formula, self.formats["text"], behavior)
            routing_formula = f'=IF(B{row + 1}=0,"inactive",IF(A{row + 1}<HashLayers,"hash","score"))'
            routing = "inactive" if not active else "hash" if index < self.p.hash_layers else "score"
            ws.write_formula(row, 5, routing_formula, self.formats["text"], routing)
        ws.data_validation(3, 1, 3 + MAX_LAYER_ROWS - 1, 1, {"validate": "list", "source": [0, 1]})
        ws.data_validation(3, 2, 3 + MAX_LAYER_ROWS - 1, 2, {"validate": "list", "source": ["window", "short", "long"]})
        ws.autofilter(2, 0, 2 + MAX_LAYER_ROWS, 5)
        ws.set_column("A:B", 12)
        ws.set_column("C:D", 20)
        ws.set_column("E:F", 34)

    def _write_helper(self, ws: Any, row: int, label: str, name: str, formula: str, value: float, unit: str) -> None:
        ws.write(row, 0, label, self.formats["text"])
        ws.write(row, 1, formula, self.formats["text"])
        ws.write_formula(row, 2, formula, self.formats["derived"], value)
        ws.write(row, 3, unit, self.formats["text"])
        self._define_cell_name(name, ws.name, row, 2)

    def write_scenario(self, mode: str) -> None:
        is_prefill = mode == "prefill"
        sheet = "Prefill_8K" if is_prefill else "Decode_1M"
        prefix = "PF" if is_prefill else "DC"
        items, helpers = scenario_items(self.p, self.layers, mode)
        ws = self.workbook.add_worksheet(sheet)
        ws.hide_gridlines(2)
        ws.write(0, 0, f"{'8K Prefill' if is_prefill else '1M-context Decode'} - Formula Model", self.formats["title"])
        ws.write(1, 0, "All numeric result cells below are Excel formulas. Change blue cells in Parameters or Layer_Config.", self.formats["note"])
        ws.write_row(2, 0, ["Helper metric", "Excel formula", "Value", "Unit"], self.formats["header"])
        helper_rows = []
        if is_prefill:
            raw_formula = "=IF(PrefillSequence<=WindowSize,PrefillSequence*(PrefillSequence+1)/2,WindowSize*(WindowSize+1)/2+(PrefillSequence-WindowSize)*WindowSize)"
            short_formula = "=IF(PrefillSequence<IndexTopK*ShortRatio,ShortRatio*INT(PrefillSequence/ShortRatio)*(INT(PrefillSequence/ShortRatio)-1)/2+INT(PrefillSequence/ShortRatio)*(MOD(PrefillSequence,ShortRatio)+1),ShortRatio*IndexTopK*(IndexTopK-1)/2+(PrefillSequence-IndexTopK*ShortRatio+1)*IndexTopK)"
            long_formula = "=LongRatio*INT(PrefillSequence/LongRatio)*(INT(PrefillSequence/LongRatio)-1)/2+INT(PrefillSequence/LongRatio)*(MOD(PrefillSequence,LongRatio)+1)"
            index_formula = "=ShortRatio*INT(PrefillSequence/ShortRatio)*(INT(PrefillSequence/ShortRatio)-1)/2+INT(PrefillSequence/ShortRatio)*(MOD(PrefillSequence,ShortRatio)+1)"
            helper_rows = [
                ("Token rows", f"{prefix}_Rows", "=PrefillBatch*PrefillSequence", helpers["rows"], "token rows"),
                ("Causal raw-window pairs", f"{prefix}_RawPairs", raw_formula, helpers["raw_pairs"], "pairs/sequence"),
                ("Capped short-compressed pairs", f"{prefix}_ShortPairs", short_formula, helpers["short_pairs"], "pairs/sequence"),
                ("Long-compressed pairs", f"{prefix}_LongPairs", long_formula, helpers["long_pairs"], "pairs/sequence"),
                ("Indexer scan pairs", f"{prefix}_IndexPairs", index_formula, helpers["index_pairs"], "pairs/sequence"),
                ("Expert activation probability", f"{prefix}_ExpertActiveProbability", "=1-(1-ActivatedExperts/RoutedExperts)^PF_Rows", helpers["expert_probability"], "probability"),
            ]
        else:
            helper_rows = [
                ("Token rows", f"{prefix}_Rows", "=DecodeBatch*DecodeTokens", helpers["rows"], "token rows"),
                ("Raw-window candidates", f"{prefix}_RawPairs", "=DecodeTokens*MIN(WindowSize,DecodeContext)", helpers["raw_pairs"], "pairs/sequence"),
                ("Short-compressed Top-K candidates", f"{prefix}_ShortPairs", "=DecodeTokens*MIN(IndexTopK,INT(DecodeContext/ShortRatio))", helpers["short_pairs"], "pairs/sequence"),
                ("Long-compressed candidates", f"{prefix}_LongPairs", "=DecodeTokens*INT(DecodeContext/LongRatio)", helpers["long_pairs"], "pairs/sequence"),
                ("Indexer scan candidates", f"{prefix}_IndexPairs", "=DecodeTokens*INT(DecodeContext/ShortRatio)", helpers["index_pairs"], "pairs/sequence"),
                ("Expert activation probability", f"{prefix}_ExpertActiveProbability", "=1-(1-ActivatedExperts/RoutedExperts)^DC_Rows", helpers["expert_probability"], "probability"),
            ]
        row = 3
        for label, name, formula, value, unit in helper_rows:
            self._write_helper(ws, row, label, name, formula, value, unit)
            row += 1

        summary_start = row + 1
        ws.merge_range(summary_start, 0, summary_start, 3, "Scenario summary", self.formats["section"])
        summary_headers_row = summary_start + 1
        ws.write_row(summary_headers_row, 0, ["Metric", "Excel formula", "Value", "Unit"], self.formats["header"])
        detail_start = summary_headers_row + 16
        detail_first_excel = detail_start + 2
        detail_last_excel = detail_first_excel + len(items) - 1

        category_sums = {
            category: {
                "global": sum(item.global_flops for item in items if item.category == category),
                "rank": sum(item.rank_flops for item in items if item.category == category),
                "read": sum(item.read_bytes for item in items if item.category == category),
                "write": sum(item.write_bytes for item in items if item.category == category),
            }
            for category in ("Attention", "MoE", "Other", "Communication")
        }
        attention_major = sum(item.rank_flops for item in items if item.category == "Attention" and item.accounting == "Major")
        moe_major = sum(item.rank_flops for item in items if item.category == "MoE" and item.accounting == "Major")
        total_rank = sum(item.rank_flops for item in items)
        total_global = sum(item.global_flops for item in items)
        total_read = sum(item.read_bytes for item in items)
        total_write = sum(item.write_bytes for item in items)
        total_network = sum(item.network_bytes for item in items)
        target_ms = self.p.prefill_target_ms if is_prefill else self.p.decode_target_ms
        summary_specs = [
            ("Attention major FLOPs/rank", f'=SUMIFS($F${detail_first_excel}:$F${detail_last_excel},$A${detail_first_excel}:$A${detail_last_excel},"Attention",$L${detail_first_excel}:$L${detail_last_excel},"Major")', attention_major, "FLOPs"),
            ("MoE major FLOPs/rank", f'=SUMIFS($F${detail_first_excel}:$F${detail_last_excel},$A${detail_first_excel}:$A${detail_last_excel},"MoE",$L${detail_first_excel}:$L${detail_last_excel},"Major")', moe_major, "FLOPs"),
            ("Total FLOPs/rank", f"=SUM($F${detail_first_excel}:$F${detail_last_excel})", total_rank, "FLOPs"),
            ("Total aggregate FLOPs", f"=SUM($E${detail_first_excel}:$E${detail_last_excel})", total_global, "FLOPs"),
            ("HBM read/rank", f"=SUM($G${detail_first_excel}:$G${detail_last_excel})", total_read, "bytes"),
            ("HBM write/rank", f"=SUM($H${detail_first_excel}:$H${detail_last_excel})", total_write, "bytes"),
            ("HBM total/rank", f"=SUM($I${detail_first_excel}:$I${detail_last_excel})", total_read + total_write, "bytes"),
            ("Interconnect/rank", f"=SUM($K${detail_first_excel}:$K${detail_last_excel})", total_network, "bytes"),
            ("Arithmetic intensity", f"=C{summary_headers_row + 4}/C{summary_headers_row + 8}", total_rank / (total_read + total_write), "FLOPs/byte"),
            ("Required compute at target", f"=C{summary_headers_row + 4}/({'PrefillTargetMs' if is_prefill else 'DecodeTargetMs'}/1000)/1E12", total_rank / (target_ms / 1000) / 1e12, "TFLOP/s"),
            ("Required HBM at target", f"=C{summary_headers_row + 8}/({'PrefillTargetMs' if is_prefill else 'DecodeTargetMs'}/1000)/1E9", (total_read + total_write) / (target_ms / 1000) / 1e9, "GB/s"),
            ("Compute lower bound", f"=C{summary_headers_row + 4}/(PeakTFLOPs*1E12)*1000", total_rank / (self.p.peak_tflops * 1e12) * 1000, "ms"),
            ("HBM lower bound", f"=C{summary_headers_row + 8}/(HBMBandwidthGBps*1E9)*1000", (total_read + total_write) / (self.p.hbm_gbps * 1e9) * 1000, "ms"),
            ("Interconnect lower bound", f"=IF(TPSize=1,0,C{summary_headers_row + 9}/(InterconnectGBps*1E9)*1000)", 0 if self.p.tp == 1 else total_network / (self.p.interconnect_gbps * 1e9) * 1000, "ms"),
        ]
        summary_values: dict[str, float] = {}
        for offset, (label, formula, value, unit) in enumerate(summary_specs):
            target_row = summary_headers_row + 1 + offset
            ws.write(target_row, 0, label, self.formats["text"])
            ws.write(target_row, 1, formula, self.formats["text"])
            ws.write_formula(target_row, 2, formula, self.formats["derived"], value)
            ws.write(target_row, 3, unit, self.formats["text"])
            summary_values[label] = value
        self.scenario_summaries[prefix] = summary_values

        ws.merge_range(detail_start - 1, 0, detail_start - 1, 12, "Detailed calculation rows", self.formats["section"])
        headers = [
            "Category",
            "Item",
            "Layer scope",
            "Readable formula",
            "Global FLOPs",
            "Per-rank FLOPs",
            "HBM read/rank (B)",
            "HBM write/rank (B)",
            "HBM total/rank (B)",
            "FLOPs/byte",
            "Interconnect/rank (B)",
            "Accounting",
            "Notes",
        ]
        ws.write_row(detail_start, 0, headers, self.formats["header"])
        for offset, item in enumerate(items):
            target_row = detail_start + 1 + offset
            ws.write(target_row, 0, item.category, self.formats["text"])
            ws.write(target_row, 1, item.name, self.formats["text"])
            ws.write(target_row, 2, item.layer_scope, self.formats["text"])
            ws.write(target_row, 3, item.readable_formula, self.formats["text"])
            ws.write_formula(target_row, 4, item.global_flops_formula, self.formats["scientific"], item.global_flops)
            ws.write_formula(target_row, 5, item.rank_flops_formula, self.formats["scientific"], item.rank_flops)
            ws.write_formula(target_row, 6, item.read_formula, self.formats["bytes"], item.read_bytes)
            ws.write_formula(target_row, 7, item.write_formula, self.formats["bytes"], item.write_bytes)
            ws.write_formula(target_row, 8, f"=G{target_row + 1}+H{target_row + 1}", self.formats["bytes"], item.read_bytes + item.write_bytes)
            intensity = item.rank_flops / (item.read_bytes + item.write_bytes) if item.read_bytes + item.write_bytes else 0
            ws.write_formula(target_row, 9, f"=IF(I{target_row + 1}=0,0,F{target_row + 1}/I{target_row + 1})", self.formats["ratio"], intensity)
            ws.write_formula(target_row, 10, item.network_formula, self.formats["bytes"], item.network_bytes)
            ws.write(target_row, 11, item.accounting, self.formats["text"])
            ws.write(target_row, 12, item.notes, self.formats["text"])
        self.scenario_ranges[prefix] = (detail_start + 1, detail_start + len(items))
        ws.autofilter(detail_start, 0, detail_start + len(items), 12)
        ws.freeze_panes(detail_start + 1, 4)
        ws.set_column("A:A", 16)
        ws.set_column("B:B", 33)
        ws.set_column("C:C", 25)
        ws.set_column("D:D", 58)
        ws.set_column("E:K", 20)
        ws.set_column("L:L", 13)
        ws.set_column("M:M", 62)

    def write_memory(self) -> dict[str, float]:
        ws = self.workbook.add_worksheet("Memory")
        ws.hide_gridlines(2)
        ws.freeze_panes(3, 0)
        ws.write(0, 0, "Per-Rank Parameter and KV Capacity", self.formats["title"])
        ws.write(1, 0, "All capacities are formulas. Runtime allocator overhead and transient kernel workspace are excluded.", self.formats["note"])
        headers = ["Category", "Item", "Readable formula", "Bytes/rank", "GB/rank", "GiB/rank", "Notes"]
        ws.write_row(2, 0, headers, self.formats["header"])
        p = self.p
        layers = self.layers
        weights = weight_values(p, layers)

        core_xl = (
            f"TotalLayers*({xl_fp8('HiddenSize', 'QLoraRank')}+"
            f"{xl_fp8('QLoraRank', 'LocalHeads*HeadDim')}+"
            f"{xl_fp8('HiddenSize', 'HeadDim')}+"
            "LocalHeads*HeadDim*OLoraRank*BF16Bytes+"
            f"{xl_fp8('LocalOGroups*OLoraRank', 'HiddenSize')})"
        )
        main_comp_xl = (
            "ShortLayers*(2*HiddenSize*(2*HeadDim)*FP32Bytes+ShortRatio*2*HeadDim*FP32Bytes+HeadDim*FP32Bytes)+"
            "LongLayers*(2*HiddenSize*HeadDim*FP32Bytes+LongRatio*HeadDim*FP32Bytes+HeadDim*FP32Bytes)"
        )
        index_xl = (
            f"ShortLayers*({xl_fp8('QLoraRank', 'LocalIndexHeads*IndexHeadDim')}+"
            "HiddenSize*LocalIndexHeads*BF16Bytes+2*HiddenSize*(2*IndexHeadDim)*FP32Bytes+"
            "ShortRatio*2*IndexHeadDim*FP32Bytes+IndexHeadDim*FP32Bytes)"
        )
        mix_hc = (2 + p.hc_slots) * p.hc_slots
        parameter_rows = [
            ("Parameters", "Embedding", "VocabSize/TPSize*HiddenSize*BF16Bytes", p.vocab / p.tp * p.hidden * p.bf16_bytes, "Vocabulary sharded."),
            ("Parameters", "LM head", "VocabSize/TPSize*HiddenSize*FP32Bytes", p.vocab / p.tp * p.hidden * p.fp32_bytes, "Independent FP32 inference head."),
            ("Parameters", "Core attention projections", core_xl, layers.total * weights["core_rank_layer"], "Includes FP8 scales; Wo_a is BF16 after conversion."),
            ("Parameters", "Main KV compressors", main_comp_xl, layers.short * weights["short_main_compressor"] + layers.long * weights["long_main_compressor"], "Replicated across TP ranks."),
            ("Parameters", "Ratio-4 Indexers", index_xl, layers.short * (weights["index_core_rank"] + weights["index_compressor"]), "Indexer projections plus its compressor."),
            ("Parameters", "Routed experts", f"TotalLayers*LocalExperts*{xl_expert_bytes()}", weights["routed_rank"], "FP4 packed weights and per-32 scales."),
            ("Parameters", "Shared experts", f"TotalLayers*SharedExperts*(2*{xl_fp8('HiddenSize', 'ExpertInter')}+{xl_fp8('ExpertInter', 'HiddenSize')})", weights["shared_rank"], "Replicated FP8 shared experts."),
            ("Parameters", "Router and hash tables", "TotalLayers*RoutedExperts*HiddenSize*BF16Bytes+(TotalLayers-ActiveHashLayers)*RoutedExperts*FP32Bytes+ActiveHashLayers*VocabSize*ActivatedExperts*INT32Bytes", weights["router_rank"], "Gate weight, score bias, and token-to-expert tables."),
            ("Parameters", "Hyper-Connections", "TotalLayers*2*(((2+HCSlots)*HCSlots)*HCSlots*HiddenSize*FP32Bytes+(((2+HCSlots)*HCSlots)+3)*FP32Bytes)", weights["hc_rank"], "Two HC parameter sets per block."),
            ("Parameters", "Norms and attention sinks", "TotalLayers*(2*HiddenSize+QLoraRank+HeadDim+LocalHeads)*FP32Bytes", weights["norms_rank"], "Block norms, Q/KV norms, local sink."),
            ("Parameters", "Tail HC and final norm", "HCSlots*HCSlots*HiddenSize*FP32Bytes+(HCSlots+1)*FP32Bytes+HiddenSize*FP32Bytes", weights["tail_rank"], "Final HC reduction and RMSNorm."),
        ]

        prefill_cache = cache_values(p, layers, p.prefill_sequence, p.prefill_batch)
        decode_cache = cache_values(p, layers, p.decode_context, p.decode_batch)
        allocated_prefill = cache_values(p, layers, p.max_context, p.prefill_batch)
        allocated_decode = cache_values(p, layers, p.max_context, p.decode_batch)
        cache_rows = [
            ("KV Cache", "Prefill effective main KV", "PrefillBatch*HeadDim*BF16Bytes*(WindowLayers*MIN(PrefillSequence,WindowSize)+ShortLayers*(MIN(PrefillSequence,WindowSize)+INT(PrefillSequence/ShortRatio))+LongLayers*(MIN(PrefillSequence,WindowSize)+INT(PrefillSequence/LongRatio)))", prefill_cache["main"], "Raw plus generated compressed main-attention KV."),
            ("KV Cache", "Prefill effective Indexer KV", "PrefillBatch*ShortLayers*INT(PrefillSequence/ShortRatio)*IndexHeadDim*BF16Bytes", prefill_cache["indexer"], "Ratio-4 index cache actually populated by 8K prefill."),
            ("KV Cache", "Prefill compressor states", "PrefillBatch*FP32Bytes*(ShortLayers*2*(2*ShortRatio)*(2*HeadDim)+LongLayers*2*LongRatio*HeadDim+ShortLayers*2*(2*ShortRatio)*(2*IndexHeadDim))", prefill_cache["states"], "Main and Indexer incremental compressor states."),
            ("KV Cache", "Prefill 1M preallocated cache", "PrefillBatch*(HeadDim*BF16Bytes*(WindowLayers*WindowSize+ShortLayers*(WindowSize+INT(MaxContext/ShortRatio))+LongLayers*(WindowSize+INT(MaxContext/LongRatio)))+ShortLayers*INT(MaxContext/ShortRatio)*IndexHeadDim*BF16Bytes)+PrefillBatch*FP32Bytes*(ShortLayers*2*(2*ShortRatio)*(2*HeadDim)+LongLayers*2*LongRatio*HeadDim+ShortLayers*2*(2*ShortRatio)*(2*IndexHeadDim))", allocated_prefill["total"], "Current implementation allocates from max context."),
            ("KV Cache", "Decode effective main KV", "DecodeBatch*HeadDim*BF16Bytes*(WindowLayers*MIN(DecodeContext,WindowSize)+ShortLayers*(MIN(DecodeContext,WindowSize)+INT(DecodeContext/ShortRatio))+LongLayers*(MIN(DecodeContext,WindowSize)+INT(DecodeContext/LongRatio)))", decode_cache["main"], "Raw plus compressed main-attention KV at DecodeContext."),
            ("KV Cache", "Decode effective Indexer KV", "DecodeBatch*ShortLayers*INT(DecodeContext/ShortRatio)*IndexHeadDim*BF16Bytes", decode_cache["indexer"], "Ratio-4 index cache at DecodeContext."),
            ("KV Cache", "Decode compressor states", "DecodeBatch*FP32Bytes*(ShortLayers*2*(2*ShortRatio)*(2*HeadDim)+LongLayers*2*LongRatio*HeadDim+ShortLayers*2*(2*ShortRatio)*(2*IndexHeadDim))", decode_cache["states"], "Main and Indexer incremental compressor states."),
            ("KV Cache", "Decode 1M preallocated cache", "DecodeBatch*(HeadDim*BF16Bytes*(WindowLayers*WindowSize+ShortLayers*(WindowSize+INT(MaxContext/ShortRatio))+LongLayers*(WindowSize+INT(MaxContext/LongRatio)))+ShortLayers*INT(MaxContext/ShortRatio)*IndexHeadDim*BF16Bytes)+DecodeBatch*FP32Bytes*(ShortLayers*2*(2*ShortRatio)*(2*HeadDim)+LongLayers*2*LongRatio*HeadDim+ShortLayers*2*(2*ShortRatio)*(2*IndexHeadDim))", allocated_decode["total"], "Allocated cache and states for DecodeBatch."),
        ]

        row = 3
        parameter_total = 0.0
        for category, item, expression, value, note in parameter_rows:
            ws.write(row, 0, category, self.formats["text"])
            ws.write(row, 1, item, self.formats["text"])
            ws.write(row, 2, expression, self.formats["text"])
            ws.write_formula(row, 3, f"={expression}", self.formats["bytes"], value)
            ws.write_formula(row, 4, f"=D{row + 1}/1E9", self.formats["number"], value / 1e9)
            ws.write_formula(row, 5, f"=D{row + 1}/2^30", self.formats["number"], value / 2**30)
            ws.write(row, 6, note, self.formats["text"])
            parameter_total += value
            row += 1
        parameter_total_row = row
        ws.write(row, 0, "Parameters", self.formats["total"])
        ws.write(row, 1, "Total parameter capacity/rank", self.formats["total"])
        ws.write(row, 2, "SUM(parameter rows)", self.formats["total"])
        ws.write_formula(row, 3, f"=SUM(D4:D{row})", self.formats["total"], parameter_total)
        ws.write_formula(row, 4, f"=D{row + 1}/1E9", self.formats["total"], parameter_total / 1e9)
        ws.write_formula(row, 5, f"=D{row + 1}/2^30", self.formats["total"], parameter_total / 2**30)
        ws.write(row, 6, "Static runtime parameter estimate; excludes MTP.", self.formats["text"])
        self.memory_rows["ParameterTotal"] = row
        self._define_cell_name("ParameterTotalPerRank", "Memory", row, 3)
        row += 2
        cache_start = row
        cache_lookup: dict[str, float] = {}
        for category, item, expression, value, note in cache_rows:
            ws.write(row, 0, category, self.formats["text"])
            ws.write(row, 1, item, self.formats["text"])
            ws.write(row, 2, expression, self.formats["text"])
            ws.write_formula(row, 3, f"={expression}", self.formats["bytes"], value)
            ws.write_formula(row, 4, f"=D{row + 1}/1E9", self.formats["number"], value / 1e9)
            ws.write_formula(row, 5, f"=D{row + 1}/2^30", self.formats["number"], value / 2**30)
            ws.write(row, 6, note, self.formats["text"])
            cache_lookup[item] = value
            self.memory_rows[item] = row
            row += 1

        transient_prefill = max(
            p.prefill_batch * p.prefill_sequence * p.hc_slots * p.hidden * p.bf16_bytes,
            p.prefill_batch * p.prefill_sequence * p.heads * p.head_dim * p.bf16_bytes,
        )
        transient_decode = max(
            p.decode_batch * p.decode_tokens * p.hc_slots * p.hidden * p.bf16_bytes,
            p.decode_batch * p.decode_tokens * p.heads * p.head_dim * p.bf16_bytes,
        )
        transient_rows = [
            ("Transient", "Prefill largest major activation", "MAX(PrefillBatch*PrefillSequence*HCSlots*HiddenSize*BF16Bytes,PrefillBatch*PrefillSequence*NumHeads*HeadDim*BF16Bytes)", transient_prefill, "Max of HC state and full Q tensor; not allocator peak."),
            ("Transient", "Decode largest major activation", "MAX(DecodeBatch*DecodeTokens*HCSlots*HiddenSize*BF16Bytes,DecodeBatch*DecodeTokens*NumHeads*HeadDim*BF16Bytes)", transient_decode, "Max of HC state and full Q tensor; not allocator peak."),
        ]
        row += 1
        for category, item, expression, value, note in transient_rows:
            ws.write(row, 0, category, self.formats["text"])
            ws.write(row, 1, item, self.formats["text"])
            ws.write(row, 2, expression, self.formats["text"])
            ws.write_formula(row, 3, f"={expression}", self.formats["bytes"], value)
            ws.write_formula(row, 4, f"=D{row + 1}/1E9", self.formats["number"], value / 1e9)
            ws.write_formula(row, 5, f"=D{row + 1}/2^30", self.formats["number"], value / 2**30)
            ws.write(row, 6, note, self.formats["text"])
            row += 1

        ws.autofilter(2, 0, row - 1, 6)
        ws.set_column("A:A", 16)
        ws.set_column("B:B", 35)
        ws.set_column("C:C", 95)
        ws.set_column("D:F", 20)
        ws.set_column("G:G", 58)
        return {
            "parameter_total": parameter_total,
            "prefill_effective_cache": prefill_cache["total"],
            "prefill_allocated_cache": allocated_prefill["total"],
            "decode_effective_cache": decode_cache["total"],
            "decode_allocated_cache": allocated_decode["total"],
            "prefill_transient": transient_prefill,
            "decode_transient": transient_decode,
        }

    def write_comparison(self, memory: dict[str, float]) -> None:
        ws = self.workbook.add_worksheet("Comparison")
        ws.hide_gridlines(2)
        ws.write(0, 0, "Prefill 8K vs Decode 1M", self.formats["title"])
        ws.write(1, 0, "Values reference the two scenario sheets and update when inputs change.", self.formats["note"])
        headers = ["Metric", "Prefill 8K", "Decode 1M", "Unit", "Interpretation"]
        ws.write_row(3, 0, headers, self.formats["header"])
        pf_start, pf_end = self.scenario_ranges["PF"]
        dc_start, dc_end = self.scenario_ranges["DC"]
        pf_first, pf_last = pf_start + 1, pf_end + 1
        dc_first, dc_last = dc_start + 1, dc_end + 1
        categories = ["Attention", "MoE", "Other"]
        rows: list[tuple[str, str, str, float, float, str, str]] = []
        pf_items, _ = scenario_items(self.p, self.layers, "prefill")
        dc_items, _ = scenario_items(self.p, self.layers, "decode")
        for category in categories:
            pf_value = sum(item.rank_flops for item in pf_items if item.category == category)
            dc_value = sum(item.rank_flops for item in dc_items if item.category == category)
            rows.append(
                (
                    f"{category} FLOPs/rank",
                    f'=SUMIF(Prefill_8K!$A${pf_first}:$A${pf_last},"{category}",Prefill_8K!$F${pf_first}:$F${pf_last})',
                    f'=SUMIF(Decode_1M!$A${dc_first}:$A${dc_last},"{category}",Decode_1M!$F${dc_first}:$F${dc_last})',
                    pf_value,
                    dc_value,
                    "FLOPs",
                    "Per-rank logical/physical calculation according to TP and kernel padding.",
                )
            )
        for category in ("Attention", "MoE", "Other"):
            pf_value = sum(item.read_bytes + item.write_bytes for item in pf_items if item.category == category)
            dc_value = sum(item.read_bytes + item.write_bytes for item in dc_items if item.category == category)
            rows.append(
                (
                    f"{category} HBM traffic/rank",
                    f'=SUMIF(Prefill_8K!$A${pf_first}:$A${pf_last},"{category}",Prefill_8K!$I${pf_first}:$I${pf_last})',
                    f'=SUMIF(Decode_1M!$A${dc_first}:$A${dc_last},"{category}",Decode_1M!$I${dc_first}:$I${dc_last})',
                    pf_value,
                    dc_value,
                    "bytes",
                    "Logical reads plus writes; cache/fusion effects are not simulated.",
                )
            )
        pf_hbm_total = sum(item.read_bytes + item.write_bytes for item in pf_items)
        dc_hbm_total = sum(item.read_bytes + item.write_bytes for item in dc_items)
        for category in ("Attention", "MoE", "Other"):
            pf_value = sum(
                item.read_bytes + item.write_bytes
                for item in pf_items
                if item.category == category
            ) / pf_hbm_total
            dc_value = sum(
                item.read_bytes + item.write_bytes
                for item in dc_items
                if item.category == category
            ) / dc_hbm_total
            rows.append(
                (
                    f"{category} HBM share",
                    f'=SUMIF(Prefill_8K!$A${pf_first}:$A${pf_last},"{category}",Prefill_8K!$I${pf_first}:$I${pf_last})/SUM(Prefill_8K!$I${pf_first}:$I${pf_last})',
                    f'=SUMIF(Decode_1M!$A${dc_first}:$A${dc_last},"{category}",Decode_1M!$I${dc_first}:$I${dc_last})/SUM(Decode_1M!$I${dc_first}:$I${dc_last})',
                    pf_value,
                    dc_value,
                    "fraction",
                    "Share of modeled per-rank logical HBM reads plus writes.",
                )
            )

        def memory_cell(label: str) -> str:
            return f"Memory!D{self.memory_rows[label] + 1}"

        prefill_effective_formula = "+".join(
            memory_cell(label)
            for label in (
                "Prefill effective main KV",
                "Prefill effective Indexer KV",
                "Prefill compressor states",
            )
        )
        decode_effective_formula = "+".join(
            memory_cell(label)
            for label in (
                "Decode effective main KV",
                "Decode effective Indexer KV",
                "Decode compressor states",
            )
        )
        rows.extend(
            [
                ("Effective KV + state", f"={prefill_effective_formula}", f"={decode_effective_formula}", memory["prefill_effective_cache"], memory["decode_effective_cache"], "bytes", "Scenario-populated cache and compressor states."),
                ("Preallocated KV + state", f"={memory_cell('Prefill 1M preallocated cache')}", f"={memory_cell('Decode 1M preallocated cache')}", memory["prefill_allocated_cache"], memory["decode_allocated_cache"], "bytes", "Capacity reserved from MaxContext and each scenario batch."),
                ("Parameters/rank", "=ParameterTotalPerRank", "=ParameterTotalPerRank", memory["parameter_total"], memory["parameter_total"], "bytes", "Weights do not scale with batch or sequence length."),
            ]
        )
        row = 4
        for label, pf_formula, dc_formula, pf_value, dc_value, unit, note in rows:
            ws.write(row, 0, label, self.formats["text"])
            ws.write_formula(row, 1, pf_formula, self.formats["scientific"], pf_value)
            ws.write_formula(row, 2, dc_formula, self.formats["scientific"], dc_value)
            ws.write(row, 3, unit, self.formats["text"])
            ws.write(row, 4, note, self.formats["text"])
            row += 1

        flops_chart = self.workbook.add_chart({"type": "column", "subtype": "stacked"})
        for column, name in ((1, "Prefill 8K"), (2, "Decode 1M")):
            flops_chart.add_series(
                {
                    "name": name,
                    "categories": ["Comparison", 4, 0, 6, 0],
                    "values": ["Comparison", 4, column, 6, column],
                }
            )
        flops_chart.set_title({"name": "Per-rank FLOPs distribution"})
        flops_chart.set_y_axis({"name": "FLOPs", "log_base": 10})
        flops_chart.set_style(10)
        ws.insert_chart("G4", flops_chart, {"x_scale": 1.25, "y_scale": 1.15})

        hbm_chart = self.workbook.add_chart({"type": "column", "subtype": "stacked"})
        for column, name in ((1, "Prefill 8K"), (2, "Decode 1M")):
            hbm_chart.add_series(
                {
                    "name": name,
                    "categories": ["Comparison", 7, 0, 9, 0],
                    "values": ["Comparison", 7, column, 9, column],
                }
            )
        hbm_chart.set_title({"name": "Per-rank logical HBM traffic"})
        hbm_chart.set_y_axis({"name": "Bytes", "log_base": 10})
        hbm_chart.set_style(11)
        ws.insert_chart("G21", hbm_chart, {"x_scale": 1.25, "y_scale": 1.15})
        ws.set_column("A:A", 34)
        ws.set_column("B:C", 22)
        ws.set_column("D:D", 14)
        ws.set_column("E:E", 65)

    def write_methodology(self) -> None:
        ws = self.workbook.add_worksheet("Methodology")
        ws.hide_gridlines(2)
        ws.write(0, 0, "Methodology and Limits", self.formats["title"])
        lines = [
            ("Scope", "43-layer main inference path by default; MTP is excluded. Layer_Config may activate/deactivate up to 64 rows."),
            ("FLOPs", "Matrix multiply-accumulate counts as 2 FLOPs. Main Attention and MoE rows are exact major GEMM estimates; HC and elementwise work is approximate."),
            ("Prefill attention", "Uses causal pair counts, not S*S blindly. Raw window pairs, compressed pairs, and Indexer scan pairs are separate formulas."),
            ("Decode attention", "Each DecodeTokens item is modeled at DecodeContext. Ratio-4 main attention uses Top-K, while its Indexer scans all compressed entries."),
            ("MoE", "Compute uses Top-K routed assignments plus SharedExperts. Routed parameter reads use the probability that an expert is touched at least once."),
            ("HBM traffic", "Logical parameter/activation/KV reads and writes. It does not model L2 reuse, fusion, tiling, allocator behavior, or compression in a vendor kernel."),
            ("Memory", "Parameter capacity follows inference runtime dtypes: routed FP4, most projections FP8, Wo_a BF16, compressor FP32, and LM head FP32."),
            ("KV cache", "Effective cache reports populated entries. Preallocated cache uses MaxContext. KV is replicated on every TP rank in the current implementation."),
            ("TP", "Wq_a, Wkv, compressors, router, HC, and shared experts are replicated. Q/O shards, routed experts, vocabulary, and LM head are divided by TP."),
            ("Communication", "Ring formulas report send+receive transfer bytes per rank. TP=1 is zero. This is not an observed GB/s or latency."),
            ("Roofline", "Compute/HBM/interconnect lower bounds use editable hardware cells and assume no overlap. Real runtime requires profiling."),
            ("Units", "GB uses 10^9 bytes; GiB uses 2^30 bytes; TFLOP uses 10^12 FLOPs."),
            ("Formula maintenance", "Blue cells are inputs. Green and result cells contain formulas and cached baseline values. Excel is configured to recalculate on open."),
        ]
        ws.write_row(2, 0, ["Topic", "Definition"], self.formats["header"])
        for row, (topic, definition) in enumerate(lines, start=3):
            ws.write(row, 0, topic, self.formats["text"])
            ws.write(row, 1, definition, self.formats["text"])
        ws.set_column("A:A", 24)
        ws.set_column("B:B", 110)

    def close(self) -> None:
        self.workbook.close()


def summarize_items(items: list[Item]) -> dict[str, Any]:
    categories = {}
    for category in sorted({item.category for item in items}):
        selected = [item for item in items if item.category == category]
        categories[category] = {
            "global_flops": sum(item.global_flops for item in selected),
            "per_rank_flops": sum(item.rank_flops for item in selected),
            "hbm_read_bytes_per_rank": sum(item.read_bytes for item in selected),
            "hbm_write_bytes_per_rank": sum(item.write_bytes for item in selected),
            "interconnect_bytes_per_rank": sum(item.network_bytes for item in selected),
        }
    return {
        "categories": categories,
        "total_global_flops": sum(item.global_flops for item in items),
        "total_per_rank_flops": sum(item.rank_flops for item in items),
        "total_hbm_read_bytes_per_rank": sum(item.read_bytes for item in items),
        "total_hbm_write_bytes_per_rank": sum(item.write_bytes for item in items),
        "total_interconnect_bytes_per_rank": sum(item.network_bytes for item in items),
        "attention_major_flops_per_rank": sum(
            item.rank_flops
            for item in items
            if item.category == "Attention" and item.accounting == "Major"
        ),
        "moe_major_flops_per_rank": sum(
            item.rank_flops
            for item in items
            if item.category == "MoE" and item.accounting == "Major"
        ),
    }


def write_reports(
    output_dir: Path,
    p: Inputs,
    ratios: list[int],
    memory: dict[str, float],
    prefill_items: list[Item],
    decode_items: list[Item],
) -> None:
    layers = count_layers(ratios, p)
    prefill = summarize_items(prefill_items)
    decode = summarize_items(decode_items)
    report = {
        "model": "DeepSeek-V4-Flash",
        "scope": "main 43-layer inference path; MTP excluded",
        "inputs": p.__dict__,
        "layer_counts": layers.__dict__,
        "prefill_8k": prefill,
        "decode_1m": decode,
        "memory": memory,
    }
    (output_dir / "baseline_results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    def tf(value: float) -> str:
        return f"{value / 1e12:,.3f} TFLOPs"

    def gb(value: float) -> str:
        return f"{value / 1e9:,.3f} GB"

    markdown = [
        "# DeepSeek V4 Flash TP1 Baseline",
        "",
        "## Assumptions",
        "",
        f"- TP: `{p.tp}`",
        f"- Prefill: batch `{p.prefill_batch}`, sequence `{p.prefill_sequence}`",
        f"- Decode: batch `{p.decode_batch}`, tokens `{p.decode_tokens}`, context `{p.decode_context}`",
        f"- Layers: `{layers.total}` = window `{layers.window}` + short `{layers.short}` + long `{layers.long}`",
        "- MAC = 2 FLOPs; MTP excluded.",
        "",
        "## Compute and HBM",
        "",
        "| Metric | Prefill 8K | Decode 1M |",
        "|---|---:|---:|",
        f"| Attention major FLOPs/rank | {tf(prefill['attention_major_flops_per_rank'])} | {tf(decode['attention_major_flops_per_rank'])} |",
        f"| MoE major FLOPs/rank | {tf(prefill['moe_major_flops_per_rank'])} | {tf(decode['moe_major_flops_per_rank'])} |",
        f"| Total modeled FLOPs/rank | {tf(prefill['total_per_rank_flops'])} | {tf(decode['total_per_rank_flops'])} |",
        f"| Attention HBM traffic/rank | {gb(prefill['categories']['Attention']['hbm_read_bytes_per_rank'] + prefill['categories']['Attention']['hbm_write_bytes_per_rank'])} | {gb(decode['categories']['Attention']['hbm_read_bytes_per_rank'] + decode['categories']['Attention']['hbm_write_bytes_per_rank'])} |",
        f"| MoE HBM traffic/rank | {gb(prefill['categories']['MoE']['hbm_read_bytes_per_rank'] + prefill['categories']['MoE']['hbm_write_bytes_per_rank'])} | {gb(decode['categories']['MoE']['hbm_read_bytes_per_rank'] + decode['categories']['MoE']['hbm_write_bytes_per_rank'])} |",
        f"| Other HBM traffic/rank | {gb(prefill['categories']['Other']['hbm_read_bytes_per_rank'] + prefill['categories']['Other']['hbm_write_bytes_per_rank'])} | {gb(decode['categories']['Other']['hbm_read_bytes_per_rank'] + decode['categories']['Other']['hbm_write_bytes_per_rank'])} |",
        f"| HBM read/rank | {gb(prefill['total_hbm_read_bytes_per_rank'])} | {gb(decode['total_hbm_read_bytes_per_rank'])} |",
        f"| HBM write/rank | {gb(prefill['total_hbm_write_bytes_per_rank'])} | {gb(decode['total_hbm_write_bytes_per_rank'])} |",
        "",
        "## Capacity",
        "",
        "| Metric | Capacity |",
        "|---|---:|",
        f"| Parameters/rank | {gb(memory['parameter_total'])} |",
        f"| Prefill effective KV + states | {gb(memory['prefill_effective_cache'])} |",
        f"| Prefill preallocated KV + states | {gb(memory['prefill_allocated_cache'])} |",
        f"| Decode effective KV + states | {gb(memory['decode_effective_cache'])} |",
        f"| Decode preallocated KV + states | {gb(memory['decode_allocated_cache'])} |",
        "",
        "The Excel workbook is authoritative for architecture exploration: all detail rows contain formulas and recalculate after input edits.",
    ]
    (output_dir / "baseline_report.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )


def validate_baseline(
    p: Inputs,
    ratios: list[int],
    memory: dict[str, float],
    prefill_items: list[Item],
    decode_items: list[Item],
    workbook_path: Path,
) -> None:
    layers = count_layers(ratios, p)
    if (layers.total, layers.window, layers.short, layers.long) != (43, 2, 21, 20):
        raise AssertionError(f"Unexpected layer mix: {layers}")
    decode = summarize_items(decode_items)
    if not math.isclose(
        decode["attention_major_flops_per_rank"], 123_969_470_464, rel_tol=0, abs_tol=1
    ):
        raise AssertionError(
            "Decode attention regression failed: "
            f"{decode['attention_major_flops_per_rank']}"
        )
    if not math.isclose(
        decode["moe_major_flops_per_rank"], 15_240_003_584, rel_tol=0, abs_tol=1
    ):
        raise AssertionError(
            f"Decode MoE regression failed: {decode['moe_major_flops_per_rank']}"
        )
    expected_main_and_index = 7_219_838_976
    decode_cache_without_states = cache_values(
        p, layers, p.decode_context, p.decode_batch
    )["main"] + cache_values(p, layers, p.decode_context, p.decode_batch)["indexer"]
    if decode_cache_without_states != expected_main_and_index:
        raise AssertionError(
            f"Decode KV regression failed: {decode_cache_without_states}"
        )
    if summarize_items(prefill_items)["total_interconnect_bytes_per_rank"] != 0:
        raise AssertionError("TP1 prefill interconnect must be zero")
    if summarize_items(decode_items)["total_interconnect_bytes_per_rank"] != 0:
        raise AssertionError("TP1 decode interconnect must be zero")
    if memory["parameter_total"] <= 0:
        raise AssertionError("Parameter capacity must be positive")
    with zipfile.ZipFile(workbook_path) as archive:
        formula_count = 0
        formula_payload = b""
        for name in archive.namelist():
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml"):
                payload = archive.read(name)
                formula_payload += payload
                formula_count += payload.count(b"<f")
        if formula_count < 150:
            raise AssertionError(
                f"Workbook contains too few formulas: {formula_count}"
            )
        if b"#REF!" in formula_payload or b"#NAME?" in formula_payload:
            raise AssertionError("Workbook contains an invalid formula reference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    p, ratios = load_inputs(args.config.resolve())
    layers = count_layers(ratios, p)
    prefill_items, _ = scenario_items(p, layers, "prefill")
    decode_items, _ = scenario_items(p, layers, "decode")

    writer = CalculatorWriter(output, p, ratios)
    writer.write_parameters()
    writer.write_layer_config()
    writer.write_scenario("prefill")
    writer.write_scenario("decode")
    memory = writer.write_memory()
    writer.write_comparison(memory)
    writer.write_methodology()
    writer.close()

    write_reports(output.parent, p, ratios, memory, prefill_items, decode_items)
    validate_baseline(p, ratios, memory, prefill_items, decode_items, output)
    print(f"Wrote {output}")
    print(f"Wrote {output.parent / 'baseline_results.json'}")
    print(f"Wrote {output.parent / 'baseline_report.md'}")


if __name__ == "__main__":
    main()