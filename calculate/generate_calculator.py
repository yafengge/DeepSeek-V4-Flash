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
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import xlsxwriter
from xlsxwriter.utility import xl_col_to_name


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "inference" / "config.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "deepseek_v4_flash_calculator.xlsx"
MAX_LAYER_ROWS = 64
MAX_RANK_ROWS = 64


@dataclass(frozen=True)
class Inputs:
    tp: int
    comparison_tp: int
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
            comparison_tp=8,
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


TP_FORMULA_NAMES = (
    "KernelLocalHeads",
    "LocalIndexHeads",
    "LocalOGroups",
    "LocalExperts",
    "LocalHeads",
    "LocalVocab",
    "TPSize",
)


def formula_for_tp(formula: str, label: str) -> str:
    """Map generic TP names in an Excel formula to TP1/TP8 named cells."""
    result = formula
    for name in TP_FORMULA_NAMES:
        replacement = f"{label}Size" if name == "TPSize" else f"{label}{name}"
        result = re.sub(rf"\b{name}\b", replacement, result)
    return result


def scaled_value(value: float, kind: str) -> tuple[float, str]:
    if kind == "flops":
        units = ((1e15, "PFLOPs"), (1e12, "TFLOPs"), (1e9, "GFLOPs"), (1e6, "MFLOPs"), (1, "FLOPs"))
    elif kind == "bytes":
        units = ((1e12, "TB"), (1e9, "GB"), (1e6, "MB"), (1e3, "KB"), (1, "B"))
    elif kind == "params":
        units = ((1e12, "T params"), (1e9, "G params"), (1e6, "M params"), (1e3, "K params"), (1, "params"))
    else:
        return value, kind
    for divisor, unit in units:
        if abs(value) >= divisor or divisor == 1:
            return value / divisor, unit
    raise AssertionError("unreachable")


def display_formula(raw_cell: str, kind: str) -> tuple[str, str]:
    if kind == "flops":
        thresholds = (("1E15", "PFLOPs"), ("1E12", "TFLOPs"), ("1E9", "GFLOPs"), ("1E6", "MFLOPs"))
    elif kind == "bytes":
        thresholds = (("1E12", "TB"), ("1E9", "GB"), ("1E6", "MB"), ("1E3", "KB"))
    elif kind == "params":
        thresholds = (("1E12", "T params"), ("1E9", "G params"), ("1E6", "M params"), ("1E3", "K params"))
    else:
        return f"={raw_cell}", f'="{kind}"'
    value_expression = raw_cell
    unit_expression = f'"{kind if kind != "params" else "params"}"'
    for threshold, unit in reversed(thresholds):
        value_expression = f"IF(ABS({raw_cell})>={threshold},{raw_cell}/{threshold},{value_expression})"
        unit_expression = f'IF(ABS({raw_cell})>={threshold},"{unit}",{unit_expression})'
    return f"={value_expression}", f"={unit_expression}"


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


@dataclass(frozen=True)
class ParameterComponent:
    category: str
    name: str
    global_count_formula: str
    global_count: float
    rank_count_formula: str
    rank_count: float
    global_bytes_formula: str
    global_bytes: float
    rank_bytes_formula: str
    rank_bytes: float
    notes: str


def parameter_components(p: Inputs, layers: LayerCounts) -> list[ParameterComponent]:
    weights = weight_values(p, layers)
    local_heads = p.heads / p.tp
    local_groups = p.o_groups / p.tp
    local_index_heads = p.index_heads / p.tp
    mix_hc = (2 + p.hc_slots) * p.hc_slots

    core_global_count = layers.total * (
        p.hidden * p.q_rank
        + p.q_rank * p.heads * p.head_dim
        + p.hidden * p.head_dim
        + p.heads * p.head_dim * p.o_rank
        + p.o_groups * p.o_rank * p.hidden
    )
    core_rank_count = layers.total * (
        p.hidden * p.q_rank
        + p.q_rank * local_heads * p.head_dim
        + p.hidden * p.head_dim
        + local_heads * p.head_dim * p.o_rank
        + local_groups * p.o_rank * p.hidden
    )
    core_global_bytes = layers.total * weights["core_global_layer"]
    core_rank_bytes = layers.total * weights["core_rank_layer"]
    core_global_count_xl = "TotalLayers*(HiddenSize*QLoraRank+QLoraRank*NumHeads*HeadDim+HiddenSize*HeadDim+NumHeads*HeadDim*OLoraRank+OGroups*OLoraRank*HiddenSize)"
    core_rank_count_xl = "TotalLayers*(HiddenSize*QLoraRank+QLoraRank*LocalHeads*HeadDim+HiddenSize*HeadDim+LocalHeads*HeadDim*OLoraRank+LocalOGroups*OLoraRank*HiddenSize)"
    core_global_bytes_xl = (
        f"TotalLayers*({xl_fp8('HiddenSize', 'QLoraRank')}+"
        f"{xl_fp8('QLoraRank', 'NumHeads*HeadDim')}+"
        f"{xl_fp8('HiddenSize', 'HeadDim')}+"
        "NumHeads*HeadDim*OLoraRank*BF16Bytes+"
        f"{xl_fp8('OGroups*OLoraRank', 'HiddenSize')})"
    )
    core_rank_bytes_xl = (
        f"TotalLayers*({xl_fp8('HiddenSize', 'QLoraRank')}+"
        f"{xl_fp8('QLoraRank', 'LocalHeads*HeadDim')}+"
        f"{xl_fp8('HiddenSize', 'HeadDim')}+"
        "LocalHeads*HeadDim*OLoraRank*BF16Bytes+"
        f"{xl_fp8('LocalOGroups*OLoraRank', 'HiddenSize')})"
    )

    compressor_count_xl = (
        "ShortLayers*(2*HiddenSize*(2*HeadDim)+ShortRatio*2*HeadDim+HeadDim)+"
        "LongLayers*(2*HiddenSize*HeadDim+LongRatio*HeadDim+HeadDim)"
    )
    compressor_count = (
        layers.short
        * (2 * p.hidden * (2 * p.head_dim) + p.short_ratio * 2 * p.head_dim + p.head_dim)
        + layers.long
        * (2 * p.hidden * p.head_dim + p.long_ratio * p.head_dim + p.head_dim)
    )
    compressor_bytes_xl = (
        "ShortLayers*(2*HiddenSize*(2*HeadDim)*FP32Bytes+ShortRatio*2*HeadDim*FP32Bytes+HeadDim*FP32Bytes)+"
        "LongLayers*(2*HiddenSize*HeadDim*FP32Bytes+LongRatio*HeadDim*FP32Bytes+HeadDim*FP32Bytes)"
    )
    compressor_bytes = (
        layers.short * weights["short_main_compressor"]
        + layers.long * weights["long_main_compressor"]
    )

    index_global_count_xl = (
        "ShortLayers*(QLoraRank*IndexHeads*IndexHeadDim+HiddenSize*IndexHeads+"
        "2*HiddenSize*(2*IndexHeadDim)+ShortRatio*2*IndexHeadDim+IndexHeadDim)"
    )
    index_rank_count_xl = (
        "ShortLayers*(QLoraRank*LocalIndexHeads*IndexHeadDim+HiddenSize*LocalIndexHeads+"
        "2*HiddenSize*(2*IndexHeadDim)+ShortRatio*2*IndexHeadDim+IndexHeadDim)"
    )
    index_global_count = layers.short * (
        p.q_rank * p.index_heads * p.index_dim
        + p.hidden * p.index_heads
        + 2 * p.hidden * (2 * p.index_dim)
        + p.short_ratio * 2 * p.index_dim
        + p.index_dim
    )
    index_rank_count = layers.short * (
        p.q_rank * local_index_heads * p.index_dim
        + p.hidden * local_index_heads
        + 2 * p.hidden * (2 * p.index_dim)
        + p.short_ratio * 2 * p.index_dim
        + p.index_dim
    )
    index_global_bytes_xl = (
        f"ShortLayers*({xl_fp8('QLoraRank', 'IndexHeads*IndexHeadDim')}+"
        "HiddenSize*IndexHeads*BF16Bytes+2*HiddenSize*(2*IndexHeadDim)*FP32Bytes+"
        "ShortRatio*2*IndexHeadDim*FP32Bytes+IndexHeadDim*FP32Bytes)"
    )
    index_rank_bytes_xl = (
        f"ShortLayers*({xl_fp8('QLoraRank', 'LocalIndexHeads*IndexHeadDim')}+"
        "HiddenSize*LocalIndexHeads*BF16Bytes+2*HiddenSize*(2*IndexHeadDim)*FP32Bytes+"
        "ShortRatio*2*IndexHeadDim*FP32Bytes+IndexHeadDim*FP32Bytes)"
    )
    index_global_bytes = layers.short * (
        weights["index_core_global"] + weights["index_compressor"]
    )
    index_rank_bytes = layers.short * (
        weights["index_core_rank"] + weights["index_compressor"]
    )

    routed_global_count = layers.total * p.routed_experts * 3 * p.hidden * p.expert_inter
    routed_rank_count = routed_global_count / p.tp
    routed_global_bytes = layers.total * p.routed_experts * weights["expert_one"]
    routed_rank_bytes = weights["routed_rank"]

    shared_global_count = layers.total * p.shared_experts * 3 * p.hidden * p.expert_inter
    shared_bytes = weights["shared_rank"]
    shared_bytes_xl = (
        f"TotalLayers*SharedExperts*(2*{xl_fp8('HiddenSize', 'ExpertInter')}+"
        f"{xl_fp8('ExpertInter', 'HiddenSize')})"
    )

    score_layers = layers.total - layers.hash
    router_count = (
        layers.total * p.routed_experts * p.hidden
        + score_layers * p.routed_experts
        + layers.hash * p.vocab * p.activated_experts
    )
    router_count_xl = (
        "TotalLayers*RoutedExperts*HiddenSize+(TotalLayers-ActiveHashLayers)*RoutedExperts+"
        "ActiveHashLayers*VocabSize*ActivatedExperts"
    )
    router_bytes_xl = (
        "TotalLayers*RoutedExperts*HiddenSize*BF16Bytes+"
        "(TotalLayers-ActiveHashLayers)*RoutedExperts*FP32Bytes+"
        "ActiveHashLayers*VocabSize*ActivatedExperts*INT32Bytes"
    )

    hc_count = layers.total * 2 * (mix_hc * p.hc_slots * p.hidden + mix_hc + 3)
    hc_count_xl = "TotalLayers*2*(((2+HCSlots)*HCSlots)*HCSlots*HiddenSize+((2+HCSlots)*HCSlots)+3)"
    hc_bytes_xl = f"({hc_count_xl})*FP32Bytes"

    norm_global_count = layers.total * (
        2 * p.hidden + p.q_rank + p.head_dim + p.heads
    )
    norm_rank_count = layers.total * (
        2 * p.hidden + p.q_rank + p.head_dim + local_heads
    )
    norm_global_count_xl = "TotalLayers*(2*HiddenSize+QLoraRank+HeadDim+NumHeads)"
    norm_rank_count_xl = "TotalLayers*(2*HiddenSize+QLoraRank+HeadDim+LocalHeads)"

    tail_count = p.hc_slots * p.hc_slots * p.hidden + p.hc_slots + 1 + p.hidden
    tail_count_xl = "HCSlots*HCSlots*HiddenSize+HCSlots+1+HiddenSize"

    return [
        ParameterComponent("Attention", "Core Q/K/O projections", core_global_count_xl, core_global_count, core_rank_count_xl, core_rank_count, core_global_bytes_xl, core_global_bytes, core_rank_bytes_xl, core_rank_bytes, "Wq_a/Wkv replicated; Q/O output dimensions TP-sharded."),
        ParameterComponent("Attention", "KV compressors", compressor_count_xl, compressor_count, compressor_count_xl, compressor_count, compressor_bytes_xl, compressor_bytes, compressor_bytes_xl, compressor_bytes, "Replicated FP32 inference compressor parameters."),
        ParameterComponent("Attention", "Ratio-4 Indexers", index_global_count_xl, index_global_count, index_rank_count_xl, index_rank_count, index_global_bytes_xl, index_global_bytes, index_rank_bytes_xl, index_rank_bytes, "Indexer Q/weight projections are sharded; index compressor is replicated."),
        ParameterComponent("MoE", "Routed experts", "TotalLayers*RoutedExperts*3*HiddenSize*ExpertInter", routed_global_count, "TotalLayers*LocalExperts*3*HiddenSize*ExpertInter", routed_rank_count, f"TotalLayers*RoutedExperts*{xl_expert_bytes()}", routed_global_bytes, f"TotalLayers*LocalExperts*{xl_expert_bytes()}", routed_rank_bytes, "FP4 routed experts are uniformly sharded across ranks."),
        ParameterComponent("MoE", "Shared experts", "TotalLayers*SharedExperts*3*HiddenSize*ExpertInter", shared_global_count, "TotalLayers*SharedExperts*3*HiddenSize*ExpertInter", shared_global_count, shared_bytes_xl, shared_bytes, shared_bytes_xl, shared_bytes, "Shared FP8 experts are replicated on every rank."),
        ParameterComponent("MoE", "Router and hash tables", router_count_xl, router_count, router_count_xl, router_count, router_bytes_xl, weights["router_rank"], router_bytes_xl, weights["router_rank"], "Gate, bias, and token-to-expert tables are replicated."),
        ParameterComponent("Other", "Embedding", "VocabSize*HiddenSize", p.vocab * p.hidden, "LocalVocab*HiddenSize", p.vocab / p.tp * p.hidden, "VocabSize*HiddenSize*BF16Bytes", p.vocab * p.hidden * p.bf16_bytes, "LocalVocab*HiddenSize*BF16Bytes", weights["embedding"], "Vocabulary-parallel embedding."),
        ParameterComponent("Other", "LM head", "VocabSize*HiddenSize", p.vocab * p.hidden, "LocalVocab*HiddenSize", p.vocab / p.tp * p.hidden, "VocabSize*HiddenSize*FP32Bytes", p.vocab * p.hidden * p.fp32_bytes, "LocalVocab*HiddenSize*FP32Bytes", weights["lm_head"], "Vocabulary-parallel FP32 inference head."),
        ParameterComponent("Other", "Hyper-Connections", hc_count_xl, hc_count, hc_count_xl, hc_count, hc_bytes_xl, weights["hc_rank"], hc_bytes_xl, weights["hc_rank"], "Replicated HC parameters; inference only."),
        ParameterComponent("Other", "Norms and attention sinks", norm_global_count_xl, norm_global_count, norm_rank_count_xl, norm_rank_count, f"({norm_global_count_xl})*FP32Bytes", norm_global_count * p.fp32_bytes, f"({norm_rank_count_xl})*FP32Bytes", weights["norms_rank"], "Only local attention-sink vectors are sharded."),
        ParameterComponent("Other", "Tail HC and final norm", tail_count_xl, tail_count, tail_count_xl, tail_count, f"({tail_count_xl})*FP32Bytes", weights["tail_rank"], f"({tail_count_xl})*FP32Bytes", weights["tail_rank"], "Replicated final inference tail."),
    ]


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
            ("Parallel", "TP1 comparison size", "TP1Size", self.p.tp, "ranks", "First tensor-parallel configuration."),
            ("Parallel", "TP8 comparison size", "TP8Size", self.p.comparison_tp, "ranks", "Second tensor-parallel configuration; editable."),
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
            ("Derived", "TP1 local Q heads", "TP1LocalHeads", "=NumHeads/TP1Size", self.p.heads / self.p.tp, "heads/rank", "Q heads on one TP1 rank."),
            ("Derived", "TP1 kernel local heads", "TP1KernelLocalHeads", "=MAX(TP1LocalHeads,KernelMinHeads)", max(self.p.heads / self.p.tp, self.p.kernel_min_heads), "heads/rank", "Sparse-kernel heads after padding."),
            ("Derived", "TP1 local index heads", "TP1LocalIndexHeads", "=IndexHeads/TP1Size", self.p.index_heads / self.p.tp, "heads/rank", "Indexer heads on one TP1 rank."),
            ("Derived", "TP1 local experts", "TP1LocalExperts", "=RoutedExperts/TP1Size", self.p.routed_experts / self.p.tp, "experts/rank", "Routed experts on one TP1 rank."),
            ("Derived", "TP1 local output groups", "TP1LocalOGroups", "=OGroups/TP1Size", self.p.o_groups / self.p.tp, "groups/rank", "Output groups on one TP1 rank."),
            ("Derived", "TP1 local vocabulary", "TP1LocalVocab", "=VocabSize/TP1Size", self.p.vocab / self.p.tp, "tokens/rank", "Vocabulary rows on one TP1 rank."),
            ("Derived", "TP8 local Q heads", "TP8LocalHeads", "=NumHeads/TP8Size", self.p.heads / self.p.comparison_tp, "heads/rank", "Q heads on one TP8 rank."),
            ("Derived", "TP8 kernel local heads", "TP8KernelLocalHeads", "=MAX(TP8LocalHeads,KernelMinHeads)", max(self.p.heads / self.p.comparison_tp, self.p.kernel_min_heads), "heads/rank", "Sparse-kernel heads after padding."),
            ("Derived", "TP8 local index heads", "TP8LocalIndexHeads", "=IndexHeads/TP8Size", self.p.index_heads / self.p.comparison_tp, "heads/rank", "Indexer heads on one TP8 rank."),
            ("Derived", "TP8 local experts", "TP8LocalExperts", "=RoutedExperts/TP8Size", self.p.routed_experts / self.p.comparison_tp, "experts/rank", "Routed experts on one TP8 rank."),
            ("Derived", "TP8 local output groups", "TP8LocalOGroups", "=OGroups/TP8Size", self.p.o_groups / self.p.comparison_tp, "groups/rank", "Output groups on one TP8 rank."),
            ("Derived", "TP8 local vocabulary", "TP8LocalVocab", "=VocabSize/TP8Size", self.p.vocab / self.p.comparison_tp, "tokens/rank", "Vocabulary rows on one TP8 rank."),
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
            ("Q heads divisible by TP1 and TP8", "=IF(AND(MOD(NumHeads,TP1Size)=0,MOD(NumHeads,TP8Size)=0),\"OK\",\"ERROR\")", self.p.heads % self.p.tp == 0 and self.p.heads % self.p.comparison_tp == 0),
            ("Experts divisible by TP1 and TP8", "=IF(AND(MOD(RoutedExperts,TP1Size)=0,MOD(RoutedExperts,TP8Size)=0),\"OK\",\"ERROR\")", self.p.routed_experts % self.p.tp == 0 and self.p.routed_experts % self.p.comparison_tp == 0),
            ("Output groups divisible by TP1 and TP8", "=IF(AND(MOD(OGroups,TP1Size)=0,MOD(OGroups,TP8Size)=0),\"OK\",\"ERROR\")", self.p.o_groups % self.p.tp == 0 and self.p.o_groups % self.p.comparison_tp == 0),
            ("Indexer heads divisible by TP1 and TP8", "=IF(AND(MOD(IndexHeads,TP1Size)=0,MOD(IndexHeads,TP8Size)=0),\"OK\",\"ERROR\")", self.p.index_heads % self.p.tp == 0 and self.p.index_heads % self.p.comparison_tp == 0),
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


class InferenceComparisonWriter(CalculatorWriter):
    """Inference-only workbook with side-by-side TP1 and TP8 analysis."""

    def __init__(self, output: Path, p: Inputs, ratios: list[int]) -> None:
        super().__init__(output, p, ratios)
        self.p_tp1 = replace(p, tp=p.tp)
        self.p_tp8 = replace(p, tp=p.comparison_tp)
        self.scenario_refs: dict[tuple[str, str, str], str] = {}
        self.memory_refs: dict[tuple[str, str], str] = {}

    def _formats(self) -> dict[str, Any]:
        formats = super()._formats()
        formats.update(
            {
                "display": self.workbook.add_format(
                    {"border": 1, "num_format": "#,##0.000", "align": "right"}
                ),
                "display_total": self.workbook.add_format(
                    {
                        "bold": True,
                        "bg_color": "#D6EAF8",
                        "border": 1,
                        "num_format": "#,##0.000",
                    }
                ),
                "formula_text": self.workbook.add_format(
                    {
                        "border": 1,
                        "font_color": "#555555",
                        "font_size": 9,
                        "text_wrap": True,
                        "valign": "top",
                    }
                ),
                "unit": self.workbook.add_format(
                    {"border": 1, "font_color": "#555555", "align": "left"}
                ),
                "percent_display": self.workbook.add_format(
                    {"border": 1, "num_format": "0.00%", "align": "right"}
                ),
            }
        )
        return formats

    @staticmethod
    def _cell(sheet: str, row: int, col: int) -> str:
        return f"'{sheet}'!${xl_col_to_name(col)}${row + 1}"

    @staticmethod
    def _formula_body(formula: str) -> str:
        return formula[1:] if formula.startswith("=") else formula

    def _write_human_value(
        self,
        ws: Any,
        row: int,
        value_col: int,
        unit_col: int,
        raw_col: int,
        raw_formula: str,
        raw_value: float,
        kind: str,
        total: bool = False,
    ) -> None:
        ws.write_formula(row, raw_col, raw_formula, self.formats["number"], raw_value)
        raw_ref = f"{xl_col_to_name(raw_col)}{row + 1}"
        if kind == "percent":
            ws.write_formula(
                row,
                value_col,
                f"={raw_ref}",
                self.formats["percent_display"],
                raw_value,
            )
            ws.write(row, unit_col, "%", self.formats["unit"])
            return
        if kind in {"flops", "bytes", "params"}:
            value_formula, unit_formula = display_formula(raw_ref, kind)
            display_value, display_unit = scaled_value(raw_value, kind)
            ws.write_formula(
                row,
                value_col,
                value_formula,
                self.formats["display_total" if total else "display"],
                display_value,
            )
            ws.write_formula(
                row,
                unit_col,
                unit_formula,
                self.formats["unit"],
                display_unit,
            )
            return
        ws.write_formula(
            row,
            value_col,
            f"={raw_ref}",
            self.formats["display_total" if total else "display"],
            raw_value,
        )
        ws.write(row, unit_col, kind, self.formats["unit"])

    def _scenario_inputs(
        self, mode: str
    ) -> tuple[list[Item], list[Item], dict[str, float]]:
        tp1_items, helpers = scenario_items(self.p_tp1, self.layers, mode)
        tp8_items, _ = scenario_items(self.p_tp8, self.layers, mode)
        if [item.name for item in tp1_items] != [item.name for item in tp8_items]:
            raise AssertionError("TP1 and TP8 item layouts differ")
        return tp1_items, tp8_items, helpers

    def _write_scenario_helpers(
        self, ws: Any, mode: str, prefix: str, helpers: dict[str, float]
    ) -> int:
        ws.write_row(
            2,
            0,
            ["Helper metric", "Excel formula (text)", "Value", "Unit"],
            self.formats["header"],
        )
        if mode == "prefill":
            raw_formula = "=IF(PrefillSequence<=WindowSize,PrefillSequence*(PrefillSequence+1)/2,WindowSize*(WindowSize+1)/2+(PrefillSequence-WindowSize)*WindowSize)"
            short_formula = "=IF(PrefillSequence<IndexTopK*ShortRatio,ShortRatio*INT(PrefillSequence/ShortRatio)*(INT(PrefillSequence/ShortRatio)-1)/2+INT(PrefillSequence/ShortRatio)*(MOD(PrefillSequence,ShortRatio)+1),ShortRatio*IndexTopK*(IndexTopK-1)/2+(PrefillSequence-IndexTopK*ShortRatio+1)*IndexTopK)"
            long_formula = "=LongRatio*INT(PrefillSequence/LongRatio)*(INT(PrefillSequence/LongRatio)-1)/2+INT(PrefillSequence/LongRatio)*(MOD(PrefillSequence,LongRatio)+1)"
            index_formula = "=ShortRatio*INT(PrefillSequence/ShortRatio)*(INT(PrefillSequence/ShortRatio)-1)/2+INT(PrefillSequence/ShortRatio)*(MOD(PrefillSequence,ShortRatio)+1)"
            rows = [
                ("Token rows", f"{prefix}_Rows", "=PrefillBatch*PrefillSequence", helpers["rows"], "token rows"),
                ("Causal raw-window pairs", f"{prefix}_RawPairs", raw_formula, helpers["raw_pairs"], "pairs/sequence"),
                ("Capped short-compressed pairs", f"{prefix}_ShortPairs", short_formula, helpers["short_pairs"], "pairs/sequence"),
                ("Long-compressed pairs", f"{prefix}_LongPairs", long_formula, helpers["long_pairs"], "pairs/sequence"),
                ("Indexer scan pairs", f"{prefix}_IndexPairs", index_formula, helpers["index_pairs"], "pairs/sequence"),
                ("Expert activation probability", f"{prefix}_ExpertActiveProbability", "=1-(1-ActivatedExperts/RoutedExperts)^PF_Rows", helpers["expert_probability"], "probability"),
            ]
        else:
            rows = [
                ("Token rows", f"{prefix}_Rows", "=DecodeBatch*DecodeTokens", helpers["rows"], "token rows"),
                ("Raw-window candidates", f"{prefix}_RawPairs", "=DecodeTokens*MIN(WindowSize,DecodeContext)", helpers["raw_pairs"], "pairs/sequence"),
                ("Short-compressed Top-K candidates", f"{prefix}_ShortPairs", "=DecodeTokens*MIN(IndexTopK,INT(DecodeContext/ShortRatio))", helpers["short_pairs"], "pairs/sequence"),
                ("Long-compressed candidates", f"{prefix}_LongPairs", "=DecodeTokens*INT(DecodeContext/LongRatio)", helpers["long_pairs"], "pairs/sequence"),
                ("Indexer scan candidates", f"{prefix}_IndexPairs", "=DecodeTokens*INT(DecodeContext/ShortRatio)", helpers["index_pairs"], "pairs/sequence"),
                ("Expert activation probability", f"{prefix}_ExpertActiveProbability", "=1-(1-ActivatedExperts/RoutedExperts)^DC_Rows", helpers["expert_probability"], "probability"),
            ]
        row = 3
        for label, name, formula, value, unit in rows:
            ws.write(row, 0, label, self.formats["text"])
            ws.write_string(row, 1, formula, self.formats["formula_text"])
            ws.write_formula(row, 2, formula, self.formats["derived"], value)
            ws.write(row, 3, unit, self.formats["unit"])
            self._define_cell_name(name, ws.name, row, 2)
            row += 1
        return row

    def write_scenario(self, mode: str) -> None:
        is_prefill = mode == "prefill"
        sheet = "Prefill_8K" if is_prefill else "Decode_1M"
        prefix = "PF" if is_prefill else "DC"
        tp1_items, tp8_items, helpers = self._scenario_inputs(mode)
        ws = self.workbook.add_worksheet(sheet)
        ws.hide_gridlines(2)
        ws.freeze_panes(3, 0)
        ws.write(
            0,
            0,
            f"{'8K Prefill' if is_prefill else '1M-context Decode'} - TP1 vs TP8 Inference",
            self.formats["title"],
        )
        ws.write(
            1,
            0,
            "Inference only: no backward, gradients, optimizer, or training-state memory. Scrollable below row 3.",
            self.formats["note"],
        )
        helper_end = self._write_scenario_helpers(ws, mode, prefix, helpers)

        detail_header_row = 31
        detail_first_row = detail_header_row + 1
        detail_last_row = detail_first_row + len(tp1_items) - 1
        detail_first_excel = detail_first_row + 1
        detail_last_excel = detail_last_row + 1
        headers = [
            "Category",
            "Item",
            "Layer scope",
            "FLOPs formula (text)",
            "HBM formula (text)",
            "TP1 FLOPs",
            "TP1 FLOPs unit",
            "TP8 FLOPs/rank",
            "TP8 FLOPs unit",
            "TP1 HBM/rank",
            "TP1 HBM unit",
            "TP8 HBM/rank",
            "TP8 HBM unit",
            "TP1 Interconnect",
            "TP1 interconnect unit",
            "TP8 Interconnect/rank",
            "TP8 interconnect unit",
            "Accounting",
            "Notes",
        ]
        ws.merge_range(
            detail_header_row - 1,
            0,
            detail_header_row - 1,
            len(headers) - 1,
            "Detailed inference calculations",
            self.formats["section"],
        )
        ws.write_row(detail_header_row, 0, headers, self.formats["header"])

        raw_global_col = 19
        raw_tp1_flops_col = 20
        raw_tp8_flops_col = 21
        raw_tp1_read_col = 22
        raw_tp1_write_col = 23
        raw_tp8_read_col = 24
        raw_tp8_write_col = 25
        raw_tp1_network_col = 26
        raw_tp8_network_col = 27
        raw_tp1_hbm_col = 28
        raw_tp8_hbm_col = 29

        for offset, (tp1_item, tp8_item) in enumerate(zip(tp1_items, tp8_items)):
            row = detail_first_row + offset
            tp1_flops = formula_for_tp(tp1_item.rank_flops_formula, "TP1")
            tp8_flops = formula_for_tp(tp8_item.rank_flops_formula, "TP8")
            tp1_read = formula_for_tp(tp1_item.read_formula, "TP1")
            tp1_write = formula_for_tp(tp1_item.write_formula, "TP1")
            tp8_read = formula_for_tp(tp8_item.read_formula, "TP8")
            tp8_write = formula_for_tp(tp8_item.write_formula, "TP8")
            tp1_network = formula_for_tp(tp1_item.network_formula, "TP1")
            tp8_network = formula_for_tp(tp8_item.network_formula, "TP8")
            tp1_hbm = f"=({self._formula_body(tp1_read)})+({self._formula_body(tp1_write)})"
            tp8_hbm = f"=({self._formula_body(tp8_read)})+({self._formula_body(tp8_write)})"

            ws.write(row, 0, tp1_item.category, self.formats["text"])
            ws.write(row, 1, tp1_item.name, self.formats["text"])
            ws.write(row, 2, tp1_item.layer_scope, self.formats["text"])
            ws.write_string(
                row,
                3,
                f"TP1 {tp1_flops}\nTP8 {tp8_flops}",
                self.formats["formula_text"],
            )
            ws.write_string(
                row,
                4,
                f"TP1 read {tp1_read}; write {tp1_write}\nTP8 read {tp8_read}; write {tp8_write}",
                self.formats["formula_text"],
            )
            ws.write_formula(
                row,
                raw_global_col,
                tp1_item.global_flops_formula,
                self.formats["number"],
                tp1_item.global_flops,
            )
            self._write_human_value(
                ws,
                row,
                5,
                6,
                raw_tp1_flops_col,
                tp1_flops,
                tp1_item.rank_flops,
                "flops",
            )
            self._write_human_value(
                ws,
                row,
                7,
                8,
                raw_tp8_flops_col,
                tp8_flops,
                tp8_item.rank_flops,
                "flops",
            )
            ws.write_formula(row, raw_tp1_read_col, tp1_read, self.formats["number"], tp1_item.read_bytes)
            ws.write_formula(row, raw_tp1_write_col, tp1_write, self.formats["number"], tp1_item.write_bytes)
            ws.write_formula(row, raw_tp8_read_col, tp8_read, self.formats["number"], tp8_item.read_bytes)
            ws.write_formula(row, raw_tp8_write_col, tp8_write, self.formats["number"], tp8_item.write_bytes)
            self._write_human_value(
                ws,
                row,
                9,
                10,
                raw_tp1_hbm_col,
                tp1_hbm,
                tp1_item.read_bytes + tp1_item.write_bytes,
                "bytes",
            )
            self._write_human_value(
                ws,
                row,
                11,
                12,
                raw_tp8_hbm_col,
                tp8_hbm,
                tp8_item.read_bytes + tp8_item.write_bytes,
                "bytes",
            )
            self._write_human_value(
                ws,
                row,
                13,
                14,
                raw_tp1_network_col,
                tp1_network,
                tp1_item.network_bytes,
                "bytes",
            )
            self._write_human_value(
                ws,
                row,
                15,
                16,
                raw_tp8_network_col,
                tp8_network,
                tp8_item.network_bytes,
                "bytes",
            )
            ws.write(row, 17, tp1_item.accounting, self.formats["text"])
            ws.write(row, 18, tp1_item.notes, self.formats["text"])

        ws.add_table(
            detail_header_row,
            0,
            detail_last_row,
            len(headers) - 1,
            {
                "name": f"{prefix}_Inference_Detail",
                "style": "Table Style Medium 2",
                "columns": [{"header": header} for header in headers],
            },
        )
        ws.set_column(raw_global_col, raw_tp8_hbm_col, None, None, {"hidden": True})

        summary_start = helper_end + 1
        ws.merge_range(summary_start, 0, summary_start, 5, "Scenario summary", self.formats["section"])
        summary_header = summary_start + 1
        ws.write_row(
            summary_header,
            0,
            ["Metric", "Excel formula (text)", "TP1 value", "Unit", "TP8 value/rank", "Unit"],
            self.formats["header"],
        )

        category_col = "$A"
        accounting_col = "$R"
        tp1_flops_col = "$U"
        tp8_flops_col = "$V"
        tp1_read_col = "$W"
        tp1_write_col = "$X"
        tp8_read_col = "$Y"
        tp8_write_col = "$Z"
        tp1_network_col = "$AA"
        tp8_network_col = "$AB"

        def sum_category(column: str, category: str, accounting: str | None = None) -> str:
            if accounting is None:
                return f'SUMIF({category_col}${detail_first_excel}:{category_col}${detail_last_excel},"{category}",{column}${detail_first_excel}:{column}${detail_last_excel})'
            return f'SUMIFS({column}${detail_first_excel}:{column}${detail_last_excel},{category_col}${detail_first_excel}:{category_col}${detail_last_excel},"{category}",{accounting_col}${detail_first_excel}:{accounting_col}${detail_last_excel},"{accounting}")'

        def category_hbm(read_col: str, write_col: str, category: str) -> str:
            return f"{sum_category(read_col, category)}+{sum_category(write_col, category)}"

        tp1_summary = summarize_items(tp1_items)
        tp8_summary = summarize_items(tp8_items)

        def hbm_value(summary: dict[str, Any], category: str) -> float:
            item = summary["categories"][category]
            return item["hbm_read_bytes_per_rank"] + item["hbm_write_bytes_per_rank"]

        total_tp1_hbm = tp1_summary["total_hbm_read_bytes_per_rank"] + tp1_summary["total_hbm_write_bytes_per_rank"]
        total_tp8_hbm = tp8_summary["total_hbm_read_bytes_per_rank"] + tp8_summary["total_hbm_write_bytes_per_rank"]
        target_ms = self.p.prefill_target_ms if is_prefill else self.p.decode_target_ms
        target_name = "PrefillTargetMs" if is_prefill else "DecodeTargetMs"

        specs: list[tuple[str, str, float, str, float, str]] = [
            ("Attention major FLOPs", f"={sum_category(tp1_flops_col, 'Attention', 'Major')}", tp1_summary["attention_major_flops_per_rank"], f"={sum_category(tp8_flops_col, 'Attention', 'Major')}", tp8_summary["attention_major_flops_per_rank"], "flops"),
            ("MoE major FLOPs", f"={sum_category(tp1_flops_col, 'MoE', 'Major')}", tp1_summary["moe_major_flops_per_rank"], f"={sum_category(tp8_flops_col, 'MoE', 'Major')}", tp8_summary["moe_major_flops_per_rank"], "flops"),
            ("Other inference FLOPs", f"={sum_category(tp1_flops_col, 'Other')}", tp1_summary["categories"]["Other"]["per_rank_flops"], f"={sum_category(tp8_flops_col, 'Other')}", tp8_summary["categories"]["Other"]["per_rank_flops"], "flops"),
            ("Total inference FLOPs", f"=SUM({tp1_flops_col}${detail_first_excel}:{tp1_flops_col}${detail_last_excel})", tp1_summary["total_per_rank_flops"], f"=SUM({tp8_flops_col}${detail_first_excel}:{tp8_flops_col}${detail_last_excel})", tp8_summary["total_per_rank_flops"], "flops"),
            ("Attention HBM traffic", f"={category_hbm(tp1_read_col, tp1_write_col, 'Attention')}", hbm_value(tp1_summary, "Attention"), f"={category_hbm(tp8_read_col, tp8_write_col, 'Attention')}", hbm_value(tp8_summary, "Attention"), "bytes"),
            ("MoE HBM traffic", f"={category_hbm(tp1_read_col, tp1_write_col, 'MoE')}", hbm_value(tp1_summary, "MoE"), f"={category_hbm(tp8_read_col, tp8_write_col, 'MoE')}", hbm_value(tp8_summary, "MoE"), "bytes"),
            ("Other HBM traffic", f"={category_hbm(tp1_read_col, tp1_write_col, 'Other')}", hbm_value(tp1_summary, "Other"), f"={category_hbm(tp8_read_col, tp8_write_col, 'Other')}", hbm_value(tp8_summary, "Other"), "bytes"),
            ("Total HBM traffic", f"=SUM({tp1_read_col}${detail_first_excel}:{tp1_write_col}${detail_last_excel})", total_tp1_hbm, f"=SUM({tp8_read_col}${detail_first_excel}:{tp8_write_col}${detail_last_excel})", total_tp8_hbm, "bytes"),
            ("Interconnect transfer", f"=SUM({tp1_network_col}${detail_first_excel}:{tp1_network_col}${detail_last_excel})", tp1_summary["total_interconnect_bytes_per_rank"], f"=SUM({tp8_network_col}${detail_first_excel}:{tp8_network_col}${detail_last_excel})", tp8_summary["total_interconnect_bytes_per_rank"], "bytes"),
            ("Arithmetic intensity", f"=SUM({tp1_flops_col}${detail_first_excel}:{tp1_flops_col}${detail_last_excel})/(SUM({tp1_read_col}${detail_first_excel}:{tp1_read_col}${detail_last_excel})+SUM({tp1_write_col}${detail_first_excel}:{tp1_write_col}${detail_last_excel}))", tp1_summary["total_per_rank_flops"] / total_tp1_hbm, f"=SUM({tp8_flops_col}${detail_first_excel}:{tp8_flops_col}${detail_last_excel})/(SUM({tp8_read_col}${detail_first_excel}:{tp8_read_col}${detail_last_excel})+SUM({tp8_write_col}${detail_first_excel}:{tp8_write_col}${detail_last_excel}))", tp8_summary["total_per_rank_flops"] / total_tp8_hbm, "FLOPs/byte"),
            ("Required compute at target", f"=SUM({tp1_flops_col}${detail_first_excel}:{tp1_flops_col}${detail_last_excel})/({target_name}/1000)/1E12", tp1_summary["total_per_rank_flops"] / (target_ms / 1000) / 1e12, f"=SUM({tp8_flops_col}${detail_first_excel}:{tp8_flops_col}${detail_last_excel})/({target_name}/1000)/1E12", tp8_summary["total_per_rank_flops"] / (target_ms / 1000) / 1e12, "TFLOP/s"),
            ("Required HBM at target", f"=(SUM({tp1_read_col}${detail_first_excel}:{tp1_read_col}${detail_last_excel})+SUM({tp1_write_col}${detail_first_excel}:{tp1_write_col}${detail_last_excel}))/({target_name}/1000)/1E9", total_tp1_hbm / (target_ms / 1000) / 1e9, f"=(SUM({tp8_read_col}${detail_first_excel}:{tp8_read_col}${detail_last_excel})+SUM({tp8_write_col}${detail_first_excel}:{tp8_write_col}${detail_last_excel}))/({target_name}/1000)/1E9", total_tp8_hbm / (target_ms / 1000) / 1e9, "GB/s"),
        ]

        raw_tp1_summary_col = 30
        raw_tp8_summary_col = 31
        for offset, (label, tp1_formula, tp1_value, tp8_formula, tp8_value, kind) in enumerate(specs):
            row = summary_header + 1 + offset
            ws.write(row, 0, label, self.formats["text"])
            ws.write_string(
                row,
                1,
                f"TP1 {tp1_formula}\nTP8 {tp8_formula}",
                self.formats["formula_text"],
            )
            self._write_human_value(
                ws,
                row,
                2,
                3,
                raw_tp1_summary_col,
                tp1_formula,
                tp1_value,
                kind,
                label.startswith("Total"),
            )
            self._write_human_value(
                ws,
                row,
                4,
                5,
                raw_tp8_summary_col,
                tp8_formula,
                tp8_value,
                kind,
                label.startswith("Total"),
            )
            self.scenario_refs[(prefix, "TP1", label)] = self._cell(sheet, row, raw_tp1_summary_col)
            self.scenario_refs[(prefix, "TP8", label)] = self._cell(sheet, row, raw_tp8_summary_col)

        ws.set_column(raw_tp1_summary_col, raw_tp8_summary_col, None, None, {"hidden": True})
        ws.set_column("A:A", 23)
        ws.set_column("B:B", 64)
        ws.set_column("C:C", 24)
        ws.set_column("D:D", 52)
        ws.set_column("E:E", 52)
        ws.set_column("F:P", 16)
        ws.set_column("Q:Q", 13)
        ws.set_column("R:R", 13)
        ws.set_column("S:S", 60)

    def write_memory(self) -> dict[str, float]:
        sheet = "Memory"
        ws = self.workbook.add_worksheet(sheet)
        ws.hide_gridlines(2)
        ws.freeze_panes(3, 0)
        ws.write(0, 0, "TP1 / TP8 每 Rank 推理参数与 KV 容量", self.formats["title"])
        ws.write(1, 0, "仅统计推理驻留数据；不含梯度、优化器、反向激活和训练状态。", self.formats["note"])
        headers = [
            "大类",
            "参数组件",
            "参数量公式（文本）",
            "容量公式（文本）",
            "全局参数量",
            "全局参数量单位",
            "TP1 参数量/rank",
            "TP1 参数量单位",
            "TP8 参数量/rank",
            "TP8 参数量单位",
            "TP1 容量/rank",
            "TP1 容量单位",
            "TP8 容量/rank",
            "TP8 容量单位",
            "说明",
        ]
        ws.write_row(2, 0, headers, self.formats["header"])
        tp1_components = parameter_components(self.p_tp1, self.layers)
        tp8_components = parameter_components(self.p_tp8, self.layers)
        detail_first_row = 3
        detail_last_row = detail_first_row + len(tp1_components) - 1
        raw_global_count_col = 15
        raw_tp1_count_col = 16
        raw_tp8_count_col = 17
        raw_global_bytes_col = 18
        raw_tp1_bytes_col = 19
        raw_tp8_bytes_col = 20

        for offset, (tp1_component, tp8_component) in enumerate(
            zip(tp1_components, tp8_components)
        ):
            row = detail_first_row + offset
            tp1_count_formula = "=" + formula_for_tp(
                tp1_component.rank_count_formula, "TP1"
            )
            tp8_count_formula = "=" + formula_for_tp(
                tp8_component.rank_count_formula, "TP8"
            )
            tp1_bytes_formula = "=" + formula_for_tp(
                tp1_component.rank_bytes_formula, "TP1"
            )
            tp8_bytes_formula = "=" + formula_for_tp(
                tp8_component.rank_bytes_formula, "TP8"
            )
            ws.write(row, 0, tp1_component.category, self.formats["text"])
            ws.write(row, 1, tp1_component.name, self.formats["text"])
            ws.write_string(
                row,
                2,
                f"Global ={tp1_component.global_count_formula}\nTP1 {tp1_count_formula}\nTP8 {tp8_count_formula}",
                self.formats["formula_text"],
            )
            ws.write_string(
                row,
                3,
                f"Global ={tp1_component.global_bytes_formula}\nTP1 {tp1_bytes_formula}\nTP8 {tp8_bytes_formula}",
                self.formats["formula_text"],
            )
            self._write_human_value(
                ws,
                row,
                4,
                5,
                raw_global_count_col,
                "=" + tp1_component.global_count_formula,
                tp1_component.global_count,
                "params",
            )
            self._write_human_value(
                ws,
                row,
                6,
                7,
                raw_tp1_count_col,
                tp1_count_formula,
                tp1_component.rank_count,
                "params",
            )
            self._write_human_value(
                ws,
                row,
                8,
                9,
                raw_tp8_count_col,
                tp8_count_formula,
                tp8_component.rank_count,
                "params",
            )
            ws.write_formula(
                row,
                raw_global_bytes_col,
                "=" + tp1_component.global_bytes_formula,
                self.formats["number"],
                tp1_component.global_bytes,
            )
            self._write_human_value(
                ws,
                row,
                10,
                11,
                raw_tp1_bytes_col,
                tp1_bytes_formula,
                tp1_component.rank_bytes,
                "bytes",
            )
            self._write_human_value(
                ws,
                row,
                12,
                13,
                raw_tp8_bytes_col,
                tp8_bytes_formula,
                tp8_component.rank_bytes,
                "bytes",
            )
            ws.write(row, 14, tp1_component.notes, self.formats["text"])

        ws.add_table(
            2,
            0,
            detail_last_row,
            len(headers) - 1,
            {
                "name": "Inference_Parameter_Detail",
                "style": "Table Style Medium 2",
                "columns": [{"header": header} for header in headers],
            },
        )
        ws.set_column(
            raw_global_count_col,
            raw_tp8_bytes_col,
            None,
            None,
            {"hidden": True},
        )

        summary_start = detail_last_row + 2
        ws.merge_range(
            summary_start,
            0,
            summary_start,
            8,
            "参数量与参数容量汇总",
            self.formats["section"],
        )
        summary_header = summary_start + 1
        ws.write_row(
            summary_header,
            0,
            [
                "大类",
                "TP1 参数量/rank",
                "单位",
                "TP8 参数量/rank",
                "单位",
                "TP1 容量/rank",
                "单位",
                "TP8 容量/rank",
                "单位",
            ],
            self.formats["header"],
        )
        detail_first_excel = detail_first_row + 1
        detail_last_excel = detail_last_row + 1
        category_values: dict[str, dict[str, float]] = {}
        raw_summary_cols = (15, 16, 17, 18)
        for offset, category in enumerate(("Attention", "MoE", "Other", "Total")):
            row = summary_header + 1 + offset
            if category == "Total":
                formulas = (
                    f"=SUM($Q${detail_first_excel}:$Q${detail_last_excel})",
                    f"=SUM($R${detail_first_excel}:$R${detail_last_excel})",
                    f"=SUM($T${detail_first_excel}:$T${detail_last_excel})",
                    f"=SUM($U${detail_first_excel}:$U${detail_last_excel})",
                )
                selected_tp1 = tp1_components
                selected_tp8 = tp8_components
            else:
                formulas = (
                    f'=SUMIF($A${detail_first_excel}:$A${detail_last_excel},"{category}",$Q${detail_first_excel}:$Q${detail_last_excel})',
                    f'=SUMIF($A${detail_first_excel}:$A${detail_last_excel},"{category}",$R${detail_first_excel}:$R${detail_last_excel})',
                    f'=SUMIF($A${detail_first_excel}:$A${detail_last_excel},"{category}",$T${detail_first_excel}:$T${detail_last_excel})',
                    f'=SUMIF($A${detail_first_excel}:$A${detail_last_excel},"{category}",$U${detail_first_excel}:$U${detail_last_excel})',
                )
                selected_tp1 = [c for c in tp1_components if c.category == category]
                selected_tp8 = [c for c in tp8_components if c.category == category]
            values = {
                "tp1_count": sum(c.rank_count for c in selected_tp1),
                "tp8_count": sum(c.rank_count for c in selected_tp8),
                "tp1_bytes": sum(c.rank_bytes for c in selected_tp1),
                "tp8_bytes": sum(c.rank_bytes for c in selected_tp8),
            }
            category_values[category] = values
            ws.write(row, 0, category, self.formats["total" if category == "Total" else "text"])
            self._write_human_value(ws, row, 1, 2, raw_summary_cols[0], formulas[0], values["tp1_count"], "params", category == "Total")
            self._write_human_value(ws, row, 3, 4, raw_summary_cols[1], formulas[1], values["tp8_count"], "params", category == "Total")
            self._write_human_value(ws, row, 5, 6, raw_summary_cols[2], formulas[2], values["tp1_bytes"], "bytes", category == "Total")
            self._write_human_value(ws, row, 7, 8, raw_summary_cols[3], formulas[3], values["tp8_bytes"], "bytes", category == "Total")
            self.memory_refs[("TP1", f"{category} Parameter Count")] = self._cell(sheet, row, raw_summary_cols[0])
            self.memory_refs[("TP8", f"{category} Parameter Count")] = self._cell(sheet, row, raw_summary_cols[1])
            self.memory_refs[("TP1", f"{category} Parameter Capacity")] = self._cell(sheet, row, raw_summary_cols[2])
            self.memory_refs[("TP8", f"{category} Parameter Capacity")] = self._cell(sheet, row, raw_summary_cols[3])

        cache_start = summary_header + 7
        ws.merge_range(cache_start, 0, cache_start, 6, "KV Cache 与 Compressor State（每 Rank，TP 间复制）", self.formats["section"])
        cache_header = cache_start + 1
        ws.write_row(
            cache_header,
            0,
            ["项目", "公式（文本）", "TP1/rank", "单位", "TP8/rank", "单位", "说明"],
            self.formats["header"],
        )
        prefill_cache = cache_values(
            self.p,
            self.layers,
            self.p.prefill_sequence,
            self.p.prefill_batch,
        )
        decode_cache = cache_values(
            self.p,
            self.layers,
            self.p.decode_context,
            self.p.decode_batch,
        )
        allocated_prefill = cache_values(
            self.p, self.layers, self.p.max_context, self.p.prefill_batch
        )
        allocated_decode = cache_values(
            self.p, self.layers, self.p.max_context, self.p.decode_batch
        )
        cache_specs = [
            ("Prefill effective main KV", "=PrefillBatch*HeadDim*BF16Bytes*(WindowLayers*MIN(PrefillSequence,WindowSize)+ShortLayers*(MIN(PrefillSequence,WindowSize)+INT(PrefillSequence/ShortRatio))+LongLayers*(MIN(PrefillSequence,WindowSize)+INT(PrefillSequence/LongRatio)))", prefill_cache["main"], "8K Prefill 已生成的主 Attention KV。"),
            ("Prefill effective Indexer KV", "=PrefillBatch*ShortLayers*INT(PrefillSequence/ShortRatio)*IndexHeadDim*BF16Bytes", prefill_cache["indexer"], "8K Prefill 已生成的 ratio-4 Indexer KV。"),
            ("Prefill compressor states", "=PrefillBatch*FP32Bytes*(ShortLayers*2*(2*ShortRatio)*(2*HeadDim)+LongLayers*2*LongRatio*HeadDim+ShortLayers*2*(2*ShortRatio)*(2*IndexHeadDim))", prefill_cache["states"], "主 Compressor 与 Indexer Compressor 状态。"),
            ("Prefill preallocated KV + states", "=PrefillBatch*(HeadDim*BF16Bytes*(WindowLayers*WindowSize+ShortLayers*(WindowSize+INT(MaxContext/ShortRatio))+LongLayers*(WindowSize+INT(MaxContext/LongRatio)))+ShortLayers*INT(MaxContext/ShortRatio)*IndexHeadDim*BF16Bytes)+PrefillBatch*FP32Bytes*(ShortLayers*2*(2*ShortRatio)*(2*HeadDim)+LongLayers*2*LongRatio*HeadDim+ShortLayers*2*(2*ShortRatio)*(2*IndexHeadDim))", allocated_prefill["total"], "按 MaxContext 预分配，适合评估实际设备容量。"),
            ("Decode effective main KV", "=DecodeBatch*HeadDim*BF16Bytes*(WindowLayers*MIN(DecodeContext,WindowSize)+ShortLayers*(MIN(DecodeContext,WindowSize)+INT(DecodeContext/ShortRatio))+LongLayers*(MIN(DecodeContext,WindowSize)+INT(DecodeContext/LongRatio)))", decode_cache["main"], "DecodeContext 下的主 Attention KV。"),
            ("Decode effective Indexer KV", "=DecodeBatch*ShortLayers*INT(DecodeContext/ShortRatio)*IndexHeadDim*BF16Bytes", decode_cache["indexer"], "DecodeContext 下的 ratio-4 Indexer KV。"),
            ("Decode compressor states", "=DecodeBatch*FP32Bytes*(ShortLayers*2*(2*ShortRatio)*(2*HeadDim)+LongLayers*2*LongRatio*HeadDim+ShortLayers*2*(2*ShortRatio)*(2*IndexHeadDim))", decode_cache["states"], "Decode Batch 对应的 Compressor 状态。"),
            ("Decode preallocated KV + states", "=DecodeBatch*(HeadDim*BF16Bytes*(WindowLayers*WindowSize+ShortLayers*(WindowSize+INT(MaxContext/ShortRatio))+LongLayers*(WindowSize+INT(MaxContext/LongRatio)))+ShortLayers*INT(MaxContext/ShortRatio)*IndexHeadDim*BF16Bytes)+DecodeBatch*FP32Bytes*(ShortLayers*2*(2*ShortRatio)*(2*HeadDim)+LongLayers*2*LongRatio*HeadDim+ShortLayers*2*(2*ShortRatio)*(2*IndexHeadDim))", allocated_decode["total"], "按 MaxContext 预分配，KV 在每个 TP Rank 复制。"),
        ]
        raw_tp1_cache_col = 15
        raw_tp8_cache_col = 16
        for offset, (label, formula, value, note) in enumerate(cache_specs):
            row = cache_header + 1 + offset
            ws.write(row, 0, label, self.formats["text"])
            ws.write_string(row, 1, formula, self.formats["formula_text"])
            self._write_human_value(ws, row, 2, 3, raw_tp1_cache_col, formula, value, "bytes")
            self._write_human_value(ws, row, 4, 5, raw_tp8_cache_col, formula, value, "bytes")
            ws.write(row, 6, note, self.formats["text"])
            self.memory_refs[("TP1", label)] = self._cell(sheet, row, raw_tp1_cache_col)
            self.memory_refs[("TP8", label)] = self._cell(sheet, row, raw_tp8_cache_col)

        rank_start = cache_header + len(cache_specs) + 3
        ws.merge_range(rank_start, 0, rank_start, 9, "逐 Rank 设备驻留容量（默认均匀分片）", self.formats["section"])
        rank_header = rank_start + 1
        rank_headers = [
            "配置",
            "Rank",
            "Attention 参数 GB",
            "MoE 参数 GB",
            "Other 参数 GB",
            "参数总计 GB",
            "Decode KV+State GB",
            "总驻留 GB",
            "逻辑参数量 G",
            "有效",
        ]
        ws.write_row(rank_header, 0, rank_headers, self.formats["header"])
        rank_row = rank_header + 1
        for label, size_name, p_config in (
            ("TP1", "TP1Size", self.p_tp1),
            ("TP8", "TP8Size", self.p_tp8),
        ):
            active_size = p_config.tp
            for rank in range(MAX_RANK_ROWS):
                active = rank < active_size
                ws.write(rank_row, 0, label, self.formats["text"])
                ws.write_number(rank_row, 1, rank, self.formats["integer"])
                attention_ref = self.memory_refs[(label, "Attention Parameter Capacity")]
                moe_ref = self.memory_refs[(label, "MoE Parameter Capacity")]
                other_ref = self.memory_refs[(label, "Other Parameter Capacity")]
                total_ref = self.memory_refs[(label, "Total Parameter Capacity")]
                kv_ref = self.memory_refs[(label, "Decode preallocated KV + states")]
                count_ref = self.memory_refs[(label, "Total Parameter Count")]
                refs = (attention_ref, moe_ref, other_ref, total_ref, kv_ref)
                values = (
                    category_values["Attention"][f"{label.lower()}_bytes"] / 1e9,
                    category_values["MoE"][f"{label.lower()}_bytes"] / 1e9,
                    category_values["Other"][f"{label.lower()}_bytes"] / 1e9,
                    category_values["Total"][f"{label.lower()}_bytes"] / 1e9,
                    allocated_decode["total"] / 1e9,
                )
                for col, (ref, value) in enumerate(zip(refs, values), start=2):
                    ws.write_formula(
                        rank_row,
                        col,
                        f'=IF(B{rank_row + 1}<{size_name},{ref}/1E9,"")',
                        self.formats["number"],
                        value if active else "",
                    )
                total_resident = values[3] + values[4]
                ws.write_formula(
                    rank_row,
                    7,
                    f'=IF(B{rank_row + 1}<{size_name},F{rank_row + 1}+G{rank_row + 1},"")',
                    self.formats["number"],
                    total_resident if active else "",
                )
                count_value = category_values["Total"][f"{label.lower()}_count"] / 1e9
                ws.write_formula(
                    rank_row,
                    8,
                    f'=IF(B{rank_row + 1}<{size_name},{count_ref}/1E9,"")',
                    self.formats["number"],
                    count_value if active else "",
                )
                ws.write_formula(
                    rank_row,
                    9,
                    f'=IF(B{rank_row + 1}<{size_name},"YES","")',
                    self.formats["text"],
                    "YES" if active else "",
                )
                rank_row += 1
        ws.add_table(
            rank_header,
            0,
            rank_row - 1,
            len(rank_headers) - 1,
            {
                "name": "Per_Rank_Inference_Memory",
                "style": "Table Style Medium 4",
                "columns": [{"header": header} for header in rank_headers],
            },
        )
        ws.set_column(15, 20, None, None, {"hidden": True})
        ws.set_column("A:A", 18)
        ws.set_column("B:B", 32)
        ws.set_column("C:D", 72)
        ws.set_column("E:N", 18)
        ws.set_column("O:O", 58)

        return {
            "tp1_parameter_total": category_values["Total"]["tp1_bytes"],
            "tp8_parameter_total": category_values["Total"]["tp8_bytes"],
            "tp1_parameter_count": category_values["Total"]["tp1_count"],
            "tp8_parameter_count": category_values["Total"]["tp8_count"],
            "tp1_attention_parameter": category_values["Attention"]["tp1_bytes"],
            "tp8_attention_parameter": category_values["Attention"]["tp8_bytes"],
            "tp1_moe_parameter": category_values["MoE"]["tp1_bytes"],
            "tp8_moe_parameter": category_values["MoE"]["tp8_bytes"],
            "tp1_other_parameter": category_values["Other"]["tp1_bytes"],
            "tp8_other_parameter": category_values["Other"]["tp8_bytes"],
            "parameter_total": category_values["Total"]["tp1_bytes"],
            "prefill_effective_cache": prefill_cache["total"],
            "prefill_allocated_cache": allocated_prefill["total"],
            "decode_effective_cache": decode_cache["total"],
            "decode_allocated_cache": allocated_decode["total"],
        }

    def write_summary(self, memory: dict[str, float]) -> None:
        sheet = "Summary"
        ws = self.workbook.add_worksheet(sheet)
        ws.activate()
        ws.hide_gridlines(2)
        ws.freeze_panes(4, 1)
        ws.write(0, 0, "每 Rank 推理硬件资源汇总", self.formats["title"])
        ws.write(1, 0, "面向 TP1 / TP8 硬件选型：计算量、HBM 流量、参数量、参数容量与 KV Cache。", self.formats["note"])
        headers = [
            "资源项目",
            "Prefill TP1",
            "单位",
            "Prefill TP8/rank",
            "单位",
            "Decode TP1",
            "单位",
            "Decode TP8/rank",
            "单位",
            "说明",
        ]
        ws.write_row(3, 0, headers, self.formats["header"])
        raw_cols = (10, 11, 12, 13)

        def scenario_ref(prefix: str, tp: str, label: str) -> str:
            return self.scenario_refs[(prefix, tp, label)]

        def memory_ref(tp: str, label: str) -> str:
            return self.memory_refs[(tp, label)]

        rows: list[tuple[str, tuple[str, float], tuple[str, float], tuple[str, float], tuple[str, float], str, str]] = []
        tp1_pf, _ = self._scenario_inputs("prefill")[:2]
        tp1_dc, _ = self._scenario_inputs("decode")[:2]
        pf1 = summarize_items(tp1_pf)
        pf8 = summarize_items(self._scenario_inputs("prefill")[1])
        dc1 = summarize_items(tp1_dc)
        dc8 = summarize_items(self._scenario_inputs("decode")[1])

        def category_hbm(summary: dict[str, Any], category: str) -> float:
            data = summary["categories"][category]
            return data["hbm_read_bytes_per_rank"] + data["hbm_write_bytes_per_rank"]

        for label, source_label, category in (
            ("Attention 计算量", "Attention major FLOPs", "Attention"),
            ("MoE 计算量", "MoE major FLOPs", "MoE"),
            ("Other 推理计算量", "Other inference FLOPs", "Other"),
            ("总推理计算量", "Total inference FLOPs", "Total"),
        ):
            if category == "Attention":
                values = (pf1["attention_major_flops_per_rank"], pf8["attention_major_flops_per_rank"], dc1["attention_major_flops_per_rank"], dc8["attention_major_flops_per_rank"])
            elif category == "MoE":
                values = (pf1["moe_major_flops_per_rank"], pf8["moe_major_flops_per_rank"], dc1["moe_major_flops_per_rank"], dc8["moe_major_flops_per_rank"])
            elif category == "Other":
                values = tuple(summary["categories"]["Other"]["per_rank_flops"] for summary in (pf1, pf8, dc1, dc8))
            else:
                values = tuple(summary["total_per_rank_flops"] for summary in (pf1, pf8, dc1, dc8))
            rows.append((label, (scenario_ref("PF", "TP1", source_label), values[0]), (scenario_ref("PF", "TP8", source_label), values[1]), (scenario_ref("DC", "TP1", source_label), values[2]), (scenario_ref("DC", "TP8", source_label), values[3]), "flops", "单次 Prefill 或 Decode step 的每 Rank 逻辑 FLOPs。"))

        for label, source_label, category in (
            ("Attention HBM 流量", "Attention HBM traffic", "Attention"),
            ("MoE HBM 流量", "MoE HBM traffic", "MoE"),
            ("Other HBM 流量", "Other HBM traffic", "Other"),
            ("总 HBM 流量", "Total HBM traffic", "Total"),
        ):
            values = (
                category_hbm(pf1, category) if category != "Total" else pf1["total_hbm_read_bytes_per_rank"] + pf1["total_hbm_write_bytes_per_rank"],
                category_hbm(pf8, category) if category != "Total" else pf8["total_hbm_read_bytes_per_rank"] + pf8["total_hbm_write_bytes_per_rank"],
                category_hbm(dc1, category) if category != "Total" else dc1["total_hbm_read_bytes_per_rank"] + dc1["total_hbm_write_bytes_per_rank"],
                category_hbm(dc8, category) if category != "Total" else dc8["total_hbm_read_bytes_per_rank"] + dc8["total_hbm_write_bytes_per_rank"],
            )
            rows.append((label, (scenario_ref("PF", "TP1", source_label), values[0]), (scenario_ref("PF", "TP8", source_label), values[1]), (scenario_ref("DC", "TP1", source_label), values[2]), (scenario_ref("DC", "TP8", source_label), values[3]), "bytes", "本地逻辑 HBM 读写量，不含缓存复用和算子融合。"))

        for label, source_label, category in (
            ("Attention 所需 HBM 带宽", "Attention HBM traffic", "Attention"),
            ("MoE 所需 HBM 带宽", "MoE HBM traffic", "MoE"),
            ("Other 所需 HBM 带宽", "Other HBM traffic", "Other"),
        ):
            pf_values = (
                category_hbm(pf1, category) / (self.p.prefill_target_ms / 1000) / 1e9,
                category_hbm(pf8, category) / (self.p.prefill_target_ms / 1000) / 1e9,
            )
            dc_values = (
                category_hbm(dc1, category) / (self.p.decode_target_ms / 1000) / 1e9,
                category_hbm(dc8, category) / (self.p.decode_target_ms / 1000) / 1e9,
            )
            rows.append(
                (
                    label,
                    (f"{scenario_ref('PF', 'TP1', source_label)}/(PrefillTargetMs/1000)/1E9", pf_values[0]),
                    (f"{scenario_ref('PF', 'TP8', source_label)}/(PrefillTargetMs/1000)/1E9", pf_values[1]),
                    (f"{scenario_ref('DC', 'TP1', source_label)}/(DecodeTargetMs/1000)/1E9", dc_values[0]),
                    (f"{scenario_ref('DC', 'TP8', source_label)}/(DecodeTargetMs/1000)/1E9", dc_values[1]),
                    "GB/s",
                    "按 Parameters 中目标时延换算；是静态需求值，不是实测带宽。",
                )
            )

        pf_required_compute = (
            pf1["total_per_rank_flops"] / (self.p.prefill_target_ms / 1000) / 1e12,
            pf8["total_per_rank_flops"] / (self.p.prefill_target_ms / 1000) / 1e12,
        )
        dc_required_compute = (
            dc1["total_per_rank_flops"] / (self.p.decode_target_ms / 1000) / 1e12,
            dc8["total_per_rank_flops"] / (self.p.decode_target_ms / 1000) / 1e12,
        )
        rows.append(
            (
                "目标时延所需计算性能",
                (scenario_ref("PF", "TP1", "Required compute at target"), pf_required_compute[0]),
                (scenario_ref("PF", "TP8", "Required compute at target"), pf_required_compute[1]),
                (scenario_ref("DC", "TP1", "Required compute at target"), dc_required_compute[0]),
                (scenario_ref("DC", "TP8", "Required compute at target"), dc_required_compute[1]),
                "TFLOP/s",
                "按目标时延折算的每 Rank 最低计算吞吐需求。",
            )
        )
        interconnect_values = (
            pf1["total_interconnect_bytes_per_rank"],
            pf8["total_interconnect_bytes_per_rank"],
            dc1["total_interconnect_bytes_per_rank"],
            dc8["total_interconnect_bytes_per_rank"],
        )
        rows.append(
            (
                "卡间互连传输量",
                (scenario_ref("PF", "TP1", "Interconnect transfer"), interconnect_values[0]),
                (scenario_ref("PF", "TP8", "Interconnect transfer"), interconnect_values[1]),
                (scenario_ref("DC", "TP1", "Interconnect transfer"), interconnect_values[2]),
                (scenario_ref("DC", "TP8", "Interconnect transfer"), interconnect_values[3]),
                "bytes",
                "Ring 集合通信每 Rank 发送加接收；TP1 为 0。",
            )
        )
        interconnect_bandwidth = (
            interconnect_values[0] / (self.p.prefill_target_ms / 1000) / 1e9,
            interconnect_values[1] / (self.p.prefill_target_ms / 1000) / 1e9,
            interconnect_values[2] / (self.p.decode_target_ms / 1000) / 1e9,
            interconnect_values[3] / (self.p.decode_target_ms / 1000) / 1e9,
        )
        rows.append(
            (
                "目标时延所需互连带宽",
                (f"{scenario_ref('PF', 'TP1', 'Interconnect transfer')}/(PrefillTargetMs/1000)/1E9", interconnect_bandwidth[0]),
                (f"{scenario_ref('PF', 'TP8', 'Interconnect transfer')}/(PrefillTargetMs/1000)/1E9", interconnect_bandwidth[1]),
                (f"{scenario_ref('DC', 'TP1', 'Interconnect transfer')}/(DecodeTargetMs/1000)/1E9", interconnect_bandwidth[2]),
                (f"{scenario_ref('DC', 'TP8', 'Interconnect transfer')}/(DecodeTargetMs/1000)/1E9", interconnect_bandwidth[3]),
                "GB/s",
                "未考虑通信与计算重叠、拓扑和协议效率。",
            )
        )

        for label, memory_label, kind, note in (
            ("Attention 逻辑参数量", "Attention Parameter Count", "params", "Attention 投影、Compressor、Indexer。"),
            ("MoE 逻辑参数量", "MoE Parameter Count", "params", "Routed/Shared Expert 与 Router。"),
            ("Other 逻辑参数量", "Other Parameter Count", "params", "Embedding、LM Head、HC 与 Norm。"),
            ("总逻辑参数量", "Total Parameter Count", "params", "不包含 MTP。"),
            ("Attention 参数容量", "Attention Parameter Capacity", "bytes", "按推理 dtype 与量化 scale 计算。"),
            ("MoE 参数容量", "MoE Parameter Capacity", "bytes", "Routed FP4、Shared FP8、Router 复制。"),
            ("Other 参数容量", "Other Parameter Capacity", "bytes", "Embedding/LM Head/HC/Norm。"),
            ("总参数容量", "Total Parameter Capacity", "bytes", "每 Rank 静态参数驻留。"),
        ):
            tp1_ref = memory_ref("TP1", memory_label)
            tp8_ref = memory_ref("TP8", memory_label)
            tp1_key = "tp1_count" if kind == "params" else "tp1_bytes"
            tp8_key = "tp8_count" if kind == "params" else "tp8_bytes"
            category = memory_label.split()[0]
            if category not in {"Attention", "MoE", "Other"}:
                category = "Total"
            tp1_value = parameter_components(self.p_tp1, self.layers)
            tp8_value = parameter_components(self.p_tp8, self.layers)
            selected1 = tp1_value if category == "Total" else [c for c in tp1_value if c.category == category]
            selected8 = tp8_value if category == "Total" else [c for c in tp8_value if c.category == category]
            value1 = sum(c.rank_count if kind == "params" else c.rank_bytes for c in selected1)
            value8 = sum(c.rank_count if kind == "params" else c.rank_bytes for c in selected8)
            rows.append((label, (tp1_ref, value1), (tp8_ref, value8), (tp1_ref, value1), (tp8_ref, value8), kind, note))

        kv_rows = (
            ("有效主 KV Cache", "Prefill effective main KV", "Decode effective main KV"),
            ("有效 Indexer KV Cache", "Prefill effective Indexer KV", "Decode effective Indexer KV"),
            ("Compressor State", "Prefill compressor states", "Decode compressor states"),
            ("预分配 KV + State", "Prefill preallocated KV + states", "Decode preallocated KV + states"),
        )
        for label, pf_label, dc_label in kv_rows:
            pf_value = (
                self.p.prefill_batch
                and cache_values(self.p, self.layers, self.p.prefill_sequence, self.p.prefill_batch)
            )
            dc_value = cache_values(self.p, self.layers, self.p.decode_context, self.p.decode_batch)
            if "main" in pf_label:
                pf_number, dc_number = pf_value["main"], dc_value["main"]
            elif "Indexer" in pf_label:
                pf_number, dc_number = pf_value["indexer"], dc_value["indexer"]
            elif "compressor" in pf_label:
                pf_number, dc_number = pf_value["states"], dc_value["states"]
            else:
                pf_number = cache_values(self.p, self.layers, self.p.max_context, self.p.prefill_batch)["total"]
                dc_number = cache_values(self.p, self.layers, self.p.max_context, self.p.decode_batch)["total"]
            rows.append((label, (memory_ref("TP1", pf_label), pf_number), (memory_ref("TP8", pf_label), pf_number), (memory_ref("TP1", dc_label), dc_number), (memory_ref("TP8", dc_label), dc_number), "bytes", "当前实现中 KV 与 Compressor State 在 TP Rank 间复制。"))

        parameter_tp1 = memory["tp1_parameter_total"]
        parameter_tp8 = memory["tp8_parameter_total"]
        pf_alloc = memory["prefill_allocated_cache"]
        dc_alloc = memory["decode_allocated_cache"]
        rows.append(("总驻留容量", (f"{memory_ref('TP1', 'Total Parameter Capacity')}+{memory_ref('TP1', 'Prefill preallocated KV + states')}", parameter_tp1 + pf_alloc), (f"{memory_ref('TP8', 'Total Parameter Capacity')}+{memory_ref('TP8', 'Prefill preallocated KV + states')}", parameter_tp8 + pf_alloc), (f"{memory_ref('TP1', 'Total Parameter Capacity')}+{memory_ref('TP1', 'Decode preallocated KV + states')}", parameter_tp1 + dc_alloc), (f"{memory_ref('TP8', 'Total Parameter Capacity')}+{memory_ref('TP8', 'Decode preallocated KV + states')}", parameter_tp8 + dc_alloc), "bytes", "参数 + 按 MaxContext 预分配的 KV/State；不含临时 workspace。"))

        for offset, (label, pf1_spec, pf8_spec, dc1_spec, dc8_spec, kind, note) in enumerate(rows):
            row = 4 + offset
            ws.write(row, 0, label, self.formats["total" if label in {"总推理计算量", "总 HBM 流量", "总参数容量", "总驻留容量"} else "text"])
            for pair_index, (formula_ref, value) in enumerate((pf1_spec, pf8_spec, dc1_spec, dc8_spec)):
                raw_col = raw_cols[pair_index]
                formula = formula_ref if formula_ref.startswith("=") else f"={formula_ref}"
                value_col = 1 + pair_index * 2
                unit_col = value_col + 1
                self._write_human_value(ws, row, value_col, unit_col, raw_col, formula, value, kind, label.startswith("总"))
            ws.write(row, 9, note, self.formats["text"])

        ws.set_column(raw_cols[0], raw_cols[-1], None, None, {"hidden": True})
        ws.set_column("A:A", 29)
        ws.set_column("B:I", 17)
        ws.set_column("J:J", 62)

    def write_comparison(self, memory: dict[str, float]) -> None:
        ws = self.workbook.add_worksheet("Comparison")
        ws.hide_gridlines(2)
        ws.write(0, 0, "TP1 / TP8 推理资源对比图", self.formats["title"])
        ws.write(1, 0, "所有图表使用线性坐标与固定 M/G/T 单位，不显示科学计数法。", self.formats["note"])
        pf1 = summarize_items(self._scenario_inputs("prefill")[0])
        pf8 = summarize_items(self._scenario_inputs("prefill")[1])
        dc1 = summarize_items(self._scenario_inputs("decode")[0])
        dc8 = summarize_items(self._scenario_inputs("decode")[1])

        def category_value(summary: dict[str, Any], category: str, metric: str) -> float:
            if metric == "flops":
                if category == "Attention":
                    return summary["attention_major_flops_per_rank"]
                if category == "MoE":
                    return summary["moe_major_flops_per_rank"]
                return summary["categories"]["Other"]["per_rank_flops"]
            data = summary["categories"][category]
            return data["hbm_read_bytes_per_rank"] + data["hbm_write_bytes_per_rank"]

        ws.write_row(3, 0, ["大类", "Prefill TP1 TFLOPs", "Prefill TP8 TFLOPs", "Decode TP1 GFLOPs", "Decode TP8 GFLOPs"], self.formats["header"])
        for offset, category in enumerate(("Attention", "MoE", "Other")):
            row = 4 + offset
            ws.write(row, 0, category, self.formats["text"])
            values = (
                category_value(pf1, category, "flops") / 1e12,
                category_value(pf8, category, "flops") / 1e12,
                category_value(dc1, category, "flops") / 1e9,
                category_value(dc8, category, "flops") / 1e9,
            )
            refs = (
                self.scenario_refs[("PF", "TP1", f"{category if category != 'Other' else 'Other inference'} {'major FLOPs' if category != 'Other' else 'FLOPs'}")],
                self.scenario_refs[("PF", "TP8", f"{category if category != 'Other' else 'Other inference'} {'major FLOPs' if category != 'Other' else 'FLOPs'}")],
                self.scenario_refs[("DC", "TP1", f"{category if category != 'Other' else 'Other inference'} {'major FLOPs' if category != 'Other' else 'FLOPs'}")],
                self.scenario_refs[("DC", "TP8", f"{category if category != 'Other' else 'Other inference'} {'major FLOPs' if category != 'Other' else 'FLOPs'}")],
            )
            divisors = (1e12, 1e12, 1e9, 1e9)
            for col, (ref, divisor, value) in enumerate(zip(refs, divisors, values), start=1):
                ws.write_formula(row, col, f"={ref}/{divisor:g}", self.formats["number"], value)

        ws.write_row(9, 0, ["大类", "Prefill TP1 GB", "Prefill TP8 GB", "Decode TP1 GB", "Decode TP8 GB"], self.formats["header"])
        for offset, category in enumerate(("Attention", "MoE", "Other")):
            row = 10 + offset
            ws.write(row, 0, category, self.formats["text"])
            source_label = f"{category} HBM traffic"
            values = tuple(category_value(summary, category, "hbm") / 1e9 for summary in (pf1, pf8, dc1, dc8))
            refs = (
                self.scenario_refs[("PF", "TP1", source_label)],
                self.scenario_refs[("PF", "TP8", source_label)],
                self.scenario_refs[("DC", "TP1", source_label)],
                self.scenario_refs[("DC", "TP8", source_label)],
            )
            for col, (ref, value) in enumerate(zip(refs, values), start=1):
                ws.write_formula(row, col, f"={ref}/1E9", self.formats["number"], value)

        ws.write_row(15, 0, ["大类", "TP1 参数 GB/rank", "TP8 参数 GB/rank"], self.formats["header"])
        for offset, category in enumerate(("Attention", "MoE", "Other")):
            row = 16 + offset
            ws.write(row, 0, category, self.formats["text"])
            for col, label in ((1, "TP1"), (2, "TP8")):
                ref = self.memory_refs[(label, f"{category} Parameter Capacity")]
                value = memory[f"{label.lower()}_{category.lower()}_parameter"] / 1e9
                ws.write_formula(row, col, f"={ref}/1E9", self.formats["number"], value)

        ws.write_row(21, 0, ["容量项", "TP1 GB/rank", "TP8 GB/rank"], self.formats["header"])
        capacity_rows = (
            ("参数", memory["tp1_parameter_total"], memory["tp8_parameter_total"]),
            ("Decode KV+State", memory["decode_allocated_cache"], memory["decode_allocated_cache"]),
        )
        for offset, (label, value1, value8) in enumerate(capacity_rows):
            row = 22 + offset
            ws.write(row, 0, label, self.formats["text"])
            formula1 = self.memory_refs[("TP1", "Total Parameter Capacity")] if label == "参数" else self.memory_refs[("TP1", "Decode preallocated KV + states")]
            formula8 = self.memory_refs[("TP8", "Total Parameter Capacity")] if label == "参数" else self.memory_refs[("TP8", "Decode preallocated KV + states")]
            ws.write_formula(row, 1, f"={formula1}/1E9", self.formats["number"], value1 / 1e9)
            ws.write_formula(row, 2, f"={formula8}/1E9", self.formats["number"], value8 / 1e9)

        def add_chart(title: str, anchor: str, categories: tuple[int, int], series: list[tuple[str, int]], y_name: str) -> None:
            chart = self.workbook.add_chart({"type": "column"})
            for name, column in series:
                chart.add_series({
                    "name": name,
                    "categories": ["Comparison", categories[0], 0, categories[1], 0],
                    "values": ["Comparison", categories[0], column, categories[1], column],
                    "data_labels": {"value": True, "num_format": "0.0"},
                })
            chart.set_title({"name": title})
            chart.set_y_axis({"name": y_name, "num_format": "0.0", "major_gridlines": {"visible": True}})
            chart.set_x_axis({"label_position": "low"})
            chart.set_legend({"position": "bottom"})
            chart.set_style(10)
            ws.insert_chart(anchor, chart, {"x_scale": 1.28, "y_scale": 1.15})

        add_chart("Prefill 每 Rank 计算量", "G3", (4, 6), [("TP1", 1), ("TP8", 2)], "TFLOPs")
        add_chart("Decode 每 Rank 计算量", "N3", (4, 6), [("TP1", 3), ("TP8", 4)], "GFLOPs")
        add_chart("Prefill 每 Rank HBM 流量", "G20", (10, 12), [("TP1", 1), ("TP8", 2)], "GB")
        add_chart("Decode 每 Rank HBM 流量", "N20", (10, 12), [("TP1", 3), ("TP8", 4)], "GB")
        add_chart("每 Rank 参数容量", "G37", (16, 18), [("TP1", 1), ("TP8", 2)], "GB")
        add_chart("Decode 每 Rank 驻留容量", "N37", (22, 23), [("TP1", 1), ("TP8", 2)], "GB")
        ws.set_column("A:A", 25)
        ws.set_column("B:E", 20)

    def write_methodology(self) -> None:
        ws = self.workbook.add_worksheet("Methodology")
        ws.hide_gridlines(2)
        ws.write(0, 0, "方法与统计边界", self.formats["title"])
        rows = [
            ("统计范围", "仅统计 43 层主推理路径；不含 MTP、反向传播、梯度、优化器状态和训练激活。Layer_Config 最多可维护 64 层。"),
            ("计算量", "矩阵乘加按 2 FLOPs 计。Attention 与 MoE 主要 GEMM 使用显式公式；HC 与逐元素运算为近似估计。"),
            ("Prefill Attention", "使用因果候选对数量，不直接使用 S×S；Raw Window、Compressed KV 和 Indexer 扫描分别统计。"),
            ("Decode Attention", "每个 DecodeTokens 在 DecodeContext 下建模；ratio-4 主 Attention 取 Top-K，但 Indexer 需要扫描全部已完成压缩项。"),
            ("MoE", "每 Token 执行 Top-K Routed Experts 和 Shared Experts。HBM 参数读取按专家至少被命中一次的概率估计。"),
            ("HBM 流量", "统计本地逻辑参数、激活和 KV 的读取与写入；不模拟 L2 命中、算子融合、分块、Allocator 或厂商 Kernel 内部复用。"),
            ("参数容量", "按推理运行 dtype 估算：Routed Expert FP4、多数投影 FP8、Wo_a BF16、Compressor FP32、LM Head FP32，并计入量化 Scale。"),
            ("KV Cache", "有效容量表示已填充项；预分配容量使用 MaxContext。当前实现的 KV 和 Compressor State 在每个 TP Rank 上完整复制。"),
            ("TP 分片", "Wq_a、Wkv、Compressor、Router、HC、Shared Expert 复制；Q/O 部分投影、Routed Experts、Embedding 和 LM Head 按 TP 分片。"),
            ("卡间通信", "Ring 公式给出每 Rank 发送加接收的数据量。TP1 为 0；该数值是传输量，不是实测 GB/s 或时延。"),
            ("硬件下界", "计算/HBM/互连下界使用 Parameters 中可编辑硬件峰值并假设互不重叠；真实运行时间必须通过 Profiling 验证。"),
            ("单位", "FLOPs 使用 M/G/T/P 十进制单位；容量与流量使用 KB/MB/GB/TB，1 GB=10^9 Bytes；Memory 明细保留原始公式。"),
            ("公式维护", "蓝色单元格为输入；结果由 Excel 公式计算并在打开时完整重算。隐藏列保留原始未缩放值，便于审计。"),
        ]
        ws.write_row(2, 0, ["主题", "定义"], self.formats["header"])
        for row, (topic, definition) in enumerate(rows, start=3):
            ws.write(row, 0, topic, self.formats["text"])
            ws.write(row, 1, definition, self.formats["text"])
        ws.set_column("A:A", 24)
        ws.set_column("B:B", 120)


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
    p_tp8 = replace(p, tp=p.comparison_tp)
    prefill_tp1 = summarize_items(prefill_items)
    decode_tp1 = summarize_items(decode_items)
    prefill_tp8 = summarize_items(scenario_items(p_tp8, layers, "prefill")[0])
    decode_tp8 = summarize_items(scenario_items(p_tp8, layers, "decode")[0])
    report = {
        "model": "DeepSeek-V4-Flash",
        "scope": "仅主推理路径；不含 MTP、反向传播、梯度和优化器",
        "inputs": p.__dict__,
        "layer_counts": layers.__dict__,
        "prefill_8k": {
            "tp1_per_rank": prefill_tp1,
            "tp8_per_rank": prefill_tp8,
        },
        "decode_1m": {
            "tp1_per_rank": decode_tp1,
            "tp8_per_rank": decode_tp8,
        },
        "memory": memory,
    }
    (output_dir / "baseline_results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    def tf(value: float) -> str:
        return f"{value / 1e12:,.3f} TFLOPs"

    def gb(value: float) -> str:
        return f"{value / 1e9:,.3f} GB"

    def category_hbm(summary: dict[str, Any], category: str) -> str:
        values = summary["categories"][category]
        return gb(
            values["hbm_read_bytes_per_rank"]
            + values["hbm_write_bytes_per_rank"]
        )

    markdown = [
        "# DeepSeek V4 Flash TP1 / TP8 推理基准",
        "",
        "## 假设",
        "",
        f"- TP 配置：`{p.tp}` 与 `{p.comparison_tp}`",
        f"- Prefill：Batch `{p.prefill_batch}`，Sequence `{p.prefill_sequence}`",
        f"- Decode：Batch `{p.decode_batch}`，Tokens `{p.decode_tokens}`，Context `{p.decode_context}`",
        f"- 层数：`{layers.total}` = window `{layers.window}` + short `{layers.short}` + long `{layers.long}`",
        "- MAC = 2 FLOPs；仅推理，不含 MTP/训练算子。",
        "",
        "## 每 Rank 计算量与 HBM",
        "",
        "| 指标 | Prefill TP1 | Prefill TP8/rank | Decode TP1 | Decode TP8/rank |",
        "|---|---:|---:|---:|---:|",
        f"| Attention FLOPs | {tf(prefill_tp1['attention_major_flops_per_rank'])} | {tf(prefill_tp8['attention_major_flops_per_rank'])} | {tf(decode_tp1['attention_major_flops_per_rank'])} | {tf(decode_tp8['attention_major_flops_per_rank'])} |",
        f"| MoE FLOPs | {tf(prefill_tp1['moe_major_flops_per_rank'])} | {tf(prefill_tp8['moe_major_flops_per_rank'])} | {tf(decode_tp1['moe_major_flops_per_rank'])} | {tf(decode_tp8['moe_major_flops_per_rank'])} |",
        f"| 总 FLOPs | {tf(prefill_tp1['total_per_rank_flops'])} | {tf(prefill_tp8['total_per_rank_flops'])} | {tf(decode_tp1['total_per_rank_flops'])} | {tf(decode_tp8['total_per_rank_flops'])} |",
        f"| Attention HBM | {category_hbm(prefill_tp1, 'Attention')} | {category_hbm(prefill_tp8, 'Attention')} | {category_hbm(decode_tp1, 'Attention')} | {category_hbm(decode_tp8, 'Attention')} |",
        f"| MoE HBM | {category_hbm(prefill_tp1, 'MoE')} | {category_hbm(prefill_tp8, 'MoE')} | {category_hbm(decode_tp1, 'MoE')} | {category_hbm(decode_tp8, 'MoE')} |",
        f"| Other HBM | {category_hbm(prefill_tp1, 'Other')} | {category_hbm(prefill_tp8, 'Other')} | {category_hbm(decode_tp1, 'Other')} | {category_hbm(decode_tp8, 'Other')} |",
        "",
        "## 每 Rank 容量",
        "",
        "| 指标 | TP1 | TP8/rank |",
        "|---|---:|---:|",
        f"| Attention 参数容量 | {gb(memory['tp1_attention_parameter'])} | {gb(memory['tp8_attention_parameter'])} |",
        f"| MoE 参数容量 | {gb(memory['tp1_moe_parameter'])} | {gb(memory['tp8_moe_parameter'])} |",
        f"| Other 参数容量 | {gb(memory['tp1_other_parameter'])} | {gb(memory['tp8_other_parameter'])} |",
        f"| 参数总容量 | {gb(memory['tp1_parameter_total'])} | {gb(memory['tp8_parameter_total'])} |",
        f"| Decode 1M KV + State | {gb(memory['decode_allocated_cache'])} | {gb(memory['decode_allocated_cache'])} |",
        f"| Decode 总驻留容量 | {gb(memory['tp1_parameter_total'] + memory['decode_allocated_cache'])} | {gb(memory['tp8_parameter_total'] + memory['decode_allocated_cache'])} |",
        "",
        "Excel 工作簿是架构探索的主要产物：修改 TP、Batch、Hidden Size、专家数或层模式后会自动重算。",
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
    p_tp8 = replace(p, tp=p.comparison_tp)
    prefill_tp8 = summarize_items(scenario_items(p_tp8, layers, "prefill")[0])
    decode_tp8 = summarize_items(scenario_items(p_tp8, layers, "decode")[0])
    if prefill_tp8["total_interconnect_bytes_per_rank"] <= 0:
        raise AssertionError("TP8 prefill interconnect must be non-zero")
    if decode_tp8["total_interconnect_bytes_per_rank"] <= 0:
        raise AssertionError("TP8 decode interconnect must be non-zero")
    tp1_weights = weight_values(p, layers)
    tp8_weights = weight_values(p_tp8, layers)
    if not math.isclose(
        tp8_weights["routed_rank"] * p.comparison_tp,
        tp1_weights["routed_rank"],
        rel_tol=0,
        abs_tol=1,
    ):
        raise AssertionError("TP8 routed-expert capacity is not a 1/8 shard")
    tp1_cache = cache_values(p, layers, p.decode_context, p.decode_batch)
    tp8_cache = cache_values(p_tp8, layers, p.decode_context, p.decode_batch)
    if tp1_cache != tp8_cache:
        raise AssertionError("KV cache must be replicated and equal per TP rank")
    if memory["tp8_parameter_total"] >= memory["tp1_parameter_total"]:
        raise AssertionError("TP8 per-rank parameter capacity must be below TP1")
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
        workbook_xml = archive.read("xl/workbook.xml")
        for required_sheet in (
            b"Summary",
            b"Prefill_8K",
            b"Decode_1M",
            b"Memory",
            b"Comparison",
            b"Methodology",
        ):
            if required_sheet not in workbook_xml:
                raise AssertionError(f"Missing workbook sheet: {required_sheet!r}")
        chart_payloads = [
            archive.read(name)
            for name in archive.namelist()
            if name.startswith("xl/charts/chart")
        ]
        if len(chart_payloads) < 6:
            raise AssertionError("Expected six readable comparison charts")
        if any(b"logBase" in payload or b"0.000E" in payload for payload in chart_payloads):
            raise AssertionError("Charts must not use log axes or scientific formats")
        workbook_root = ET.fromstring(workbook_xml)
        namespace = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        sheet_names = [
            sheet.attrib["name"]
            for sheet in workbook_root.find("m:sheets", namespace)
        ]
        for target in ("Prefill_8K", "Decode_1M"):
            sheet_index = sheet_names.index(target) + 1
            sheet_root = ET.fromstring(
                archive.read(f"xl/worksheets/sheet{sheet_index}.xml")
            )
            pane = sheet_root.find(".//m:pane", namespace)
            if pane is None or pane.attrib.get("ySplit") != "3":
                raise AssertionError(f"{target} must freeze only the top three rows")


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

    writer = InferenceComparisonWriter(output, p, ratios)
    writer.write_parameters()
    writer.write_layer_config()
    writer.write_scenario("prefill")
    writer.write_scenario("decode")
    memory = writer.write_memory()
    writer.write_summary(memory)
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