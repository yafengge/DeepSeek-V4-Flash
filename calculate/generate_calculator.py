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
CHART_CATEGORY_GAP = 30


def display_label(value: str) -> str:
    return value


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
            "Wq_a 和 Wkv 在每个 TP Rank 上复制；Wq_b/Wo_a/Wo_b 按 TP 分片。",
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
            "KV 在各头之间共享；每个 Rank 的内核头数包含最小头数补齐。",
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
            "Indexer Q 和令牌权重投影；仅 ratio-4 层使用。",
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
            "在选取 Top-K 前扫描所有已完成的 ratio-4 索引条目。",
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
            "每个 TP Rank 都复制；ratio-4 使用重叠的 2 倍投影。",
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
            "复制的辅助 Compressor，用于构建 Indexer 缓存。",
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
            "即使是哈希路由层，源码仍会计算 Gate 分数。",
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
            "每个 Rank 的均衡期望分配；参数读取使用专家至少被触发一次的概率。",
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
            "共享专家在每个 TP Rank 上复制，并独立计算。",
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
            "HC 计算为近似值；sigmoid/exp 仅做定性跟踪。",
        )
    )

    lm_global = 2 * (p.prefill_batch if mode == "prefill" else rows) * p.hidden * p.vocab
    lm_rank = lm_global / p.tp
    lm_read = weights["lm_head"] + (p.prefill_batch if mode == "prefill" else rows) * p.hidden * p.fp32_bytes
    lm_write = (p.prefill_batch if mode == "prefill" else rows) * (p.vocab / p.tp) * p.fp32_bytes
    items.append(
        Item(
            "Other",
            "LM Head",
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
            "推理路径仅根据最后一个隐藏令牌计算 logits。",
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
            "TP=1 时为零。这是传输量，不是 HBM 流量。",
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
        ParameterComponent("Attention", "Core Q/K/O projections", core_global_count_xl, core_global_count, core_rank_count_xl, core_rank_count, core_global_bytes_xl, core_global_bytes, core_rank_bytes_xl, core_rank_bytes, "Wq_a/Wkv 复制；Q/O 输出维度按 TP 分片。"),
        ParameterComponent("Attention", "KV compressors", compressor_count_xl, compressor_count, compressor_count_xl, compressor_count, compressor_bytes_xl, compressor_bytes, compressor_bytes_xl, compressor_bytes, "FP32 推理 Compressor 在每个 TP Rank 上复制。"),
        ParameterComponent("Attention", "Ratio-4 Indexers", index_global_count_xl, index_global_count, index_rank_count_xl, index_rank_count, index_global_bytes_xl, index_global_bytes, index_rank_bytes_xl, index_rank_bytes, "Indexer Q/权重投影按 TP 分片；Indexer Compressor 复制。"),
        ParameterComponent("MoE", "Routed experts", "TotalLayers*RoutedExperts*3*HiddenSize*ExpertInter", routed_global_count, "TotalLayers*LocalExperts*3*HiddenSize*ExpertInter", routed_rank_count, f"TotalLayers*RoutedExperts*{xl_expert_bytes()}", routed_global_bytes, f"TotalLayers*LocalExperts*{xl_expert_bytes()}", routed_rank_bytes, "FP4 路由专家在各 Rank 间均匀分片。"),
        ParameterComponent("MoE", "Shared experts", "TotalLayers*SharedExperts*3*HiddenSize*ExpertInter", shared_global_count, "TotalLayers*SharedExperts*3*HiddenSize*ExpertInter", shared_global_count, shared_bytes_xl, shared_bytes, shared_bytes_xl, shared_bytes, "FP8 共享专家在每个 Rank 上复制。"),
        ParameterComponent("MoE", "Router and hash tables", router_count_xl, router_count, router_count_xl, router_count, router_bytes_xl, weights["router_rank"], router_bytes_xl, weights["router_rank"], "Gate、偏置和令牌到专家的映射表在每个 Rank 上复制。"),
        ParameterComponent("Other", "Embedding", "VocabSize*HiddenSize", p.vocab * p.hidden, "LocalVocab*HiddenSize", p.vocab / p.tp * p.hidden, "VocabSize*HiddenSize*BF16Bytes", p.vocab * p.hidden * p.bf16_bytes, "LocalVocab*HiddenSize*BF16Bytes", weights["embedding"], "Embedding 按词表维度并行。"),
        ParameterComponent("Other", "LM head", "VocabSize*HiddenSize", p.vocab * p.hidden, "LocalVocab*HiddenSize", p.vocab / p.tp * p.hidden, "VocabSize*HiddenSize*FP32Bytes", p.vocab * p.hidden * p.fp32_bytes, "LocalVocab*HiddenSize*FP32Bytes", weights["lm_head"], "FP32 推理 LM Head 按词表维度并行。"),
        ParameterComponent("Other", "Hyper-Connections", hc_count_xl, hc_count, hc_count_xl, hc_count, hc_bytes_xl, weights["hc_rank"], hc_bytes_xl, weights["hc_rank"], "HC 参数在每个 Rank 上复制；仅用于推理。"),
        ParameterComponent("Other", "Norms and attention sinks", norm_global_count_xl, norm_global_count, norm_rank_count_xl, norm_rank_count, f"({norm_global_count_xl})*FP32Bytes", norm_global_count * p.fp32_bytes, f"({norm_rank_count_xl})*FP32Bytes", weights["norms_rank"], "仅本地注意力 Sink 向量按 TP 分片。"),
        ParameterComponent("Other", "Tail HC and final norm", tail_count_xl, tail_count, tail_count_xl, tail_count, f"({tail_count_xl})*FP32Bytes", weights["tail_rank"], f"({tail_count_xl})*FP32Bytes", weights["tail_rank"], "模型尾部推理参数在每个 Rank 上复制。"),
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

    def _unit_number_format(self, style: str, unit: str) -> Any:
        options: dict[str, Any] = {
            "border": 1,
            "num_format": f'#,##0.000 "{unit}"',
        }
        if style == "input":
            options.update({"bg_color": "#D9EAF7", "num_format": f'0.######## "{unit}"'})
        elif style == "derived":
            options.update({"bg_color": "#E8F5E9", "num_format": f'0.######## "{unit}"'})
        elif style == "total":
            options.update({"bold": True, "bg_color": "#D6EAF8"})
        return self.workbook.add_format(options)

    def write_parameters(self) -> None:
        ws = self.workbook.add_worksheet("Parameters")
        ws.hide_gridlines(2)
        ws.freeze_panes(3, 0)
        ws.write(0, 0, "DeepSeek V4 Flash - Editable Architecture Parameters", self.formats["title"])
        ws.write(1, 0, "蓝色单元格为可编辑输入，绿色单元格为 Excel 派生公式。", self.formats["note"])
        headers = ["Category", "Parameter", "Excel name", "Value", "Description"]
        for col, header in enumerate(headers):
            ws.write(2, col, header, self.formats["header"])

        records: list[tuple[str, str, str, float, str, str]] = [
            ("Parallel", "TP1 comparison size", "TP1Size", self.p.tp, "ranks", "第一组张量并行配置。"),
            ("Parallel", "TP8 comparison size", "TP8Size", self.p.comparison_tp, "ranks", "第二组张量并行配置，可编辑。"),
            ("Scenario", "Prefill batch size", "PrefillBatch", self.p.prefill_batch, "sequences", "可编辑的预填充批大小。"),
            ("Scenario", "Prefill sequence length", "PrefillSequence", self.p.prefill_sequence, "tokens", "一次预填充处理的提示词令牌数。"),
            ("Scenario", "Decode batch size", "DecodeBatch", self.p.decode_batch, "sequences", "可编辑的解码批大小。"),
            ("Scenario", "Decode tokens per step", "DecodeTokens", self.p.decode_tokens, "tokens", "通常为 1；公式按解码上下文长度建模每个令牌。"),
            ("Scenario", "Decode context", "DecodeContext", self.p.decode_context, "tokens", "解码阶段可见的有效上下文长度。"),
            ("Scenario", "Maximum allocated context", "MaxContext", self.p.max_context, "tokens", "KV 缓存的预分配容量。"),
            ("Model", "Hidden size", "HiddenSize", self.p.hidden, "elements", "Transformer 残差宽度。"),
            ("Model", "Vocabulary size", "VocabSize", self.p.vocab, "tokens", "Embedding 与 LM Head 使用的词表大小。"),
            ("Attention", "Query heads", "NumHeads", self.p.heads, "heads", "全局 Query 头数量。"),
            ("Attention", "Head dimension", "HeadDim", self.p.head_dim, "elements", "共享潜在 KV 宽度及每个 Query 头的宽度。"),
            ("Attention", "RoPE dimension", "RopeDim", self.p.rope_dim, "elements", "HeadDim 中用于旋转位置编码的维度。"),
            ("Attention", "Q LoRA rank", "QLoraRank", self.p.q_rank, "elements", "Q 投影的低秩中间宽度。"),
            ("Attention", "Output groups", "OGroups", self.p.o_groups, "groups", "分组低秩输出投影的分组数量。"),
            ("Attention", "Output LoRA rank", "OLoraRank", self.p.o_rank, "elements", "每个输出分组的低秩宽度。"),
            ("MoE", "Routed experts", "RoutedExperts", self.p.routed_experts, "experts", "每个专家块的全局路由专家数量。"),
            ("MoE", "Activated experts", "ActivatedExperts", self.p.activated_experts, "experts/token", "每个令牌选取的 Top-K 路由专家数量。"),
            ("MoE", "Shared experts", "SharedExperts", self.p.shared_experts, "experts", "每个专家块复制的共享专家数量。"),
            ("MoE", "Expert intermediate size", "ExpertInter", self.p.expert_inter, "elements", "SwiGLU 专家的中间维度。"),
            ("Cache", "Raw sliding window", "WindowSize", self.p.window, "tokens", "每层保留的原始 KV 位置数量。"),
            ("Cache", "Short compression ratio", "ShortRatio", self.p.short_ratio, "tokens/entry", "默认的 ratio-4 模式。"),
            ("Cache", "Long compression ratio", "LongRatio", self.p.long_ratio, "tokens/entry", "默认的 ratio-128 模式。"),
            ("Indexer", "Indexer heads", "IndexHeads", self.p.index_heads, "heads", "全局 ratio-4 Indexer 头数量。"),
            ("Indexer", "Indexer head dimension", "IndexHeadDim", self.p.index_dim, "elements", "Indexer Q/K 的维度。"),
            ("Indexer", "Indexer Top-K", "IndexTopK", self.p.index_topk, "entries", "ratio-4 注意力保留的压缩条目数量。"),
            ("HC", "HC slots", "HCSlots", self.p.hc_slots, "streams", "Hyper-Connection 残差流数量。"),
            ("HC", "Sinkhorn iterations", "HCIters", self.p.hc_iters, "iterations", "HC 组合归一化的迭代次数。"),
            ("Routing", "Hash-routed layers", "HashLayers", self.p.hash_layers, "layers", "使用令牌编号到专家编号的路由起始层数量。"),
            ("DType", "BF16 bytes", "BF16Bytes", self.p.bf16_bytes, "bytes", "每个 BF16 元素占用的存储字节数。"),
            ("DType", "FP32 bytes", "FP32Bytes", self.p.fp32_bytes, "bytes", "每个 FP32 元素占用的存储字节数。"),
            ("DType", "FP8 bytes", "FP8Bytes", self.p.fp8_bytes, "bytes", "每个 FP8 元素占用的存储字节数。"),
            ("DType", "FP4 bytes", "FP4Bytes", self.p.fp4_bytes, "bytes", "每个 FP4 元素的逻辑打包字节数。"),
            ("DType", "Scale bytes", "ScaleBytes", self.p.scale_bytes, "bytes", "E8M0 Scale 的存储字节数。"),
            ("DType", "INT32 bytes", "INT32Bytes", self.p.int32_bytes, "bytes", "每个 INT32 元素占用的存储字节数。"),
            ("DType", "INT64 bytes", "INT64Bytes", self.p.int64_bytes, "bytes", "每个 INT64 元素占用的存储字节数。"),
            ("Quantization", "FP8 scale block", "FP8Block", self.p.fp8_block, "elements", "二维 FP8 权重 Scale 的分块大小。"),
            ("Quantization", "FP4 scale block", "FP4Block", self.p.fp4_block, "elements", "沿 K 维度的 FP4 专家 Scale 分块大小。"),
            ("Kernel", "Minimum sparse-attention heads", "KernelMinHeads", self.p.kernel_min_heads, "heads", "当本地头数不足时，内核补齐到该头数。"),
            ("Hardware", "Peak compute", "PeakTFLOPs", self.p.peak_tflops, "TFLOP/s", "可编辑的示例硬件峰值算力。"),
            ("Hardware", "HBM bandwidth", "HBMBandwidthGBps", self.p.hbm_gbps, "GB/s", "可编辑的持续/目标 HBM 带宽。"),
            ("Hardware", "Interconnect bandwidth", "InterconnectGBps", self.p.interconnect_gbps, "GB/s", "可编辑的有效双向互连带宽。"),
            ("Hardware", "Prefill target latency", "PrefillTargetMs", self.p.prefill_target_ms, "ms", "仅用于推导所需 HBM 带宽和卡间互连带宽。"),
            ("Hardware", "Decode target latency", "DecodeTargetMs", self.p.decode_target_ms, "ms", "仅用于推导所需 HBM 带宽和卡间互连带宽。"),
        ]
        row = 3
        for category, label, name, value, unit, description in records:
            ws.write(row, 0, category, self.formats["text"])
            ws.write(row, 1, label, self.formats["text"])
            ws.write(row, 2, name, self.formats["text"])
            ws.write_number(row, 3, value, self._unit_number_format("input", display_label(unit)))
            ws.write(row, 4, description, self.formats["text"])
            self.parameter_rows[name] = row
            self._define_cell_name(name, "Parameters", row, 3)
            row += 1

        derived = [
            ("Derived", "Active layers", "TotalLayers", "=SUM(Layer_Config!$B$4:$B$67)", self.layers.total, "layers", "Layer_Config 中启用的层数。"),
            ("Derived", "Window-only layers", "WindowLayers", '=COUNTIFS(Layer_Config!$B$4:$B$67,1,Layer_Config!$C$4:$C$67,"window")', self.layers.window, "layers", "启用的窗口模式层数。"),
            ("Derived", "Short-compression layers", "ShortLayers", '=COUNTIFS(Layer_Config!$B$4:$B$67,1,Layer_Config!$C$4:$C$67,"short")', self.layers.short, "layers", "启用的 ratio-4/短压缩层数。"),
            ("Derived", "Long-compression layers", "LongLayers", '=COUNTIFS(Layer_Config!$B$4:$B$67,1,Layer_Config!$C$4:$C$67,"long")', self.layers.long, "layers", "启用的 ratio-128/长压缩层数。"),
            ("Derived", "Active hash layers", "ActiveHashLayers", '=COUNTIFS(Layer_Config!$B$4:$B$67,1,Layer_Config!$A$4:$A$67,"<"&HashLayers)', self.layers.hash, "layers", "层 ID 小于 HashLayers 且已启用的层数。"),
            ("Derived", "TP1 local Q heads", "TP1LocalHeads", "=NumHeads/TP1Size", self.p.heads / self.p.tp, "heads/rank", "单个 TP1 Rank 上的 Q 头数。"),
            ("Derived", "TP1 kernel local heads", "TP1KernelLocalHeads", "=MAX(TP1LocalHeads,KernelMinHeads)", max(self.p.heads / self.p.tp, self.p.kernel_min_heads), "heads/rank", "补齐后的稀疏内核本地头数。"),
            ("Derived", "TP1 local index heads", "TP1LocalIndexHeads", "=IndexHeads/TP1Size", self.p.index_heads / self.p.tp, "heads/rank", "单个 TP1 Rank 上的 Indexer 头数。"),
            ("Derived", "TP1 local experts", "TP1LocalExperts", "=RoutedExperts/TP1Size", self.p.routed_experts / self.p.tp, "experts/rank", "单个 TP1 Rank 上的路由专家数。"),
            ("Derived", "TP1 local output groups", "TP1LocalOGroups", "=OGroups/TP1Size", self.p.o_groups / self.p.tp, "groups/rank", "单个 TP1 Rank 上的输出分组数。"),
            ("Derived", "TP1 local vocabulary", "TP1LocalVocab", "=VocabSize/TP1Size", self.p.vocab / self.p.tp, "tokens/rank", "单个 TP1 Rank 上的词表行数。"),
            ("Derived", "TP8 local Q heads", "TP8LocalHeads", "=NumHeads/TP8Size", self.p.heads / self.p.comparison_tp, "heads/rank", "单个 TP8 Rank 上的 Q 头数。"),
            ("Derived", "TP8 kernel local heads", "TP8KernelLocalHeads", "=MAX(TP8LocalHeads,KernelMinHeads)", max(self.p.heads / self.p.comparison_tp, self.p.kernel_min_heads), "heads/rank", "补齐后的稀疏内核本地头数。"),
            ("Derived", "TP8 local index heads", "TP8LocalIndexHeads", "=IndexHeads/TP8Size", self.p.index_heads / self.p.comparison_tp, "heads/rank", "单个 TP8 Rank 上的 Indexer 头数。"),
            ("Derived", "TP8 local experts", "TP8LocalExperts", "=RoutedExperts/TP8Size", self.p.routed_experts / self.p.comparison_tp, "experts/rank", "单个 TP8 Rank 上的路由专家数。"),
            ("Derived", "TP8 local output groups", "TP8LocalOGroups", "=OGroups/TP8Size", self.p.o_groups / self.p.comparison_tp, "groups/rank", "单个 TP8 Rank 上的输出分组数。"),
            ("Derived", "TP8 local vocabulary", "TP8LocalVocab", "=VocabSize/TP8Size", self.p.vocab / self.p.comparison_tp, "tokens/rank", "单个 TP8 Rank 上的词表行数。"),
        ]
        for category, label, name, formula, value, unit, description in derived:
            ws.write(row, 0, category, self.formats["text"])
            ws.write(row, 1, label, self.formats["text"])
            ws.write(row, 2, name, self.formats["text"])
            ws.write_formula(row, 3, formula, self._unit_number_format("derived", display_label(unit)), value)
            ws.write(row, 4, description, self.formats["text"])
            self.parameter_rows[name] = row
            self._define_cell_name(name, "Parameters", row, 3)
            row += 1

        ws.autofilter(2, 0, 2 + len(records) + len(derived), 4)
        ws.set_column("A:A", 16)
        ws.set_column("B:B", 29)
        ws.set_column("C:C", 25)
        ws.set_column("D:D", 18)
        ws.set_column("E:E", 58)

    def write_layer_config(self) -> None:
        ws = self.workbook.add_worksheet("Layer_Config")
        ws.hide_gridlines(2)
        ws.freeze_panes(3, 0)
        ws.write(0, 0, "Per-Layer Attention and Routing Configuration", self.formats["title"])
        ws.write(1, 0, "可编辑启用状态和模式；压缩比例与路由行为会自动更新。", self.formats["note"])
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
        self.scenario_worksheets: dict[str, Any] = {}
        self.scenario_next_rows: dict[str, int] = {}

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
        _unit_col: int,
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
                f'=TEXT({raw_ref},"0.00%")',
                self.formats["percent_display"],
                f"{raw_value:.2%}",
            )
            return
        if kind in {"flops", "bytes", "params"}:
            value_formula, unit_formula = display_formula(raw_ref, kind)
            display_value, display_unit = scaled_value(raw_value, kind)
            if kind == "params":
                display_unit = display_unit.removesuffix(" params")
            unit_formula_body = self._formula_body(unit_formula)
            if kind == "params":
                for source, target in (
                    ("T params", "T"),
                    ("G params", "G"),
                    ("M params", "M"),
                    ("K params", "K"),
                    ("params", ""),
                ):
                    unit_formula_body = unit_formula_body.replace(
                        f'"{source}"', f'"{target}"'
                    )
            formula = (
                f'=TEXT({value_formula[1:]},"#,##0.000")'
                f'&IF(({unit_formula_body})="",""," "&({unit_formula_body}))'
            )
            cached_value = f"{display_value:,.3f} {display_unit}".rstrip()
            ws.write_formula(
                row,
                value_col,
                formula,
                self.formats["display_total" if total else "display"],
                cached_value,
            )
            return
        unit = kind
        ws.write_formula(
            row,
            value_col,
            f'=TEXT({raw_ref},"#,##0.000")&" {unit}"',
            self.formats["display_total" if total else "display"],
            f"{raw_value:,.3f} {unit}",
        )

    def _write_combined_param_value(
        self,
        ws: Any,
        row: int,
        value_col: int,
        raw_col: int,
        raw_formula: str,
        raw_value: float,
        total: bool = False,
    ) -> None:
        ws.write_formula(row, raw_col, raw_formula, self.formats["number"], raw_value)
        raw_ref = f"{xl_col_to_name(raw_col)}{row + 1}"
        value_formula, unit_formula = display_formula(raw_ref, "params")
        display_value, display_unit = scaled_value(raw_value, "params")
        unit_formula_body = unit_formula[1:]
        for source, target in (
            ("T params", "T"),
            ("G params", "G"),
            ("M params", "M"),
            ("K params", "K"),
            ("params", ""),
        ):
            unit_formula_body = unit_formula_body.replace(
                f'"{source}"', f'"{target}"'
            )
        formula = (
            f'=TEXT({value_formula[1:]},"#,##0.000")'
            f'&IF(({unit_formula_body})="",""," "&({unit_formula_body}))'
        )
        display_unit = display_unit.removesuffix(" params")
        cached_value = f"{display_value:,.3f} {display_unit}".rstrip()
        ws.write_formula(
            row,
            value_col,
            formula,
            self.formats["display_total" if total else "display"],
            cached_value,
        )

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
            ["Helper metric", "Value", "Unit", "Description"],
            self.formats["header"],
        )
        descriptions = {
            "Token rows": "当前场景需要处理的令牌总数；预填充为批大小乘序列长度，解码为批大小乘每步令牌数。",
            "Causal raw-window pairs": "因果原始窗口路径中可参与注意力计算的令牌对数量，按每个序列统计。",
            "Capped short-compressed pairs": "短压缩路径的候选令牌对数量；按短压缩比例折叠，并受 Top-K 候选上限约束。",
            "Long-compressed pairs": "长压缩路径按长压缩比例折叠后的候选令牌对数量。",
            "Indexer scan pairs": "ratio-4 Indexer 评分时需要扫描的压缩条目数量，作为索引器计算量输入。",
            "Raw-window candidates": "解码时原始滑动窗口可见的候选条目数量，受窗口大小和当前上下文长度限制。",
            "Short-compressed Top-K candidates": "解码时短压缩路径最终保留的 Top-K 候选数量，取 Top-K 与已生成压缩条目数的较小值。",
            "Long-compressed candidates": "解码时长压缩路径按长压缩比例生成的候选条目数量。",
            "Indexer scan candidates": "解码时 ratio-4 Indexer 需要扫描的压缩条目数量，用于估算索引器评分计算量。",
            "Expert activation probability": "在当前场景令牌数下，单个路由专家至少被激活一次的概率，用于估算路由专家参数读取量。",
        }
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
            ws.write(row, 0, display_label(label), self.formats["text"])
            ws.write_formula(row, 1, formula, self._unit_number_format("derived", display_label(unit)), value)
            ws.write(row, 2, display_label(unit), self.formats["text"])
            ws.write(row, 3, descriptions[label], self.formats["text"])
            ws.set_row(row, 54)
            self._define_cell_name(name, ws.name, row, 1)
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
            f"{'8K Prefill' if is_prefill else '1M-context Decode'} - TP1 vs TP8 inference comparison",
            self.formats["title"],
        )
        ws.write(
            1,
            0,
            "仅推理：不包含反向传播、梯度、优化器或训练状态内存。第 3 行以下可滚动查看。",
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
            "Operator/item",
            "Layer scope",
            "TP1 FLOPs",
            "TP8 FLOPs/rank",
            "TP1 HBM/rank",
            "TP8 HBM/rank",
            "TP1 Interconnect",
            "TP8 Interconnect/rank",
            "Accounting",
            "Notes",
        ]
        ws.merge_range(
            detail_header_row - 1,
            0,
            detail_header_row - 1,
            len(headers) - 1,
            "Inference calculation detail",
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

            ws.write(row, 0, display_label(tp1_item.category), self.formats["text"])
            ws.write(row, 1, tp1_item.name, self.formats["text"])
            ws.write(row, 2, display_label(tp1_item.layer_scope), self.formats["text"])
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
                3,
                4,
                raw_tp1_flops_col,
                tp1_flops,
                tp1_item.rank_flops,
                "flops",
            )
            self._write_human_value(
                ws,
                row,
                4,
                4,
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
                5,
                5,
                raw_tp1_hbm_col,
                tp1_hbm,
                tp1_item.read_bytes + tp1_item.write_bytes,
                "bytes",
            )
            self._write_human_value(
                ws,
                row,
                6,
                6,
                raw_tp8_hbm_col,
                tp8_hbm,
                tp8_item.read_bytes + tp8_item.write_bytes,
                "bytes",
            )
            self._write_human_value(
                ws,
                row,
                7,
                7,
                raw_tp1_network_col,
                tp1_network,
                tp1_item.network_bytes,
                "bytes",
            )
            self._write_human_value(
                ws,
                row,
                8,
                8,
                raw_tp8_network_col,
                tp8_network,
                tp8_item.network_bytes,
                "bytes",
            )
            ws.write(row, 9, display_label(tp1_item.accounting), self.formats["text"])
            ws.write(row, 10, tp1_item.notes, self.formats["text"])

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
        ws.merge_range(summary_start, 0, summary_start, 2, "Scenario summary", self.formats["section"])
        summary_header = summary_start + 1
        ws.write_row(
            summary_header,
            0,
            ["Metric", "TP1", "TP8/rank"],
            self.formats["header"],
        )

        category_col = "$A"
        accounting_col = "$J"
        tp1_flops_col = "$U"
        tp8_flops_col = "$V"
        tp1_read_col = "$W"
        tp1_write_col = "$X"
        tp8_read_col = "$Y"
        tp8_write_col = "$Z"
        tp1_network_col = "$AA"
        tp8_network_col = "$AB"

        def sum_category(column: str, category: str, accounting: str | None = None) -> str:
            category_criteria = display_label(category)
            if accounting is None:
                return f'SUMIF({category_col}${detail_first_excel}:{category_col}${detail_last_excel},"{category_criteria}",{column}${detail_first_excel}:{column}${detail_last_excel})'
            accounting_criteria = display_label(accounting)
            return f'SUMIFS({column}${detail_first_excel}:{column}${detail_last_excel},{category_col}${detail_first_excel}:{category_col}${detail_last_excel},"{category_criteria}",{accounting_col}${detail_first_excel}:{accounting_col}${detail_last_excel},"{accounting_criteria}")'

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
            ("One-inference compute demand", f"=SUM({tp1_flops_col}${detail_first_excel}:{tp1_flops_col}${detail_last_excel})", tp1_summary["total_per_rank_flops"], f"=SUM({tp8_flops_col}${detail_first_excel}:{tp8_flops_col}${detail_last_excel})", tp8_summary["total_per_rank_flops"], "flops"),
            ("Required HBM at target", f"=(SUM({tp1_read_col}${detail_first_excel}:{tp1_read_col}${detail_last_excel})+SUM({tp1_write_col}${detail_first_excel}:{tp1_write_col}${detail_last_excel}))/({target_name}/1000)/1E9", total_tp1_hbm / (target_ms / 1000) / 1e9, f"=(SUM({tp8_read_col}${detail_first_excel}:{tp8_read_col}${detail_last_excel})+SUM({tp8_write_col}${detail_first_excel}:{tp8_write_col}${detail_last_excel}))/({target_name}/1000)/1E9", total_tp8_hbm / (target_ms / 1000) / 1e9, "GB/s"),
        ]

        raw_tp1_summary_col = 30
        raw_tp8_summary_col = 31
        for offset, (label, tp1_formula, tp1_value, tp8_formula, tp8_value, kind) in enumerate(specs):
            row = summary_header + 1 + offset
            ws.write(row, 0, display_label(label), self.formats["text"])
            self._write_human_value(
                ws,
                row,
                1,
                2,
                raw_tp1_summary_col,
                tp1_formula,
                tp1_value,
                kind,
                label.startswith("Total"),
            )
            self._write_human_value(
                ws,
                row,
                2,
                3,
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
        ws.set_column("B:B", 33)
        ws.set_column("C:C", 25)
        ws.set_column("D:I", 16)
        ws.set_column("J:J", 13)
        ws.set_column("K:K", 60)
        ws.set_column("L:S", None, None, {"hidden": True})
        self.scenario_worksheets[mode] = ws
        self.scenario_next_rows[mode] = detail_last_row + 3

    def write_embedded_memory(self) -> dict[str, float]:
        tp1_components = parameter_components(self.p_tp1, self.layers)
        tp8_components = parameter_components(self.p_tp8, self.layers)
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
            self.p,
            self.layers,
            self.p.max_context,
            self.p.prefill_batch,
        )
        allocated_decode = cache_values(
            self.p,
            self.layers,
            self.p.max_context,
            self.p.decode_batch,
        )

        memory = {
            "tp1_parameter_total": sum(component.rank_bytes for component in tp1_components),
            "tp8_parameter_total": sum(component.rank_bytes for component in tp8_components),
            "tp1_parameter_count": sum(component.rank_count for component in tp1_components),
            "tp8_parameter_count": sum(component.rank_count for component in tp8_components),
            "tp1_attention_parameter": sum(component.rank_bytes for component in tp1_components if component.category == "Attention"),
            "tp8_attention_parameter": sum(component.rank_bytes for component in tp8_components if component.category == "Attention"),
            "tp1_moe_parameter": sum(component.rank_bytes for component in tp1_components if component.category == "MoE"),
            "tp8_moe_parameter": sum(component.rank_bytes for component in tp8_components if component.category == "MoE"),
            "tp1_other_parameter": sum(component.rank_bytes for component in tp1_components if component.category == "Other"),
            "tp8_other_parameter": sum(component.rank_bytes for component in tp8_components if component.category == "Other"),
            "parameter_total": sum(component.rank_bytes for component in tp1_components),
            "prefill_effective_cache": prefill_cache["total"],
            "prefill_allocated_cache": allocated_prefill["total"],
            "decode_effective_cache": decode_cache["total"],
            "decode_allocated_cache": allocated_decode["total"],
        }
        cache_data = {
            "prefill": (prefill_cache, allocated_prefill),
            "decode": (decode_cache, allocated_decode),
        }

        for mode in ("prefill", "decode"):
            ws = self.scenario_worksheets[mode]
            prefix = "PF" if mode == "prefill" else "DC"
            current_cache, allocated_cache = cache_data[mode]
            start_row = self.scenario_next_rows[mode]
            ws.merge_range(
                start_row,
                0,
                start_row,
                7,
                f"Parameters and cache: {('8K Prefill' if mode == 'prefill' else '1M-context Decode')} parameters and capacity",
                self.formats["section"],
            )
            detail_header = start_row + 1
            detail_first_row = detail_header + 1
            detail_last_row = detail_first_row + len(tp1_components) - 1
            detail_headers = [
                "Category",
                "Parameter component",
                "Global parameter count",
                "TP1 parameter count/rank",
                "TP8 parameter count/rank",
                "TP1 capacity/rank",
                "TP8 capacity/rank",
                "Notes",
            ]
            ws.write_row(detail_header, 0, detail_headers, self.formats["header"])
            raw_global_count_col = 32
            raw_tp1_count_col = 33
            raw_tp8_count_col = 34
            raw_tp1_bytes_col = 35
            raw_tp8_bytes_col = 36
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
                ws.write(row, 0, display_label(tp1_component.category), self.formats["text"])
                ws.write(row, 1, tp1_component.name, self.formats["text"])
                self._write_combined_param_value(
                    ws,
                    row,
                    2,
                    raw_global_count_col,
                    "=" + tp1_component.global_count_formula,
                    tp1_component.global_count,
                )
                self._write_combined_param_value(
                    ws,
                    row,
                    3,
                    raw_tp1_count_col,
                    tp1_count_formula,
                    tp1_component.rank_count,
                )
                self._write_human_value(
                    ws,
                    row,
                    4,
                    4,
                    raw_tp8_count_col,
                    tp8_count_formula,
                    tp8_component.rank_count,
                    "params",
                )
                self._write_human_value(
                    ws,
                    row,
                    5,
                    5,
                    raw_tp1_bytes_col,
                    tp1_bytes_formula,
                    tp1_component.rank_bytes,
                    "bytes",
                )
                self._write_human_value(
                    ws,
                    row,
                    6,
                    6,
                    raw_tp8_bytes_col,
                    tp8_bytes_formula,
                    tp8_component.rank_bytes,
                    "bytes",
                )
                ws.write(row, 7, tp1_component.notes, self.formats["text"])
            ws.add_table(
                detail_header,
                0,
                detail_last_row,
                len(detail_headers) - 1,
                {
                    "name": f"{prefix}_Embedded_Parameter_Detail",
                    "style": "Table Style Medium 2",
                    "columns": [{"header": header} for header in detail_headers],
                },
            )

            summary_start = detail_last_row + 2
            ws.merge_range(
                summary_start,
                0,
                summary_start,
                4,
                "Parameter count and capacity summary",
                self.formats["section"],
            )
            summary_header = summary_start + 1
            summary_headers = [
                "Category",
                "TP1 parameter count/rank",
                "TP8 parameter count/rank",
                "TP1 capacity/rank",
                "TP8 capacity/rank",
            ]
            ws.write_row(summary_header, 0, summary_headers, self.formats["header"])
            detail_first_excel = detail_first_row + 1
            detail_last_excel = detail_last_row + 1
            tp1_count_col = xl_col_to_name(raw_tp1_count_col)
            tp8_count_col = xl_col_to_name(raw_tp8_count_col)
            tp1_bytes_col = xl_col_to_name(raw_tp1_bytes_col)
            tp8_bytes_col = xl_col_to_name(raw_tp8_bytes_col)
            raw_summary_cols = (37, 38, 39, 40)
            local_memory_refs: dict[tuple[str, str], str] = {}
            category_values: dict[str, dict[str, float]] = {}
            for offset, category in enumerate(("Attention", "MoE", "Other", "Total")):
                row = summary_header + 1 + offset
                if category == "Total":
                    selected_tp1 = tp1_components
                    selected_tp8 = tp8_components
                else:
                    selected_tp1 = [
                        component
                        for component in tp1_components
                        if component.category == category
                    ]
                    selected_tp8 = [
                        component
                        for component in tp8_components
                        if component.category == category
                    ]
                if category == "Total":
                    formulas = (
                        f"=SUM(${tp1_count_col}${detail_first_excel}:${tp1_count_col}${detail_last_excel})",
                        f"=SUM(${tp8_count_col}${detail_first_excel}:${tp8_count_col}${detail_last_excel})",
                        f"=SUM(${tp1_bytes_col}${detail_first_excel}:${tp1_bytes_col}${detail_last_excel})",
                        f"=SUM(${tp8_bytes_col}${detail_first_excel}:${tp8_bytes_col}${detail_last_excel})",
                    )
                else:
                    category_criteria = display_label(category)
                    formulas = (
                        f'=SUMIF($A${detail_first_excel}:$A${detail_last_excel},"{category_criteria}",${tp1_count_col}${detail_first_excel}:${tp1_count_col}${detail_last_excel})',
                        f'=SUMIF($A${detail_first_excel}:$A${detail_last_excel},"{category_criteria}",${tp8_count_col}${detail_first_excel}:${tp8_count_col}${detail_last_excel})',
                        f'=SUMIF($A${detail_first_excel}:$A${detail_last_excel},"{category_criteria}",${tp1_bytes_col}${detail_first_excel}:${tp1_bytes_col}${detail_last_excel})',
                        f'=SUMIF($A${detail_first_excel}:$A${detail_last_excel},"{category_criteria}",${tp8_bytes_col}${detail_first_excel}:${tp8_bytes_col}${detail_last_excel})',
                    )
                values = {
                    "tp1_count": sum(component.rank_count for component in selected_tp1),
                    "tp8_count": sum(component.rank_count for component in selected_tp8),
                    "tp1_bytes": sum(component.rank_bytes for component in selected_tp1),
                    "tp8_bytes": sum(component.rank_bytes for component in selected_tp8),
                }
                category_values[category] = values
                ws.write(
                    row,
                    0,
                    display_label(category),
                    self.formats["total" if category == "Total" else "text"],
                )
                self._write_human_value(
                    ws,
                    row,
                    1,
                    1,
                    raw_summary_cols[0],
                    formulas[0],
                    values["tp1_count"],
                    "params",
                    category == "Total",
                )
                self._write_human_value(
                    ws,
                    row,
                    2,
                    2,
                    raw_summary_cols[1],
                    formulas[1],
                    values["tp8_count"],
                    "params",
                    category == "Total",
                )
                self._write_human_value(
                    ws,
                    row,
                    3,
                    3,
                    raw_summary_cols[2],
                    formulas[2],
                    values["tp1_bytes"],
                    "bytes",
                    category == "Total",
                )
                self._write_human_value(
                    ws,
                    row,
                    4,
                    4,
                    raw_summary_cols[3],
                    formulas[3],
                    values["tp8_bytes"],
                    "bytes",
                    category == "Total",
                )
                for label, raw_col in (
                    (f"{category} Parameter Count", raw_summary_cols[0]),
                    (f"{category} Parameter Capacity", raw_summary_cols[2]),
                ):
                    local_memory_refs[("TP1", label)] = self._cell(
                        ws.name, row, raw_col
                    )
                for label, raw_col in (
                    (f"{category} Parameter Count", raw_summary_cols[1]),
                    (f"{category} Parameter Capacity", raw_summary_cols[3]),
                ):
                    local_memory_refs[("TP8", label)] = self._cell(
                        ws.name, row, raw_col
                    )
                for key, value in local_memory_refs.items():
                    self.memory_refs.setdefault(key, value)
            ws.add_table(
                summary_header,
                0,
                summary_header + 4,
                len(summary_headers) - 1,
                {
                    "name": f"{prefix}_Embedded_Parameter_Summary",
                    "style": "Table Style Medium 4",
                    "columns": [{"header": header} for header in summary_headers],
                },
            )

            cache_start = summary_header + 6
            ws.merge_range(
                cache_start,
                0,
                cache_start,
                3,
                "Current scenario KV cache and Compressor state (per Rank, replicated across TP)",
                self.formats["section"],
            )
            cache_header = cache_start + 1
            cache_headers = ["Item", "TP1/rank", "TP8/rank", "Notes"]
            ws.write_row(cache_header, 0, cache_headers, self.formats["header"])
            if mode == "prefill":
                cache_specs = [
                    ("Prefill effective main KV", current_cache["main"], "预填充阶段已生成的主注意力 KV。", "=PrefillBatch*HeadDim*BF16Bytes*(WindowLayers*MIN(PrefillSequence,WindowSize)+ShortLayers*(MIN(PrefillSequence,WindowSize)+INT(PrefillSequence/ShortRatio))+LongLayers*(MIN(PrefillSequence,WindowSize)+INT(PrefillSequence/LongRatio)))"),
                    ("Prefill effective Indexer KV", current_cache["indexer"], "预填充阶段已生成的 ratio-4 Indexer KV。", "=PrefillBatch*ShortLayers*INT(PrefillSequence/ShortRatio)*IndexHeadDim*BF16Bytes"),
                    ("Prefill compressor states", current_cache["states"], "预填充阶段的 Compressor 与 Indexer 状态。", "=PrefillBatch*FP32Bytes*(ShortLayers*2*(2*ShortRatio)*(2*HeadDim)+LongLayers*2*LongRatio*HeadDim+ShortLayers*2*(2*ShortRatio)*(2*IndexHeadDim))"),
                    ("Prefill preallocated KV + states", allocated_cache["total"], "按 MaxContext 为预填充阶段预分配。", "=PrefillBatch*(HeadDim*BF16Bytes*(WindowLayers*WindowSize+ShortLayers*(WindowSize+INT(MaxContext/ShortRatio))+LongLayers*(WindowSize+INT(MaxContext/LongRatio)))+ShortLayers*INT(MaxContext/ShortRatio)*IndexHeadDim*BF16Bytes)+PrefillBatch*FP32Bytes*(ShortLayers*2*(2*ShortRatio)*(2*HeadDim)+LongLayers*2*LongRatio*HeadDim+ShortLayers*2*(2*ShortRatio)*(2*IndexHeadDim))"),
                ]
            else:
                cache_specs = [
                    ("Decode effective main KV", current_cache["main"], "解码阶段的主注意力 KV。", "=DecodeBatch*HeadDim*BF16Bytes*(WindowLayers*MIN(DecodeContext,WindowSize)+ShortLayers*(MIN(DecodeContext,WindowSize)+INT(DecodeContext/ShortRatio))+LongLayers*(MIN(DecodeContext,WindowSize)+INT(DecodeContext/LongRatio)))"),
                    ("Decode effective Indexer KV", current_cache["indexer"], "解码阶段的 ratio-4 Indexer KV。", "=DecodeBatch*ShortLayers*INT(DecodeContext/ShortRatio)*IndexHeadDim*BF16Bytes"),
                    ("Decode compressor states", current_cache["states"], "解码阶段的 Compressor 与 Indexer 状态。", "=DecodeBatch*FP32Bytes*(ShortLayers*2*(2*ShortRatio)*(2*HeadDim)+LongLayers*2*LongRatio*HeadDim+ShortLayers*2*(2*ShortRatio)*(2*IndexHeadDim))"),
                    ("Decode preallocated KV + states", allocated_cache["total"], "按 MaxContext 为解码阶段预分配。", "=DecodeBatch*(HeadDim*BF16Bytes*(WindowLayers*WindowSize+ShortLayers*(WindowSize+INT(MaxContext/ShortRatio))+LongLayers*(WindowSize+INT(MaxContext/LongRatio)))+ShortLayers*INT(MaxContext/ShortRatio)*IndexHeadDim*BF16Bytes)+DecodeBatch*FP32Bytes*(ShortLayers*2*(2*ShortRatio)*(2*HeadDim)+LongLayers*2*LongRatio*HeadDim+ShortLayers*2*(2*ShortRatio)*(2*IndexHeadDim))"),
                ]
            raw_tp1_cache_col = 41
            raw_tp8_cache_col = 42
            for offset, (label, value, note, formula) in enumerate(cache_specs):
                row = cache_header + 1 + offset
                ws.write(row, 0, display_label(label), self.formats["text"])
                self._write_human_value(
                    ws,
                    row,
                    1,
                    1,
                    raw_tp1_cache_col,
                    formula,
                    value,
                    "bytes",
                )
                self._write_human_value(
                    ws,
                    row,
                    2,
                    2,
                    raw_tp8_cache_col,
                    formula,
                    value,
                    "bytes",
                )
                ws.write(row, 3, note, self.formats["text"])
                local_memory_refs[("TP1", label)] = self._cell(
                    ws.name, row, raw_tp1_cache_col
                )
                local_memory_refs[("TP8", label)] = self._cell(
                    ws.name, row, raw_tp8_cache_col
                )
                self.memory_refs[("TP1", label)] = local_memory_refs[("TP1", label)]
                self.memory_refs[("TP8", label)] = local_memory_refs[("TP8", label)]
            ws.add_table(
                cache_header,
                0,
                cache_header + len(cache_specs),
                len(cache_headers) - 1,
                {
                    "name": f"{prefix}_Embedded_Cache",
                    "style": "Table Style Medium 2",
                    "columns": [{"header": header} for header in cache_headers],
                },
            )

            rank_start = cache_header + len(cache_specs) + 3
            ws.merge_range(
                rank_start,
                0,
                rank_start,
                9,
                "Per-rank device resident capacity (default even sharding)",
                self.formats["section"],
            )
            rank_header = rank_start + 1
            rank_headers = [
                "Configuration",
                "Rank",
                "Attention parameters",
                "MoE parameters",
                "Other parameters",
                "Total parameters",
                "Current scenario KV+state",
                "Total resident",
                "Logical parameter count",
                "Status",
            ]
            ws.write_row(rank_header, 0, rank_headers, self.formats["header"])
            rank_row = rank_header + 1
            for label, p_config in (("TP1", self.p_tp1), ("TP8", self.p_tp8)):
                ws.write(rank_row, 0, label, self.formats["text"])
                ws.write(rank_row, 1, "Any Rank", self.formats["text"])
                category_refs = [
                    local_memory_refs[(label, "Attention Parameter Capacity")],
                    local_memory_refs[(label, "MoE Parameter Capacity")],
                    local_memory_refs[(label, "Other Parameter Capacity")],
                    local_memory_refs[(label, "Total Parameter Capacity")],
                ]
                category_values_for_rank = [
                    category_values["Attention"][f"{label.lower()}_bytes"] / 1e9,
                    category_values["MoE"][f"{label.lower()}_bytes"] / 1e9,
                    category_values["Other"][f"{label.lower()}_bytes"] / 1e9,
                    category_values["Total"][f"{label.lower()}_bytes"] / 1e9,
                ]
                for col, (ref, value) in enumerate(
                    zip(category_refs, category_values_for_rank), start=2
                ):
                    ws.write_formula(
                        rank_row,
                        col,
                        f"={ref}/1E9",
                        self._unit_number_format("number", "GB"),
                        value,
                    )
                cache_ref = local_memory_refs[(label, cache_specs[-1][0])]
                cache_value = allocated_cache["total"] / 1e9
                ws.write_formula(
                    rank_row,
                    6,
                    f"={cache_ref}/1E9",
                    self._unit_number_format("number", "GB"),
                    cache_value,
                )
                ws.write_formula(
                    rank_row,
                    7,
                    f"=F{rank_row + 1}+G{rank_row + 1}",
                    self._unit_number_format("number", "GB"),
                    category_values_for_rank[3] + cache_value,
                )
                count_ref = local_memory_refs[(label, "Total Parameter Count")]
                ws.write_formula(
                    rank_row,
                    8,
                    f"={count_ref}/1E9",
                    self._unit_number_format("number", "G"),
                    category_values["Total"][f"{label.lower()}_count"] / 1e9,
                )
                ws.write(rank_row, 9, f"All {p_config.tp} ranks identical", self.formats["text"])
                rank_row += 1
            ws.add_table(
                rank_header,
                0,
                rank_row - 1,
                len(rank_headers) - 1,
                {
                    "name": f"{prefix}_Embedded_Rank_Memory",
                    "style": "Table Style Medium 4",
                    "columns": [{"header": header} for header in rank_headers],
                },
            )
            ws.set_column(32, raw_tp8_cache_col, None, None, {"hidden": True})
            ws.set_column("A:A", 20)
            ws.set_column("B:B", 34)
            ws.set_column("C:C", 18)
            ws.set_column("D:D", 44)
            ws.set_column("E:G", 18)
            ws.set_column("H:H", 58)
            ws.set_column("I:J", 18)
            self.scenario_next_rows[mode] = rank_row + 2
        return memory

    def write_dtype(self) -> None:
        dtype_rows = [
            ("Global config", "torch_dtype / main activations", "BF16", "BF16", "config.json 的 torch_dtype=bfloat16；隐藏状态与残差主路径使用 BF16。"),
            ("Global config", "Standard quantized linear layers", "FP8 E4M3 + FP8 E8M0 scale factors", "Dynamic FP8 activation quantization", "quantization_config：quant_method=fp8，fmt=e4m3，scale_fmt=ue8m0。"),
            ("All layers 0-42", "Main attention projections wq_a/wq_b/wkv/wo_b", "FP8 E4M3 + FP8 E8M0 scale factors", "FP8 GEMM; BF16 input dynamically quantized", "大多数注意力线性权重。"),
            ("All layers 0-42", "Attention output projection wo_a", "BF16", "BF16 tensor product (einsum)", "检查点中为 FP8，转换后在推理实现中按 BF16 使用。"),
            ("All layers 0-42", "RMSNorm weights and normalization", "BF16 weights; FP32 implementation parameters", "FP32 compute, output cast to BF16", "Norm 计算先转 FP32 以提高稳定性。"),
            ("Ratio-4 / ratio-128 layers", "Main Compressor wkv/wgate/ape", "Checkpoint mainly BF16; FP32 implementation parameters", "FP32 compression/softmax", "Compressor 参数在推理实现中提升到 FP32。"),
            ("Ratio-4 layers", "Indexer projections and scoring", "wq_b FP8; weights_proj BF16", "QAT/FP4 simulated QKV; FP32 scoring", "仅 ratio-4 层启用 Indexer。"),
            ("All layers 0-42", "Routed experts w1/w2/w3", "FP4 E2M1 packed + FP8 E8M0 scale factors", "FP4 GEMM; SwiGLU FP32; output cast to BF16", "专家数据类型为 fp4；每个令牌激活 6 个路由专家。"),
            ("All layers 0-42", "Shared experts w1/w2/w3", "FP8 E4M3 + FP8 E8M0 scale factors", "FP8 GEMM; SwiGLU FP32", "每层 1 个共享专家，跨 TP Rank 复制。"),
            ("All layers 0-42", "Router scores/routing weights", "FP32 bias; INT32 hash table", "FP32", "sqrt(softplus) 评分、Top-K 和归一化在 FP32 中完成。"),
            ("Layers 0-2", "Hash routing table", "INT32", "INT32 indexing", "前 3 层使用令牌编号到专家编号的预计算路由。"),
            ("All layers 0-42", "mHC parameters / attention sinks", "FP32", "FP32", "HC 混合、Sinkhorn 与 Sink 参数使用 FP32。"),
            ("All layers 0-42", "KV cache", "BF16", "BF16; partial non-RoPE dimensions simulated with FP8", "当前推理实现中的 KV 缓存为 BF16。"),
            ("Model tail", "Final normalization / HC head", "FP32 inference parameters", "FP32 compute, output cast to BF16", "最终 HC 归约与 RMSNorm。"),
            ("Model tail", "LM head / Logits", "Checkpoint BF16; FP32 implementation parameters", "FP32 linear layer and Logits", "词表并行；Logits 以 FP32 计算。"),
        ]
        module_headers = ["Scope", "Module/tensor", "Checkpoint/parameter storage", "Activation/intermediate computation", "Notes"]
        layer_headers = [
            "Layer ID",
            "Attention mode",
            "Compression ratio",
            "Routing mode",
            "Hidden state / residual",
            "Main attention projections",
            "Routed experts",
            "Shared experts",
            "Normalization / HC / Router computation",
        ]
        ws = self.workbook.add_worksheet("dtype")
        ws.hide_gridlines(2)
        ws.freeze_panes(3, 0)
        ws.write(0, 0, "Inference-Path Data Types (dtype) Overview", self.formats["title"])
        ws.write(
            1,
            0,
            "集中记录各算子模块的参数存储、激活类型和逐层注意力/路由模式。",
            self.formats["note"],
        )

        module_section = 3
        ws.merge_range(
            module_section,
            0,
            module_section,
            4,
            "Data types (dtype): module-level storage and compute types",
            self.formats["section"],
        )
        module_header = module_section + 1
        module_first_row = module_header + 1
        module_last_row = module_first_row + len(dtype_rows) - 1
        ws.write_row(module_header, 0, module_headers, self.formats["header"])
        for offset, values in enumerate(dtype_rows):
            ws.write_row(module_first_row + offset, 0, values, self.formats["text"])
        ws.add_table(
            module_header,
            0,
            module_last_row,
            len(module_headers) - 1,
            {
                "name": "DType_Module",
                "style": "Table Style Medium 2",
                "columns": [{"header": header} for header in module_headers],
            },
        )

        layer_section = module_last_row + 2
        ws.merge_range(
            layer_section,
            0,
            layer_section,
            8,
            "43-layer dtype, attention, and routing modes",
            self.formats["section"],
        )
        layer_header = layer_section + 1
        layer_first_row = layer_header + 1
        layer_last_row = layer_first_row + self.layers.total - 1
        ws.write_row(layer_header, 0, layer_headers, self.formats["header"])
        for layer_id, ratio in enumerate(self.ratios[: self.layers.total]):
            if ratio == 0:
                attention_mode = "Window"
            elif ratio == self.p.short_ratio:
                attention_mode = "Short-compression sparse"
            elif ratio == self.p.long_ratio:
                attention_mode = "Long-compression"
            else:
                attention_mode = "Custom ratio"
            router_mode = (
                "Hash routing (INT32)"
                if layer_id < self.p.hash_layers
                else "Score routing (FP32)"
            )
            ws.write_row(
                layer_first_row + layer_id,
                0,
                [
                    layer_id,
                    attention_mode,
                    ratio,
                    router_mode,
                    "BF16",
                    "FP8 E4M3; wo_a BF16",
                    "FP4 E2M1",
                    "FP8 E4M3",
                    "FP32",
                ],
                self.formats["text"],
            )
        ws.add_table(
            layer_header,
            0,
            layer_last_row,
            len(layer_headers) - 1,
            {
                "name": "DType_Layer",
                "style": "Table Style Medium 4",
                "columns": [{"header": header} for header in layer_headers],
            },
        )
        ws.set_column("A:A", 20)
        ws.set_column("B:B", 34)
        ws.set_column("C:C", 20)
        ws.set_column("D:D", 28)
        ws.set_column("E:I", 24)

    def write_memory(self) -> dict[str, float]:
        return self.write_embedded_memory()

    def write_methodology(self) -> None:
        ws = self.workbook.add_worksheet("Methodology")
        ws.hide_gridlines(2)
        ws.write(0, 0, "Methodology and Limits", self.formats["title"])
        rows = [
            ("Scope", "仅统计 43 层主推理路径；不含 MTP、反向传播、梯度、优化器状态和训练激活。Layer_Config 最多可维护 64 层。"),
            ("FLOPs", "矩阵乘加按 2 FLOPs 计。注意力与 MoE 主要 GEMM 使用显式公式；HC 与逐元素运算为近似估计。"),
            ("Prefill attention", "使用因果候选对数量，不直接使用 S×S；原始窗口、压缩 KV 和 Indexer 扫描分别统计。"),
            ("Decode attention", "每个解码令牌在解码上下文长度下建模；ratio-4 主注意力取 Top-K，但 Indexer 需要扫描全部已完成压缩项。"),
            ("MoE", "每个令牌执行 Top-K 路由专家和共享专家。HBM 参数读取按专家至少被命中一次的概率估计。"),
            ("HBM traffic", "统计本地逻辑参数、激活和 KV 的读取与写入；不模拟 L2 命中、算子融合、分块、Allocator 或厂商 Kernel 内部复用。"),
            ("Memory", "按推理运行 dtype 估算：路由专家 FP4、多数投影 FP8、Wo_a BF16、Compressor FP32、LM Head FP32，并计入量化 Scale。"),
            ("KV cache", "有效容量表示已填充项；预分配容量使用 MaxContext。当前实现的 KV 和 Compressor 状态在每个 TP Rank 上完整复制。"),
            ("TP", "Wq_a、Wkv、Compressor、Router、HC、共享专家复制；Q/O 部分投影、路由专家、Embedding 和 LM Head 按 TP 分片。"),
            ("Communication", "Ring 公式给出每 Rank 发送加接收的数据量。TP1 为 0；该数值是传输量，不是实测 GB/s 或时延。"),
            ("Roofline", "计算/HBM/互连下界使用参数页中的可编辑硬件峰值并假设互不重叠；真实运行时间必须通过性能分析验证。"),
            ("Units", "FLOPs 使用 M/G/T/P 十进制单位；容量与流量使用 KB/MB/GB/TB，1 GB=10^9 字节；场景页容量明细保留原始公式。"),
            ("Formula maintenance", "蓝色单元格为输入；结果由 Excel 公式计算并在打开时完整重算。隐藏列保留原始未缩放值，便于审计。"),
        ]
        ws.write_row(2, 0, ["Topic", "Definition"], self.formats["header"])
        for row, (topic, definition) in enumerate(rows, start=3):
            ws.write(row, 0, topic, self.formats["text"])
            ws.write(row, 1, definition, self.formats["text"])
        ws.set_column("A:A", 24)
        ws.set_column("B:B", 120)

    def write_comparison(self, memory: dict[str, float]) -> None:
        ws = self.workbook.add_worksheet("Comparison")
        ws.hide_gridlines(2)
        ws.write(0, 0, "TP1 / TP8 Inference Resource Comparison Charts", self.formats["title"])
        ws.write(
            1,
            0,
            "计算量、HBM、参数、缓存与驻留容量集中在一张资源表；隐藏列保留公式原值和图表缩放源。",
            self.formats["note"],
        )

        pf1_items, pf8_items, _ = self._scenario_inputs("prefill")
        dc1_items, dc8_items, _ = self._scenario_inputs("decode")
        pf1 = summarize_items(pf1_items)
        pf8 = summarize_items(pf8_items)
        dc1 = summarize_items(dc1_items)
        dc8 = summarize_items(dc8_items)

        def scenario_ref(prefix: str, tp: str, label: str) -> str:
            return self.scenario_refs[(prefix, tp, label)]

        def memory_ref(tp: str, label: str) -> str:
            return self.memory_refs[(tp, label)]

        def category_hbm(summary: dict[str, Any], category: str) -> float:
            values = summary["categories"][category]
            return values["hbm_read_bytes_per_rank"] + values["hbm_write_bytes_per_rank"]

        def parameter_value(tp: str, category: str, kind: str) -> float:
            parameters = parameter_components(
                self.p_tp1 if tp == "TP1" else self.p_tp8,
                self.layers,
            )
            if category != "Total":
                parameters = [
                    parameter
                    for parameter in parameters
                    if parameter.category == category
                ]
            return sum(
                parameter.rank_count if kind == "params" else parameter.rank_bytes
                for parameter in parameters
            )

        cache_values_by_mode = {
            "prefill": (
                cache_values(
                    self.p,
                    self.layers,
                    self.p.prefill_sequence,
                    self.p.prefill_batch,
                ),
                cache_values(
                    self.p,
                    self.layers,
                    self.p.max_context,
                    self.p.prefill_batch,
                )["total"],
            ),
            "decode": (
                cache_values(
                    self.p,
                    self.layers,
                    self.p.decode_context,
                    self.p.decode_batch,
                ),
                cache_values(
                    self.p,
                    self.layers,
                    self.p.max_context,
                    self.p.decode_batch,
                )["total"],
            ),
        }

        def cache_number(label: str, mode: str) -> float:
            effective, allocated = cache_values_by_mode[mode]
            if "main" in label:
                return effective["main"]
            if "Indexer" in label:
                return effective["indexer"]
            if "compressor" in label:
                return effective["states"]
            return allocated

        rows: list[
            tuple[
                str,
                list[tuple[str, float, str]],
                str,
                bool,
            ]
        ] = []

        def add_row(
            label: str,
            entries: list[tuple[str, float, str]],
            note: str,
            total: bool = False,
        ) -> None:
            rows.append((label, entries, note, total))

        for label, source_label, category in (
            ("Attention FLOPs", "Attention major FLOPs", "Attention"),
            ("MoE FLOPs", "MoE major FLOPs", "MoE"),
            ("Other inference FLOPs", "Other inference FLOPs", "Other"),
            ("Total inference FLOPs", "Total inference FLOPs", "Total"),
        ):
            if category == "Attention":
                values = (
                    pf1["attention_major_flops_per_rank"],
                    pf8["attention_major_flops_per_rank"],
                    dc1["attention_major_flops_per_rank"],
                    dc8["attention_major_flops_per_rank"],
                )
            elif category == "MoE":
                values = (
                    pf1["moe_major_flops_per_rank"],
                    pf8["moe_major_flops_per_rank"],
                    dc1["moe_major_flops_per_rank"],
                    dc8["moe_major_flops_per_rank"],
                )
            elif category == "Other":
                values = tuple(
                    summary["categories"]["Other"]["per_rank_flops"]
                    for summary in (pf1, pf8, dc1, dc8)
                )
            else:
                values = tuple(
                    summary["total_per_rank_flops"]
                    for summary in (pf1, pf8, dc1, dc8)
                )
            add_row(
                label,
                [
                    (scenario_ref("PF", "TP1", source_label), values[0], "flops"),
                    (scenario_ref("PF", "TP8", source_label), values[1], "flops"),
                    (scenario_ref("DC", "TP1", source_label), values[2], "flops"),
                    (scenario_ref("DC", "TP8", source_label), values[3], "flops"),
                ],
                "单次预填充或解码步骤的每 Rank 逻辑 FLOPs。",
                category == "Total",
            )

        for label, source_label, category in (
            ("Attention HBM traffic", "Attention HBM traffic", "Attention"),
            ("MoE HBM traffic", "MoE HBM traffic", "MoE"),
            ("Other HBM traffic", "Other HBM traffic", "Other"),
            ("Total HBM traffic", "Total HBM traffic", "Total"),
        ):
            values = tuple(
                (
                    summary["total_hbm_read_bytes_per_rank"]
                    + summary["total_hbm_write_bytes_per_rank"]
                    if category == "Total"
                    else category_hbm(summary, category)
                )
                for summary in (pf1, pf8, dc1, dc8)
            )
            add_row(
                label,
                [
                    (scenario_ref("PF", "TP1", source_label), values[0], "bytes"),
                    (scenario_ref("PF", "TP8", source_label), values[1], "bytes"),
                    (scenario_ref("DC", "TP1", source_label), values[2], "bytes"),
                    (scenario_ref("DC", "TP8", source_label), values[3], "bytes"),
                ],
                "本地逻辑 HBM 读写量，不含缓存复用和算子融合。",
                category == "Total",
            )

        for label, source_label, category in (
            ("Attention required HBM bandwidth", "Attention HBM traffic", "Attention"),
            ("MoE required HBM bandwidth", "MoE HBM traffic", "MoE"),
            ("Other required HBM bandwidth", "Other HBM traffic", "Other"),
        ):
            prefill_values = (
                category_hbm(pf1, category)
                / (self.p.prefill_target_ms / 1000)
                / 1e9,
                category_hbm(pf8, category)
                / (self.p.prefill_target_ms / 1000)
                / 1e9,
            )
            decode_values = (
                category_hbm(dc1, category)
                / (self.p.decode_target_ms / 1000)
                / 1e9,
                category_hbm(dc8, category)
                / (self.p.decode_target_ms / 1000)
                / 1e9,
            )
            add_row(
                label,
                [
                    (
                        f"{scenario_ref('PF', 'TP1', source_label)}/(PrefillTargetMs/1000)/1E9",
                        prefill_values[0],
                        "GB/s",
                    ),
                    (
                        f"{scenario_ref('PF', 'TP8', source_label)}/(PrefillTargetMs/1000)/1E9",
                        prefill_values[1],
                        "GB/s",
                    ),
                    (
                        f"{scenario_ref('DC', 'TP1', source_label)}/(DecodeTargetMs/1000)/1E9",
                        decode_values[0],
                        "GB/s",
                    ),
                    (
                        f"{scenario_ref('DC', 'TP8', source_label)}/(DecodeTargetMs/1000)/1E9",
                        decode_values[1],
                        "GB/s",
                    ),
                ],
                "按参数页中的目标时延换算；不是实测带宽。",
            )

        required_compute_values = (
            pf1["total_per_rank_flops"] / (self.p.prefill_target_ms / 1000) / 1e12,
            pf8["total_per_rank_flops"] / (self.p.prefill_target_ms / 1000) / 1e12,
            dc1["total_per_rank_flops"] / (self.p.decode_target_ms / 1000) / 1e12,
            dc8["total_per_rank_flops"] / (self.p.decode_target_ms / 1000) / 1e12,
        )
        add_row(
            "Compute required at target latency",
            [
                (
                    f"{scenario_ref('PF', 'TP1', 'Total inference FLOPs')}/(PrefillTargetMs/1000)/1E12",
                    required_compute_values[0],
                    "TFLOP/s",
                ),
                (
                    f"{scenario_ref('PF', 'TP8', 'Total inference FLOPs')}/(PrefillTargetMs/1000)/1E12",
                    required_compute_values[1],
                    "TFLOP/s",
                ),
                (
                    f"{scenario_ref('DC', 'TP1', 'Total inference FLOPs')}/(DecodeTargetMs/1000)/1E12",
                    required_compute_values[2],
                    "TFLOP/s",
                ),
                (
                    f"{scenario_ref('DC', 'TP8', 'Total inference FLOPs')}/(DecodeTargetMs/1000)/1E12",
                    required_compute_values[3],
                    "TFLOP/s",
                ),
            ],
            "按目标时延折算的每 Rank 最低计算吞吐需求。",
        )

        interconnect_values = (
            pf1["total_interconnect_bytes_per_rank"],
            pf8["total_interconnect_bytes_per_rank"],
            dc1["total_interconnect_bytes_per_rank"],
            dc8["total_interconnect_bytes_per_rank"],
        )
        add_row(
            "Interconnect transfer",
            [
                (scenario_ref("PF", "TP1", "Interconnect transfer"), interconnect_values[0], "bytes"),
                (scenario_ref("PF", "TP8", "Interconnect transfer"), interconnect_values[1], "bytes"),
                (scenario_ref("DC", "TP1", "Interconnect transfer"), interconnect_values[2], "bytes"),
                (scenario_ref("DC", "TP8", "Interconnect transfer"), interconnect_values[3], "bytes"),
            ],
            "Ring 集合通信每 Rank 发送加接收；TP1 为 0。",
        )
        interconnect_bandwidth = (
            interconnect_values[0] / (self.p.prefill_target_ms / 1000) / 1e9,
            interconnect_values[1] / (self.p.prefill_target_ms / 1000) / 1e9,
            interconnect_values[2] / (self.p.decode_target_ms / 1000) / 1e9,
            interconnect_values[3] / (self.p.decode_target_ms / 1000) / 1e9,
        )
        add_row(
            "Required interconnect bandwidth",
            [
                (
                    f"{scenario_ref('PF', 'TP1', 'Interconnect transfer')}/(PrefillTargetMs/1000)/1E9",
                    interconnect_bandwidth[0],
                    "GB/s",
                ),
                (
                    f"{scenario_ref('PF', 'TP8', 'Interconnect transfer')}/(PrefillTargetMs/1000)/1E9",
                    interconnect_bandwidth[1],
                    "GB/s",
                ),
                (
                    f"{scenario_ref('DC', 'TP1', 'Interconnect transfer')}/(DecodeTargetMs/1000)/1E9",
                    interconnect_bandwidth[2],
                    "GB/s",
                ),
                (
                    f"{scenario_ref('DC', 'TP8', 'Interconnect transfer')}/(DecodeTargetMs/1000)/1E9",
                    interconnect_bandwidth[3],
                    "GB/s",
                ),
            ],
            "未考虑通信与计算重叠、拓扑和协议效率。",
        )

        for label, category, kind, note in (
            ("Attention logical parameter count", "Attention", "params", "注意力投影、Compressor、Indexer。"),
            ("MoE logical parameter count", "MoE", "params", "路由/共享专家与 Router。"),
            ("Other logical parameter count", "Other", "params", "Embedding、LM Head、HC 与 Norm。"),
            ("Total logical parameter count", "Total", "params", "不包含 MTP。"),
            ("Attention parameter capacity", "Attention", "bytes", "按推理 dtype 与量化 Scale 计算。"),
            ("MoE parameter capacity", "MoE", "bytes", "路由专家 FP4、共享专家 FP8、Router 复制。"),
            ("Other parameter capacity", "Other", "bytes", "Embedding、LM Head、HC 与 Norm。"),
            ("Total parameter capacity", "Total", "bytes", "每 Rank 静态参数驻留。"),
        ):
            memory_label = f"{category} Parameter {'Count' if kind == 'params' else 'Capacity'}"
            values = (
                parameter_value("TP1", category, kind),
                parameter_value("TP8", category, kind),
                parameter_value("TP1", category, kind),
                parameter_value("TP8", category, kind),
            )
            add_row(
                label,
                [
                    (memory_ref("TP1", memory_label), values[0], kind),
                    (memory_ref("TP8", memory_label), values[1], kind),
                    (memory_ref("TP1", memory_label), values[2], kind),
                    (memory_ref("TP8", memory_label), values[3], kind),
                ],
                note,
                category == "Total",
            )

        for label, prefill_label, decode_label in (
            ("Effective main KV cache", "Prefill effective main KV", "Decode effective main KV"),
            ("Effective Indexer KV cache", "Prefill effective Indexer KV", "Decode effective Indexer KV"),
            ("Compressor State", "Prefill compressor states", "Decode compressor states"),
            ("Preallocated KV + state", "Prefill preallocated KV + states", "Decode preallocated KV + states"),
        ):
            values = (
                cache_number(prefill_label, "prefill"),
                cache_number(prefill_label, "prefill"),
                cache_number(decode_label, "decode"),
                cache_number(decode_label, "decode"),
            )
            add_row(
                label,
                [
                    (memory_ref("TP1", prefill_label), values[0], "bytes"),
                    (memory_ref("TP8", prefill_label), values[1], "bytes"),
                    (memory_ref("TP1", decode_label), values[2], "bytes"),
                    (memory_ref("TP8", decode_label), values[3], "bytes"),
                ],
                "KV 与 Compressor 状态在当前实现中每个 TP Rank 复制。",
            )

        resident_values = (
            memory["tp1_parameter_total"] + memory["prefill_allocated_cache"],
            memory["tp8_parameter_total"] + memory["prefill_allocated_cache"],
            memory["tp1_parameter_total"] + memory["decode_allocated_cache"],
            memory["tp8_parameter_total"] + memory["decode_allocated_cache"],
        )
        add_row(
            "Total resident capacity",
            [
                (
                    f"{memory_ref('TP1', 'Total Parameter Capacity')}+{memory_ref('TP1', 'Prefill preallocated KV + states')}",
                    resident_values[0],
                    "bytes",
                ),
                (
                    f"{memory_ref('TP8', 'Total Parameter Capacity')}+{memory_ref('TP8', 'Prefill preallocated KV + states')}",
                    resident_values[1],
                    "bytes",
                ),
                (
                    f"{memory_ref('TP1', 'Total Parameter Capacity')}+{memory_ref('TP1', 'Decode preallocated KV + states')}",
                    resident_values[2],
                    "bytes",
                ),
                (
                    f"{memory_ref('TP8', 'Total Parameter Capacity')}+{memory_ref('TP8', 'Decode preallocated KV + states')}",
                    resident_values[3],
                    "bytes",
                ),
            ],
                "参数 + 按 MaxContext 预分配的 KV/状态；不含临时工作区。",
            True,
        )
        total_flops_values = (
            pf1["total_per_rank_flops"],
            pf8["total_per_rank_flops"],
            dc1["total_per_rank_flops"],
            dc8["total_per_rank_flops"],
        )
        add_row(
            "One-inference compute demand",
            [
                (scenario_ref("PF", "TP1", "Total inference FLOPs"), total_flops_values[0], "flops"),
                (scenario_ref("PF", "TP8", "Total inference FLOPs"), total_flops_values[1], "flops"),
                (scenario_ref("DC", "TP1", "Total inference FLOPs"), total_flops_values[2], "flops"),
                (scenario_ref("DC", "TP8", "Total inference FLOPs"), total_flops_values[3], "flops"),
            ],
            "单次推理调用的单 Rank 总 FLOPs；不除以目标时延。",
            True,
        )

        headers = [
            "Resource",
            "Prefill TP1",
            "Prefill TP8/rank",
            "Decode TP1",
            "Decode TP8/rank",
            "Notes",
        ]
        table_header = 3
        table_first_row = table_header + 1
        raw_cols = (6, 7, 8, 9)
        row_by_label: dict[str, int] = {}
        ws.write_row(table_header, 0, headers, self.formats["header"])
        for offset, (label, entries, note, total) in enumerate(rows):
            row = table_first_row + offset
            row_by_label[label] = row
            ws.write(row, 0, display_label(label), self.formats["total" if total else "text"])
            for value_index, (reference, value, kind) in enumerate(entries):
                formula = reference if reference.startswith("=") else f"={reference}"
                self._write_human_value(
                    ws,
                    row,
                    value_index + 1,
                    value_index + 1,
                    raw_cols[value_index],
                    formula,
                    value,
                    kind,
                    total,
                )
            ws.write(row, 5, note, self.formats["text"])
        table_last_row = table_first_row + len(rows) - 1
        ws.add_table(
            table_header,
            0,
            table_last_row,
            len(headers) - 1,
            {
                "name": "Comparison_Resource_Overview",
                "style": "Table Style Medium 2",
                "columns": [{"header": header} for header in headers],
            },
        )

        ws.set_column(raw_cols[0], raw_cols[-1], None, None, {"hidden": True})
        helper_start_col = 18
        helper_next_row = table_last_row + 3

        def add_chart(
            title: str,
            anchor: str,
            category_labels: list[str],
            series_sources: list[tuple[str, list[tuple[int, int, float, float]]]],
            y_name: str,
            number_format: str,
        ) -> None:
            nonlocal helper_next_row
            block_header = helper_next_row
            ws.write(block_header, helper_start_col, "Category", self.formats["header"])
            for series_index, (series_name, _) in enumerate(series_sources, start=1):
                ws.write(
                    block_header,
                    helper_start_col + series_index,
                    series_name,
                    self.formats["header"],
                )
            for category_index, category_label in enumerate(category_labels):
                row = block_header + 1 + category_index
                ws.write(row, helper_start_col, category_label, self.formats["text"])
                for series_index, (_, source_rows) in enumerate(series_sources, start=1):
                    table_row, raw_col, divisor, cached_value = source_rows[category_index]
                    raw_ref = f"{xl_col_to_name(raw_col)}{table_row + 1}"
                    ws.write_formula(
                        row,
                        helper_start_col + series_index,
                        f"={raw_ref}/{divisor:g}",
                        self.formats["number"],
                        cached_value / divisor,
                    )
            chart = self.workbook.add_chart({"type": "column"})
            for series_index, (series_name, _) in enumerate(series_sources, start=1):
                source_col = helper_start_col + series_index
                chart.add_series(
                    {
                        "name": series_name,
                        "gap": CHART_CATEGORY_GAP,
                        "categories": [
                            "Comparison",
                            block_header + 1,
                            helper_start_col,
                            block_header + len(category_labels),
                            helper_start_col,
                        ],
                        "values": [
                            "Comparison",
                            block_header + 1,
                            source_col,
                            block_header + len(category_labels),
                            source_col,
                        ],
                        "data_labels": {
                            "value": True,
                            "num_format": number_format,
                        },
                    }
                )
            chart.show_hidden_data()
            chart.set_title({"name": title})
            chart.set_y_axis(
                {
                    "name": y_name,
                    "num_format": number_format,
                    "major_gridlines": {"visible": True},
                }
            )
            chart.set_x_axis({"label_position": "low"})
            chart.set_legend({"position": "bottom"})
            chart.set_style(10)
            ws.insert_chart(anchor, chart, {"x_scale": 1.28, "y_scale": 1.15})
            helper_next_row = block_header + max(len(category_labels), 4) + 3

        def source_for(
            label: str,
            raw_col: int,
            divisor: float,
        ) -> tuple[int, int, float, float]:
            table_row = row_by_label[label]
            raw_value = rows[table_row - table_first_row][1][raw_cols.index(raw_col)][1]
            return table_row, raw_col, divisor, raw_value

        anchor_row = table_last_row + 4
        add_chart(
            "Total compute per inference / rank",
            f"A{anchor_row}",
            ["Prefill TP1", "Prefill TP8", "Decode TP1", "Decode TP8"],
            [
                (
                    "One-inference TFLOPs",
                    [
                        source_for("One-inference compute demand", raw_col, 1e12)
                        for raw_col in raw_cols
                    ],
                )
            ],
            "TFLOPs",
            '0.0" TFLOPs"',
        )
        add_chart(
            "Prefill compute per rank",
            f"A{anchor_row + 17}",
            ["Attention", "MoE", "Other"],
            [
                (
                    "TP1 TFLOPs",
                    [
                        source_for(label, raw_cols[0], 1e12)
                        for label in ("Attention FLOPs", "MoE FLOPs", "Other inference FLOPs")
                    ],
                ),
                (
                    "TP8 TFLOPs",
                    [
                        source_for(label, raw_cols[1], 1e12)
                        for label in ("Attention FLOPs", "MoE FLOPs", "Other inference FLOPs")
                    ],
                ),
            ],
            "TFLOPs",
            '0.0" TFLOPs"',
        )
        add_chart(
            "Decode compute per rank",
            f"J{anchor_row + 17}",
            ["Attention", "MoE", "Other"],
            [
                (
                    "TP1 GFLOPs",
                    [
                        source_for(label, raw_cols[2], 1e9)
                        for label in ("Attention FLOPs", "MoE FLOPs", "Other inference FLOPs")
                    ],
                ),
                (
                    "TP8 GFLOPs",
                    [
                        source_for(label, raw_cols[3], 1e9)
                        for label in ("Attention FLOPs", "MoE FLOPs", "Other inference FLOPs")
                    ],
                ),
            ],
            "GFLOPs",
            '0.0" GFLOPs"',
        )
        add_chart(
            "Prefill HBM traffic per rank",
            f"A{anchor_row + 34}",
            ["Attention", "MoE", "Other"],
            [
                (
                    "TP1 GB",
                    [
                        source_for(label, raw_cols[0], 1e9)
                        for label in ("Attention HBM traffic", "MoE HBM traffic", "Other HBM traffic")
                    ],
                ),
                (
                    "TP8 GB",
                    [
                        source_for(label, raw_cols[1], 1e9)
                        for label in ("Attention HBM traffic", "MoE HBM traffic", "Other HBM traffic")
                    ],
                ),
            ],
            "GB",
            '0.0" GB"',
        )
        add_chart(
            "Decode HBM traffic per rank",
            f"J{anchor_row + 34}",
            ["Attention", "MoE", "Other"],
            [
                (
                    "TP1 GB",
                    [
                        source_for(label, raw_cols[2], 1e9)
                        for label in ("Attention HBM traffic", "MoE HBM traffic", "Other HBM traffic")
                    ],
                ),
                (
                    "TP8 GB",
                    [
                        source_for(label, raw_cols[3], 1e9)
                        for label in ("Attention HBM traffic", "MoE HBM traffic", "Other HBM traffic")
                    ],
                ),
            ],
            "GB",
            '0.0" GB"',
        )
        add_chart(
            "Static parameter capacity per rank",
            f"A{anchor_row + 51}",
            ["Attention", "MoE", "Other"],
            [
                (
                    "TP1 GB",
                    [
                        source_for(label, raw_cols[0], 1e9)
                        for label in ("Attention parameter capacity", "MoE parameter capacity", "Other parameter capacity")
                    ],
                ),
                (
                    "TP8 GB",
                    [
                        source_for(label, raw_cols[1], 1e9)
                        for label in ("Attention parameter capacity", "MoE parameter capacity", "Other parameter capacity")
                    ],
                ),
            ],
            "GB",
            '0.0" GB"',
        )
        add_chart(
            "TP1 vs TP8: inference resident capacity per rank",
            f"J{anchor_row + 51}",
            ["Prefill", "Decode"],
            [
                (
                    "TP1 GB",
                    [
                        source_for("Total resident capacity", raw_cols[0], 1e9),
                        source_for("Total resident capacity", raw_cols[2], 1e9),
                    ],
                ),
                (
                    "TP8 GB",
                    [
                        source_for("Total resident capacity", raw_cols[1], 1e9),
                        source_for("Total resident capacity", raw_cols[3], 1e9),
                    ],
                ),
            ],
            "GB",
            '0.0" GB"',
        )
        ws.set_column("A:A", 30)
        ws.set_column("B:E", 20)
        ws.set_column("F:F", 72)
        ws.set_column(helper_start_col, helper_start_col + 2, None, None, {"hidden": True})

    def write_summary(self, memory: dict[str, float]) -> None:
        """Write two independent per-rank TP sections; never aggregate TP1 and TP8."""
        sheet = "Summary"
        ws = self.workbook.add_worksheet(sheet)
        ws.activate()
        ws.hide_gridlines(2)
        ws.freeze_panes(3, 0)
        ws.write(0, 0, "Per-Rank Inference Hardware Resource Summary", self.formats["title"])
        ws.write(
            1,
            0,
            "TP1 与 TP8 分区独立展示；所有值均为单个 Rank，不进行跨 Rank 或跨 TP 求和。",
            self.formats["note"],
        )

        pf1 = summarize_items(self._scenario_inputs("prefill")[0])
        pf8 = summarize_items(self._scenario_inputs("prefill")[1])
        dc1 = summarize_items(self._scenario_inputs("decode")[0])
        dc8 = summarize_items(self._scenario_inputs("decode")[1])
        raw_prefill_col = 6
        raw_decode_col = 7

        def scenario_ref(prefix: str, tp: str, label: str) -> str:
            return self.scenario_refs[(prefix, tp, label)]

        def memory_ref(tp: str, label: str) -> str:
            return self.memory_refs[(tp, label)]

        def category_hbm(summary: dict[str, Any], category: str) -> float:
            data = summary["categories"][category]
            return data["hbm_read_bytes_per_rank"] + data["hbm_write_bytes_per_rank"]

        def parameter_value(tp: str, category: str, kind: str) -> float:
            p_config = self.p_tp1 if tp == "TP1" else self.p_tp8
            selected = parameter_components(p_config, self.layers)
            if category != "Total":
                selected = [item for item in selected if item.category == category]
            return sum(
                item.rank_count if kind == "params" else item.rank_bytes
                for item in selected
            )

        def cache_number(label: str, mode: str) -> float:
            if mode == "prefill":
                values = cache_values(
                    self.p,
                    self.layers,
                    self.p.prefill_sequence,
                    self.p.prefill_batch,
                )
                allocated = cache_values(
                    self.p,
                    self.layers,
                    self.p.max_context,
                    self.p.prefill_batch,
                )["total"]
            else:
                values = cache_values(
                    self.p,
                    self.layers,
                    self.p.decode_context,
                    self.p.decode_batch,
                )
                allocated = cache_values(
                    self.p,
                    self.layers,
                    self.p.max_context,
                    self.p.decode_batch,
                )["total"]
            if "main" in label:
                return values["main"]
            if "Indexer" in label:
                return values["indexer"]
            if "compressor" in label:
                return values["states"]
            return allocated

        def write_tp_section(
            start_row: int,
            tp: str,
            prefill_summary: dict[str, Any],
            decode_summary: dict[str, Any],
        ) -> int:
            size = self.p.tp if tp == "TP1" else self.p.comparison_tp
            ws.merge_range(
                start_row,
                0,
                start_row,
                3,
                f"{tp} (configured size={size}) per-rank resources",
                self.formats["section"],
            )
            header_row = start_row + 1
            ws.write_row(
                header_row,
                0,
                ["Resource", "Prefill/rank", "Decode/rank", "Notes"],
                self.formats["header"],
            )
            specs: list[
                tuple[str, tuple[str, float], tuple[str, float], str, str]
            ] = []
            for label, source_label, category in (
                ("Attention FLOPs", "Attention major FLOPs", "Attention"),
                ("MoE FLOPs", "MoE major FLOPs", "MoE"),
                ("Other inference FLOPs", "Other inference FLOPs", "Other"),
                ("Total inference FLOPs", "Total inference FLOPs", "Total"),
            ):
                if category == "Attention":
                    values = (
                        prefill_summary["attention_major_flops_per_rank"],
                        decode_summary["attention_major_flops_per_rank"],
                    )
                elif category == "MoE":
                    values = (
                        prefill_summary["moe_major_flops_per_rank"],
                        decode_summary["moe_major_flops_per_rank"],
                    )
                elif category == "Other":
                    values = (
                        prefill_summary["categories"]["Other"]["per_rank_flops"],
                        decode_summary["categories"]["Other"]["per_rank_flops"],
                    )
                else:
                    values = (
                        prefill_summary["total_per_rank_flops"],
                        decode_summary["total_per_rank_flops"],
                    )
                specs.append(
                    (
                        label,
                        (scenario_ref("PF", tp, source_label), values[0]),
                        (scenario_ref("DC", tp, source_label), values[1]),
                        "flops",
                        "单次推理调用的单 Rank 计算量。",
                    )
                )

            for label, source_label, category in (
                ("Attention HBM traffic", "Attention HBM traffic", "Attention"),
                ("MoE HBM traffic", "MoE HBM traffic", "MoE"),
                ("Other HBM traffic", "Other HBM traffic", "Other"),
                ("Total HBM traffic", "Total HBM traffic", "Total"),
            ):
                if category == "Total":
                    values = (
                        prefill_summary["total_hbm_read_bytes_per_rank"]
                        + prefill_summary["total_hbm_write_bytes_per_rank"],
                        decode_summary["total_hbm_read_bytes_per_rank"]
                        + decode_summary["total_hbm_write_bytes_per_rank"],
                    )
                else:
                    values = (
                        category_hbm(prefill_summary, category),
                        category_hbm(decode_summary, category),
                    )
                specs.append(
                    (
                        label,
                        (scenario_ref("PF", tp, source_label), values[0]),
                        (scenario_ref("DC", tp, source_label), values[1]),
                        "bytes",
                        "本地逻辑 HBM 读写量，不含缓存复用。",
                    )
                )

            for label, source_label, category in (
                ("Attention required HBM bandwidth", "Attention HBM traffic", "Attention"),
                ("MoE required HBM bandwidth", "MoE HBM traffic", "MoE"),
                ("Other required HBM bandwidth", "Other HBM traffic", "Other"),
            ):
                pf_value = (
                    category_hbm(prefill_summary, category)
                    / (self.p.prefill_target_ms / 1000)
                    / 1e9
                )
                dc_value = (
                    category_hbm(decode_summary, category)
                    / (self.p.decode_target_ms / 1000)
                    / 1e9
                )
                specs.append(
                    (
                        label,
                        (
                            f"{scenario_ref('PF', tp, source_label)}/(PrefillTargetMs/1000)/1E9",
                            pf_value,
                        ),
                        (
                            f"{scenario_ref('DC', tp, source_label)}/(DecodeTargetMs/1000)/1E9",
                            dc_value,
                        ),
                        "GB/s",
                        "按参数页中的目标时延换算。",
                    )
                )

            specs.append(
                (
                    "One-inference compute demand",
                    (
                        scenario_ref("PF", tp, "Total inference FLOPs"),
                        prefill_summary["total_per_rank_flops"],
                    ),
                    (
                        scenario_ref("DC", tp, "Total inference FLOPs"),
                        decode_summary["total_per_rank_flops"],
                    ),
                    "flops",
                    "单次推理调用的单 Rank 总 FLOPs；不除以目标时延，延迟由芯片峰值算力另行估算。",
                )
            )
            interconnect = (
                prefill_summary["total_interconnect_bytes_per_rank"],
                decode_summary["total_interconnect_bytes_per_rank"],
            )
            specs.append(
                (
                    "Interconnect transfer",
                    (
                        scenario_ref("PF", tp, "Interconnect transfer"),
                        interconnect[0],
                    ),
                    (
                        scenario_ref("DC", tp, "Interconnect transfer"),
                        interconnect[1],
                    ),
                    "bytes",
                    "Ring 集合通信单 Rank 发送加接收；TP1 为 0。",
                )
            )
            specs.append(
                (
                    "Required interconnect bandwidth",
                    (
                        f"{scenario_ref('PF', tp, 'Interconnect transfer')}/(PrefillTargetMs/1000)/1E9",
                        interconnect[0]
                        / (self.p.prefill_target_ms / 1000)
                        / 1e9,
                    ),
                    (
                        f"{scenario_ref('DC', tp, 'Interconnect transfer')}/(DecodeTargetMs/1000)/1E9",
                        interconnect[1]
                        / (self.p.decode_target_ms / 1000)
                        / 1e9,
                    ),
                    "GB/s",
                    "未考虑通信与计算重叠、拓扑和协议效率。",
                )
            )

            for label, category, kind, note in (
                ("Attention logical parameter count", "Attention", "params", "投影、Compressor、Indexer。"),
                ("MoE logical parameter count", "MoE", "params", "路由/共享专家与 Router。"),
                ("Other logical parameter count", "Other", "params", "Embedding、LM Head、HC、Norm。"),
                ("Total logical parameter count", "Total", "params", "不含 MTP。"),
                ("Attention parameter capacity", "Attention", "bytes", "按推理 dtype 与 Scale 计算。"),
                ("MoE parameter capacity", "MoE", "bytes", "路由专家 FP4，共享专家 FP8。"),
                ("Other parameter capacity", "Other", "bytes", "Embedding、LM Head、HC、Norm。"),
                ("Total parameter capacity", "Total", "bytes", "单 Rank 静态参数驻留。"),
            ):
                memory_label = f"{category} Parameter {'Count' if kind == 'params' else 'Capacity'}"
                value = parameter_value(tp, category, kind)
                ref = memory_ref(tp, memory_label)
                specs.append((label, (ref, value), (ref, value), kind, note))

            for label, pf_label, dc_label in (
                ("Effective main KV cache", "Prefill effective main KV", "Decode effective main KV"),
                ("Effective Indexer KV cache", "Prefill effective Indexer KV", "Decode effective Indexer KV"),
                ("Compressor State", "Prefill compressor states", "Decode compressor states"),
                ("Preallocated KV + state", "Prefill preallocated KV + states", "Decode preallocated KV + states"),
            ):
                specs.append(
                    (
                        label,
                        (memory_ref(tp, pf_label), cache_number(pf_label, "prefill")),
                        (memory_ref(tp, dc_label), cache_number(dc_label, "decode")),
                        "bytes",
                        "KV/状态在当前实现中每个 TP Rank 复制。",
                    )
                )

            total_parameter = parameter_value(tp, "Total", "bytes")
            pf_allocated = cache_number("Prefill preallocated KV + states", "prefill")
            dc_allocated = cache_number("Decode preallocated KV + states", "decode")
            specs.append(
                (
                    "Total resident capacity",
                    (
                        f"{memory_ref(tp, 'Total Parameter Capacity')}+{memory_ref(tp, 'Prefill preallocated KV + states')}",
                        total_parameter + pf_allocated,
                    ),
                    (
                        f"{memory_ref(tp, 'Total Parameter Capacity')}+{memory_ref(tp, 'Decode preallocated KV + states')}",
                        total_parameter + dc_allocated,
                    ),
                    "bytes",
                    "参数 + 预分配 KV/状态；不含临时工作区。",
                )
            )

            for offset, (label, pf_spec, dc_spec, kind, note) in enumerate(specs):
                row = header_row + 1 + offset
                total = label.startswith("Total")
                ws.write(
                    row,
                    0,
                    display_label(label),
                    self.formats["total" if total else "text"],
                )
                for value_col, unit_col, raw_col, spec in (
                    (1, 2, raw_prefill_col, pf_spec),
                    (2, 3, raw_decode_col, dc_spec),
                ):
                    formula_ref, value = spec
                    formula = (
                        formula_ref
                        if formula_ref.startswith("=")
                        else f"={formula_ref}"
                    )
                    self._write_human_value(
                        ws,
                        row,
                        value_col,
                        unit_col,
                        raw_col,
                        formula,
                        value,
                        kind,
                        total,
                    )
                ws.write(row, 3, note, self.formats["text"])
            return header_row + len(specs) + 2

        next_row = write_tp_section(3, "TP1", pf1, dc1)
        write_tp_section(next_row, "TP8", pf8, dc8)
        ws.set_column(raw_prefill_col, raw_decode_col, None, None, {"hidden": True})
        ws.set_column("A:A", 30)
        ws.set_column("B:C", 18)
        ws.set_column("D:D", 62)


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
    architecture_checks = (
        (p.heads % p.tp == 0 and p.heads % p.comparison_tp == 0, "Q heads are not divisible by TP1 and TP8"),
        (p.routed_experts % p.tp == 0 and p.routed_experts % p.comparison_tp == 0, "Experts are not divisible by TP1 and TP8"),
        (p.o_groups % p.tp == 0 and p.o_groups % p.comparison_tp == 0, "Output groups are not divisible by TP1 and TP8"),
        (p.index_heads % p.tp == 0 and p.index_heads % p.comparison_tp == 0, "Indexer heads are not divisible by TP1 and TP8"),
        (p.activated_experts <= p.routed_experts, "Activated experts exceed routed experts"),
        (p.rope_dim <= p.head_dim, "RoPE dimension exceeds head dimension"),
    )
    for valid, message in architecture_checks:
        if not valid:
            raise AssertionError(message)
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
        if b"Memory!" in formula_payload or b"dtype!" in formula_payload:
            raise AssertionError("Workbook contains a stale standalone-sheet reference")
        workbook_xml = archive.read("xl/workbook.xml")
        for required_sheet in (
            b"Summary",
            b"Prefill_8K",
            b"Decode_1M",
            b"dtype",
            b"Comparison",
            b"Methodology",
        ):
            if required_sheet not in workbook_xml:
                raise AssertionError(f"Missing workbook sheet: {required_sheet!r}")
        if b'name="Memory"' in workbook_xml:
            raise AssertionError("Standalone Memory sheet must be absent")
        shared_strings = archive.read("xl/sharedStrings.xml")
        if b"Architecture validation" in shared_strings:
            raise AssertionError("Architecture validation must remain code-only")
        chart_payloads = [
            archive.read(name)
            for name in archive.namelist()
            if name.startswith("xl/charts/chart")
        ]
        if len(chart_payloads) < 7:
            raise AssertionError("Expected seven readable comparison charts")
        if any(b"logBase" in payload or b"0.000E" in payload for payload in chart_payloads):
            raise AssertionError("Charts must not use log axes or scientific formats")
        chart_namespace = {
            "c": "http://schemas.openxmlformats.org/drawingml/2006/chart"
        }
        chart_formulas = [
            formula.text or ""
            for payload in chart_payloads
            for formula in ET.fromstring(payload).findall(".//c:f", chart_namespace)
        ]
        if any(
            any(f"${column}$" in formula for column in ("G", "H", "I", "J"))
            for formula in chart_formulas
        ):
            raise AssertionError("Charts must use scaled hidden sources, not raw columns")
        first_chart_formats = [
            node.attrib.get("formatCode", "")
            for node in ET.fromstring(chart_payloads[0]).findall(
                ".//c:numFmt", chart_namespace
            )
        ]
        if not any("TFLOPs" in format_code for format_code in first_chart_formats):
            raise AssertionError("Total FLOPs chart must use a readable TFLOPs format")
        if any(
            b'<c:gapWidth val="30"/>' not in payload for payload in chart_payloads
        ):
            raise AssertionError("Charts must use a 1/5 category gap")
        workbook_root = ET.fromstring(workbook_xml)
        namespace = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        sheet_names = [
            sheet.attrib["name"]
            for sheet in workbook_root.find("m:sheets", namespace)
        ]
        expected_sheet_order = [
            "Parameters",
            "Prefill_8K",
            "Decode_1M",
            "dtype",
            "Comparison",
            "Summary",
            "Layer_Config",
            "Methodology",
        ]
        if sheet_names != expected_sheet_order:
            raise AssertionError(f"Unexpected worksheet order: {sheet_names}")
        for target in ("Prefill_8K", "Decode_1M"):
            sheet_index = sheet_names.index(target) + 1
            sheet_root = ET.fromstring(
                archive.read(f"xl/worksheets/sheet{sheet_index}.xml")
            )
            pane = sheet_root.find(".//m:pane", namespace)
            if pane is None or pane.attrib.get("ySplit") != "3":
                raise AssertionError(f"{target} must freeze only the top three rows")
        rank_tables: dict[str, ET.Element] = {}
        comparison_tables: list[ET.Element] = []
        for name in archive.namelist():
            if not name.startswith("xl/tables/table"):
                continue
            candidate = ET.fromstring(archive.read(name))
            if candidate.attrib.get("name") == "Comparison_Resource_Overview":
                comparison_tables.append(candidate)
            if candidate.attrib.get("name") in {
                "PF_Embedded_Rank_Memory",
                "DC_Embedded_Rank_Memory",
            }:
                rank_tables[candidate.attrib["name"]] = candidate
        if len(comparison_tables) != 1:
            raise AssertionError("Comparison must contain exactly one resource table")
        comparison_headers = [
            column.attrib.get("name", "")
            for column in comparison_tables[0].findall("{*}tableColumns/{*}tableColumn")
        ]
        if comparison_headers != [
            "Resource",
            "Prefill TP1",
            "Prefill TP8/rank",
            "Decode TP1",
            "Decode TP8/rank",
            "Notes",
        ]:
            raise AssertionError("Comparison resource table has unexpected columns")
        if any(header in {"Unit", "单位"} for header in comparison_headers):
            raise AssertionError("Comparison must not contain a standalone Unit column")
        if set(rank_tables) != {"PF_Embedded_Rank_Memory", "DC_Embedded_Rank_Memory"}:
            raise AssertionError("Missing embedded per-rank memory tables")
        embedded_tables = {
            candidate.attrib.get("name", "")
            for name in archive.namelist()
            if name.startswith("xl/tables/table")
            for candidate in [ET.fromstring(archive.read(name))]
        }
        expected_embedded_tables = {
            "DType_Module",
            "DType_Layer",
            "Comparison_Resource_Overview",
        }
        if not expected_embedded_tables.issubset(embedded_tables):
            raise AssertionError("Missing embedded dtype or Comparison tables")
        for rank_table in rank_tables.values():
            first_cell, last_cell = rank_table.attrib["ref"].split(":")
            first_row = int(re.search(r"\d+", first_cell).group())
            last_row = int(re.search(r"\d+", last_cell).group())
            if last_row - first_row != 2:
                raise AssertionError("Embedded per-rank memory table must contain TP1 and TP8 only")


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
    writer.write_scenario("prefill")
    writer.write_scenario("decode")
    writer.write_dtype()
    memory = writer.write_memory()
    writer.write_comparison(memory)
    writer.write_summary(memory)
    writer.write_layer_config()
    writer.write_methodology()
    writer.close()

    write_reports(output.parent, p, ratios, memory, prefill_items, decode_items)
    validate_baseline(p, ratios, memory, prefill_items, decode_items, output)
    print(f"Wrote {output}")
    print(f"Wrote {output.parent / 'baseline_results.json'}")
    print(f"Wrote {output.parent / 'baseline_report.md'}")


if __name__ == "__main__":
    main()