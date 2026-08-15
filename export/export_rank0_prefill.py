#!/usr/bin/env python3
"""Build a weight-free ONNX structure graph without executing PyTorch forward."""

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import onnx
from onnx import TensorProto, checker, helper


BATCH_SIZE = 1
SEQUENCE_LENGTH = 8
WORLD_SIZE = 8
RANK = 0
LAYER_ID = 21
CONTEXT_LENGTH = 1 << 20
ONNX_OPSET = 18

BF16 = TensorProto.BFLOAT16
FP32 = TensorProto.FLOAT
FP8 = TensorProto.FLOAT8E4M3FN
E8M0 = TensorProto.FLOAT8E8M0
UINT8 = TensorProto.UINT8
INT32 = TensorProto.INT32
INT64 = TensorProto.INT64
COMPLEX64 = TensorProto.COMPLEX64

DTYPE_NAMES = {
    BF16: "bfloat16",
    FP32: "float32",
    FP8: "float8_e4m3fn",
    E8M0: "float8_e8m0fnu",
    UINT8: "uint8",
    INT32: "int32",
    INT64: "int64",
    COMPLEX64: "complex64",
}

DTYPE_BYTES = {
    BF16: 2,
    FP32: 4,
    FP8: 1,
    E8M0: 1,
    UINT8: 1,
    INT32: 4,
    INT64: 8,
    COMPLEX64: 8,
}

Shape = list[int | str]


@dataclass(frozen=True)
class TensorDesc:
    name: str
    dtype: int
    shape: Shape
    kind: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": DTYPE_NAMES[self.dtype],
            "shape": self.shape,
            "kind": self.kind,
        }


class GraphBuilder:
    def __init__(self) -> None:
        self.tensors: dict[str, TensorDesc] = {}
        self.graph_inputs: list[str] = []
        self.graph_outputs: list[str] = []
        self.nodes: list[onnx.NodeProto] = []
        self.operators: list[dict[str, Any]] = []
        self.parameters: list[dict[str, Any]] = []
        self.communications: list[dict[str, Any]] = []

    def add_tensor(self, name: str, dtype: int, shape: Shape, kind: str) -> str:
        if name in self.tensors:
            raise ValueError(f"Tensor already exists: {name}")
        self.tensors[name] = TensorDesc(name, dtype, list(shape), kind)
        return name

    def input(self, name: str, dtype: int, shape: Shape, kind: str = "data") -> str:
        self.add_tensor(name, dtype, shape, kind)
        self.graph_inputs.append(name)
        return name

    def parameter(
        self,
        name: str,
        dtype: int,
        shape: Shape,
        logical_shape: Shape | None = None,
        logical_dtype: str | None = None,
        active_for_selected_call: bool = True,
        inactive_reason: str | None = None,
    ) -> str:
        self.input(name, dtype, shape, "parameter")
        record = self.tensors[name].as_dict()
        record["logical_shape"] = logical_shape or list(shape)
        record["logical_dtype"] = logical_dtype or DTYPE_NAMES[dtype]
        record["active_for_selected_call"] = active_for_selected_call
        if inactive_reason is not None:
            record["inactive_reason"] = inactive_reason
        if all(isinstance(dim, int) for dim in shape):
            record["storage_bytes"] = math.prod(shape) * DTYPE_BYTES[dtype]
        self.parameters.append(record)
        return name

    def node(
        self,
        name: str,
        op_type: str,
        inputs: Iterable[str],
        outputs: Iterable[tuple[str, int, Shape]],
        domain: str = "",
        attributes: dict[str, Any] | None = None,
        analysis: dict[str, Any] | None = None,
    ) -> list[str]:
        input_names = list(inputs)
        for input_name in input_names:
            if input_name not in self.tensors:
                raise ValueError(f"Unknown input {input_name} for node {name}")
        output_specs = list(outputs)
        output_names = [
            self.add_tensor(output_name, dtype, shape, "intermediate")
            for output_name, dtype, shape in output_specs
        ]
        attributes = attributes or {}
        self.nodes.append(
            helper.make_node(
                op_type,
                input_names,
                output_names,
                name=name,
                domain=domain,
                **attributes,
            )
        )
        self.operators.append(
            {
                "name": name,
                "domain": domain or "ai.onnx",
                "op_type": op_type,
                "inputs": [self.tensors[item].as_dict() for item in input_names],
                "outputs": [self.tensors[item].as_dict() for item in output_names],
                "attributes": attributes,
                "analysis": analysis or {},
            }
        )
        return output_names

    def output(self, name: str) -> None:
        if name not in self.tensors:
            raise ValueError(f"Unknown graph output: {name}")
        self.graph_outputs.append(name)

    def value_info(self, name: str) -> onnx.ValueInfoProto:
        tensor = self.tensors[name]
        return helper.make_tensor_value_info(name, tensor.dtype, tensor.shape)

    def make_model(self) -> onnx.ModelProto:
        input_set = set(self.graph_inputs)
        output_set = set(self.graph_outputs)
        value_names = [
            name for name in self.tensors if name not in input_set and name not in output_set
        ]
        graph = helper.make_graph(
            self.nodes,
            "DeepSeek-V4-Flash_rank0_prefill_b1_s8_layer21_tp8",
            [self.value_info(name) for name in self.graph_inputs],
            [self.value_info(name) for name in self.graph_outputs],
            initializer=[],
            value_info=[self.value_info(name) for name in value_names],
        )
        return helper.make_model(
            graph,
            producer_name="deepseek-v4-structure-export",
            producer_version="1.0",
            opset_imports=[
                helper.make_opsetid("", ONNX_OPSET),
                helper.make_opsetid("ai.deepseek", 1),
                helper.make_opsetid("ai.deepseek.distributed", 1),
            ],
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_numel(shape: Shape) -> int:
    if not all(isinstance(dim, int) for dim in shape):
        raise ValueError(f"Expected a static shape, received {shape}")
    return math.prod(shape)


def cast(builder: GraphBuilder, name: str, source: str, dtype: int, output: str) -> str:
    shape = builder.tensors[source].shape
    return builder.node(
        name,
        "Cast",
        [source],
        [(output, dtype, shape)],
        attributes={"to": dtype},
    )[0]


def view(
    builder: GraphBuilder,
    name: str,
    source: str,
    shape: Shape,
    output: str,
) -> str:
    return builder.node(
        name,
        "View",
        [source],
        [(output, builder.tensors[source].dtype, shape)],
        domain="ai.deepseek",
        attributes={"shape": [str(item) for item in shape]},
    )[0]


def rms_norm(
    builder: GraphBuilder,
    prefix: str,
    source: str,
    dim: int,
    epsilon: float,
) -> str:
    weight = builder.parameter(f"{prefix}.weight", FP32, [dim])
    return builder.node(
        prefix,
        "RMSNorm",
        [source, weight],
        [(f"{prefix}.output", builder.tensors[source].dtype, builder.tensors[source].shape)],
        domain="ai.deepseek",
        attributes={"axis": -1, "epsilon": epsilon},
    )[0]


def fp8_linear(
    builder: GraphBuilder,
    prefix: str,
    source: str,
    in_features: int,
    out_features: int,
) -> str:
    source_shape = builder.tensors[source].shape
    if source_shape[-1] != in_features or in_features % 128:
        raise ValueError(f"{prefix}: expected K={in_features}, received {source_shape}")
    activation, activation_scale = builder.node(
        f"{prefix}.act_quant",
        "ActQuantFP8",
        [source],
        [
            (f"{prefix}.activation_fp8", FP8, source_shape),
            (
                f"{prefix}.activation_scale",
                E8M0,
                [*source_shape[:-1], in_features // 128],
            ),
        ],
        domain="ai.deepseek",
        attributes={"block_size": 128, "round_scale": "ue8m0"},
    )
    weight = builder.parameter(f"{prefix}.weight", FP8, [out_features, in_features])
    scale = builder.parameter(
        f"{prefix}.scale",
        E8M0,
        [math.ceil(out_features / 128), math.ceil(in_features / 128)],
    )
    output_shape = [*source_shape[:-1], out_features]
    batch_elements = "*".join(str(item) for item in source_shape[:-1]) or "1"
    return builder.node(
        prefix,
        "FP8Gemm",
        [activation, activation_scale, weight, scale],
        [(f"{prefix}.output", BF16, output_shape)],
        domain="ai.deepseek",
        attributes={
            "in_features": in_features,
            "out_features": out_features,
            "scale_block": 128,
            "weight_layout": "OI",
        },
        analysis={"flops": f"2*({batch_elements})*{in_features}*{out_features}"},
    )[0]


def fp4_linear(
    builder: GraphBuilder,
    prefix: str,
    source: str,
    in_features: int,
    out_features: int,
) -> str:
    source_shape = builder.tensors[source].shape
    if source_shape[-1] != in_features or in_features % 32:
        raise ValueError(f"{prefix}: invalid FP4 K dimension {source_shape}")
    if in_features % 128:
        raise ValueError(f"{prefix}: FP4 activation K must be divisible by 128")
    activation, activation_scale = builder.node(
        f"{prefix}.act_quant",
        "ActQuantFP8",
        [source],
        [
            (f"{prefix}.activation_fp8", FP8, source_shape),
            (
                f"{prefix}.activation_scale",
                E8M0,
                [*source_shape[:-1], in_features // 128],
            ),
        ],
        domain="ai.deepseek",
        attributes={"block_size": 128, "round_scale": "ue8m0"},
    )
    weight = builder.parameter(
        f"{prefix}.weight_packed",
        UINT8,
        [out_features, in_features // 2],
        logical_shape=[out_features, in_features],
        logical_dtype="float4_e2m1fn_x2",
    )
    scale = builder.parameter(
        f"{prefix}.scale",
        E8M0,
        [out_features, in_features // 32],
    )
    output_shape = [*source_shape[:-1], out_features]
    batch_elements = "*".join(str(item) for item in source_shape[:-1]) or "1"
    return builder.node(
        prefix,
        "FP4Gemm",
        [activation, activation_scale, weight, scale],
        [(f"{prefix}.output", BF16, output_shape)],
        domain="ai.deepseek",
        attributes={
            "in_features": in_features,
            "out_features": out_features,
            "packed_k": in_features // 2,
            "scale_block": 32,
            "weight_layout": "OI_packed_x2",
        },
        analysis={"flops": f"2*({batch_elements})*{in_features}*{out_features}"},
    )[0]


def fp32_linear(
    builder: GraphBuilder,
    prefix: str,
    source: str,
    in_features: int,
    out_features: int,
) -> str:
    source_shape = builder.tensors[source].shape
    if source_shape[-1] != in_features:
        raise ValueError(f"{prefix}: expected K={in_features}, received {source_shape}")
    weight = builder.parameter(f"{prefix}.weight", FP32, [out_features, in_features])
    output_shape = [*source_shape[:-1], out_features]
    return builder.node(
        prefix,
        "Gemm",
        [source, weight],
        [(f"{prefix}.output", FP32, output_shape)],
        attributes={"transB": 1},
        analysis={
            "flops": 2 * numeric_numel(source_shape[:-1]) * in_features * out_features
            if all(isinstance(item, int) for item in source_shape)
            else f"2*{'*'.join(str(item) for item in source_shape[:-1])}*{in_features}*{out_features}"
        },
    )[0]


def all_reduce(
    builder: GraphBuilder,
    name: str,
    source: str,
    output: str,
    purpose: str,
) -> str:
    tensor = builder.tensors[source]
    payload = numeric_numel(tensor.shape) * DTYPE_BYTES[tensor.dtype]
    ring_directional_bytes = 2 * (WORLD_SIZE - 1) * payload // WORLD_SIZE
    result = builder.node(
        name,
        "AllReduce",
        [source],
        [(output, tensor.dtype, tensor.shape)],
        domain="ai.deepseek.distributed",
        attributes={"group_size": WORLD_SIZE, "op": "sum", "rank": RANK},
        analysis={"logical_payload_bytes": payload},
    )[0]
    builder.communications.append(
        {
            "name": name,
            "collective": "AllReduce",
            "purpose": purpose,
            "dtype": DTYPE_NAMES[tensor.dtype],
            "shape": tensor.shape,
            "logical_payload_bytes": payload,
            "ring_send_bytes_per_rank": ring_directional_bytes,
            "ring_receive_bytes_per_rank": ring_directional_bytes,
            "ring_transfer_bytes_per_rank": 2 * ring_directional_bytes,
        }
    )
    return result


def all_gather(
    builder: GraphBuilder,
    name: str,
    source: str,
    output: str,
    output_shape: Shape,
    purpose: str,
) -> str:
    tensor = builder.tensors[source]
    payload = numeric_numel(tensor.shape) * DTYPE_BYTES[tensor.dtype]
    ring_directional_bytes = (WORLD_SIZE - 1) * payload
    result = builder.node(
        name,
        "AllGather",
        [source],
        [(output, tensor.dtype, output_shape)],
        domain="ai.deepseek.distributed",
        attributes={"axis": -1, "group_size": WORLD_SIZE, "rank": RANK},
        analysis={"local_payload_bytes": payload},
    )[0]
    builder.communications.append(
        {
            "name": name,
            "collective": "AllGather",
            "purpose": purpose,
            "dtype": DTYPE_NAMES[tensor.dtype],
            "local_shape": tensor.shape,
            "output_shape": output_shape,
            "local_payload_bytes": payload,
            "ring_send_bytes_per_rank": ring_directional_bytes,
            "ring_receive_bytes_per_rank": ring_directional_bytes,
            "ring_transfer_bytes_per_rank": 2 * ring_directional_bytes,
        }
    )
    return result


def hc_pre(
    builder: GraphBuilder,
    prefix: str,
    source: str,
    batch_size: int,
    sequence_length: int,
    hc_mult: int,
    dim: int,
    epsilon: float,
    iterations: int,
) -> tuple[str, str, str]:
    rows = batch_size * sequence_length
    hc_dim = hc_mult * dim
    mix_hc = (2 + hc_mult) * hc_mult
    function = builder.parameter(f"{prefix}.function", FP32, [mix_hc, hc_dim])
    scale = builder.parameter(f"{prefix}.scale", FP32, [3])
    base = builder.parameter(f"{prefix}.base", FP32, [mix_hc])
    flat = view(builder, f"{prefix}.flatten", source, [rows, hc_dim], f"{prefix}.flat")
    flat_fp32 = cast(builder, f"{prefix}.cast", flat, FP32, f"{prefix}.flat_fp32")
    reciprocal = builder.node(
        f"{prefix}.rms_reciprocal",
        "RMSReciprocal",
        [flat_fp32],
        [(f"{prefix}.rms_reciprocal.output", FP32, [rows, 1])],
        domain="ai.deepseek",
        attributes={"epsilon": epsilon},
    )[0]
    mixes = builder.node(
        f"{prefix}.projection",
        "Gemm",
        [flat_fp32, function],
        [(f"{prefix}.mixes_unscaled", FP32, [rows, mix_hc])],
        attributes={"transB": 1},
        analysis={"flops": 2 * rows * hc_dim * mix_hc},
    )[0]
    scaled = builder.node(
        f"{prefix}.scale_mixes",
        "Mul",
        [mixes, reciprocal],
        [(f"{prefix}.mixes", FP32, [rows, mix_hc])],
    )[0]
    pre, post, combination = builder.node(
        f"{prefix}.sinkhorn",
        "HCSplitSinkhorn",
        [scaled, scale, base],
        [
            (f"{prefix}.pre", FP32, [batch_size, sequence_length, hc_mult]),
            (f"{prefix}.post", FP32, [batch_size, sequence_length, hc_mult]),
            (
                f"{prefix}.combination",
                FP32,
                [batch_size, sequence_length, hc_mult, hc_mult],
            ),
        ],
        domain="ai.deepseek",
        attributes={"epsilon": epsilon, "hc_mult": hc_mult, "iterations": iterations},
    )
    reduced = builder.node(
        f"{prefix}.reduce",
        "HCReduce",
        [source, pre],
        [(f"{prefix}.reduced", BF16, [batch_size, sequence_length, dim])],
        domain="ai.deepseek",
    )[0]
    return reduced, post, combination


def hc_post(
    builder: GraphBuilder,
    prefix: str,
    update: str,
    residual: str,
    post: str,
    combination: str,
    batch_size: int,
    sequence_length: int,
    hc_mult: int,
    dim: int,
) -> str:
    return builder.node(
        prefix,
        "HCPost",
        [update, residual, post, combination],
        [(f"{prefix}.output", BF16, [batch_size, sequence_length, hc_mult, dim])],
        domain="ai.deepseek",
    )[0]


def build_attention(
    builder: GraphBuilder,
    source: str,
    config: dict[str, Any],
) -> tuple[str, str, str, str]:
    dim = config["dim"]
    q_rank = config["q_lora_rank"]
    head_dim = config["head_dim"]
    rope_dim = config["rope_head_dim"]
    local_heads = config["n_heads"] // WORLD_SIZE
    local_groups = config["o_groups"] // WORLD_SIZE
    o_rank = config["o_lora_rank"]
    ratio = config["compress_ratios"][LAYER_ID]
    window = config["window_size"]
    epsilon = config.get("norm_eps", 1e-6)
    cache_slots = window + CONTEXT_LENGTH // ratio

    q_low = fp8_linear(builder, "block21.attn.wq_a", source, dim, q_rank)
    q_low = rms_norm(builder, "block21.attn.q_norm", q_low, q_rank, epsilon)
    q_flat = fp8_linear(
        builder,
        "block21.attn.wq_b",
        q_low,
        q_rank,
        local_heads * head_dim,
    )
    freqs_cis = builder.input(
        "block21.attn.freqs_cis",
        COMPLEX64,
        [CONTEXT_LENGTH, rope_dim // 2],
        "buffer",
    )
    rope = builder.node(
        "block21.attn.rope_slice",
        "RoPESlice",
        [freqs_cis],
        [("block21.attn.rope_freqs", COMPLEX64, [SEQUENCE_LENGTH, rope_dim // 2])],
        domain="ai.deepseek",
        attributes={"length": SEQUENCE_LENGTH, "start_pos": 0},
    )[0]
    query = builder.node(
        "block21.attn.query_rms_normalize",
        "QueryRMSNormalize",
        [q_flat],
        [("block21.attn.query_normalized", BF16, [BATCH_SIZE, SEQUENCE_LENGTH, local_heads, head_dim])],
        domain="ai.deepseek",
        attributes={"epsilon": epsilon, "head_dim": head_dim},
    )[0]
    query = builder.node(
        "block21.attn.query_rope",
        "RoPE",
        [query, rope],
        [("block21.attn.query", BF16, [BATCH_SIZE, SEQUENCE_LENGTH, local_heads, head_dim])],
        domain="ai.deepseek",
        attributes={"rope_dim": rope_dim},
    )[0]

    current_kv = fp8_linear(builder, "block21.attn.wkv", source, dim, head_dim)
    current_kv = rms_norm(builder, "block21.attn.kv_norm", current_kv, head_dim, epsilon)
    current_kv = builder.node(
        "block21.attn.kv_rope",
        "RoPE",
        [current_kv, rope],
        [("block21.attn.kv_rotated", BF16, [BATCH_SIZE, SEQUENCE_LENGTH, head_dim])],
        domain="ai.deepseek",
        attributes={"rope_dim": rope_dim},
    )[0]
    current_kv = builder.node(
        "block21.attn.kv_act_quant_dequant",
        "ActQuantDequantFP8",
        [current_kv],
        [("block21.attn.current_kv", BF16, [BATCH_SIZE, SEQUENCE_LENGTH, head_dim])],
        domain="ai.deepseek",
        attributes={
            "block_size": 64,
            "quantized_features": head_dim - rope_dim,
            "round_scale": "ue8m0",
        },
    )[0]

    cache_input = builder.input(
        "block21.attn.kv_cache_in",
        BF16,
        [BATCH_SIZE, cache_slots, head_dim],
        "state",
    )
    cache_output = builder.node(
        "block21.attn.prefill_cache_write",
        "PrefillKVCacheWrite",
        [cache_input, current_kv],
        [("block21.attn.kv_cache_out", BF16, [BATCH_SIZE, cache_slots, head_dim])],
        domain="ai.deepseek",
        attributes={
            "compressed_capacity": CONTEXT_LENGTH // ratio,
            "compress_ratio": ratio,
            "start_pos": 0,
            "window_size": window,
        },
    )[0]

    source_fp32 = cast(
        builder,
        "block21.attn.compressor.cast",
        source,
        FP32,
        "block21.attn.compressor.input_fp32",
    )
    projected_kv = fp32_linear(
        builder,
        "block21.attn.compressor.wkv",
        source_fp32,
        dim,
        head_dim,
    )
    projected_score = fp32_linear(
        builder,
        "block21.attn.compressor.wgate",
        source_fp32,
        dim,
        head_dim,
    )
    ape = builder.parameter("block21.attn.compressor.ape", FP32, [ratio, head_dim])
    builder.parameter(
        "block21.attn.compressor.norm.weight",
        FP32,
        [head_dim],
        active_for_selected_call=False,
        inactive_reason="S=8 is smaller than compress_ratio=128, so Compressor.forward returns before norm",
    )
    state_kv = builder.input(
        "block21.attn.compressor.kv_state_in",
        FP32,
        [BATCH_SIZE, ratio, head_dim],
        "state",
    )
    state_score = builder.input(
        "block21.attn.compressor.score_state_in",
        FP32,
        [BATCH_SIZE, ratio, head_dim],
        "state",
    )
    state_kv_out, state_score_out = builder.node(
        "block21.attn.compressor.prefill_state_update",
        "CompressorPrefillStateUpdate",
        [projected_kv, projected_score, ape, state_kv, state_score],
        [
            ("block21.attn.compressor.kv_state_out", FP32, [BATCH_SIZE, ratio, head_dim]),
            (
                "block21.attn.compressor.score_state_out",
                FP32,
                [BATCH_SIZE, ratio, head_dim],
            ),
        ],
        domain="ai.deepseek",
        attributes={
            "compress_ratio": ratio,
            "sequence_length": SEQUENCE_LENGTH,
            "should_compress": 0,
            "start_pos": 0,
            "compressor_norm_executed": 0,
        },
    )
    attention_kv = current_kv
    window_indices = builder.node(
        "block21.attn.window_indices",
        "PrefillWindowIndices",
        [],
        [("block21.attn.window_indices.int64", INT64, [BATCH_SIZE, SEQUENCE_LENGTH, SEQUENCE_LENGTH])],
        domain="ai.deepseek",
        attributes={"start_pos": 0, "window_size": window},
    )[0]
    compressed_indices = builder.node(
        "block21.attn.compressed_indices",
        "PrefillCompressedIndices",
        [],
        [("block21.attn.compressed_indices.int64", INT64, [BATCH_SIZE, SEQUENCE_LENGTH, 0])],
        domain="ai.deepseek",
        attributes={"compress_ratio": ratio, "start_pos": 0, "offset": SEQUENCE_LENGTH},
    )[0]
    indices = builder.node(
        "block21.attn.concat_indices",
        "Concat",
        [window_indices, compressed_indices],
        [("block21.attn.topk_indices.int64", INT64, [BATCH_SIZE, SEQUENCE_LENGTH, SEQUENCE_LENGTH])],
        attributes={"axis": -1},
    )[0]
    indices = cast(
        builder,
        "block21.attn.indices_cast_int32",
        indices,
        INT32,
        "block21.attn.topk_indices",
    )
    sink = builder.parameter("block21.attn.attn_sink", FP32, [local_heads])
    padded_query, padded_sink = builder.node(
        "block21.attn.pad_heads",
        "PadAttentionHeads",
        [query, sink],
        [
            ("block21.attn.query_padded", BF16, [BATCH_SIZE, SEQUENCE_LENGTH, 16, head_dim]),
            ("block21.attn.sink_padded", FP32, [16]),
        ],
        domain="ai.deepseek",
        attributes={"kernel_heads": 16, "local_heads": local_heads},
    )
    attended = builder.node(
        "block21.attn.sparse_attention",
        "SparseAttention",
        [padded_query, attention_kv, padded_sink, indices],
        [("block21.attn.context_padded", BF16, [BATCH_SIZE, SEQUENCE_LENGTH, 16, head_dim])],
        domain="ai.deepseek",
        attributes={
            "causal": 1,
            "kernel_heads": 16,
            "kv_heads": 1,
            "query_heads": local_heads,
            "softmax_scale": head_dim**-0.5,
            "topk": SEQUENCE_LENGTH,
        },
        analysis={
            "attention_flops": 4
            * BATCH_SIZE
            * SEQUENCE_LENGTH
            * 16
            * SEQUENCE_LENGTH
            * head_dim
        },
    )[0]
    attended = builder.node(
        "block21.attn.unpad_heads",
        "UnpadAttentionHeads",
        [attended],
        [("block21.attn.context", BF16, [BATCH_SIZE, SEQUENCE_LENGTH, local_heads, head_dim])],
        domain="ai.deepseek",
        attributes={"kernel_heads": 16, "local_heads": local_heads},
    )[0]
    context = builder.node(
        "block21.attn.inverse_rope",
        "InverseRoPE",
        [attended, rope],
        [("block21.attn.context_derotated", BF16, [BATCH_SIZE, SEQUENCE_LENGTH, local_heads, head_dim])],
        domain="ai.deepseek",
        attributes={"rope_dim": rope_dim},
    )[0]
    grouped = view(
        builder,
        "block21.attn.group_view",
        context,
        [BATCH_SIZE, SEQUENCE_LENGTH, local_groups, local_heads * head_dim // local_groups],
        "block21.attn.grouped_context",
    )
    wo_a = builder.parameter(
        "block21.attn.wo_a.weight",
        BF16,
        [local_groups * o_rank, config["n_heads"] * head_dim // config["o_groups"]],
    )
    wo_a_grouped = view(
        builder,
        "block21.attn.wo_a.weight_view",
        wo_a,
        [local_groups, o_rank, local_heads * head_dim // local_groups],
        "block21.attn.wo_a.weight_grouped",
    )
    projected = builder.node(
        "block21.attn.wo_a",
        "GroupedOutputProjection",
        [grouped, wo_a_grouped],
        [("block21.attn.wo_a.output", BF16, [BATCH_SIZE, SEQUENCE_LENGTH, local_groups, o_rank])],
        domain="ai.deepseek",
        attributes={"equation": "bsgd,grd->bsgr"},
        analysis={
            "flops": 2
            * BATCH_SIZE
            * SEQUENCE_LENGTH
            * local_groups
            * o_rank
            * (local_heads * head_dim // local_groups)
        },
    )[0]
    projected = view(
        builder,
        "block21.attn.wo_a.flatten",
        projected,
        [BATCH_SIZE, SEQUENCE_LENGTH, local_groups * o_rank],
        "block21.attn.wo_a.flat",
    )
    local_output = fp8_linear(
        builder,
        "block21.attn.wo_b",
        projected,
        local_groups * o_rank,
        dim,
    )
    local_output_fp32 = cast(
        builder,
        "block21.attn.wo_b.cast_fp32",
        local_output,
        FP32,
        "block21.attn.row_parallel_partial_fp32",
    )
    reduced = all_reduce(
        builder,
        "block21.attn.row_parallel_all_reduce",
        local_output_fp32,
        "block21.attn.row_parallel_reduced_fp32",
        "row-parallel attention output",
    )
    output = cast(
        builder,
        "block21.attn.row_parallel_cast_bf16",
        reduced,
        BF16,
        "block21.attn.output",
    )
    return output, cache_output, state_kv_out, state_score_out


def build_expert(
    builder: GraphBuilder,
    expert_index: int,
    tokens: str,
    route_weights: str,
    dim: int,
    inter_dim: int,
    swiglu_limit: float,
) -> str:
    prefix = f"block21.moe.experts.{expert_index}"
    gate = fp4_linear(builder, f"{prefix}.w1", tokens, dim, inter_dim)
    up = fp4_linear(builder, f"{prefix}.w3", tokens, dim, inter_dim)
    gate_fp32 = cast(builder, f"{prefix}.w1.cast", gate, FP32, f"{prefix}.gate_fp32")
    up_fp32 = cast(builder, f"{prefix}.w3.cast", up, FP32, f"{prefix}.up_fp32")
    activated = builder.node(
        f"{prefix}.swiglu_weighted",
        "SwiGLUWeighted",
        [gate_fp32, up_fp32, route_weights],
        [(f"{prefix}.activated_fp32", FP32, [f"N_expert_{expert_index}", inter_dim])],
        domain="ai.deepseek",
        attributes={"swiglu_limit": swiglu_limit},
    )[0]
    activated = cast(
        builder,
        f"{prefix}.activated.cast",
        activated,
        BF16,
        f"{prefix}.activated_bf16",
    )
    return fp4_linear(builder, f"{prefix}.w2", activated, inter_dim, dim)


def build_shared_expert(
    builder: GraphBuilder,
    source: str,
    dim: int,
    inter_dim: int,
    swiglu_limit: float,
) -> str:
    prefix = "block21.moe.shared_expert"
    gate = fp8_linear(builder, f"{prefix}.w1", source, dim, inter_dim)
    up = fp8_linear(builder, f"{prefix}.w3", source, dim, inter_dim)
    activated = builder.node(
        f"{prefix}.swiglu",
        "SwiGLU",
        [gate, up],
        [(f"{prefix}.activated_fp32", FP32, [BATCH_SIZE * SEQUENCE_LENGTH, inter_dim])],
        domain="ai.deepseek",
        attributes={"swiglu_limit": swiglu_limit},
    )[0]
    activated = cast(
        builder,
        f"{prefix}.activated.cast",
        activated,
        BF16,
        f"{prefix}.activated_bf16",
    )
    return fp8_linear(builder, f"{prefix}.w2", activated, inter_dim, dim)


def build_moe(builder: GraphBuilder, source: str, config: dict[str, Any]) -> str:
    dim = config["dim"]
    inter_dim = config["moe_inter_dim"]
    routed_experts = config["n_routed_experts"]
    local_experts = routed_experts // WORLD_SIZE
    topk = config["n_activated_experts"]
    rows = BATCH_SIZE * SEQUENCE_LENGTH
    swiglu_limit = float(config["swiglu_limit"])

    tokens = view(builder, "block21.moe.flatten", source, [rows, dim], "block21.moe.tokens")
    tokens_fp32 = cast(
        builder,
        "block21.moe.gate.input_cast",
        tokens,
        FP32,
        "block21.moe.gate.input_fp32",
    )
    gate_weight = builder.parameter("block21.moe.gate.weight", BF16, [routed_experts, dim])
    gate_weight_fp32 = cast(
        builder,
        "block21.moe.gate.weight_cast",
        gate_weight,
        FP32,
        "block21.moe.gate.weight_fp32",
    )
    scores = builder.node(
        "block21.moe.gate.linear",
        "Gemm",
        [tokens_fp32, gate_weight_fp32],
        [("block21.moe.gate.scores_linear", FP32, [rows, routed_experts])],
        attributes={"transB": 1},
        analysis={"flops": 2 * rows * dim * routed_experts},
    )[0]
    scores = builder.node(
        "block21.moe.gate.softplus",
        "Softplus",
        [scores],
        [("block21.moe.gate.scores_softplus", FP32, [rows, routed_experts])],
    )[0]
    original_scores = builder.node(
        "block21.moe.gate.sqrt",
        "Sqrt",
        [scores],
        [("block21.moe.gate.original_scores", FP32, [rows, routed_experts])],
    )[0]
    gate_bias = builder.parameter("block21.moe.gate.bias", FP32, [routed_experts])
    selection_scores = builder.node(
        "block21.moe.gate.add_bias",
        "Add",
        [original_scores, gate_bias],
        [("block21.moe.gate.selection_scores", FP32, [rows, routed_experts])],
    )[0]
    _, indices = builder.node(
        "block21.moe.gate.topk",
        "TopKStatic",
        [selection_scores],
        [
            ("block21.moe.gate.topk_values", FP32, [rows, topk]),
            ("block21.moe.gate.topk_indices", INT64, [rows, topk]),
        ],
        domain="ai.deepseek",
        attributes={"axis": -1, "k": topk, "sorted": 1},
    )
    route_weights = builder.node(
        "block21.moe.gate.route_weights",
        "GatherNormalizeRouteWeights",
        [original_scores, indices],
        [("block21.moe.gate.route_weights", FP32, [rows, topk])],
        domain="ai.deepseek",
        attributes={"route_scale": float(config["route_scale"])},
    )[0]

    dispatch_outputs: list[tuple[str, int, Shape]] = []
    for expert_index in range(local_experts):
        symbolic = f"N_expert_{expert_index}"
        dispatch_outputs.extend(
            [
                (f"block21.moe.dispatch.expert_{expert_index}.tokens", BF16, [symbolic, dim]),
                (f"block21.moe.dispatch.expert_{expert_index}.weights", FP32, [symbolic, 1]),
                (f"block21.moe.dispatch.expert_{expert_index}.positions", INT64, [symbolic]),
            ]
        )
    dispatched = builder.node(
        "block21.moe.local_dispatch",
        "MoELocalDispatch",
        [tokens, indices, route_weights],
        dispatch_outputs,
        domain="ai.deepseek",
        attributes={
            "expert_end": local_experts,
            "expert_start": 0,
            "max_local_assignments": rows * topk,
            "topk": topk,
            "world_size": WORLD_SIZE,
        },
        analysis={
            "local_assignments_expected_balanced": rows * topk / WORLD_SIZE,
            "local_assignments_max": rows * topk,
            "local_assignments_min": 0,
        },
    )

    expert_outputs: list[str] = []
    expert_positions: list[str] = []
    for expert_index in range(local_experts):
        offset = expert_index * 3
        expert_outputs.append(
            build_expert(
                builder,
                expert_index,
                dispatched[offset],
                dispatched[offset + 1],
                dim,
                inter_dim,
                swiglu_limit,
            )
        )
        expert_positions.append(dispatched[offset + 2])

    local_combined = builder.node(
        "block21.moe.local_combine",
        "MoELocalCombine",
        [item for pair in zip(expert_outputs, expert_positions) for item in pair],
        [("block21.moe.local_output_fp32", FP32, [rows, dim])],
        domain="ai.deepseek",
        attributes={"local_experts": local_experts, "output_rows": rows},
    )[0]
    routed_output = all_reduce(
        builder,
        "block21.moe.expert_parallel_all_reduce",
        local_combined,
        "block21.moe.routed_output_fp32",
        "sum routed expert outputs",
    )
    shared_output = build_shared_expert(builder, tokens, dim, inter_dim, swiglu_limit)
    shared_output = cast(
        builder,
        "block21.moe.shared_expert.output_cast",
        shared_output,
        FP32,
        "block21.moe.shared_output_fp32",
    )
    combined = builder.node(
        "block21.moe.add_shared",
        "Add",
        [routed_output, shared_output],
        [("block21.moe.output_fp32", FP32, [rows, dim])],
    )[0]
    combined = cast(
        builder,
        "block21.moe.output_cast",
        combined,
        BF16,
        "block21.moe.output_flat",
    )
    return view(
        builder,
        "block21.moe.output_view",
        combined,
        [BATCH_SIZE, SEQUENCE_LENGTH, dim],
        "block21.moe.output",
    )


def build_hc_head(builder: GraphBuilder, source: str, config: dict[str, Any]) -> str:
    dim = config["dim"]
    hc_mult = config["hc_mult"]
    rows = BATCH_SIZE * SEQUENCE_LENGTH
    hc_dim = hc_mult * dim
    epsilon = config.get("hc_eps", 1e-6)
    function = builder.parameter("tail.hc_head.function", FP32, [hc_mult, hc_dim])
    scale = builder.parameter("tail.hc_head.scale", FP32, [1])
    base = builder.parameter("tail.hc_head.base", FP32, [hc_mult])
    flat = view(builder, "tail.hc_head.flatten", source, [rows, hc_dim], "tail.hc_head.flat")
    flat_fp32 = cast(builder, "tail.hc_head.cast", flat, FP32, "tail.hc_head.flat_fp32")
    reciprocal = builder.node(
        "tail.hc_head.rms_reciprocal",
        "RMSReciprocal",
        [flat_fp32],
        [("tail.hc_head.rms_reciprocal.output", FP32, [rows, 1])],
        domain="ai.deepseek",
        attributes={"epsilon": config.get("norm_eps", 1e-6)},
    )[0]
    mixes = builder.node(
        "tail.hc_head.projection",
        "Gemm",
        [flat_fp32, function],
        [("tail.hc_head.mixes_unscaled", FP32, [rows, hc_mult])],
        attributes={"transB": 1},
        analysis={"flops": 2 * rows * hc_dim * hc_mult},
    )[0]
    mixes = builder.node(
        "tail.hc_head.scale_mixes",
        "Mul",
        [mixes, reciprocal],
        [("tail.hc_head.mixes", FP32, [rows, hc_mult])],
    )[0]
    weights = builder.node(
        "tail.hc_head.weights",
        "HCHeadWeights",
        [mixes, scale, base],
        [("tail.hc_head.weights.output", FP32, [BATCH_SIZE, SEQUENCE_LENGTH, hc_mult])],
        domain="ai.deepseek",
        attributes={"epsilon": epsilon},
    )[0]
    return builder.node(
        "tail.hc_head.reduce",
        "HCReduce",
        [source, weights],
        [("tail.hc_head.output", BF16, [BATCH_SIZE, SEQUENCE_LENGTH, dim])],
        domain="ai.deepseek",
    )[0]


def build_graph(config: dict[str, Any]) -> GraphBuilder:
    dim = config["dim"]
    vocab_size = config["vocab_size"]
    hc_mult = config["hc_mult"]
    epsilon = config.get("norm_eps", 1e-6)
    local_vocab = vocab_size // WORLD_SIZE
    ratio = config["compress_ratios"][LAYER_ID]
    if ratio != 128:
        raise ValueError(f"Layer {LAYER_ID} must use compress_ratio=128, received {ratio}")
    if SEQUENCE_LENGTH >= ratio:
        raise ValueError("This exporter models the no-compression S=8 prefill branch")
    if vocab_size % WORLD_SIZE or config["n_heads"] % WORLD_SIZE:
        raise ValueError("Vocabulary and attention heads must be divisible by TP world size")
    if config["n_routed_experts"] % WORLD_SIZE or config["o_groups"] % WORLD_SIZE:
        raise ValueError("Experts and output groups must be divisible by TP world size")

    builder = GraphBuilder()
    input_ids = builder.input("input_ids", INT64, [BATCH_SIZE, SEQUENCE_LENGTH])
    embedding = builder.parameter("embedding.weight", BF16, [local_vocab, dim])
    embedded_partial = builder.node(
        "embedding.local_lookup",
        "VocabParallelEmbedding",
        [input_ids, embedding],
        [("embedding.partial", BF16, [BATCH_SIZE, SEQUENCE_LENGTH, dim])],
        domain="ai.deepseek",
        attributes={
            "vocab_end": local_vocab,
            "vocab_start": 0,
            "world_size": WORLD_SIZE,
        },
    )[0]
    embedded = all_reduce(
        builder,
        "embedding.all_reduce",
        embedded_partial,
        "embedding.output",
        "combine vocabulary-parallel embedding",
    )
    block_input = builder.node(
        "embedding.hc_expand",
        "HCExpand",
        [embedded],
        [("block21.input", BF16, [BATCH_SIZE, SEQUENCE_LENGTH, hc_mult, dim])],
        domain="ai.deepseek",
        attributes={"hc_mult": hc_mult},
    )[0]

    attn_input, attn_post, attn_combination = hc_pre(
        builder,
        "block21.hc_attn",
        block_input,
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        hc_mult,
        dim,
        epsilon,
        config["hc_sinkhorn_iters"],
    )
    attn_input = rms_norm(builder, "block21.attn_norm", attn_input, dim, epsilon)
    attn_output, cache_output, compressor_kv, compressor_score = build_attention(
        builder, attn_input, config
    )
    after_attn = hc_post(
        builder,
        "block21.hc_attn_post",
        attn_output,
        block_input,
        attn_post,
        attn_combination,
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        hc_mult,
        dim,
    )

    moe_input, moe_post, moe_combination = hc_pre(
        builder,
        "block21.hc_moe",
        after_attn,
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        hc_mult,
        dim,
        epsilon,
        config["hc_sinkhorn_iters"],
    )
    moe_input = rms_norm(builder, "block21.moe_norm", moe_input, dim, epsilon)
    moe_output = build_moe(builder, moe_input, config)
    block_output = hc_post(
        builder,
        "block21.hc_moe_post",
        moe_output,
        after_attn,
        moe_post,
        moe_combination,
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        hc_mult,
        dim,
    )

    tail = build_hc_head(builder, block_output, config)
    tail = rms_norm(builder, "tail.final_norm", tail, dim, epsilon)
    last_token = builder.node(
        "tail.last_token",
        "LastToken",
        [tail],
        [("tail.last_hidden", BF16, [BATCH_SIZE, dim])],
        domain="ai.deepseek",
        attributes={"axis": 1, "index": -1},
    )[0]
    last_token = cast(
        builder,
        "tail.last_token_cast",
        last_token,
        FP32,
        "tail.last_hidden_fp32",
    )
    head_weight = builder.parameter("tail.lm_head.weight", FP32, [local_vocab, dim])
    local_logits = builder.node(
        "tail.lm_head.local",
        "Gemm",
        [last_token, head_weight],
        [("tail.local_logits", FP32, [BATCH_SIZE, local_vocab])],
        attributes={"transB": 1},
        analysis={"flops": 2 * BATCH_SIZE * dim * local_vocab},
    )[0]
    logits = all_gather(
        builder,
        "tail.lm_head.all_gather",
        local_logits,
        "logits",
        [BATCH_SIZE, vocab_size],
        "assemble vocabulary-parallel logits",
    )

    for output in [logits, cache_output, compressor_kv, compressor_score]:
        builder.output(output)
    return builder


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_dimensions(path: Path, config: dict[str, Any], builder: GraphBuilder) -> None:
    ratio = config["compress_ratios"][LAYER_ID]
    cache_slots = config["window_size"] + CONTEXT_LENGTH // ratio
    lines = [
        "# Rank 0 Prefill Dimensions",
        "",
        f"- Batch: `{BATCH_SIZE}`",
        f"- Sequence length: `{SEQUENCE_LENGTH}`",
        f"- Tensor-parallel world size: `{WORLD_SIZE}`",
        f"- Rank: `{RANK}`",
        f"- Source transformer layer: `{LAYER_ID}`",
        f"- Compression ratio: `{ratio}`",
        f"- Logical context capacity: `{CONTEXT_LENGTH}` tokens",
        f"- Main KV cache: `[1, {cache_slots}, {config['head_dim']}]`",
        f"- Local attention heads: `{config['n_heads'] // WORLD_SIZE}`",
        f"- Local routed experts: `{config['n_routed_experts'] // WORLD_SIZE}`",
        f"- Local vocabulary: `{config['vocab_size'] // WORLD_SIZE}`",
        "",
        "The sequence is shorter than the compression ratio, so this prefill call",
        "updates compressor state but produces a `[1, 0, 512]` compressed-KV tensor.",
        "Sparse attention reads the eight current KV positions.",
        "",
        "All weights are graph inputs. The ONNX graph has no initializers and",
        "contains custom DeepSeek and distributed nodes for structure analysis only.",
        "",
        f"- Operators: `{len(builder.operators)}`",
        f"- Parameter inputs: `{len(builder.parameters)}`",
        f"- Collective operations: `{len(builder.communications)}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


Metric = dict[str, float]
METRIC_KEYS = ("minimum", "balanced_expected", "maximum")


def fixed_metric(value: int | float) -> Metric:
    number = float(value)
    return {key: number for key in METRIC_KEYS}


def range_metric(minimum: int | float, balanced_expected: int | float, maximum: int | float) -> Metric:
    return {
        "minimum": float(minimum),
        "balanced_expected": float(balanced_expected),
        "maximum": float(maximum),
    }


def add_metrics(*metrics: Metric) -> Metric:
    return {key: sum(metric[key] for metric in metrics) for key in METRIC_KEYS}


def scale_metric(metric: Metric, factor: int | float) -> Metric:
    return {key: metric[key] * factor for key in METRIC_KEYS}


def multiply_metrics(left: Metric, right: Metric) -> Metric:
    return {key: left[key] * right[key] for key in METRIC_KEYS}


def metrics_equal(metric: Metric) -> bool:
    return metric["minimum"] == metric["balanced_expected"] == metric["maximum"]


def static_tensor_bytes(tensor: TensorDesc) -> int | None:
    if not all(isinstance(dim, int) for dim in tensor.shape):
        return None
    return math.prod(tensor.shape) * DTYPE_BYTES[tensor.dtype]


def tensor_elements_metric(tensor: TensorDesc, config: dict[str, Any]) -> Metric:
    result = fixed_metric(1)
    rows = BATCH_SIZE * SEQUENCE_LENGTH
    expected_per_expert = rows * config["n_activated_experts"] / config["n_routed_experts"]
    for dim in tensor.shape:
        if isinstance(dim, int):
            result = scale_metric(result, dim)
        elif dim.startswith("N_expert_"):
            result = multiply_metrics(result, range_metric(0, expected_per_expert, rows))
        else:
            raise ValueError(f"Unsupported symbolic tensor dimension: {dim}")
    return result


def tensor_bytes_metric(tensor: TensorDesc, config: dict[str, Any]) -> Metric:
    return scale_metric(tensor_elements_metric(tensor, config), DTYPE_BYTES[tensor.dtype])


def format_number(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.3f}".rstrip("0").rstrip(".")


def human_bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = value
    for unit in units:
        if abs(amount) < 1024 or unit == units[-1]:
            return f"{format_number(amount)} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def human_flops(value: float) -> str:
    units = ("FLOP", "KFLOP", "MFLOP", "GFLOP", "TFLOP", "PFLOP")
    amount = value
    for unit in units:
        if abs(amount) < 1000 or unit == units[-1]:
            return f"{format_number(amount)} {unit}"
        amount /= 1000
    raise AssertionError("unreachable")


def format_metric(metric: Metric, formatter) -> str:
    if metrics_equal(metric):
        return formatter(metric["balanced_expected"])
    return "/".join(formatter(metric[key]) for key in METRIC_KEYS)


def tensor_metric_sum(names: Iterable[str], builder: GraphBuilder, config: dict[str, Any]) -> Metric:
    return add_metrics(*(tensor_bytes_metric(builder.tensors[name], config) for name in names))


def estimate_operator(
    index: int,
    operator: dict[str, Any],
    builder: GraphBuilder,
    config: dict[str, Any],
    communications: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    input_names = [item["name"] for item in operator["inputs"]]
    output_names = [item["name"] for item in operator["outputs"]]
    inputs = [builder.tensors[name] for name in input_names]
    outputs = [builder.tensors[name] for name in output_names]
    op_type = operator["op_type"]
    name = operator["name"]
    attributes = operator["attributes"]

    is_conditional_expert = name.startswith("block21.moe.experts.")
    routed_assignments = BATCH_SIZE * SEQUENCE_LENGTH * config["n_activated_experts"]
    expert_activation_probability = 1 - (
        1 - config["n_activated_experts"] / config["n_routed_experts"]
    ) ** (BATCH_SIZE * SEQUENCE_LENGTH)
    conditional_parameter_read = range_metric(0, expert_activation_probability, 1)
    input_read_metrics = []
    conditional_parameter_read_bytes = fixed_metric(0)
    for tensor in inputs:
        if is_conditional_expert and tensor.kind == "parameter":
            static_bytes = static_tensor_bytes(tensor)
            if static_bytes is None:
                raise ValueError(f"Expert parameter must have a static shape: {tensor.name}")
            parameter_read = scale_metric(conditional_parameter_read, static_bytes)
            input_read_metrics.append(parameter_read)
            conditional_parameter_read_bytes = add_metrics(
                conditional_parameter_read_bytes, parameter_read
            )
        else:
            input_read_metrics.append(tensor_bytes_metric(tensor, config))
    read_bytes = add_metrics(*input_read_metrics)
    write_bytes = tensor_metric_sum(output_names, builder, config)
    flops = fixed_metric(0)
    special_math_elements = fixed_metric(0)
    selection_elements = fixed_metric(0)
    network_send_bytes = fixed_metric(0)
    network_receive_bytes = fixed_metric(0)
    notes: list[str] = []

    if op_type in {"View", "RoPESlice", "LastToken"}:
        read_bytes = fixed_metric(0)
        write_bytes = fixed_metric(0)
        notes.append("view/alias; no materialized tensor copy")
    elif op_type == "VocabParallelEmbedding":
        lookup_bytes = fixed_metric(
            BATCH_SIZE * SEQUENCE_LENGTH * config["dim"] * DTYPE_BYTES[BF16]
        )
        read_bytes = add_metrics(tensor_bytes_metric(inputs[0], config), lookup_bytes)
        notes.append("logical lookup reads one embedding row per token; masked out-of-range IDs may reuse row zero in the upstream code")
    elif op_type == "PrefillKVCacheWrite":
        current_kv = tensor_bytes_metric(inputs[1], config)
        read_bytes = current_kv
        write_bytes = current_kv
        notes.append("in-place write of the first S window-cache slots; full cache is not copied")
    elif op_type == "CompressorPrefillStateUpdate":
        active_ape = fixed_metric(BATCH_SIZE * SEQUENCE_LENGTH * config["head_dim"] * DTYPE_BYTES[FP32])
        read_bytes = add_metrics(
            tensor_bytes_metric(inputs[0], config),
            tensor_bytes_metric(inputs[1], config),
            active_ape,
        )
        write_bytes = fixed_metric(
            2 * BATCH_SIZE * SEQUENCE_LENGTH * config["head_dim"] * DTYPE_BYTES[FP32]
        )
        notes.append("S=8 < ratio=128: state update only; Compressor.norm and compressed-cache write are inactive")
    elif op_type in {"FP8Gemm", "FP4Gemm"}:
        matrix_rows = tensor_elements_metric(
            TensorDesc("rows", inputs[0].dtype, inputs[0].shape[:-1], "derived"), config
        )
        flops = scale_metric(
            matrix_rows,
            2 * attributes["in_features"] * attributes["out_features"],
        )
    elif op_type == "Gemm":
        matrix_rows = tensor_elements_metric(
            TensorDesc("rows", inputs[0].dtype, inputs[0].shape[:-1], "derived"), config
        )
        flops = scale_metric(matrix_rows, 2 * inputs[0].shape[-1] * outputs[0].shape[-1])
    elif op_type == "GroupedOutputProjection":
        flops = scale_metric(
            tensor_elements_metric(outputs[0], config),
            2 * inputs[0].shape[-1],
        )
    elif op_type == "SparseAttention":
        query = inputs[0]
        batch_seq_heads = tensor_elements_metric(
            TensorDesc("attention_rows", query.dtype, query.shape[:-1], "derived"), config
        )
        flops = scale_metric(batch_seq_heads, 4 * attributes["topk"] * query.shape[-1])
        notes.append("includes QK and AV matrix products; softmax transcendental work is tracked separately")
    elif op_type == "RMSNorm":
        dim = inputs[0].shape[-1]
        vectors = tensor_elements_metric(
            TensorDesc("norm_vectors", inputs[0].dtype, inputs[0].shape[:-1], "derived"), config
        )
        flops = scale_metric(vectors, 4 * dim + 1)
        special_math_elements = vectors
    elif op_type == "RMSReciprocal":
        dim = inputs[0].shape[-1]
        vectors = tensor_elements_metric(
            TensorDesc("norm_vectors", inputs[0].dtype, inputs[0].shape[:-1], "derived"), config
        )
        flops = scale_metric(vectors, 2 * dim + 1)
        special_math_elements = vectors
    elif op_type == "QueryRMSNormalize":
        head_dim = attributes["head_dim"]
        vectors = tensor_elements_metric(
            TensorDesc("query_vectors", outputs[0].dtype, outputs[0].shape[:-1], "derived"), config
        )
        flops = scale_metric(vectors, 3 * head_dim + 1)
        special_math_elements = vectors
    elif op_type in {"RoPE", "InverseRoPE"}:
        vectors = tensor_elements_metric(
            TensorDesc("rope_vectors", outputs[0].dtype, outputs[0].shape[:-1], "derived"), config
        )
        flops = scale_metric(vectors, 3 * attributes["rope_dim"])
        notes.append("six real arithmetic operations per complex rotary pair")
    elif op_type == "ActQuantFP8":
        elements = tensor_elements_metric(inputs[0], config)
        flops = scale_metric(elements, 3)
        notes.append("approximate absmax/scale/quantize arithmetic; reduction scheduling is kernel-specific")
    elif op_type == "ActQuantDequantFP8":
        rows = tensor_elements_metric(
            TensorDesc("quant_rows", inputs[0].dtype, inputs[0].shape[:-1], "derived"), config
        )
        flops = scale_metric(rows, 3 * attributes["quantized_features"])
        notes.append("approximate FP8 quantize-dequantize work on non-RoPE features only")
    elif op_type == "HCSplitSinkhorn":
        hc_mult = attributes["hc_mult"]
        iterations = attributes["iterations"]
        rows = tensor_elements_metric(
            TensorDesc("sinkhorn_rows", outputs[0].dtype, outputs[0].shape[:-1], "derived"), config
        )
        initial_normalization = 6 * hc_mult * hc_mult + 4 * hc_mult * (hc_mult - 1)
        repeat_normalization = 4 * hc_mult * hc_mult + 4 * hc_mult * (hc_mult - 1)
        flops = scale_metric(
            rows,
            4 * hc_mult
            + 2 * hc_mult * hc_mult
            + initial_normalization
            + (iterations - 1) * repeat_normalization,
        )
        special_math_elements = scale_metric(rows, 2 * hc_mult + hc_mult * hc_mult)
        notes.append("Sinkhorn arithmetic estimate; sigmoid/exp are counted as special math elements")
    elif op_type == "HCReduce":
        hc_mult = inputs[0].shape[-2]
        flops = scale_metric(tensor_elements_metric(outputs[0], config), 2 * hc_mult - 1)
    elif op_type == "HCPost":
        hc_mult = outputs[0].shape[-2]
        flops = scale_metric(tensor_elements_metric(outputs[0], config), 2 * hc_mult + 1)
    elif op_type == "HCHeadWeights":
        flops = scale_metric(tensor_elements_metric(outputs[0], config), 3)
        special_math_elements = tensor_elements_metric(outputs[0], config)
    elif op_type == "SwiGLU":
        flops = scale_metric(tensor_elements_metric(outputs[0], config), 2)
        special_math_elements = tensor_elements_metric(outputs[0], config)
    elif op_type == "SwiGLUWeighted":
        flops = scale_metric(tensor_elements_metric(outputs[0], config), 3)
        special_math_elements = tensor_elements_metric(outputs[0], config)
    elif op_type in {"Softplus", "Sqrt", "Sigmoid"}:
        special_math_elements = tensor_elements_metric(outputs[0], config)
    elif op_type == "GatherNormalizeRouteWeights":
        rows = outputs[0].shape[0]
        topk = outputs[0].shape[1]
        flops = fixed_metric(rows * (3 * topk - 1))
    elif op_type == "TopKStatic":
        selection_elements = tensor_elements_metric(inputs[0], config)
        notes.append("selection/comparison work is reported separately from floating-point arithmetic")
    elif op_type == "MoELocalCombine":
        assignments = range_metric(
            0,
            BATCH_SIZE * SEQUENCE_LENGTH * config["n_activated_experts"] / WORLD_SIZE,
            BATCH_SIZE * SEQUENCE_LENGTH * config["n_activated_experts"],
        )
        flops = scale_metric(assignments, config["dim"])
        read_bytes = scale_metric(assignments, config["dim"] * DTYPE_BYTES[BF16] + DTYPE_BYTES[INT64])
        write_bytes = fixed_metric(BATCH_SIZE * SEQUENCE_LENGTH * config["dim"] * DTYPE_BYTES[FP32])
        notes.append("aggregate local-expert output/position reads are bounded by total local routed assignments")
    elif op_type == "MoELocalDispatch":
        assignments = range_metric(
            0,
            BATCH_SIZE * SEQUENCE_LENGTH * config["n_activated_experts"] / WORLD_SIZE,
            BATCH_SIZE * SEQUENCE_LENGTH * config["n_activated_experts"],
        )
        read_bytes = add_metrics(
            tensor_bytes_metric(inputs[0], config),
            tensor_bytes_metric(inputs[1], config),
            tensor_bytes_metric(inputs[2], config),
        )
        write_bytes = scale_metric(
            assignments,
            config["dim"] * DTYPE_BYTES[BF16] + DTYPE_BYTES[FP32] + DTYPE_BYTES[INT64],
        )
        notes.append("aggregate dispatch outputs are bounded by total local routed assignments")
    elif op_type in {"Add", "Mul", "Div"}:
        flops = tensor_elements_metric(outputs[0], config)
    elif op_type == "PrefillWindowIndices":
        read_bytes = fixed_metric(0)
        notes.append("index generation does not consume token IDs in the upstream start_pos=0 helper")

    if op_type in {"AllReduce", "AllGather"}:
        communication = communications[name]
        network_send_bytes = fixed_metric(communication["ring_send_bytes_per_rank"])
        network_receive_bytes = fixed_metric(communication["ring_receive_bytes_per_rank"])
        notes.append("ring collective volume per rank; local HBM traffic is reported separately")
    if is_conditional_expert:
        notes.append(
            "conditional expert branch; expected parameter read uses uniform-route activation probability "
            f"{expert_activation_probability:.6f}"
        )

    return {
        "index": index,
        "name": name,
        "domain": operator["domain"],
        "op_type": op_type,
        "flops": flops,
        "special_math_elements": special_math_elements,
        "selection_elements": selection_elements,
        "local_memory_read_bytes": read_bytes,
        "conditional_parameter_read_bytes": conditional_parameter_read_bytes,
        "local_memory_write_bytes": write_bytes,
        "logical_output_capacity_bytes": tensor_metric_sum(output_names, builder, config),
        "interconnect_send_bytes": network_send_bytes,
        "interconnect_receive_bytes": network_receive_bytes,
        "notes": notes,
    }


def build_performance_report(builder: GraphBuilder, config: dict[str, Any]) -> dict[str, Any]:
    communications = {item["name"]: item for item in builder.communications}
    operators = []
    for index, operator in enumerate(builder.operators):
        performance = estimate_operator(index, operator, builder, config, communications)
        operator["performance"] = performance
        operators.append(performance)

    parameter_capacity = sum(item.get("storage_bytes", 0) for item in builder.parameters)
    conditional_expert_parameter_capacity = sum(
        item.get("storage_bytes", 0)
        for item in builder.parameters
        if item["name"].startswith("block21.moe.experts.")
    )
    inactive_parameter_capacity = sum(
        item.get("storage_bytes", 0)
        for item in builder.parameters
        if not item.get("active_for_selected_call", True)
    )
    unconditional_active_parameter_capacity = (
        parameter_capacity - conditional_expert_parameter_capacity - inactive_parameter_capacity
    )
    buffers = [tensor for tensor in builder.tensors.values() if tensor.kind == "buffer"]
    mutable_states = [
        tensor
        for tensor in builder.tensors.values()
        if tensor.kind == "state" and tensor.name.endswith("_in")
    ]
    buffer_capacity = sum(static_tensor_bytes(tensor) or 0 for tensor in buffers)
    mutable_state_capacity = sum(static_tensor_bytes(tensor) or 0 for tensor in mutable_states)
    state_outputs = {
        "block21.attn.kv_cache_out",
        "block21.attn.compressor.kv_state_out",
        "block21.attn.compressor.score_state_out",
    }
    alias_output_names = {
        output["name"]
        for operator in builder.operators
        if operator["op_type"] in {"View", "RoPESlice", "LastToken"}
        for output in operator["outputs"]
    }
    transient_tensors = [
        tensor
        for tensor in builder.tensors.values()
        if (
            tensor.kind == "intermediate"
            and tensor.name not in state_outputs
            and tensor.name not in alias_output_names
        )
    ]
    largest_transient = max(
        (
            (tensor.name, static_tensor_bytes(tensor) or 0)
            for tensor in transient_tensors
            if static_tensor_bytes(tensor) is not None
        ),
        key=lambda item: item[1],
    )

    regular_operators = [item for item in operators if not item["name"].startswith("block21.moe.experts.")]
    conditional_operators = [item for item in operators if item["name"].startswith("block21.moe.experts.")]
    routed_assignments = BATCH_SIZE * SEQUENCE_LENGTH * config["n_activated_experts"]
    individual_expert_max_assignments = BATCH_SIZE * SEQUENCE_LENGTH
    max_correlation_scale = routed_assignments / (
        (config["n_routed_experts"] // WORLD_SIZE) * individual_expert_max_assignments
    )

    def total_metric(field: str, preserve_conditional_parameter_reads: bool = False) -> Metric:
        regular = add_metrics(*(item[field] for item in regular_operators))
        if not conditional_operators:
            return regular
        conditional = add_metrics(*(item[field] for item in conditional_operators))
        if field == "local_memory_read_bytes" and preserve_conditional_parameter_reads:
            parameter_reads = add_metrics(
                *(item["conditional_parameter_read_bytes"] for item in conditional_operators)
            )
            dynamic_reads = {
                key: conditional[key] - parameter_reads[key] for key in METRIC_KEYS
            }
            corrected_conditional = {
                "minimum": dynamic_reads["minimum"] + parameter_reads["minimum"],
                "balanced_expected": dynamic_reads["balanced_expected"] + parameter_reads["balanced_expected"],
                "maximum": dynamic_reads["maximum"] * max_correlation_scale + parameter_reads["maximum"],
            }
        else:
            corrected_conditional = {
                "minimum": conditional["minimum"],
                "balanced_expected": conditional["balanced_expected"],
                "maximum": conditional["maximum"] * max_correlation_scale,
            }
        return add_metrics(regular, corrected_conditional)

    summary = {
        "static_parameter_capacity_bytes": parameter_capacity,
        "unconditional_active_parameter_capacity_bytes": unconditional_active_parameter_capacity,
        "conditional_expert_parameter_capacity_bytes": conditional_expert_parameter_capacity,
        "inactive_parameter_capacity_bytes": inactive_parameter_capacity,
        "read_only_buffer_capacity_bytes": buffer_capacity,
        "mutable_state_capacity_bytes": mutable_state_capacity,
        "persistent_capacity_bytes": parameter_capacity + buffer_capacity + mutable_state_capacity,
        "largest_nonpersistent_intermediate": {
            "name": largest_transient[0],
            "bytes": largest_transient[1],
        },
        "local_memory_read_bytes": total_metric("local_memory_read_bytes", True),
        "local_memory_write_bytes": total_metric("local_memory_write_bytes"),
        "interconnect_send_bytes": total_metric("interconnect_send_bytes"),
        "interconnect_receive_bytes": total_metric("interconnect_receive_bytes"),
        "flops": total_metric("flops"),
        "special_math_elements": total_metric("special_math_elements"),
        "selection_elements": total_metric("selection_elements"),
        "moe_summary_correlation": {
            "total_routed_assignments": routed_assignments,
            "max_assignments_on_one_local_expert": individual_expert_max_assignments,
            "local_expert_count": config["n_routed_experts"] // WORLD_SIZE,
            "maximum_correlation_scale": max_correlation_scale,
        },
    }
    return {
        "methodology": {
            "flops": "GEMM and attention use multiply-add = 2 FLOPs; custom/elementwise estimates are documented per node.",
            "local_memory": "Logical HBM reads/writes with no cache reuse or kernel fusion model; state updates count changed slots, not full state copies.",
            "interconnect": "Per-rank ring send/receive volumes. They are transfer volumes, not a bandwidth-utilization percentage or runtime estimate.",
            "moe": "Dynamic routed-expert quantities use minimum/balanced_expected/maximum over valid TopK routing outcomes.",
            "capacity": "Persistent capacity includes physical parameter storage, read-only buffers, and mutable state inputs; it is not an allocator-level peak-memory measurement.",
        },
        "operators": operators,
        "capacity_items": {
            "read_only_buffers": [
                {"name": tensor.name, "bytes": static_tensor_bytes(tensor)} for tensor in buffers
            ],
            "mutable_states": [
                {"name": tensor.name, "bytes": static_tensor_bytes(tensor)}
                for tensor in mutable_states
            ],
        },
        "summary": summary,
    }


def append_performance_ascii(lines: list[str], performance: dict[str, Any]) -> None:
    summary = performance["summary"]
    lines.extend(
        [
            "",
            "统计口径",
            "--------",
            "- FLOPs: GEMM/Attention 按一次乘加等于 2 FLOPs；自定义算子采用下方说明中的近似公式。",
            "- 本地内存: 逻辑 HBM 读写量，不模拟 L2/共享内存缓存、算子融合或分块调度。",
            "- MoE: 三元数值依次为 最小/均衡期望/最大；均衡期望假设每个 token 在 256 个专家中均匀、不重复地选择 Top-6。",
            "- 互连: 假设 Ring 集合通信，分别给出单卡发送和接收字节；不是实际带宽利用率或运行时间。",
            "- 容量: 持久容量为物理参数、只读 buffer 和可变 state 的逻辑和，不等于框架分配器峰值。",
            "",
            "逐算子计算、内存与互连统计",
            "----------------------------",
            "格式: 序号 | 节点 | 算子 | FLOPs[最小/期望/最大] | HBM读 | HBM写 | 网络发送 | 网络接收 | 附注",
        ]
    )
    for item in performance["operators"]:
        notes = list(item["notes"])
        if item["special_math_elements"] != fixed_metric(0):
            notes.append(
                "special_math="
                + format_metric(item["special_math_elements"], format_number)
            )
        if item["selection_elements"] != fixed_metric(0):
            notes.append(
                "selection_elements="
                + format_metric(item["selection_elements"], format_number)
            )
        lines.append(
            f"{item['index']:03d} | {item['name']} | {item['op_type']} | "
            f"{format_metric(item['flops'], human_flops)} | "
            f"{format_metric(item['local_memory_read_bytes'], human_bytes)} | "
            f"{format_metric(item['local_memory_write_bytes'], human_bytes)} | "
            f"{format_metric(item['interconnect_send_bytes'], human_bytes)} | "
            f"{format_metric(item['interconnect_receive_bytes'], human_bytes)} | "
            f"{' ; '.join(notes) if notes else '-'}"
        )

    total_interconnect = add_metrics(
        summary["interconnect_send_bytes"], summary["interconnect_receive_bytes"]
    )
    lines.extend(
        [
            "",
            "本卡汇总",
            "--------",
            f"静态参数容量: {human_bytes(summary['static_parameter_capacity_bytes'])}",
            "固定执行参数容量: "
            f"{human_bytes(summary['unconditional_active_parameter_capacity_bytes'])}",
            "条件路由专家参数驻留容量: "
            f"{human_bytes(summary['conditional_expert_parameter_capacity_bytes'])}",
            f"短 Prefill 非活跃参数容量: {human_bytes(summary['inactive_parameter_capacity_bytes'])}",
            f"只读 buffer 容量: {human_bytes(summary['read_only_buffer_capacity_bytes'])}",
            f"可变状态容量: {human_bytes(summary['mutable_state_capacity_bytes'])}",
            f"持久逻辑容量: {human_bytes(summary['persistent_capacity_bytes'])}",
            "最大非持久中间张量: "
            f"{summary['largest_nonpersistent_intermediate']['name']} "
            f"({human_bytes(summary['largest_nonpersistent_intermediate']['bytes'])})",
            f"计算量 FLOPs[最小/期望/最大]: {format_metric(summary['flops'], human_flops)}",
            "特殊数学元素[最小/期望/最大]: "
            + format_metric(summary["special_math_elements"], format_number),
            "TopK 选择元素[最小/期望/最大]: "
            + format_metric(summary["selection_elements"], format_number),
            "本地逻辑 HBM 读取[最小/期望/最大]: "
            + format_metric(summary["local_memory_read_bytes"], human_bytes),
            "本地逻辑 HBM 写入[最小/期望/最大]: "
            + format_metric(summary["local_memory_write_bytes"], human_bytes),
            "Ring 互连发送[最小/期望/最大]: "
            + format_metric(summary["interconnect_send_bytes"], human_bytes),
            "Ring 互连接收[最小/期望/最大]: "
            + format_metric(summary["interconnect_receive_bytes"], human_bytes),
            "Ring 互连总传输[最小/期望/最大]: "
            + format_metric(total_interconnect, human_bytes),
            "",
            "带宽占用说明: 上述是单次 Prefill 的传输字节数。实际占用率需要给定 GPU 间链路带宽、",
            "NCCL 算法、拓扑、并发流与测得时延，不能仅由静态 ONNX 图推导为百分比。",
        ]
    )


def format_shape(shape: Shape) -> str:
    return "[" + ", ".join(str(item) for item in shape) + "]"


def write_ascii_structure(
    path: Path,
    config: dict[str, Any],
    builder: GraphBuilder,
    performance: dict[str, Any],
) -> None:
    dim = config["dim"]
    head_dim = config["head_dim"]
    q_rank = config["q_lora_rank"]
    o_rank = config["o_lora_rank"]
    inter_dim = config["moe_inter_dim"]
    hc_mult = config["hc_mult"]
    local_heads = config["n_heads"] // WORLD_SIZE
    local_groups = config["o_groups"] // WORLD_SIZE
    local_experts = config["n_routed_experts"] // WORLD_SIZE
    local_vocab = config["vocab_size"] // WORLD_SIZE
    ratio = config["compress_ratios"][LAYER_ID]
    cache_slots = config["window_size"] + CONTEXT_LENGTH // ratio
    rows = BATCH_SIZE * SEQUENCE_LENGTH

    lines = [
        "DeepSeek-V4-Flash - Prefill ONNX 结构图",
        "====================================",
        "",
        "导出范围",
        "--------",
        f"TP world_size={WORLD_SIZE}, rank={RANK}, source_layer={LAYER_ID} (zero-based)",
        f"batch_size={BATCH_SIZE}, sequence_length={SEQUENCE_LENGTH}, start_pos=0 (prefill)",
        f"logical_context_capacity={CONTEXT_LENGTH} tokens",
        "sequence_length is STATIC at 8; context capacity is not a dynamic 1M input.",
        "weights are typed graph inputs with full shapes; no random placeholders.",
        "initializers=0, external_weight_data=0, forward_executed=false.",
        "Custom DeepSeek/distributed nodes preserve structure; this is not an ORT model.",
        "",
        "导出图",
        "------",
        "",
        f"input_ids INT64 [{BATCH_SIZE}, {SEQUENCE_LENGTH}]",
        "  |",
        "+- VocabParallelEmbedding (first stage)",
        f"|    rank-{RANK} vocabulary range: [{RANK * local_vocab}, {(RANK + 1) * local_vocab})",
        f"|    embedding.weight: BF16 [{local_vocab}, {dim}]",
        f"|    local output: BF16 [{BATCH_SIZE}, {SEQUENCE_LENGTH}, {dim}]",
        f"|    AllReduce(sum, TP={WORLD_SIZE}): BF16 [{BATCH_SIZE}, {SEQUENCE_LENGTH}, {dim}]",
        "|",
        f"+- HCExpand: BF16 [{BATCH_SIZE}, {SEQUENCE_LENGTH}, {hc_mult}, {dim}]",
        "|",
        f"+- TransformerBlock(source layer {LAYER_ID}, compress_ratio={ratio})",
        "|  |",
        "|  +- mHC attention pre-mix",
        f"|  |    input: [{BATCH_SIZE}, {SEQUENCE_LENGTH}, {hc_mult}, {dim}]",
        f"|  |    function: FP32 [{(2 + hc_mult) * hc_mult}, {hc_mult * dim}]",
        f"|  |    HCSplitSinkhorn pre/post: [{BATCH_SIZE}, {SEQUENCE_LENGTH}, {hc_mult}]",
        f"|  |    HCSplitSinkhorn combination: [{BATCH_SIZE}, {SEQUENCE_LENGTH}, {hc_mult}, {hc_mult}]",
        f"|  |    HCReduce output: BF16 [{BATCH_SIZE}, {SEQUENCE_LENGTH}, {dim}]",
        "|  |",
        f"|  +- Attention RMSNorm: gamma FP32 [{dim}]",
        f"|  |    output: BF16 [{BATCH_SIZE}, {SEQUENCE_LENGTH}, {dim}]",
        "|  |",
        "|  +- Multi-head Latent Attention",
        f"|  |    wq_a: FP8 [{q_rank}, {dim}]",
        f"|  |      -> q_low BF16 [{BATCH_SIZE}, {SEQUENCE_LENGTH}, {q_rank}]",
        f"|  |    q_norm: gamma FP32 [{q_rank}]",
        f"|  |    wq_b rank-{RANK}: FP8 [{local_heads * head_dim}, {q_rank}]",
        f"|  |      -> Q BF16 [{BATCH_SIZE}, {SEQUENCE_LENGTH}, {local_heads}, {head_dim}]",
        f"|  |    wkv: FP8 [{head_dim}, {dim}]",
        f"|  |      -> current KV BF16 [{BATCH_SIZE}, {SEQUENCE_LENGTH}, {head_dim}]",
        f"|  |    RoPE dims: {config['rope_head_dim']} of each {head_dim}-wide Q/K vector",
        f"|  |    freqs_cis read-only buffer: COMPLEX64 [{CONTEXT_LENGTH}, {config['rope_head_dim'] // 2}]",
        "|  |",
        f"|  |    compressor ratio={ratio} (no Indexer on ratio-128 layers)",
        f"|  |      compressor.wkv: FP32 [{head_dim}, {dim}]",
        f"|  |      compressor.wgate: FP32 [{head_dim}, {dim}]",
        f"|  |      ape: FP32 [{ratio}, {head_dim}]",
        f"|  |      norm.weight: FP32 [{head_dim}] (allocated, inactive because S < ratio)",
        f"|  |      state inputs/outputs: FP32 [{BATCH_SIZE}, {ratio}, {head_dim}] each",
        f"|  |      S={SEQUENCE_LENGTH} < ratio={ratio}: compressor returns None; no compressed KV is appended",
        f"|  |      internal zero-length compressed intermediate: [{BATCH_SIZE}, 0, {head_dim}]",
        f"|  |      logical main KV cache: BF16 [{BATCH_SIZE}, {cache_slots}, {head_dim}]",
        f"|  |      sparse-attention KV for this call: BF16 [{BATCH_SIZE}, {SEQUENCE_LENGTH}, {head_dim}]",
        "|  |",
        f"|  |    window indices: INT32 [{BATCH_SIZE}, {SEQUENCE_LENGTH}, {SEQUENCE_LENGTH}]",
        f"|  |    SparseAttention kernel pads {local_heads} local heads to 16 heads",
        f"|  |    SparseAttention(Q=[{BATCH_SIZE},{SEQUENCE_LENGTH},16,{head_dim}],",
        f"|  |                    KV=[{BATCH_SIZE},{SEQUENCE_LENGTH},{head_dim}])",
        f"|  |      -> unpadded context BF16 [{BATCH_SIZE}, {SEQUENCE_LENGTH}, {local_heads}, {head_dim}]",
        f"|  |    group view: [{BATCH_SIZE}, {SEQUENCE_LENGTH}, {local_groups}, {local_heads * head_dim}]",
        f"|  |    wo_a rank-{RANK}: BF16 [{local_groups * o_rank}, {local_heads * head_dim}]",
        f"|  |      -> [{BATCH_SIZE}, {SEQUENCE_LENGTH}, {local_groups}, {o_rank}]",
        f"|  |    wo_b rank-{RANK}: FP8 [{dim}, {local_groups * o_rank}]",
        f"|  |      -> partial FP32 [{BATCH_SIZE}, {SEQUENCE_LENGTH}, {dim}]",
        f"|  |    AllReduce(sum, TP={WORLD_SIZE}) -> BF16 [{BATCH_SIZE}, {SEQUENCE_LENGTH}, {dim}]",
        "|  |",
        f"|  +- mHC attention post-mix -> BF16 [{BATCH_SIZE}, {SEQUENCE_LENGTH}, {hc_mult}, {dim}]",
        "|  |",
        "|  +- mHC MoE pre-mix",
        f"|  |    HCReduce output: BF16 [{BATCH_SIZE}, {SEQUENCE_LENGTH}, {dim}]",
        "|  |",
        f"|  +- MoE RMSNorm: gamma FP32 [{dim}]",
        "|  |",
        "|  +- MoE",
        f"|  |    tokens: BF16 [{rows}, {dim}]",
        f"|  |    gate.weight: BF16 [{config['n_routed_experts']}, {dim}] (computed in FP32)",
        f"|  |    gate.bias: FP32 [{config['n_routed_experts']}]",
        f"|  |    scores: FP32 [{rows}, {config['n_routed_experts']}]",
        f"|  |    TopK: indices/weights [{rows}, {config['n_activated_experts']}]",
        f"|  |    rank-{RANK} local experts: [{RANK * local_experts}, {(RANK + 1) * local_experts})",
        f"|  |    dynamic local assignment count: 0..{rows * config['n_activated_experts']}",
        f"|  |    balanced expected local assignments: {rows * config['n_activated_experts'] / WORLD_SIZE:g}",
        "|  |",
        f"|  |    each routed expert e has token shape [N_e, {dim}]",
        f"|  |      w1/w3 logical FP4: [{inter_dim}, {dim}]",
        f"|  |      w2 logical FP4: [{dim}, {inter_dim}]",
        f"|  |      SwiGLU -> weighted expert output [N_e, {dim}]",
        f"|  |    local combine: FP32 [{rows}, {dim}]",
        f"|  |    AllReduce(sum, TP={WORLD_SIZE}): FP32 [{rows}, {dim}]",
        "|  |",
        f"|  |    shared expert (replicated): tokens [{rows}, {dim}]",
        f"|  |      w1/w3 FP8: [{inter_dim}, {dim}]",
        f"|  |      w2 FP8: [{dim}, {inter_dim}]",
        f"|  |    routed + shared -> BF16 [{BATCH_SIZE}, {SEQUENCE_LENGTH}, {dim}]",
        "|  |",
        f"|  +- mHC MoE post-mix -> BF16 [{BATCH_SIZE}, {SEQUENCE_LENGTH}, {hc_mult}, {dim}]",
        "|",
        "+- Tail stage 1/2: HC Head + Final RMSNorm",
        f"|    HC head function: FP32 [{hc_mult}, {hc_mult * dim}]",
        f"|    HC reduce: BF16 [{BATCH_SIZE}, {SEQUENCE_LENGTH}, {dim}]",
        f"|    final_norm gamma: FP32 [{dim}]",
        f"|    normalized hidden: BF16 [{BATCH_SIZE}, {SEQUENCE_LENGTH}, {dim}]",
        "|",
        "+- Tail stage 2/2: VocabParallel LM Head",
        f"|    LastToken -> BF16 [{BATCH_SIZE}, {dim}]",
        f"|    lm_head.weight: FP32 [{local_vocab}, {dim}] (independent from embedding.weight)",
        f"|    local logits: FP32 [{BATCH_SIZE}, {local_vocab}]",
        f"|    AllGather(concat, TP={WORLD_SIZE}) -> FP32 [{BATCH_SIZE}, {config['vocab_size']}]",
        "|",
        f"+-> logits FP32 [{BATCH_SIZE}, {config['vocab_size']}]",
        "",
        "状态输出",
        "--------",
        f"kv_cache_out: BF16 [{BATCH_SIZE}, {cache_slots}, {head_dim}]",
        f"compressor.kv_state_out: FP32 [{BATCH_SIZE}, {ratio}, {head_dim}]",
        f"compressor.score_state_out: FP32 [{BATCH_SIZE}, {ratio}, {head_dim}]",
        "The cache tensors are logical graph state; this direct exporter allocates no cache data.",
        "",
        "集合通信",
        "--------",
    ]

    for communication in builder.communications:
        if communication["collective"] == "AllReduce":
            lines.append(
                f"{communication['collective']}: {communication['name']} "
                f"{communication['dtype']} {format_shape(communication['shape'])} "
                f"payload={communication['logical_payload_bytes']} bytes/rank "
                f"ring_send={communication['ring_send_bytes_per_rank']} bytes/rank "
                f"ring_receive={communication['ring_receive_bytes_per_rank']} bytes/rank"
            )
        else:
            lines.append(
                f"{communication['collective']}: {communication['name']} "
                f"{communication['dtype']} local={format_shape(communication['local_shape'])} "
                f"output={format_shape(communication['output_shape'])} "
                f"ring_send={communication['ring_send_bytes_per_rank']} bytes/rank "
                f"ring_receive={communication['ring_receive_bytes_per_rank']} bytes/rank"
            )

    lines.extend(
        [
            "",
            "完整 43 层参考（本文件未导出）",
            "------------------------------",
            "The upstream source does not name the two compression modes CSA/HCA.",
            "The authoritative distinction is compress_ratio=0, 4, or 128.",
            "Each block also contains mHC pre/post mixing and MoE(top-6 + shared expert).",
            "",
        ]
    )
    for layer_index, layer_ratio in enumerate(
        config["compress_ratios"][: config["n_layers"]]
    ):
        if layer_ratio == 0:
            attention_mode = "window-only"
        elif layer_ratio == 4:
            attention_mode = "compressed-KV + learned Indexer"
        else:
            attention_mode = "compressed-KV, deterministic compressed indices"
        routing_mode = "hash routing" if layer_index < config["n_hash_layers"] else "score routing"
        marker = "  <-- exported source block" if layer_index == LAYER_ID else ""
        lines.append(
            f"Block {layer_index:02d}/42: ratio={layer_ratio:<3} "
            f"attention={attention_mode}; MoE={routing_mode}{marker}"
        )

    lines.extend(
        [
            "",
            "常见误读更正",
            "------------",
            "- Layer 21 (zero-based) has compress_ratio=128, not ratio=4.",
            "- KV cache is explicit state, not None; S=8 only leaves compressed KV empty.",
            "- Sequence length is fixed at 8; a 1M cache capacity does not make S dynamic.",
            "- RMSNorm semantically reduces mean(x^2); it is a custom node in this graph.",
            "- Attention uses wkv [512,4096] and factorized wo_a/wo_b, not o_proj [4096,4096].",
            "- mHC is Sinkhorn-based pre/post mixing, not a scalar alpha/beta residual.",
            "- Routed expert FFN width is 2048, not 4096.",
            "- LM head is independent from embedding and consumes only the last token.",
            "- Full-vocabulary logits use AllGather/concatenation, not AllReduce.",
            "- Collective nodes are explicit, not Identity placeholders.",
            "",
            "导出审计",
            "--------",
            f"operators={len(builder.operators)}",
            f"parameter_inputs={len(builder.parameters)}",
            f"collectives={len(builder.communications)}",
            "initializers=0",
            "weights_embedded=false",
            "forward_executed=false",
        ]
    )
    append_performance_ascii(lines, performance)
    content = "\n".join(lines) + "\n"
    path.write_text(content, encoding="utf-8")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=script_dir.parent / "inference" / "config.json",
    )
    parser.add_argument("--output-dir", type=Path, default=script_dir / "output")
    args = parser.parse_args()

    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    builder = build_graph(config)
    performance = build_performance_report(builder, config)
    model = builder.make_model()
    helper.set_model_props(
        model,
        {
            "batch_size": str(BATCH_SIZE),
            "context_length": str(CONTEXT_LENGTH),
            "forward_executed": "false",
            "layer_id": str(LAYER_ID),
            "purpose": "structure_compute_bandwidth_analysis",
            "rank": str(RANK),
            "sequence_length": str(SEQUENCE_LENGTH),
            "weights_embedded": "false",
            "world_size": str(WORLD_SIZE),
        },
    )

    onnx_path = output_dir / "rank0-prefill-b1-s8-layer21-tp8.onnx"
    onnx.save_model(model, onnx_path)
    checker.check_model(str(onnx_path), full_check=True)
    loaded = onnx.load(onnx_path, load_external_data=False)
    if loaded.graph.initializer or loaded.graph.sparse_initializer:
        raise RuntimeError("Weight-free export unexpectedly contains initializers")

    source_model = config_path.parent / "model.py"
    ascii_name = f"prefill_rank{RANK}_graph_ascii.asc"
    legacy_ascii_name = f"prefill_rank{RANK:03d}_layer{LAYER_ID:03d}_ascii.asc"
    manifest = {
        "onnx": onnx_path.name,
        "ascii_structure": ascii_name,
        "ascii_encoding": "utf-8",
        "performance_report": "rank0-performance.json",
        "mode": "direct_weight_free_structure_graph",
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "context_length": CONTEXT_LENGTH,
        "world_size": WORLD_SIZE,
        "rank": RANK,
        "layer_id": LAYER_ID,
        "weights_embedded": False,
        "forward_executed": False,
        "initializer_count": len(loaded.graph.initializer),
        "operator_count": len(builder.operators),
        "parameter_input_count": len(builder.parameters),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "source_model": str(source_model),
        "source_model_sha256": sha256(source_model),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(
        output_dir / "rank0-operators.json",
        {"manifest": manifest, "parameters": builder.parameters, "operators": builder.operators},
    )
    write_json(
        output_dir / "rank0-communications.json",
        {"world_size": WORLD_SIZE, "rank": RANK, "communications": builder.communications},
    )
    write_json(output_dir / "rank0-performance.json", {"manifest": manifest, **performance})
    write_dimensions(output_dir / "rank0-dimensions.md", config, builder)
    ascii_path = output_dir / ascii_name
    write_ascii_structure(ascii_path, config, builder, performance)
    legacy_ascii_path = output_dir / legacy_ascii_name
    if legacy_ascii_path != ascii_path:
        legacy_ascii_path.unlink(missing_ok=True)
    print(f"Wrote {onnx_path}")
    print(f"Wrote {ascii_path}")
    print(
        f"operators={len(builder.operators)} parameters={len(builder.parameters)} "
        f"collectives={len(builder.communications)} initializers=0"
    )


if __name__ == "__main__":
    main()