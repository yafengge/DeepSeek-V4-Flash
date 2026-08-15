# Rank 0 Prefill 结构图导出

本目录中的脚本直接构造无权重 ONNX，不调用 `Transformer.forward`，也不读取
任何 `.safetensors` 文件。

固定建模范围：

- 单机 8 卡张量并行中的 Rank 0
- Batch size：1
- Prefill token 数：8
- 逻辑上下文容量：1,048,576 tokens
- 词表并行 Embedding
- 原始第 21 个 Transformer Block（从 0 开始，`compress_ratio=128`）
- HC Head、Final RMSNorm 和词表并行 LM Head

运行：

```bash
$HOME/.local/bin/micromamba run -n deepseek-onnx \
  python export/export_rank0_prefill.py
```

输出位于 `export/output/`：

- `rank0-prefill-b1-s8-layer21-tp8.onnx`
- `rank0-operators.json`
- `rank0-communications.json`
- `rank0-performance.json`
- `rank0-dimensions.md`
- `prefill_rank0_graph_ascii.asc`
- `manifest.json`

权重仅作为带 shape 和 dtype 的 ONNX graph input，图中没有 initializer 或外部
权重数据。自定义 DeepSeek 与分布式节点用于保留 Sparse Attention、MoE、HC 和
集合通信语义，因此该图用于结构、算力和带宽分析，不保证可由 ONNX Runtime 执行。

脚本完成后会执行 `onnx.checker.check_model(..., full_check=True)`，并再次确认
initializer 数量为 0。

`prefill_rank0_graph_ascii.asc` 是 UTF-8 纯文本结构图。它包含中文分节标题、逐算子
FLOPs/逻辑 HBM 读写/Ring 互连收发统计，以及本卡汇总。MoE 相关统计使用
最小/均衡期望/最大三种路由负载，而非虚构固定专家 token 数。