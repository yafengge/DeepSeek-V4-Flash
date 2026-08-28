# DeepSeek V4 Flash 推理架构计算器

生成公式驱动的 Excel 工作簿：

```bash
$HOME/.local/bin/micromamba run -n deepseek-onnx \
  python calculate/generate_calculator.py
```

输出文件：

- `deepseek_v4_flash_calculator.xlsx`：可编辑的推理架构计算器。
- `baseline_results.json`：TP1/TP8 基准数据。
- `baseline_report.md`：中文汇总报告。

工作表：

- `Parameters`：可调整 TP1/TP8、Prefill/Decode Batch、Hidden Size、
  Head、专家数、Cache、dtype 和硬件参数；架构约束由生成器代码校验，
  不在工作簿中单独展示。
- `Prefill_8K`：恰好两张表：整网资源汇总，以及按典型层/模块拆分的明细；两张表均横向列出 TP1/TP8 每 Rank 容量、算力、集合通信数据量、参数类型和激活类型。
- `Decode_1M`：恰好两张表：整网资源汇总，以及按典型层/模块拆分的明细；两张表均横向列出 TP1/TP8 每 Rank 容量、算力、集合通信数据量、参数类型和激活类型。
- 场景分类：Attention、MoE、Embedding、LM Head、Norm、HC、residual；典型明细进一步拆分 Window/Short/Long 注意力、压缩器、Indexer、专家和缓存状态。
- `dtype`：各种算子和张量的参数存储、激活/中间计算类型，以及 43 层逐层 dtype 与 Attention / Router 模式总表。
- `Summary`：面向每 Rank 硬件配置的单次推理 FLOPs、HBM 流量、集合通信数据量和容量总览；单次推理 FLOPs 不按目标时延换算。
- `Comparison`：一张统一的 TP1/TP8 资源对比表和七张图；一次推理的总 FLOPs 与单位显示在同一数量单元格中。
- `Layer_Config`：逐层调整 `window`、`short`、`long` 模式。
- `Methodology`：中文统计口径与限制。

蓝色单元格是输入项，绿色及结果单元格由 Excel 公式计算。打开工作簿时会完整
重算，因此修改 TP、Batch、Hidden Size、Head、专家数、序列长度或层模式后，
所有推理结果都会更新。

本计算器只考虑推理，不包含反向传播、梯度、优化器或训练状态。场景页可见的集合
通信数据量是每 Rank 的原始字节数，不除以目标时延；HBM 数据仍作为独立的逻辑读写量，
用于 HBM 流量、算术强度和目标 HBM 带宽估算。两者都不模拟 Cache 命中、Kernel 融合或
实测运行时间。目标时延不用于一次推理总 FLOPs 的计算。