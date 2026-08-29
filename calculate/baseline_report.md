# DeepSeek V4 Flash TP1 / TP8 推理基准

## 假设

- TP 配置：`1` 与 `8`
- Prefill：Batch `1`，Sequence `8192`
- Decode：Batch `1`，Tokens `1`，Context `1048576`
- 层数：`43` = window `2` + short `21` + long `20`
- MAC = 2 FLOPs；仅推理，不含 MTP/训练算子。

## 每 Rank 计算量与集合通信

| 指标 | Prefill TP1 | Prefill TP8/rank | Decode TP1 | Decode TP8/rank |
|---|---:|---:|---:|---:|
| Attention FLOPs | 102.836 TFLOPs | 23.172 TFLOPs | 0.124 TFLOPs | 0.019 TFLOPs |
| MoE FLOPs | 124.846 TFLOPs | 31.766 TFLOPs | 0.015 TFLOPs | 0.004 TFLOPs |
| 总 FLOPs | 228.398 TFLOPs | 55.653 TFLOPs | 0.140 TFLOPs | 0.024 TFLOPs |
| 通信量 | 0.000 GB | 40.635 GB | 0.000 GB | 0.006 GB |

## 每 Rank 容量

| 指标 | TP1 | TP8/rank |
|---|---:|---:|
| Attention 参数容量 | 7.451 GB | 2.238 GB |
| MoE 参数容量 | 148.351 GB | 19.578 GB |
| Other 参数容量 | 3.314 GB | 0.534 GB |
| 参数总容量 | 159.117 GB | 22.350 GB |
| Decode 1M KV + State | 7.232 GB | 7.232 GB |
| Decode 总驻留容量 | 166.349 GB | 29.582 GB |

Excel 工作簿是架构探索的主要产物：修改 TP、Batch、Hidden Size、专家数或层模式后会自动重算。
