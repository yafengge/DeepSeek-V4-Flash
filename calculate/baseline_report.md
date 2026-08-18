# DeepSeek V4 Flash TP1 Baseline

## Assumptions

- TP: `1`
- Prefill: batch `1`, sequence `8192`
- Decode: batch `1`, tokens `1`, context `1048576`
- Layers: `43` = window `2` + short `21` + long `20`
- MAC = 2 FLOPs; MTP excluded.

## Compute and HBM

| Metric | Prefill 8K | Decode 1M |
|---|---:|---:|
| Attention major FLOPs/rank | 102.836 TFLOPs | 0.124 TFLOPs |
| MoE major FLOPs/rank | 124.846 TFLOPs | 0.015 TFLOPs |
| Total modeled FLOPs/rank | 228.384 TFLOPs | 0.140 TFLOPs |
| Attention HBM traffic/rank | 358.038 GB | 10.492 GB |
| MoE HBM traffic/rank | 191.989 GB | 4.627 GB |
| Other HBM traffic/rank | 94.596 GB | 2.265 GB |
| HBM read/rank | 493.459 GB | 16.642 GB |
| HBM write/rank | 151.163 GB | 0.743 GB |

## Capacity

| Metric | Capacity |
|---|---:|
| Parameters/rank | 159.117 GB |
| Prefill effective KV + states | 0.074 GB |
| Prefill preallocated KV + states | 7.232 GB |
| Decode effective KV + states | 7.232 GB |
| Decode preallocated KV + states | 7.232 GB |

The Excel workbook is authoritative for architecture exploration: all detail rows contain formulas and recalculate after input edits.
