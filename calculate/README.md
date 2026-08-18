# DeepSeek V4 Flash Architecture Calculator

Generate the formula-driven Excel workbook:

```bash
$HOME/.local/bin/micromamba run -n deepseek-onnx \
  python calculate/generate_calculator.py
```

Outputs:

- `deepseek_v4_flash_calculator.xlsx`: editable architecture calculator.
- `baseline_results.json`: machine-readable baseline values.
- `baseline_report.md`: concise TP1 comparison report.

Workbook sheets:

- `Parameters`: editable TP, Prefill/Decode batch sizes, model dimensions,
  expert counts, cache settings, dtypes, and hardware assumptions.
- `Layer_Config`: editable per-layer mode (`window`, `short`, or `long`).
- `Prefill_8K`: formula rows for Batch-1, 8192-token prefill by default.
- `Decode_1M`: formula rows for Batch-1, one-token decode at 1M context.
- `Memory`: per-rank weights, effective/preallocated KV cache, and states.
- `Comparison`: FLOPs, HBM distribution, capacity, and charts.
- `Methodology`: calculation boundaries and limitations.

Blue cells are editable. Green and result cells contain Excel formulas. The
workbook recalculates when opened, so changing TP, either batch size, hidden
size, heads, expert counts, sequence lengths, or layer modes updates all sheets.

The report is a static architecture model, not measured runtime. HBM traffic is
logical read/write volume and does not model cache reuse or kernel fusion.