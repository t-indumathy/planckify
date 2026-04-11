# planckify

> On-device LLM inference experiments using [Google LiteRT-LM](https://ai.google.dev/edge/litert-lm/overview) — quantized down to the smallest meaningful unit.

## Overview

`planckify` is a structured experiment repo for running and benchmarking LLMs on-device using Google's **LiteRT-LM** runtime. The first experiment targets **Gemma 4 E2B** (2B parameter, int4 quantized) running entirely on **CPU** via the XNNPACK delegate.

## Experiments

| Experiment | Model | Backend | Status |
|---|---|---|---|
| `gemma4_e2b_cpu` | Gemma 4 E2B it | CPU (XNNPACK) | 🟢 Active |

## Repo Structure

```
planckify/
├── experiments/
│   └── gemma4_e2b_cpu/
│       ├── requirements.txt       # litert-lm + dependencies
│       ├── download_model.py      # Pull model from HuggingFace
│       ├── run_inference_cpu.py   # Single-turn inference with latency logging
│       └── benchmark.py          # Multi-prompt benchmark with stats
├── .github/
│   └── workflows/
│       └── test.yml               # CI smoke test on CPU (Linux)
└── .gitignore
```

## Quickstart

### 1. Install dependencies

```bash
cd experiments/gemma4_e2b_cpu
pip install -r requirements.txt
```

### 2. Download the model

```bash
python download_model.py
```

This pulls `litert-community/gemma-4-E2B-it-litert-lm` from HuggingFace (~2.6 GB, int4 weights).

### 3. Run inference

```bash
python run_inference_cpu.py --prompt "Explain quantization in neural networks"
```

### 4. Run benchmark

```bash
python benchmark.py
```

## Expected CPU Baselines (Linux x86-64, 4 threads)

| Metric | Expected |
|---|---|
| Prefill speed | ~260 tokens/sec |
| Decode speed | ~35 tokens/sec |
| Time to first token (TTFT) | ~4 seconds |
| Peak RAM | ~3.5 GB |

## Requirements

- Python 3.10+
- `litert-lm >= 0.2.0`
- `huggingface-hub`
- CPU with AVX2 support (for XNNPACK acceleration)
- ~4 GB free RAM

## References

- [LiteRT-LM Python API](https://ai.google.dev/edge/litert-lm/python)
- [LiteRT-LM Overview](https://ai.google.dev/edge/litert-lm/overview)
- [gemma-4-E2B-it-litert-lm on HuggingFace](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm)
- [LiteRT-LM GitHub](https://github.com/google-ai-edge/LiteRT-LM)
