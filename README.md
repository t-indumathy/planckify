# planckify

> On-device LLM inference experiments using [Google LiteRT-LM](https://ai.google.dev/edge/litert-lm/overview) and [ONNX Runtime GenAI](https://onnxruntime.ai/docs/genai/) — quantized down to the smallest meaningful unit.

## Overview

`planckify` is a structured experiment repo for running and benchmarking LLMs on-device across multiple inference frameworks. Both experiments target **Gemma 4 E2B** running entirely on **CPU**, using equivalent int4/q4 quantized weights — enabling a fair apples-to-apples framework comparison.

## Experiments

| Experiment | Framework | Model | Quantization | Backend | Status |
|---|---|---|---|---|---|
| `gemma4_e2b_cpu` | LiteRT-LM | Gemma 4 E2B it | int4 (baked into `.litertlm`) | CPU (XNNPACK) | 🟢 Active |
| `gemma4_e2b_onnx` | ONNX Runtime GenAI | Gemma 4 E2B it | q4 (decoder_model_merged_q4.onnx) | CPU (ORT) | 🟢 Active |

## Quantization: int4 (LiteRT-LM) vs q4 (ONNX)

Both experiments run **4-bit quantized decoder weights** — the naming difference is purely a framework convention, not a difference in precision:

| Property | LiteRT-LM (`int4`) | ONNX Runtime GenAI (`q4`) |
|---|---|---|
| Weight bit-width | 4-bit integer | 4-bit integer |
| What's quantized | Decoder weights (0.79 GB) | Decoder weights |
| What's NOT quantized | Embeddings — memory-mapped fp32 (1.12 GB) | Varies by ONNX export config |
| Grouping scheme | Fixed internally by Google, tuned for XNNPACK | Explicit at export (q4_0, q4_K_M, etc.) |
| Runtime control | None — baked into `.litertlm` at export | Chosen via model file selection |
| Comparable to | `q4_K` in llama.cpp | `q4_K_M` in llama.cpp |

> **Benchmark implication:** TPS and TTFT numbers from both experiments are directly comparable — same model, same bit-width, same CPU hardware (Ubuntu x86-64 runner). Any difference in throughput reflects **framework overhead**, not quantization level.

## Repo Structure

```
planckify/
├── experiments/
│   ├── gemma4_e2b_cpu/          # LiteRT-LM experiment
│   │   ├── requirements.txt
│   │   ├── download_model.py    # Pulls gemma-4-E2B-it.litertlm from HF
│   │   ├── run_inference_cpu.py # Engine -> create_conversation -> send_message_async
│   │   └── benchmark.py        # 5-prompt benchmark, min/avg/max stats
│   └── gemma4_e2b_onnx/         # ONNX Runtime GenAI experiment
│       ├── requirements.txt
│       ├── download_model.py    # Pulls decoder_model_merged_q4.onnx from HF
│       ├── run_inference_cpu.py # og.Model -> og.Generator streaming decode
│       └── benchmark.py        # 5-prompt benchmark, min/avg/max stats
├── .github/
│   └── workflows/
│       └── test.yml             # Parallel CI: litert-lm-cpu + onnx-genai-cpu
└── .gitignore
```

## Quickstart — LiteRT-LM

### 1. Install dependencies
```bash
cd experiments/gemma4_e2b_cpu
pip install -r requirements.txt
```

### 2. Download the model
```bash
export HF_TOKEN=your_token
python download_model.py
```
Pulls `litert-community/gemma-4-E2B-it-litert-lm` (~2.58 GB, int4).

### 3. Run inference
```bash
python run_inference_cpu.py --prompt "Explain quantization in neural networks"
```

### 4. Run benchmark
```bash
python benchmark.py --runs 1
```

## Quickstart — ONNX Runtime GenAI

### 1. Install dependencies
```bash
cd experiments/gemma4_e2b_onnx
pip install -r requirements.txt
```

### 2. Download the model
```bash
export HF_TOKEN=your_token
python download_model.py
```
Pulls `onnx-community/gemma-4-E2B-it-ONNX` q4 decoder + embedder files.

### 3. Run inference
```bash
python run_inference_cpu.py --prompt "Explain quantization in neural networks"
```

### 4. Run benchmark
```bash
python benchmark.py --runs 1
```

## Expected CPU Baselines (Linux x86-64)

| Metric | LiteRT-LM (int4) | ONNX Runtime GenAI (q4) |
|---|---|---|
| Decode speed | ~35 tokens/sec | TBD |
| TTFT | ~4 seconds | TBD |
| Peak RAM | ~3.5 GB | TBD |

> ONNX baselines will be updated after CI benchmark runs complete.

## Requirements

- Python 3.10+
- `huggingface-hub`
- CPU with AVX2 support (for XNNPACK / ORT acceleration)
- ~4 GB free RAM
- `HF_TOKEN` with read access to `litert-community` and `onnx-community` repos

## References

- [LiteRT-LM Python API](https://ai.google.dev/edge/litert-lm/python)
- [LiteRT-LM Overview](https://ai.google.dev/edge/litert-lm/overview)
- [gemma-4-E2B-it-litert-lm on HuggingFace](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm)
- [onnx-community/gemma-4-E2B-it-ONNX on HuggingFace](https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX)
- [ONNX Runtime GenAI Python API](https://onnxruntime.ai/docs/genai/api/python.html)
- [LiteRT-LM GitHub](https://github.com/google-ai-edge/LiteRT-LM)
- [ONNX Runtime GenAI GitHub](https://github.com/microsoft/onnxruntime-genai)
