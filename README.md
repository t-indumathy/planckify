# planckify

> On-device LLM inference experiments using [Google LiteRT-LM](https://ai.google.dev/edge/litert-lm/overview) and [raw ONNX Runtime](https://onnxruntime.ai/) — quantized down to the smallest meaningful unit.

## Overview

`planckify` is a structured experiment repo for running and benchmarking LLMs on-device across multiple inference frameworks. Experiments target **Gemma 4 E2B** and **Gemma 3 4B IT** running entirely on **CPU**, using equivalent quantized weights — enabling fair apples-to-apples framework comparisons.

## Experiments

| Experiment | Framework | Model | Quantization | Backend | Status |
|---|---|---|---|---|---|
| `gemma4_e2b_litertlm` | LiteRT-LM | Gemma 4 E2B it | int4 (baked into `.litertlm`) | CPU (XNNPACK) | 🟢 Active |
| `gemma3_4b_litertlm` | LiteRT-LM | Gemma 3 4B IT | int8 (`.task` / MediaPipe) | CPU (XNNPACK) | ⚠️ Format incompatible — not benchmarked |
| `gemma4_e2b_onnx` | ONNX Runtime (raw) | Gemma 4 E2B it | q4 (`decoder_model_merged_q4.onnx`) | CPU (ORT) | 🟢 Active |
| `gemma3_4b_onnx` | ONNX Runtime (raw) | Gemma 3 4B IT | int8 (`decoder_model_merged_quantized.onnx`) | CPU (ORT) | 🟢 Active |

> **Note on `gemma3_4b_litertlm`:** `litert-community/Gemma3-4B-IT` only publishes `.task` files (MediaPipe/WebGPU format). The `litert_lm.Engine` expects a `.litertlm` zip archive — these formats are incompatible. A `.litertlm` for Gemma 3 4B IT has not been published. This experiment directory is kept for reference.

## Why not onnxruntime-genai?

The natural first choice for running a Gemma 4 ONNX model would be `onnxruntime-genai` (OGA) — Microsoft's high-level GenAI loop built on top of ONNX Runtime. However, **Gemma 4 is not supported by `onnxruntime-genai` v0.12.2** (the latest release as of April 2026) due to three architectural features introduced in the Gemma 4 family that OGA's runtime and model builder do not yet handle:

### 1. Per-Layer Embeddings (PLE)

Gemma 4's `embed_tokens` session produces **two outputs** instead of one:

- `inputs_embeds` — standard `[batch, seq, hidden_size]` embedding
- `per_layer_inputs` — `[batch, seq, num_hidden_layers, hidden_size_per_layer_input]` (e.g. `[batch, seq, 35, 256]` for E2B), where each transformer layer receives its own embedding slice

OGA's inference loop expects a single embedding tensor to flow into the decoder stack. There is no mechanism to route per-layer inputs to individual attention blocks.

### 2. Variable Attention Head Dimensions

Gemma 4 uses **two different head dimensions** depending on the attention pattern:

- Sliding attention layers (most layers): `head_dim = 256`
- Full attention layers (every 5th layer — indices 4, 9, 14, 19, 24, 29, 34): `global_head_dim = 512`

`genai_config.json` only supports a single `head_size` field. OGA allocates KV cache buffers using this single value for all layers, causing shape mismatches at full-attention layers at load time.

### 3. KV Cache Sharing

Gemma 4 E2B has 35 decoder layers but only **15 unique KV cache pairs** (controlled by `num_kv_shared_layers: 20`). OGA expects one `past_key_values.N` input/output pair per layer; when the ONNX model only exposes 15 unique KV outputs for 35 layers, the KV cache management breaks.

### Workarounds attempted (and failed)

- Patching `builder.py` to route `Gemma4ForConditionalGeneration` through the Gemma 3 pipeline → produces ONNX but fails at runtime with `ShapeInferenceError` at full-attention layers (layer 4, `global_head_dim=512` vs expected `256`)
- Changing `genai_config.json` model type from `gemma4` to `gemma3_text` → same shape error
- Loading `onnx-community/gemma-4-E2B-it-ONNX` directly via OGA → incompatible I/O contract (separate `embed_tokens.onnx` + `per_layer_inputs` tensor not understood by OGA's KV cache manager)

### Result

The `gemma4_e2b_onnx` experiment therefore drives the ONNX model using **two raw `onnxruntime.InferenceSession` objects** — one for `embed_tokens_q4.onnx` and one for `decoder_model_merged_q4.onnx` — and manages the KV cache manually in Python. This bypasses OGA entirely and works today.

### Tracking issues

- [microsoft/onnxruntime-genai#2062](https://github.com/microsoft/onnxruntime-genai/issues/2062) — Feature request: Gemma 4 support (PLE, variable head dims, KV cache sharing) — **Open**
- [microsoft/onnxruntime-genai#2059](https://github.com/microsoft/onnxruntime-genai/issues/2059) — General question: any plans for Gemma 4 support? — **Open**

---

## Quantization: int4 (LiteRT-LM) vs q4 (ONNX)

Both experiments run **4-bit quantized decoder weights** — the naming difference is purely a framework convention, not a difference in precision:

| Property | LiteRT-LM (`int4`) | ONNX Runtime (`q4`) |
|---|---|---|
| Weight bit-width | 4-bit integer | 4-bit integer |
| What's quantized | Decoder weights (0.79 GB) | Decoder weights |
| What's NOT quantized | Embeddings — memory-mapped fp32 (1.12 GB) | Varies by ONNX export config |
| Grouping scheme | Fixed internally by Google, tuned for XNNPACK | Explicit at export (q4_0, q4_K_M, etc.) |
| Runtime control | None — baked into `.litertlm` at export | Chosen via model file selection |
| Comparable to | `q4_K` in llama.cpp | `q4_K_M` in llama.cpp |

> **Benchmark implication:** TPS and latency numbers from both experiments are directly comparable — same model, same bit-width, same CPU hardware (Ubuntu x86-64 runner). Any difference in throughput reflects **framework overhead**, not quantization level.

## Repo Structure

```
planckify/
├── experiments/
│   ├── gemma4_e2b_litertlm/     # LiteRT-LM experiment
│   │   ├── requirements.txt
│   │   ├── download_model.py    # Pulls gemma-4-E2B-it.litertlm from HF
│   │   ├── run_inference_cpu.py # Engine -> create_conversation -> send_message_async
│   │   └── benchmark.py         # 5-prompt benchmark, min/avg/max stats
│   ├── gemma3_4b_litertlm/     # LiteRT-LM experiment (Gemma 3 4B IT int8)
│   │   ├── requirements.txt
│   │   ├── download_model.py    # Pulls gemma-3-4b-it-int8.litertlm from HF
│   │   ├── run_inference_cpu.py # Engine -> create_conversation -> send_message_async
│   │   └── benchmark.py         # 5-prompt benchmark, min/avg/max stats
│   └── gemma4_e2b_onnx/         # Raw ONNX Runtime experiment
│       ├── requirements.txt
│       ├── download_model.py    # Pulls decoder_model_merged_q4.onnx from HF
│       ├── run_inference_cpu.py # Two ORT sessions: embed_tokens + decoder (KV cache)
│       └── benchmark.py         # 5-prompt benchmark, min/avg/max stats
│   └── gemma3_4b_onnx/          # Raw ONNX Runtime experiment (Gemma 3 4B IT int8)
│       ├── requirements.txt
│       ├── download_model.py    # Pulls embed_tokens_quantized + decoder_model_merged_quantized
│       ├── run_inference_cpu.py # Two ORT sessions: embed_tokens + decoder (KV cache, int8)
│       └── benchmark.py         # 5-prompt benchmark, min/avg/max stats
├── .github/
│   └── workflows/
│       └── test.yml             # Parallel CI: litert-lm-cpu + onnx-raw-cpu
└── .gitignore
```

## Quickstart — Gemma 3 4B IT (ONNX Runtime int8)

### 1. Install dependencies

```bash
cd experiments/gemma3_4b_onnx
pip install -r requirements.txt
```

### 2. Download the model

```bash
export HF_TOKEN=your_token
python download_model.py
```

Pulls `onnx-community/gemma-3-4b-it-ONNX` — int8/quantized subset only (`embed_tokens_quantized.onnx` + `decoder_model_merged_quantized.onnx`, ~5.5 GB).

### 3. Run inference

```bash
python run_inference_cpu.py --prompt "Explain quantization in neural networks"
```

### 4. Run benchmark

```bash
python benchmark.py --runs 1
```

## Quickstart — LiteRT-LM

### 1. Install dependencies

```bash
cd experiments/gemma4_e2b_litertlm
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

## Quickstart — ONNX Runtime (raw)

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

Pulls `onnx-community/gemma-4-E2B-it-ONNX` — `embed_tokens_q4.onnx` + `decoder_model_merged_q4.onnx`.

### 3. Run inference

```bash
python run_inference_cpu.py --prompt "Explain quantization in neural networks"
```

### 4. Run benchmark

```bash
python benchmark.py --runs 1
```

## CI Baselines — GitHub Actions (Ubuntu x86-64, 2 vCPU)

> Measured on run [#36](https://github.com/t-indumathy/planckify/actions/runs/24287930708) · 5 prompts · 64 max tokens each

| Metric | LiteRT-LM (int4) | ONNX Runtime raw (q4) |
|---|---|---|
| Decode speed avg | 14.7 tok/s | 5.4 tok/s |
| Decode speed min/max | 11.5 / 15.7 tok/s | 5.3 / 5.4 tok/s |
| TTFT / latency avg | 0.998 s | 11.91 s |
| TTFT / latency min/max | 0.836 / 1.636 s | 11.84 / 12.01 s |
| Peak RAM | ~3.5 GB | ~3.5 GB |

> **Note:** ONNX raw runs a full dual-session KV-cache decode loop with no GenAI-level fusion. The 5.4 tok/s on a 2-vCPU runner is the baseline floor — native AVX-512 hardware or a GPU EP will be significantly faster. LiteRT-LM's XNNPACK kernel is highly optimised for this workload, hence the ~2.7× throughput advantage on the same runner.

## Local Machine Baselines (macOS Apple Silicon)

> Measured locally on macOS (Apple Silicon) · Python 3.14 · `.venv` native · 10 prompts · 64 max tokens each · **partial run (stopped early)**

| Metric | Gemma 3 4B ONNX int8 |
|---|---|
| Avg decode speed | 0.39 tok/s |
| Min / Max decode speed | 0.31 / 0.43 tok/s |
| Avg decode time | 164.6 s |
| Min / Max decode time | 147.3 / 204.3 s |
| Tokens generated | 64 (all hit cap) |
| Peak RAM | ~4.2 GB |

> **Note:** macOS Apple Silicon uses ARM NEON — not AVX-512 VNNI. Expect **2–4× higher throughput** on a Linux x86_64 node with AVX-512 VNNI (e.g. Intel Ice Lake / Sapphire Rapids). See [benchmark report](locust_tests/results/benchmark_report.md) for projected OCP numbers.

## Local Machine Baselines (x86-64)

> Measured locally on Intel i7-9750H (12 vCPU, 2.60GHz) · ORT pinned to 4 threads · 5 prompts · 64 max tokens each

| Metric | LiteRT-LM (int4) | ONNX Runtime raw (q4) |
|---|---|---|
| Decode speed avg | 8.2 tok/s | 6.1 tok/s |
| Decode speed min/max | 6.9 / 9.0 tok/s | 5.9 / 6.4 tok/s |
| TTFT / latency avg | 0.994 s | 10.53 s |
| TTFT / latency min/max | 0.693 / 1.998 s | 10.04 / 10.80 s |
| Peak RAM | ~3.5 GB | ~3.5 GB |

> **Note:** LiteRT-LM retains a ~1.3× TPS advantage and ~10× TTFT advantage over raw ONNX Runtime on the same hardware.

## Requirements

- Python 3.10+
- `huggingface-hub`
- CPU with AVX2 support (for XNNPACK / ORT acceleration)
- ~4 GB free RAM
- `HF_TOKEN` with read access to `litert-community` and `onnx-community` repos

## References

- [onnx-community/gemma-3-4b-it-ONNX on HuggingFace](https://huggingface.co/onnx-community/gemma-3-4b-it-ONNX)
- [LiteRT-LM Python API](https://ai.google.dev/edge/litert-lm/python)
- [LiteRT-LM Overview](https://ai.google.dev/edge/litert-lm/overview)
- [gemma-4-E2B-it-litert-lm on HuggingFace](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm)
- [onnx-community/gemma-4-E2B-it-ONNX on HuggingFace](https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX)
- [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)
- [LiteRT-LM GitHub](https://github.com/google-ai-edge/LiteRT-LM)
- [ONNX Runtime GitHub](https://github.com/microsoft/onnxruntime)
- [onnxruntime-genai#2062 — Gemma 4 support request](https://github.com/microsoft/onnxruntime-genai/issues/2062)
- [onnxruntime-genai#2059 — Gemma 4 plans?](https://github.com/microsoft/onnxruntime-genai/issues/2059)
