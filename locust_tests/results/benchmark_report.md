# Planckify — Locust Benchmark Report

---

## Gemma 4 E2B (LiteRT-LM int4 vs ONNX q4)

**Date:** 2026-04-12  
**Model:** Gemma 4 E2B Instruction-tuned  
**Prompts:** 100 random prompts across 16 topic categories  
**Token cap:** 64 tokens per response (both flavours)  
**Concurrency:** 1 virtual user, sequential requests  
**Environment:** WSL2 (Python 3.12 Linux) on Windows host

---

## Flavours Under Test

Both flavours run entirely on **CPU** — no GPU is used in either case.

| | LiteRT-LM (CPU) | ONNX Runtime (CPU) |
|---|---|---|
| **Package** | `litert-lm-api-nightly 0.10.0.dev20260410` | `onnxruntime 1.24.4` |
| **CPU backend** | XNNPACK (kernel-fused, INT4-optimised) | CPUExecutionProvider (generic) |
| **KV cache** | Managed internally by LiteRT-LM | Manual numpy arrays in Python |
| **Model files** | `gemma-4-E2B-it.litertlm` (single file) | `embed_tokens_q4.onnx` + `decoder_model_merged_q4.onnx` |
| **Quantisation** | INT4 via LiteRT | INT4 ONNX |
| **Runtime** | WSL2 (Linux-only binary) | WSL2 |

---

## Locust Aggregate Stats

| Metric | LiteRT-LM (CPU) | ONNX Runtime (CPU) |
|---|--:|--:|
| Completed requests | 89 / 100 ¹ | 98 / 100 ¹ |
| Failures | 0 | 0 |
| Avg latency | 5 552 ms | 25 240 ms |
| p50 latency | 5 500 ms | 26 000 ms |
| p95 latency | 6 400 ms | 28 000 ms |
| p99 latency | 7 100 ms | 29 000 ms |
| Throughput | 0.169 req/s | 0.038 req/s |

> ¹ Locust flushes its stats snapshot before the final requests are written to the CSV; the `--csv` counters lag the actual 100 completed rows in `*_detailed.csv`.

---

## Per-Request Detailed Metrics

### Generation Time (seconds)

| | LiteRT-LM (CPU) | ONNX Runtime (CPU) |
|---|--:|--:|
| avg | **5.60 s** | 24.70 s |
| p50 | **5.52 s** | 24.92 s |
| p95 | **6.52 s** | 27.54 s |
| min | **4.90 s** | 15.13 s |
| max | **7.11 s** | 64.00 s |

### Decode Speed (tokens / second)

| | LiteRT-LM (CPU) | ONNX Runtime (CPU) |
|---|--:|--:|
| avg | **9.01 tok/s** | 2.67 tok/s |
| p50 | **9.02 tok/s** | 2.58 tok/s |
| p95 | **11.01 tok/s** | 3.23 tok/s |

### Tokens Generated per Response

| | LiteRT-LM (CPU) | ONNX Runtime (CPU) |
|---|--:|--:|
| avg | 50.1 tokens | 63.4 tokens |
| p50 | 51 tokens | 64 tokens |
| p95 | 56 tokens | 64 tokens |

> LiteRT-LM stops via `cancel_process()` after the first chunk that exceeds the word-count threshold (~50 words ≈ 50 tokens). ONNX uses a hard `max_new_tokens=64` loop counter, so it consistently generates closer to the cap.

### Time to First Token — LiteRT-LM only

| | LiteRT-LM (CPU) |
|---|--:|
| avg TTFT | 0.744 s |
| p50 TTFT | 0.699 s |
| p95 TTFT | 0.967 s |
| min TTFT | 0.647 s |
| max TTFT | 2.039 s |

> ONNX Runtime runs non-streaming (full decode loop returns at once), so TTFT is not measured.

---

## Delta Summary

| Metric | LiteRT-LM (CPU) | ONNX Runtime (CPU) | Ratio |
|---|--:|--:|--:|
| Avg gen time | 5.60 s | 24.70 s | **LiteRT-LM 4.4× faster** |
| Avg decode speed | 9.0 tok/s | 2.7 tok/s | **LiteRT-LM 3.4× faster** |
| Throughput | 0.169 req/s | 0.038 req/s | **LiteRT-LM 4.4× higher** |
| p95 latency | 6.5 s | 27.5 s | **LiteRT-LM 4.2× lower** |

---

## Analysis

**Why LiteRT-LM is faster:**

- Uses XNNPACK, Google's hand-optimised CPU math kernel library with INT4 weight packing and batch-matrix-multiply fusion — the entire decode step runs as a single native op.
- The `.litertlm` file bundles pre-compiled TFLite subgraphs for `decode`, `prefill_128`, `prefill_1024`, and `verify`, eliminating Python-level graph traversal.

**Why ONNX Runtime (CPU) is slower:**

- The manual KV cache is managed in Python numpy arrays, copied to and from the ONNX session on every decode step — this Python overhead adds up across 64 decode iterations.
- No kernel fusion across attention + feed-forward: each ONNX node runs as a separate CPU kernel dispatch via `CPUExecutionProvider`.
- `embed_tokens_q4.onnx` is a separate session with its own session warmup on the first call.
- XNNPACK (used by LiteRT-LM) is a significantly more optimised CPU backend than ONNX Runtime's generic `CPUExecutionProvider` for this class of INT4 transformer models.

**Measurement caveats:**

- Both tests ran on the same WSL2 instance sharing one physical machine — CPU contention could inflate latencies (Locust itself reported >90% CPU usage during the LiteRT-LM run).
- Word-count used as a proxy for token count in LiteRT-LM (streaming API does not expose token IDs). Actual token counts may differ slightly from word counts.
- ONNX outlier `max=64.00 s` was the first request (cold session / JIT compilation on first run).

---

## Files

| File | Description |
|---|---|
| `results/litertlm_detailed.csv` | Per-request: seq, prompt, TTFT, gen_time, approx_tokens, tok/s |
| `results/litertlm_stats.csv` | Locust aggregate stats for LiteRT-LM |
| `results/onnx_detailed.csv` | Per-request: seq, prompt, gen_time, tokens, tok/s |
| `results/onnx_stats.csv` | Locust aggregate stats for ONNX |
| `results/comparison.json` | Machine-readable summary of all metrics |
| `results/gemma3_4b_onnx_detailed.csv` | Per-request: seq, prompt, gen_time, tokens, tok/s (Gemma 3 4B int8) |
| `locustfile_cpu.py` | LiteRT-LM Locust test (1 user, cancel_process() token cap) |
| `locustfile_onnx.py` | ONNX Runtime Locust test (1 user, max_new_tokens=64) |
| `locustfile_gemma3_4b_onnx.py` | Gemma 3 4B ONNX int8 Locust test (1 user, max_new_tokens=64) |
| `prompts.py` | 100 shared prompts across 16 topic categories |

---

## Gemma 3 4B IT — ONNX Runtime int8 (partial run)

**Date:** 2026-04-19  
**Model:** `onnx-community/gemma-3-4b-it-ONNX` — quantized (int8) flavour  
**Files:** `embed_tokens_quantized.onnx` + `decoder_model_merged_quantized.onnx`  
**Prompts:** 10 / 100 completed before early stop  
**Token cap:** 64 tokens per response (`max_new_tokens=64`)  
**Concurrency:** 1 virtual user, sequential requests  
**Environment:** macOS (Apple Silicon), Python 3.14, `.venv` native (no WSL)

> **Note:** Run stopped early after 10 requests. All 10 hit the 64-token cap — decode speed is stable with low variance.

### Per-Request Decode Metrics

| Metric | Gemma 3 4B ONNX int8 (macOS) |
|---|--:|
| Completed requests | 10 / 100 (partial) |
| Tokens generated | 64 (all hit cap) |
| Avg decode time | 164.6 s |
| Min decode time | 147.3 s |
| Max decode time | 204.3 s |
| Avg decode speed | **0.39 tok/s** |
| Min decode speed | 0.31 tok/s |
| Max decode speed | 0.43 tok/s |

> Decode time = autoregressive loop only (after first token). Does not include prefill / TTFT.

### vs Gemma 4 E2B ONNX q4 (WSL2 x86-64 baseline)

| Metric | Gemma 4 E2B ONNX q4 (WSL2) | Gemma 3 4B ONNX int8 (macOS) |
|---|--:|--:|
| Avg decode speed | 2.67 tok/s | **0.39 tok/s** |
| Avg decode time | 24.70 s | **164.6 s** |
| Tokens generated | ~63.4 | 64 |

> **Important caveat:** These runs are on different hardware (WSL2 x86-64 vs macOS Apple Silicon) and different model sizes (2B vs 4B) and quantization schemes (int4 vs int8). The comparison is informational only — not apples-to-apples.

### Expected on Linux x86-64 OCP (projected)

| CPU type | Projected decode speed |
|---|---|
| Intel Xeon w/ AVX-512 VNNI | ~1.0–2.0 tok/s |
| AMD EPYC w/ AVX2 | ~0.8–1.5 tok/s |

> Projection based on AVX-512 VNNI int8 GEMM throughput vs observed macOS baseline. ORT threading tuning (`OMP_NUM_THREADS`, `ORT_NUM_INTRA_THREADS`) can improve this further.
