"""Multi-prompt CPU benchmark for Gemma 3 4B IT via LiteRT-LM.

Uses the official LiteRT-LM Python API:
  Engine -> create_conversation() -> send_message_async

Docs: https://ai.google.dev/edge/litert-lm/python

Usage:
    python benchmark.py
    python benchmark.py --model-path ./models/gemma3-4b-it-int8.litertlm
    python benchmark.py --runs 3
"""
import argparse
import statistics
import time
from pathlib import Path

import litert_lm

DEFAULT_MODEL_PATH = Path("./models/gemma3-4b-it-int8.litertlm")

BENCHMARK_PROMPTS = [
    "What is quantization in machine learning?",
    "Explain the attention mechanism in transformers.",
    "Write a Python function to compute Fibonacci numbers.",
    "What are the trade-offs between model size and inference speed on CPU?",
    "Summarize the key differences between GGUF and LiteRT model formats.",
]


def run_benchmark(model_path: Path, runs: int = 1) -> None:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at: {model_path}\nRun download_model.py first."
        )

    litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

    print(f"Model : {model_path}")
    print("Backend : CPU (XNNPACK)")
    print(f"Prompts : {len(BENCHMARK_PROMPTS)} | Runs per prompt: {runs}")
    print("=" * 70)

    all_ttft: list[float] = []
    all_tps: list[float] = []

    with litert_lm.Engine(
        str(model_path),
        backend=litert_lm.Backend.CPU,
        cache_dir="/tmp/planckify-litert-cache",
    ) as engine:
        for i, prompt in enumerate(BENCHMARK_PROMPTS):
            print(f"\n[{i+1}/{len(BENCHMARK_PROMPTS)}] Prompt: {prompt[:60]}...")
            prompt_ttfts: list[float] = []
            prompt_tps: list[float] = []

            for run in range(runs):
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": "You are a helpful AI assistant."}]},
                ]
                with engine.create_conversation(messages=messages) as conversation:
                    chunks: list[str] = []
                    first_chunk_time: float | None = None
                    start = time.perf_counter()

                    for chunk in conversation.send_message_async(prompt):
                        for item in chunk.get("content", []):
                            if item.get("type") == "text":
                                chunks.append(item["text"])
                                if first_chunk_time is None:
                                    first_chunk_time = time.perf_counter()

                    end = time.perf_counter()
                    total = end - start
                    ttft = (first_chunk_time - start) if first_chunk_time else total
                    response = "".join(chunks)
                    approx_tokens = len(response.split())
                    tps = approx_tokens / total if total > 0 else 0
                    prompt_ttfts.append(ttft)
                    prompt_tps.append(tps)
                    print(f"  Run {run+1}: TTFT={ttft:.3f}s TPS={tps:.1f} tokens~={approx_tokens}")

            all_ttft.extend(prompt_ttfts)
            all_tps.extend(prompt_tps)

    print("\n" + "=" * 70)
    print("[Overall Benchmark Summary]")
    print(f"  TTFT avg={statistics.mean(all_ttft):.3f}s min={min(all_ttft):.3f}s max={max(all_ttft):.3f}s")
    print(f"  TPS  avg={statistics.mean(all_tps):.1f} min={min(all_tps):.1f} max={max(all_tps):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Gemma 3 4B IT int8 on CPU via LiteRT-LM")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--runs", type=int, default=1, help="Runs per prompt")
    args = parser.parse_args()
    run_benchmark(model_path=args.model_path, runs=args.runs)
