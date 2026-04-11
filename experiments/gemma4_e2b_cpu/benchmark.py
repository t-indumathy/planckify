"""Multi-prompt CPU benchmark for Gemma 4 E2B via LiteRT-LM.

Uses the official LiteRT-LM Python API:
  Engine -> create_conversation() -> send_message_async

Docs:  https://ai.google.dev/edge/litert-lm/python

Usage:
    python benchmark.py
    python benchmark.py --model-path ./models/gemma-4-E2B-it-litert-lm.litertlm
    python benchmark.py --runs 3
"""

import argparse
import statistics
import time
from pathlib import Path

import litert_lm

DEFAULT_MODEL_PATH = Path("./models/gemma-4-E2B-it-litert-lm.litertlm")

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

    print(f"Model   : {model_path}")
    print(f"Backend : CPU (XNNPACK)")
    print(f"Prompts : {len(BENCHMARK_PROMPTS)} | Runs per prompt: {runs}")
    print("=" * 70)

    load_start = time.perf_counter()
    with litert_lm.Engine(
        str(model_path),
        backend=litert_lm.Backend.CPU,
        cache_dir="/tmp/planckify-litert-cache",
    ) as engine:
        load_latency = time.perf_counter() - load_start
        print(f"Model loaded in {load_latency:.2f}s\n")

        all_ttft: list[float] = []
        all_decode_tps: list[float] = []

        for i, prompt_text in enumerate(BENCHMARK_PROMPTS, 1):
            print(f"[{i}/{len(BENCHMARK_PROMPTS)}] {prompt_text[:65]}")

            for run_idx in range(runs):
                # Fresh conversation per run (no history carry-over)
                with engine.create_conversation() as conversation:
                    chunks: list[str] = []
                    first_chunk_time: float | None = None
                    start = time.perf_counter()

                    for chunk in conversation.send_message_async(prompt_text):
                        for item in chunk.get("content", []):
                            if item.get("type") == "text":
                                chunks.append(item["text"])
                                if first_chunk_time is None:
                                    first_chunk_time = time.perf_counter()

                    end = time.perf_counter()

                ttft = (first_chunk_time - start) if first_chunk_time else (end - start)
                total = end - start
                approx_tokens = len("".join(chunks).split())
                decode_tps = approx_tokens / total if total > 0 else 0

                all_ttft.append(ttft)
                all_decode_tps.append(decode_tps)

                print(
                    f"  run {run_idx + 1}: TTFT={ttft:.2f}s | "
                    f"~{decode_tps:.1f} tk/s | ~{approx_tokens} tokens"
                )

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  Total runs         : {len(all_ttft)}")
    print(f"  Avg TTFT           : {statistics.mean(all_ttft):.2f}s")
    print(f"  Min / Max TTFT     : {min(all_ttft):.2f}s / {max(all_ttft):.2f}s")
    print(f"  Avg decode speed   : {statistics.mean(all_decode_tps):.1f} tk/s")
    print(f"  Min / Max decode   : {min(all_decode_tps):.1f} / {max(all_decode_tps):.1f} tk/s")
    if len(all_decode_tps) > 1:
        print(f"  Stddev decode TPS  : {statistics.stdev(all_decode_tps):.2f}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Gemma 4 E2B CPU inference via LiteRT-LM"
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--runs", type=int, default=1, help="Runs per prompt (default: 1)")
    args = parser.parse_args()
    run_benchmark(args.model_path, args.runs)
