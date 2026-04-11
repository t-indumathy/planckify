"""Multi-prompt CPU benchmark for Gemma 4 E2B via LiteRT-LM.

Runs a suite of prompts, collects per-prompt latency stats,
and prints a summary table.

Usage:
    python benchmark.py
    python benchmark.py --model-path ./models/gemma-4-E2B-it-litert-lm.litertlm
    python benchmark.py --max-tokens 128 --runs 3
"""

import argparse
import statistics
import time
from pathlib import Path

import litert_lm

DEFAULT_MODEL_PATH = Path("./models/gemma-4-E2B-it-litert-lm.litertlm")
CPU_NUM_THREADS = 4

BENCHMARK_PROMPTS = [
    "What is quantization in machine learning?",
    "Explain the attention mechanism in transformers.",
    "Write a Python function to compute Fibonacci numbers.",
    "What are the trade-offs between model size and inference speed?",
    "Summarize the key differences between GGUF and LiteRT model formats.",
]


def build_prompt(user_text: str) -> str:
    return (
        "<start_of_turn>user\n"
        f"{user_text}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def run_benchmark(
    model_path: Path,
    max_tokens: int = 256,
    runs: int = 1,
) -> None:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}.\nRun download_model.py first."
        )

    print(f"Loading model: {model_path}")
    print(f"Backend: CPU (XNNPACK, {CPU_NUM_THREADS} threads)")
    print(f"Max tokens per prompt: {max_tokens}")
    print(f"Runs per prompt: {runs}")
    print("=" * 70)

    engine = litert_lm.Engine(
        model_path=str(model_path),
        backend=litert_lm.Backend.CPU,
        num_threads=CPU_NUM_THREADS,
    )

    all_ttft = []
    all_prefill_tps = []
    all_decode_tps = []

    for i, prompt_text in enumerate(BENCHMARK_PROMPTS, 1):
        print(f"\n[{i}/{len(BENCHMARK_PROMPTS)}] {prompt_text[:60]}...")
        prompt_ttfts = []
        prompt_decode_tps = []

        for run_idx in range(runs):
            formatted = build_prompt(prompt_text)
            session = engine.create_session()

            prefill_start = time.perf_counter()
            session.add_query_chunk(formatted)
            prefill_end = time.perf_counter()

            tokens = []
            first_token_time = None
            decode_start = time.perf_counter()

            for token in session.generate(max_new_tokens=max_tokens):
                tokens.append(token)
                if first_token_time is None:
                    first_token_time = time.perf_counter()

            decode_end = time.perf_counter()

            ttft = (first_token_time - prefill_start) if first_token_time else 0
            decode_latency = decode_end - decode_start
            decode_tps = len(tokens) / decode_latency if decode_latency > 0 else 0

            prompt_ttfts.append(ttft)
            prompt_decode_tps.append(decode_tps)

            print(
                f"  Run {run_idx + 1}: TTFT={ttft:.2f}s | "
                f"Decode={decode_tps:.1f} tk/s | "
                f"Tokens={len(tokens)}"
            )

        all_ttft.extend(prompt_ttfts)
        all_decode_tps.extend(prompt_decode_tps)

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  Prompts run        : {len(BENCHMARK_PROMPTS)}")
    print(f"  Runs per prompt    : {runs}")
    print(f"  Avg TTFT           : {statistics.mean(all_ttft):.2f}s")
    print(f"  Min TTFT           : {min(all_ttft):.2f}s")
    print(f"  Max TTFT           : {max(all_ttft):.2f}s")
    print(f"  Avg Decode speed   : {statistics.mean(all_decode_tps):.1f} tokens/sec")
    print(f"  Min Decode speed   : {min(all_decode_tps):.1f} tokens/sec")
    print(f"  Max Decode speed   : {max(all_decode_tps):.1f} tokens/sec")
    if len(all_decode_tps) > 1:
        print(f"  Stddev Decode TPS  : {statistics.stdev(all_decode_tps):.2f}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Gemma 4 E2B CPU inference via LiteRT-LM"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens to generate per prompt (default: 256)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per prompt for averaging (default: 1)",
    )
    args = parser.parse_args()
    run_benchmark(args.model_path, args.max_tokens, args.runs)
