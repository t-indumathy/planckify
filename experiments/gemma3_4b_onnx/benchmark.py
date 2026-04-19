"""Multi-prompt CPU benchmark for Gemma 3 4B IT via raw ONNX Runtime (int8).

Usage:
    python benchmark.py
    python benchmark.py --model-dir ./models/gemma-3-4b-it-ONNX
    python benchmark.py --runs 3
"""
import argparse
import statistics
from pathlib import Path

from run_inference_cpu import DEFAULT_MODEL_DIR, run_inference

BENCHMARK_PROMPTS = [
    "What is quantization in the context of neural networks?",
    "Explain the difference between ONNX and TensorFlow Lite.",
    "What is a transformer model in machine learning?",
    "How does CPU inference differ from GPU inference?",
    "Describe the Gemma model architecture briefly.",
]


def run_benchmark(model_dir: Path, runs: int = 1, max_tokens: int = 64) -> None:
    print("\n" + "=" * 60)
    print("planckify - Gemma 3 4B IT ONNX int8 Benchmark (raw onnxruntime)")
    print(f"Model dir : {model_dir}")
    print(f"Prompts   : {len(BENCHMARK_PROMPTS)}")
    print(f"Runs each : {runs}")
    print(f"Max tokens: {max_tokens}")
    print("=" * 60 + "\n")

    all_tps: list[float] = []
    all_latency: list[float] = []

    for run_idx in range(runs):
        print(f"\n--- Run {run_idx + 1}/{runs} ---")
        for i, prompt in enumerate(BENCHMARK_PROMPTS, 1):
            print(f"\n[{i}/{len(BENCHMARK_PROMPTS)}] {prompt[:60]}...")
            result = run_inference(model_dir, prompt, max_tokens)
            gen_time = result["gen_time_s"]
            tokens = result["tokens_generated"]
            tps = result["decode_speed_tok_s"]
            all_latency.append(gen_time)
            all_tps.append(tps)
            print(f"  gen_time={gen_time:.2f}s tokens={tokens} tps={tps:.1f}")

    print("\n" + "=" * 60)
    print("[Overall Benchmark Summary]")
    if all_tps:
        print(
            f"  TPS  avg={statistics.mean(all_tps):.1f}"
            f" min={min(all_tps):.1f}"
            f" max={max(all_tps):.1f}"
        )
    if all_latency:
        print(
            f"  Lat  avg={statistics.mean(all_latency):.2f}s"
            f" min={min(all_latency):.2f}s"
            f" max={max(all_latency):.2f}s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Gemma 3 4B IT ONNX int8 CPU")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    run_benchmark(model_dir=args.model_dir, runs=args.runs, max_tokens=args.max_tokens)
