"""Multi-prompt CPU benchmark for Gemma 4 E2B via raw ONNX Runtime.

Uses run_inference_cpu.run_inference() under the hood (raw onnxruntime sessions).

Usage:
    python benchmark.py
    python benchmark.py --model-dir ./models/gemma-4-E2B-it-ONNX
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

SYSTEM_PROMPT = "You are a concise and accurate AI assistant."


def run_benchmark(model_dir: Path, runs: int = 1, max_tokens: int = 128) -> None:
    """Run all benchmark prompts and print aggregate stats."""
    print(f"\n{'=' * 60}")
    print(f"planckify — ONNX Runtime Benchmark (raw onnxruntime)")
    print(f"Model dir : {model_dir}")
    print(f"Prompts   : {len(BENCHMARK_PROMPTS)}")
    print(f"Runs each : {runs}")
    print(f"Max tokens: {max_tokens}")
    print(f"{'=' * 60}\n")

    all_tps: list[float] = []
    all_ttft: list[float] = []
    all_latency: list[float] = []

    for run_idx in range(runs):
        print(f"\n--- Run {run_idx + 1}/{runs} ---")
        for i, prompt in enumerate(BENCHMARK_PROMPTS, 1):
            print(f"\n[{i}/{len(BENCHMARK_PROMPTS)}] {prompt[:60]}...")
            metrics = run_inference(
                model_dir=model_dir,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            all_tps.append(metrics["decode_tps"])
            all_ttft.append(metrics["ttft_s"])
            all_latency.append(metrics["total_latency_s"])

    print(f"\n{'=' * 60}")
    print("Aggregate Statistics")
    print(f"{'=' * 60}")
    print(f"  Total prompts    : {len(all_tps)}")
    print(f"  Avg decode TPS   : {statistics.mean(all_tps):.1f} tokens/sec")
    print(f"  Median TPS       : {statistics.median(all_tps):.1f} tokens/sec")
    print(f"  Avg TTFT         : {statistics.mean(all_ttft):.3f}s")
    print(f"  Avg total latency: {statistics.mean(all_latency):.2f}s")
    if len(all_tps) > 1:
        print(f"  Stdev TPS        : {statistics.stdev(all_tps):.1f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Gemma 4 E2B on CPU via raw ONNX Runtime"
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    run_benchmark(model_dir=args.model_dir, runs=args.runs, max_tokens=args.max_tokens)
