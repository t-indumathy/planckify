"""Multi-prompt CPU benchmark for Gemma 4 E2B via ONNX Runtime GenAI.

Uses onnxruntime-genai high-level Python API:
  onnxruntime_genai.Model -> Generator -> streaming decode

Docs:  https://onnxruntime.ai/docs/genai/api/python.html

Usage:
    python benchmark.py
    python benchmark.py --model-dir ./models/gemma-4-E2B-it-ONNX
    python benchmark.py --runs 3
"""
import argparse
import statistics
import time
from pathlib import Path

import onnxruntime_genai as og

DEFAULT_MODEL_DIR = Path("./models/gemma-4-E2B-it-ONNX")
SYSTEM_PROMPT = "You are a helpful AI assistant."

BENCHMARK_PROMPTS = [
    "What is quantization in machine learning?",
    "Explain the attention mechanism in transformers.",
    "Write a Python function to compute Fibonacci numbers.",
    "What are the trade-offs between model size and inference speed on CPU?",
    "Summarize the key differences between GGUF and ONNX model formats.",
]


def run_benchmark(model_dir: Path, runs: int = 1) -> None:
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model not found at: {model_dir}\nRun download_model.py first."
        )

    print(f"Model   : {model_dir}")
    print(f"Backend : CPU (ONNX Runtime GenAI)")
    print(f"Prompts : {len(BENCHMARK_PROMPTS)} | Runs per prompt: {runs}")
    print("=" * 70)

    model = og.Model(str(model_dir))
    tokenizer = og.Tokenizer(model)

    all_ttft: list[float] = []
    all_tps: list[float] = []

    for i, prompt in enumerate(BENCHMARK_PROMPTS):
        print(f"\n[{i+1}/{len(BENCHMARK_PROMPTS)}] Prompt: {prompt[:60]}...")
        prompt_ttfts: list[float] = []
        prompt_tps: list[float] = []

        for run in range(runs):
            chat_template = (
                f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{prompt}<end_of_turn>\n"
                "<start_of_turn>model\n"
            )
            input_tokens = tokenizer.encode(chat_template)
            tokenizer_stream = tokenizer.create_stream()

            params = og.GeneratorParams(model)
            params.set_search_options(
                max_length=512,
                do_sample=False,
            )
            params.input_ids = input_tokens
            generator = og.Generator(model, params)

            chunks: list[str] = []
            first_chunk_time: float | None = None
            start = time.perf_counter()

            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()
                token = tokenizer_stream.decode(generator.get_next_tokens()[0])
                chunks.append(token)
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter()

            end = time.perf_counter()
            del generator

            total = end - start
            ttft = (first_chunk_time - start) if first_chunk_time else total
            response = "".join(chunks)
            approx_tokens = len(response.split())
            tps = approx_tokens / total if total > 0 else 0

            prompt_ttfts.append(ttft)
            prompt_tps.append(tps)
            print(f"  Run {run+1}: TTFT={ttft:.3f}s  TPS={tps:.1f}  tokens~={approx_tokens}")

        all_ttft.extend(prompt_ttfts)
        all_tps.extend(prompt_tps)

    print("\n" + "=" * 70)
    print("[Overall Benchmark Summary]")
    print(f"  TTFT  avg={statistics.mean(all_ttft):.3f}s  min={min(all_ttft):.3f}s  max={max(all_ttft):.3f}s")
    print(f"  TPS   avg={statistics.mean(all_tps):.1f}  min={min(all_tps):.1f}  max={max(all_tps):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Gemma 4 E2B on CPU via ONNX Runtime GenAI")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--runs", type=int, default=1, help="Runs per prompt")
    args = parser.parse_args()
    run_benchmark(model_dir=args.model_dir, runs=args.runs)
