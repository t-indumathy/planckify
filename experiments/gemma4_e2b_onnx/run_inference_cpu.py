"""Single-turn inference with Gemma 4 E2B via ONNX Runtime GenAI on CPU.

Uses onnxruntime-genai high-level Python API:
  onnxruntime_genai.Model -> Generator -> streaming decode

Docs:  https://onnxruntime.ai/docs/genai/api/python.html
Model: onnx-community/gemma-4-E2B-it-ONNX (q4 quantized)

Usage:
    python run_inference_cpu.py
    python run_inference_cpu.py --prompt "Explain transformers" --max-tokens 256
"""
import argparse
import time
from pathlib import Path

import onnxruntime_genai as og

DEFAULT_MODEL_DIR = Path("./models/gemma-4-E2B-it-ONNX")
DEFAULT_MAX_TOKENS = 512
SYSTEM_PROMPT = "You are a helpful AI assistant running on-device via ONNX Runtime GenAI."


def run_inference(model_dir: Path, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    """
    Run a single streaming inference pass on CPU using onnxruntime-genai.
    Returns a dict with the response text and latency metrics.
    """
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model not found at: {model_dir}\n"
            "Run download_model.py first."
        )

    print(f"Loading model : {model_dir}")
    print(f"Backend       : CPU (ONNX Runtime GenAI)")
    print("-" * 60)

    load_start = time.perf_counter()
    model = og.Model(str(model_dir))
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    load_latency = time.perf_counter() - load_start
    print(f"Model loaded in {load_latency:.2f}s")

    # Build chat prompt using Gemma instruct template
    chat_template = (
        f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{prompt}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

    input_tokens = tokenizer.encode(chat_template)

    params = og.GeneratorParams(model)
    params.set_search_options(
        max_length=max_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=False,
    )
    params.input_ids = input_tokens

    generator = og.Generator(model, params)

    print(f"Prompt: {prompt}")
    print("-" * 60)
    print("Response:")

    response_chunks: list[str] = []
    first_chunk_time: float | None = None
    infer_start = time.perf_counter()

    while not generator.is_done():
        generator.compute_logits()
        generator.generate_next_token()
        token = tokenizer_stream.decode(generator.get_next_tokens()[0])
        print(token, end="", flush=True)
        response_chunks.append(token)
        if first_chunk_time is None:
            first_chunk_time = time.perf_counter()

    infer_end = time.perf_counter()
    del generator
    print("\n" + "-" * 60)

    full_response = "".join(response_chunks)
    total_latency = infer_end - infer_start
    ttft = (first_chunk_time - infer_start) if first_chunk_time else total_latency
    approx_tokens = len(full_response.split())
    decode_tps = approx_tokens / total_latency if total_latency > 0 else 0

    metrics = {
        "prompt": prompt,
        "response": full_response,
        "model_load_latency_s": round(load_latency, 2),
        "ttft_s": round(ttft, 3),
        "total_latency_s": round(total_latency, 2),
        "approx_tokens_generated": approx_tokens,
        "approx_decode_tps": round(decode_tps, 1),
    }

    print("\n[Metrics]")
    print(f"  Model load time : {metrics['model_load_latency_s']}s")
    print(f"  TTFT            : {metrics['ttft_s']}s")
    print(f"  Total decode    : {metrics['total_latency_s']}s")
    print(f"  Approx tokens   : {metrics['approx_tokens_generated']}")
    print(f"  Approx TPS      : {metrics['approx_decode_tps']} tokens/sec")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Gemma 4 E2B inference on CPU via ONNX Runtime GenAI"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain quantization in neural networks in simple terms.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
    )
    args = parser.parse_args()
    run_inference(model_dir=args.model_dir, prompt=args.prompt, max_tokens=args.max_tokens)
