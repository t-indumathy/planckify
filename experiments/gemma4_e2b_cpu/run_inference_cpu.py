"""Single-turn inference with Gemma 4 E2B via LiteRT-LM on CPU.

Uses the official LiteRT-LM Python API:
  Engine -> create_conversation() -> send_message / send_message_async

Docs:  https://ai.google.dev/edge/litert-lm/python
Model: litert-community/gemma-4-E2B-it-litert-lm

Usage:
    python run_inference_cpu.py
    python run_inference_cpu.py --prompt "Explain transformers" --max-tokens 256
"""

import argparse
import time
from pathlib import Path

import litert_lm

DEFAULT_MODEL_PATH = Path("./models/gemma-4-E2B-it-litert-lm.litertlm")
DEFAULT_MAX_TOKENS = 512
SYSTEM_PROMPT = "You are a helpful AI assistant running on-device via LiteRT-LM."


def run_inference(model_path: Path, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    """
    Run a single streaming inference pass on CPU using LiteRT-LM.
    Returns a dict with the response text and latency metrics.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            "Run download_model.py first."
        )

    # Suppress verbose internal logs
    litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

    print(f"Loading model : {model_path}")
    print(f"Backend       : CPU (XNNPACK)")
    print("-" * 60)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    ]

    load_start = time.perf_counter()

    with litert_lm.Engine(
        str(model_path),
        backend=litert_lm.Backend.CPU,
        cache_dir="/tmp/planckify-litert-cache",
    ) as engine:
        load_latency = time.perf_counter() - load_start
        print(f"Model loaded in {load_latency:.2f}s")
        print(f"Prompt: {prompt}")
        print("-" * 60)
        print("Response:")

        with engine.create_conversation(messages=messages) as conversation:
            response_chunks: list[str] = []
            first_chunk_time: float | None = None
            infer_start = time.perf_counter()

            # Streaming decode via send_message_async
            for chunk in conversation.send_message_async(prompt):
                for item in chunk.get("content", []):
                    if item.get("type") == "text":
                        text = item["text"]
                        print(text, end="", flush=True)
                        response_chunks.append(text)
                        if first_chunk_time is None:
                            first_chunk_time = time.perf_counter()

            infer_end = time.perf_counter()

    print("\n" + "-" * 60)

    full_response = "".join(response_chunks)
    total_latency = infer_end - infer_start
    ttft = (first_chunk_time - infer_start) if first_chunk_time else total_latency
    # Approximate token count (word-level proxy)
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
    print(f"  Model load time   : {metrics['model_load_latency_s']}s")
    print(f"  TTFT              : {metrics['ttft_s']}s")
    print(f"  Total decode time : {metrics['total_latency_s']}s")
    print(f"  Approx tokens out : {metrics['approx_tokens_generated']}")
    print(f"  Approx decode TPS : {metrics['approx_decode_tps']} tokens/sec")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Gemma 4 E2B inference on CPU via LiteRT-LM"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain quantization in neural networks in simple terms.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
    )
    args = parser.parse_args()
    run_inference(model_path=args.model_path, prompt=args.prompt, max_tokens=args.max_tokens)
