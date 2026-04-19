"""Single-turn inference with Gemma 3 4B IT via LiteRT-LM on CPU.

Uses the official LiteRT-LM Python API:
  Engine -> create_conversation() -> send_message / send_message_async

Docs: https://ai.google.dev/edge/litert-lm/python
Model: litert-community/gemma-3-4b-it-litert-lm

Usage:
    python run_inference_cpu.py
    python run_inference_cpu.py --prompt "Explain transformers" --max-tokens 256
"""
import argparse
import time
from pathlib import Path

import litert_lm

DEFAULT_MODEL_PATH = Path("./models/gemma-3-4b-it-int8.litertlm")
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
    print("Backend       : CPU (XNNPACK via LiteRT-LM)")
    print("-" * 60)

    load_start = time.perf_counter()

    with litert_lm.Engine(
        str(model_path),
        backend=litert_lm.Backend.CPU,
        cache_dir="/tmp/planckify-litert-cache",
    ) as engine:
        load_time = time.perf_counter() - load_start

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        ]

        gen_start = time.perf_counter()
        chunks: list[str] = []
        first_chunk_time: float | None = None

        with engine.create_conversation(messages=messages) as conversation:
            for chunk in conversation.send_message_async(prompt):
                for item in chunk.get("content", []):
                    if item.get("type") == "text":
                        chunks.append(item["text"])
                        if first_chunk_time is None:
                            first_chunk_time = time.perf_counter()

        gen_end = time.perf_counter()
        response = "".join(chunks)
        total_time = gen_end - gen_start
        ttft = (first_chunk_time - gen_start) if first_chunk_time else total_time

        # Approximate token count from word count
        approx_tokens = len(response.split())
        decode_speed = approx_tokens / total_time if total_time > 0 else 0.0

    print(f"Prompt        : {prompt}")
    print(f"Response      : {response}")
    print("-" * 60)
    print(f"Load time     : {load_time:.2f}s")
    print(f"TTFT          : {ttft:.3f}s")
    print(f"Total gen time: {total_time:.2f}s")
    print(f"Approx tokens : {approx_tokens}")
    print(f"Decode speed  : {decode_speed:.1f} tok/s (approx)")

    return {
        "prompt": prompt,
        "response": response,
        "load_time_s": round(load_time, 3),
        "ttft_s": round(ttft, 3),
        "gen_time_s": round(total_time, 3),
        "approx_tokens": approx_tokens,
        "decode_speed_tok_s": round(decode_speed, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Gemma 3 4B IT int8 LiteRT-LM CPU inference")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prompt", type=str, default="What is the Planck constant?")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()

    result = run_inference(args.model_path, args.prompt, args.max_tokens)
    print(f"\nResult JSON: {result}")


if __name__ == "__main__":
    main()
