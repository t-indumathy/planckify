"""Single-turn inference with Gemma 4 E2B via LiteRT-LM on CPU.

Usage:
    python run_inference_cpu.py --prompt "Your prompt here"
    python run_inference_cpu.py --prompt "Explain transformers" --max-tokens 256

Docs:
    https://ai.google.dev/edge/litert-lm/python
"""

import argparse
import time
from pathlib import Path

import litert_lm

DEFAULT_MODEL_PATH = Path("./models/gemma-4-E2B-it-litert-lm.litertlm")
DEFAULT_MAX_TOKENS = 512
CPU_NUM_THREADS = 4  # XNNPACK thread count — tune to your core count


def build_prompt(user_text: str) -> str:
    """Wrap user text in Gemma instruct chat template."""
    return (
        "<start_of_turn>user\n"
        f"{user_text}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def run_inference(
    model_path: Path,
    prompt: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """
    Run a single inference pass on CPU using LiteRT-LM.

    Returns a dict with the response and latency metrics.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}.\n"
            "Run download_model.py first."
        )

    print(f"Loading model from: {model_path}")
    print(f"Backend: CPU (XNNPACK, {CPU_NUM_THREADS} threads)")
    print("-" * 60)

    # Initialise LiteRT-LM engine on CPU
    engine = litert_lm.Engine(
        model_path=str(model_path),
        backend=litert_lm.Backend.CPU,
        num_threads=CPU_NUM_THREADS,
    )

    formatted_prompt = build_prompt(prompt)

    # Prefill phase
    prefill_start = time.perf_counter()
    session = engine.create_session()
    session.add_query_chunk(formatted_prompt)
    prefill_end = time.perf_counter()

    prefill_tokens = len(formatted_prompt.split())  # approximate
    prefill_latency = prefill_end - prefill_start
    ttft = prefill_latency  # time-to-first-token

    print(f"Prompt: {prompt}")
    print("-" * 60)
    print("Response:")

    # Decode phase — stream tokens
    response_tokens = []
    decode_start = time.perf_counter()

    for token in session.generate(max_new_tokens=max_tokens):
        print(token, end="", flush=True)
        response_tokens.append(token)
        if len(response_tokens) == 1:
            ttft = time.perf_counter() - prefill_start

    decode_end = time.perf_counter()
    print("\n" + "-" * 60)

    decode_latency = decode_end - decode_start
    n_decoded = len(response_tokens)
    decode_tps = n_decoded / decode_latency if decode_latency > 0 else 0
    prefill_tps = prefill_tokens / prefill_latency if prefill_latency > 0 else 0

    metrics = {
        "prompt": prompt,
        "response": "".join(response_tokens),
        "prefill_tokens": prefill_tokens,
        "prefill_latency_s": round(prefill_latency, 3),
        "prefill_tps": round(prefill_tps, 1),
        "decode_tokens": n_decoded,
        "decode_latency_s": round(decode_latency, 3),
        "decode_tps": round(decode_tps, 1),
        "ttft_s": round(ttft, 3),
    }

    print("\n[Metrics]")
    print(f"  TTFT              : {metrics['ttft_s']}s")
    print(f"  Prefill speed     : {metrics['prefill_tps']} tokens/sec")
    print(f"  Decode speed      : {metrics['decode_tps']} tokens/sec")
    print(f"  Tokens generated  : {metrics['decode_tokens']}")
    print(f"  Total decode time : {metrics['decode_latency_s']}s")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Gemma 4 E2B inference on CPU via LiteRT-LM"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain quantization in neural networks in simple terms.",
        help="Input prompt",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to .litertlm model file (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens to generate (default: {DEFAULT_MAX_TOKENS})",
    )
    args = parser.parse_args()

    run_inference(
        model_path=args.model_path,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
    )
