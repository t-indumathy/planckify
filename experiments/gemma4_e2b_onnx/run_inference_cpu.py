"""Single-turn inference with Gemma 4 E2B via raw ONNX Runtime on CPU.

Uses onnxruntime (raw sessions) + transformers tokenizer:
  ort.InferenceSession -> tokenizer.encode -> greedy decode loop

Docs:  https://onnxruntime.ai/docs/
Model: onnx-community/gemma-4-E2B-it-ONNX (q4 quantized)

Usage:
    python run_inference_cpu.py
    python run_inference_cpu.py --prompt "Explain transformers" --max-tokens 256
"""
import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

DEFAULT_MODEL_DIR = Path("./models/gemma-4-E2B-it-ONNX")
DEFAULT_MAX_TOKENS = 512
SYSTEM_PROMPT = "You are a helpful AI assistant running on-device via ONNX Runtime."


def run_inference(model_dir: Path, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    """
    Run a single greedy inference pass on CPU using raw onnxruntime sessions.
    Returns a dict with the response text and latency metrics.
    """
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model not found at: {model_dir}\n"
            "Run download_model.py first."
        )

    print(f"Loading model : {model_dir}")
    print(f"Backend       : CPU (raw ONNX Runtime)")
    print("-" * 60)

    load_start = time.perf_counter()

    # Load tokenizer from the model directory
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    # Locate the ONNX model file
    onnx_files = list(model_dir.rglob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No .onnx files found under {model_dir}")
    onnx_path = onnx_files[0]
    print(f"ONNX model    : {onnx_path}")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    load_latency = time.perf_counter() - load_start
    print(f"Model loaded in {load_latency:.2f}s")

    # Build chat prompt using Gemma instruct template
    chat_template = (
        f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{prompt}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

    input_ids = tokenizer.encode(chat_template, return_tensors="np").astype(np.int64)

    print(f"Prompt: {prompt}")
    print("-" * 60)
    print("Response:")

    response_tokens: list[int] = []
    first_token_time: float | None = None
    infer_start = time.perf_counter()

    # Greedy decode loop
    generated = input_ids.copy()
    for _ in range(max_tokens):
        inputs = {"input_ids": generated}
        # Some ONNX Gemma exports also need attention_mask
        attention_mask = np.ones_like(generated, dtype=np.int64)
        inputs["attention_mask"] = attention_mask

        try:
            outputs = session.run(None, inputs)
        except Exception:
            # Try without attention_mask if the model doesn't accept it
            outputs = session.run(None, {"input_ids": generated})

        logits = outputs[0]  # shape: [batch, seq_len, vocab_size]
        next_token_id = int(np.argmax(logits[0, -1, :]))

        if first_token_time is None:
            first_token_time = time.perf_counter()

        response_tokens.append(next_token_id)
        token_text = tokenizer.decode([next_token_id], skip_special_tokens=True)
        print(token_text, end="", flush=True)

        # Stop on EOS
        if next_token_id == tokenizer.eos_token_id:
            break

        generated = np.concatenate(
            [generated, np.array([[next_token_id]], dtype=np.int64)], axis=1
        )

    infer_end = time.perf_counter()
    print("\n" + "-" * 60)

    full_response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    total_latency = infer_end - infer_start
    ttft = (first_token_time - infer_start) if first_token_time else total_latency
    tokens_generated = len(response_tokens)
    decode_tps = tokens_generated / total_latency if total_latency > 0 else 0

    metrics = {
        "prompt": prompt,
        "response": full_response,
        "model_load_latency_s": round(load_latency, 2),
        "ttft_s": round(ttft, 3),
        "total_latency_s": round(total_latency, 2),
        "tokens_generated": tokens_generated,
        "decode_tps": round(decode_tps, 1),
    }
    print("\n[Metrics]")
    print(f"  Model load time : {metrics['model_load_latency_s']}s")
    print(f"  TTFT            : {metrics['ttft_s']}s")
    print(f"  Total decode    : {metrics['total_latency_s']}s")
    print(f"  Tokens out      : {metrics['tokens_generated']}")
    print(f"  Decode TPS      : {metrics['decode_tps']} tokens/sec")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Gemma 4 E2B inference on CPU via raw ONNX Runtime"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain quantization in neural networks in simple terms.",
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()
    run_inference(model_dir=args.model_dir, prompt=args.prompt, max_tokens=args.max_tokens)
