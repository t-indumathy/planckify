"""Single-turn text inference with Gemma 4 E2B via raw ONNX Runtime on CPU.

Follows the official ONNX Runtime Python example from the model card:
  https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX

Uses two ONNX sessions (text-only, no vision/audio):
  1. embed_tokens_q4.onnx  - input_ids -> inputs_embeds + per_layer_inputs
  2. decoder_model_merged_q4.onnx - autoregressive decode with KV cache

Usage:
    python run_inference_cpu.py
    python run_inference_cpu.py --prompt "Explain transformers" --max-tokens 64
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoConfig, AutoProcessor, GenerationConfig

DEFAULT_MODEL_DIR = Path("./models/gemma-4-E2B-it-ONNX")
DEFAULT_MAX_TOKENS = 64


def run_inference(model_dir: Path, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    """Run text inference using embed_tokens + decoder ONNX sessions with KV cache."""
    if not model_dir.exists():
        raise FileNotFoundError(f"Model not found at: {model_dir}\nRun download_model.py first.")

    print(f"Loading model : {model_dir}")
    print(f"Backend       : CPU (raw ONNX Runtime, text-only)")
    print("-" * 60)

    load_start = time.perf_counter()

    processor = AutoProcessor.from_pretrained(str(model_dir))
    config = AutoConfig.from_pretrained(str(model_dir))
    generation_config = GenerationConfig.from_pretrained(str(model_dir))

    providers = ["CPUExecutionProvider"]
    embed_path = model_dir / "onnx" / "embed_tokens_q4.onnx"
    decoder_path = model_dir / "onnx" / "decoder_model_merged_q4.onnx"

    if not embed_path.exists():
        raise FileNotFoundError(f"embed_tokens_q4.onnx not found at {embed_path}")
    if not decoder_path.exists():
        raise FileNotFoundError(f"decoder_model_merged_q4.onnx not found at {decoder_path}")

    embed_session = ort.InferenceSession(str(embed_path), providers=providers)
    decoder_session = ort.InferenceSession(str(decoder_path), providers=providers)

    load_latency = time.perf_counter() - load_start
    print(f"Model loaded in {load_latency:.2f}s")

    # Build text-only chat prompt
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="np",
    )

    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)
    position_ids = (np.cumsum(attention_mask, axis=-1) - 1).astype(np.int64)

    # Initialize KV cache (past_key_values)
    past_key_values = {
        inp.name: np.zeros(
            [1, inp.shape[1], 0, inp.shape[3]],
            dtype=np.float32 if inp.type == "tensor(float)" else np.float16,
        )
        for inp in decoder_session.get_inputs()
        if inp.name.startswith("past_key_values")
    }
    num_logits_to_keep = np.array(1, dtype=np.int64)
    eos_token_id = generation_config.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_ids = set(eos_token_id)
    else:
        eos_token_ids = {eos_token_id}

    print(f"Prompt: {prompt}")
    print("-" * 60)
    print("Response:")

    generated_tokens = []
    first_token_time = None
    infer_start = time.perf_counter()

    for _ in range(max_tokens):
        # Step 1: embed input_ids
        inputs_embeds, per_layer_inputs = embed_session.run(None, {"input_ids": input_ids})

        # Step 2: run decoder with KV cache
        decoder_inputs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            per_layer_inputs=per_layer_inputs,
            position_ids=position_ids,
            num_logits_to_keep=num_logits_to_keep,
            **past_key_values,
        )
        decoder_outputs = decoder_session.run(None, decoder_inputs)
        logits = decoder_outputs[0]
        present_key_values = decoder_outputs[1:]

        # Greedy token selection
        next_token_id = int(np.argmax(logits[:, -1, :]))

        if first_token_time is None:
            first_token_time = time.perf_counter()

        generated_tokens.append(next_token_id)
        token_text = processor.decode([next_token_id], skip_special_tokens=True)
        print(token_text, end="", flush=True)

        if next_token_id in eos_token_ids:
            break

        # Update KV cache and position
        input_ids = np.array([[next_token_id]], dtype=np.int64)
        attention_mask = np.concatenate(
            [attention_mask, np.ones((1, 1), dtype=np.int64)], axis=-1
        )
        position_ids = np.array([[position_ids[0, -1] + 1]], dtype=np.int64)
        for j, key in enumerate(past_key_values):
            past_key_values[key] = present_key_values[j]

    infer_end = time.perf_counter()
    print("\n" + "-" * 60)

    full_response = processor.decode(generated_tokens, skip_special_tokens=True)
    total_latency = infer_end - infer_start
    ttft = (first_token_time - infer_start) if first_token_time else total_latency
    decode_tps = len(generated_tokens) / total_latency if total_latency > 0 else 0

    metrics = {
        "prompt": prompt,
        "response": full_response,
        "model_load_latency_s": round(load_latency, 2),
        "ttft_s": round(ttft, 3),
        "total_latency_s": round(total_latency, 2),
        "tokens_generated": len(generated_tokens),
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
        description="Run Gemma 4 E2B text inference on CPU via raw ONNX Runtime"
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
