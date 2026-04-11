"""Single-turn text inference with Gemma 4 E2B via raw ONNX Runtime on CPU.

Uses AutoTokenizer (text-only, no vision/audio processor) to avoid
TorchVision/PyTorch dependency chain.

Two ONNX sessions:
  1. embed_tokens_q4.onnx        - input_ids -> inputs_embeds + per_layer_inputs
  2. decoder_model_merged_q4.onnx - autoregressive decode with KV cache

Usage:
    python run_inference_cpu.py
    python run_inference_cpu.py --prompt "Explain transformers" --max-tokens 64
"""
import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

DEFAULT_MODEL_DIR = Path("./models/gemma-4-E2B-it-ONNX")
DEFAULT_MAX_TOKENS = 64


def build_session(model_path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 4
    opts.intra_op_num_threads = 4
    return ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])


def run_inference(model_dir: Path, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    """Run text inference using embed_tokens + decoder ONNX sessions with KV cache."""
    if not model_dir.exists():
        raise FileNotFoundError(f"Model not found at: {model_dir}\nRun download_model.py first.")

    print(f"Loading model : {model_dir}")
    print(f"Backend       : CPU (raw ONNX Runtime, text-only)")
    print("-" * 60)

    load_start = time.perf_counter()

    # Load tokenizer only (no Processor -> no torchvision/torch dependency)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    embed_path = model_dir / "onnx" / "embed_tokens_q4.onnx"
    decoder_path = model_dir / "onnx" / "decoder_model_merged_q4.onnx"

    if not embed_path.exists() or not decoder_path.exists():
        raise FileNotFoundError(
            f"ONNX files not found under {model_dir}/onnx/.\n"
            "Expected: embed_tokens_q4.onnx, decoder_model_merged_q4.onnx"
        )

    embed_sess = build_session(embed_path)
    decoder_sess = build_session(decoder_path)
    load_time = time.perf_counter() - load_start

    # Tokenize
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="np")
    input_ids = inputs["input_ids"].astype(np.int64)  # shape: (1, seq_len)

    # --- Embedding pass ---
    embed_inputs = {"input_ids": input_ids}
    embed_outputs = embed_sess.run(None, embed_inputs)
    embed_names = [o.name for o in embed_sess.get_outputs()]
    embed_map = dict(zip(embed_names, embed_outputs))
    inputs_embeds = embed_map["inputs_embeds"]  # (1, seq_len, hidden)

    # Build per-layer KV cache inputs from embed session outputs
    per_layer_inputs = {k: v for k, v in embed_map.items() if k != "inputs_embeds"}

    # Inspect decoder to set up KV cache buffers
    decoder_input_names = [inp.name for inp in decoder_sess.get_inputs()]
    decoder_output_names = [out.name for out in decoder_sess.get_outputs()]

    # Determine num_layers from present_* outputs
    num_layers = sum(1 for n in decoder_output_names if n.startswith("present.") and n.endswith(".key"))
    if num_layers == 0:
        num_layers = 18  # Gemma 4 E2B default

    # Initialise empty KV cache
    batch, heads, kv_seq, head_dim = 1, 8, 0, 256
    past_kv = {}
    for layer in range(num_layers):
        past_kv[f"past_key_values.{layer}.key"] = np.zeros((batch, heads, kv_seq, head_dim), dtype=np.float32)
        past_kv[f"past_key_values.{layer}.value"] = np.zeros((batch, heads, kv_seq, head_dim), dtype=np.float32)

    # --- Prefill pass ---
    seq_len = input_ids.shape[1]
    attention_mask = np.ones((1, seq_len), dtype=np.int64)
    position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
    use_cache_branch = np.array([False], dtype=bool)

    prefill_inputs = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "use_cache_branch": use_cache_branch,
    }
    prefill_inputs.update(past_kv)
    prefill_inputs.update(per_layer_inputs)

    prefill_outputs = decoder_sess.run(None, prefill_inputs)
    prefill_map = dict(zip(decoder_output_names, prefill_outputs))

    # Update KV cache from prefill
    for layer in range(num_layers):
        past_kv[f"past_key_values.{layer}.key"] = prefill_map[f"present.{layer}.key"]
        past_kv[f"past_key_values.{layer}.value"] = prefill_map[f"present.{layer}.value"]

    logits = prefill_map["logits"]  # (1, seq_len, vocab)
    next_token = int(np.argmax(logits[0, -1, :]))

    # --- Autoregressive decode ---
    gen_start = time.perf_counter()
    generated_ids = [next_token]
    total_seq = seq_len + 1

    for _ in range(max_tokens - 1):
        if next_token == tokenizer.eos_token_id:
            break

        tok_ids = np.array([[next_token]], dtype=np.int64)
        # Embed single token
        tok_embed = embed_sess.run(None, {"input_ids": tok_ids})
        tok_embed_map = dict(zip(embed_names, tok_embed))
        step_embeds = tok_embed_map["inputs_embeds"]
        step_per_layer = {k: v for k, v in tok_embed_map.items() if k != "inputs_embeds"}

        attn_mask = np.ones((1, total_seq), dtype=np.int64)
        pos_ids = np.array([[total_seq - 1]], dtype=np.int64)
        use_cache_branch = np.array([True], dtype=bool)

        decode_inputs = {
            "inputs_embeds": step_embeds,
            "attention_mask": attn_mask,
            "position_ids": pos_ids,
            "use_cache_branch": use_cache_branch,
        }
        decode_inputs.update(past_kv)
        decode_inputs.update(step_per_layer)

        decode_outputs = decoder_sess.run(None, decode_inputs)
        decode_map = dict(zip(decoder_output_names, decode_outputs))

        # Update KV cache
        for layer in range(num_layers):
            past_kv[f"past_key_values.{layer}.key"] = decode_map[f"present.{layer}.key"]
            past_kv[f"past_key_values.{layer}.value"] = decode_map[f"present.{layer}.value"]

        logits = decode_map["logits"]
        next_token = int(np.argmax(logits[0, -1, :]))
        generated_ids.append(next_token)
        total_seq += 1

    gen_time = time.perf_counter() - gen_start
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    tokens_generated = len(generated_ids)
    decode_speed = tokens_generated / gen_time if gen_time > 0 else 0.0

    print(f"Prompt        : {prompt}")
    print(f"Response      : {response}")
    print("-" * 60)
    print(f"Load time     : {load_time:.2f}s")
    print(f"Prefill seq   : {seq_len} tokens")
    print(f"Generated     : {tokens_generated} tokens in {gen_time:.2f}s")
    print(f"Decode speed  : {decode_speed:.1f} tok/s")

    return {
        "prompt": prompt,
        "response": response,
        "load_time_s": round(load_time, 3),
        "gen_time_s": round(gen_time, 3),
        "tokens_generated": tokens_generated,
        "decode_speed_tok_s": round(decode_speed, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Gemma 4 E2B ONNX CPU inference")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prompt", type=str, default="What is the Planck constant?")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()

    result = run_inference(args.model_dir, args.prompt, args.max_tokens)
    print(f"\nResult JSON: {result}")


if __name__ == "__main__":
    main()
