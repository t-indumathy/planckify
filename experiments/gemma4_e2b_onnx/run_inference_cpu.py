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


def _build_kv_cache(decoder_sess: ort.InferenceSession, num_layers: int) -> dict:
    """Initialise empty KV cache based on decoder session's past_key_values inputs."""
    past_kv: dict = {}
    for inp in decoder_sess.get_inputs():
        if inp.name.startswith("past_key_values."):
            # Shape is dynamic but we initialise to (1, num_heads, 0, head_dim)
            # ORT will accept this for merged model prefill pass
            past_kv[inp.name] = np.zeros(
                [1 if (d is None or isinstance(d, str)) else d for d in inp.shape],
                dtype=np.float32,
            )
    return past_kv


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
    embed_outputs = embed_sess.run(None, {"input_ids": input_ids})
    embed_names = [o.name for o in embed_sess.get_outputs()]
    embed_map = dict(zip(embed_names, embed_outputs))
    inputs_embeds = embed_map["inputs_embeds"]  # (1, seq_len, hidden)
    per_layer_inputs = {k: v for k, v in embed_map.items() if k != "inputs_embeds"}

    # Build KV cache from decoder session's input schema
    past_kv = _build_kv_cache(decoder_sess, num_layers=15)
    decoder_input_names = {inp.name for inp in decoder_sess.get_inputs()}
    decoder_output_names = [out.name for out in decoder_sess.get_outputs()]

    # --- Prefill pass ---
    seq_len = input_ids.shape[1]
    attention_mask = np.ones((1, seq_len), dtype=np.int64)
    position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
    use_cache_branch = np.array([False], dtype=bool)
    num_logits_to_keep = np.array([1], dtype=np.int64)

    prefill_inputs: dict = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "use_cache_branch": use_cache_branch,
        "num_logits_to_keep": num_logits_to_keep,
    }
    prefill_inputs.update(past_kv)
    prefill_inputs.update(per_layer_inputs)

    # Only feed inputs the decoder actually expects
    prefill_inputs = {k: v for k, v in prefill_inputs.items() if k in decoder_input_names}

    prefill_outputs = decoder_sess.run(None, prefill_inputs)
    prefill_map = dict(zip(decoder_output_names, prefill_outputs))

    # Update KV cache from prefill presents
    for key in prefill_map:
        if key.startswith("present."):
            past_key = key.replace("present.", "past_key_values.")
            if past_key in decoder_input_names:
                past_kv[past_key] = prefill_map[key]

    logits = prefill_map["logits"]  # (1, 1, vocab) with num_logits_to_keep=1
    next_token = int(np.argmax(logits[0, -1, :]))

    # --- Autoregressive decode ---
    gen_start = time.perf_counter()
    generated_ids = [next_token]
    total_seq = seq_len + 1

    for _ in range(max_tokens - 1):
        if next_token == tokenizer.eos_token_id:
            break

        tok_ids = np.array([[next_token]], dtype=np.int64)
        tok_embed_out = embed_sess.run(None, {"input_ids": tok_ids})
        tok_embed_map = dict(zip(embed_names, tok_embed_out))
        step_embeds = tok_embed_map["inputs_embeds"]
        step_per_layer = {k: v for k, v in tok_embed_map.items() if k != "inputs_embeds"}

        attn_mask = np.ones((1, total_seq), dtype=np.int64)
        pos_ids = np.array([[total_seq - 1]], dtype=np.int64)
        use_cache_branch_decode = np.array([True], dtype=bool)
        num_logits_to_keep_decode = np.array([1], dtype=np.int64)

        decode_inputs: dict = {
            "inputs_embeds": step_embeds,
            "attention_mask": attn_mask,
            "position_ids": pos_ids,
            "use_cache_branch": use_cache_branch_decode,
            "num_logits_to_keep": num_logits_to_keep_decode,
        }
        decode_inputs.update(past_kv)
        decode_inputs.update(step_per_layer)
        decode_inputs = {k: v for k, v in decode_inputs.items() if k in decoder_input_names}

        decode_outputs = decoder_sess.run(None, decode_inputs)
        decode_map = dict(zip(decoder_output_names, decode_outputs))

        for key in decode_map:
            if key.startswith("present."):
                past_key = key.replace("present.", "past_key_values.")
                if past_key in decoder_input_names:
                    past_kv[past_key] = decode_map[key]

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
