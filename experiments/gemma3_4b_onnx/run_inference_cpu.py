"""Single-turn text inference with Gemma 3 4B IT via raw ONNX Runtime on CPU.

Uses AutoTokenizer (text-only) and the int8-quantized ONNX sessions:
    1. embed_tokens_quantized.onnx        - input_ids -> inputs_embeds
    2. decoder_model_merged_quantized.onnx - autoregressive decode with KV cache

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

DEFAULT_MODEL_DIR = Path("./models/gemma-3-4b-it-ONNX")
DEFAULT_MAX_TOKENS = 64
SYSTEM_PROMPT = "You are a helpful AI assistant."


def build_session(model_path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 4
    opts.intra_op_num_threads = 4
    return ort.InferenceSession(
        str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
    )


def _init_kv_cache(decoder_sess: ort.InferenceSession) -> dict:
    """Build zero-initialised past_key_values from the decoder session's input schema."""
    past_kv: dict = {}
    for inp in decoder_sess.get_inputs():
        if not inp.name.startswith("past_key_values."):
            continue
        shape = []
        for idx, d in enumerate(inp.shape):
            if isinstance(d, int) and d > 0:
                shape.append(d)
            elif idx == 0:  # batch
                shape.append(1)
            elif idx == 2:  # sequence length -> empty
                shape.append(0)
            else:
                shape.append(1)
        past_kv[inp.name] = np.zeros(shape, dtype=np.float32)
    return past_kv


def _num_logits_to_keep(decoder_sess: ort.InferenceSession) -> np.ndarray:
    """Return num_logits_to_keep in the shape the model expects."""
    for inp in decoder_sess.get_inputs():
        if inp.name == "num_logits_to_keep":
            declared = inp.shape
            if declared:
                return np.array([1], dtype=np.int64)
            else:
                return np.array(1, dtype=np.int64)
    return np.array([1], dtype=np.int64)


def run_inference(
    model_dir: Path,
    prompt: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """Run text inference using embed_tokens + decoder ONNX sessions with KV cache."""
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model not found at: {model_dir}\nRun download_model.py first."
        )

    print(f"Loading model : {model_dir}")
    print("Backend       : CPU (raw ONNX Runtime, int8/quantized, text-only)")
    print("-" * 60)

    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    embed_path = model_dir / "onnx" / "embed_tokens_quantized.onnx"
    decoder_path = model_dir / "onnx" / "decoder_model_merged_quantized.onnx"
    if not embed_path.exists() or not decoder_path.exists():
        raise FileNotFoundError(
            f"ONNX files not found under {model_dir}/onnx/.\n"
            "Expected: embed_tokens_quantized.onnx, decoder_model_merged_quantized.onnx"
        )

    embed_sess = build_session(embed_path)
    decoder_sess = build_session(decoder_path)
    load_time = time.perf_counter() - load_start

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="np")
    input_ids = inputs["input_ids"].astype(np.int64)

    embed_outputs = embed_sess.run(None, {"input_ids": input_ids})
    embed_names = [o.name for o in embed_sess.get_outputs()]
    embed_map = dict(zip(embed_names, embed_outputs))
    inputs_embeds = embed_map["inputs_embeds"]
    per_layer_inputs = {k: v for k, v in embed_map.items() if k != "inputs_embeds"}

    past_kv = _init_kv_cache(decoder_sess)
    decoder_input_names: set = {inp.name for inp in decoder_sess.get_inputs()}
    decoder_output_names = [out.name for out in decoder_sess.get_outputs()]

    seq_len = input_ids.shape[1]
    nlk = _num_logits_to_keep(decoder_sess)

    prefill_feed: dict = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": np.ones((1, seq_len), dtype=np.int64),
        "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, -1),
        "use_cache_branch": np.array([False], dtype=bool),
        "num_logits_to_keep": nlk,
    }
    prefill_feed.update(past_kv)
    prefill_feed.update(per_layer_inputs)
    prefill_feed = {k: v for k, v in prefill_feed.items() if k in decoder_input_names}

    first_token_start = time.perf_counter()
    prefill_outs = decoder_sess.run(None, prefill_feed)
    first_token_time = time.perf_counter() - first_token_start
    prefill_map = dict(zip(decoder_output_names, prefill_outs))

    for key, val in prefill_map.items():
        if key.startswith("present."):
            pkey = key.replace("present.", "past_key_values.")
            if pkey in decoder_input_names:
                past_kv[pkey] = val

    next_token = int(np.argmax(prefill_map["logits"][0, -1, :]))
    gen_start = time.perf_counter()
    generated_ids = [next_token]
    total_seq = seq_len + 1

    for _ in range(max_tokens - 1):
        if next_token == tokenizer.eos_token_id:
            break

        tok_ids = np.array([[next_token]], dtype=np.int64)
        tok_embed_outs = embed_sess.run(None, {"input_ids": tok_ids})
        tok_embed_map = dict(zip(embed_names, tok_embed_outs))
        step_embeds = tok_embed_map["inputs_embeds"]
        step_per_layer = {k: v for k, v in tok_embed_map.items() if k != "inputs_embeds"}

        decode_feed: dict = {
            "inputs_embeds": step_embeds,
            "attention_mask": np.ones((1, total_seq), dtype=np.int64),
            "position_ids": np.array([[total_seq - 1]], dtype=np.int64),
            "use_cache_branch": np.array([True], dtype=bool),
            "num_logits_to_keep": nlk,
        }
        decode_feed.update(past_kv)
        decode_feed.update(step_per_layer)
        decode_feed = {k: v for k, v in decode_feed.items() if k in decoder_input_names}

        decode_outs = decoder_sess.run(None, decode_feed)
        decode_map = dict(zip(decoder_output_names, decode_outs))

        for key, val in decode_map.items():
            if key.startswith("present."):
                pkey = key.replace("present.", "past_key_values.")
                if pkey in decoder_input_names:
                    past_kv[pkey] = val

        next_token = int(np.argmax(decode_map["logits"][0, -1, :]))
        generated_ids.append(next_token)
        total_seq += 1

    gen_time = time.perf_counter() - gen_start
    total_time = first_token_time + gen_time
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    tokens_generated = len(generated_ids)
    decode_speed = tokens_generated / gen_time if gen_time > 0 else 0.0

    print(f"Prompt        : {prompt}")
    print(f"Response      : {response}")
    print("-" * 60)
    print(f"Load time     : {load_time:.2f}s")
    print(f"TTFT          : {first_token_time:.3f}s")
    print(f"Prefill seq   : {seq_len} tokens")
    print(f"Generated     : {tokens_generated} tokens in {gen_time:.2f}s")
    print(f"Decode speed  : {decode_speed:.1f} tok/s")

    return {
        "prompt": prompt,
        "response": response,
        "load_time_s": round(load_time, 3),
        "ttft_s": round(first_token_time, 3),
        "gen_time_s": round(gen_time, 3),
        "tokens_generated": tokens_generated,
        "decode_speed_tok_s": round(decode_speed, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma 3 4B IT int8 ONNX CPU inference")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prompt", type=str, default="What is the Planck constant?")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()
    result = run_inference(args.model_dir, args.prompt, args.max_tokens)
    print(f"\nResult JSON: {result}")


if __name__ == "__main__":
    main()
