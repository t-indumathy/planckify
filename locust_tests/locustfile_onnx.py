"""Locust load test — Gemma 4 E2B raw ONNX Runtime CPU flavour.

ONNX sessions (embed_tokens_q4 + decoder_model_merged_q4) are loaded ONCE per
virtual user in on_start() and reused across all iterations.

Stopping: the test stops automatically after TARGET_REQUESTS (default 100)
requests via an @events.request hook that calls runner.quit().  Pass the env
var LOCUST_TARGET_REQUESTS=N to override.

Run (100 sequential prompts, 1 virtual user — via run_locust_tests.py):
    python run_locust_tests.py --flavour onnx --iterations 100

Or directly:
    wsl bash -c "
      source .wsl-venv/bin/activate
      cd locust_tests
      locust -f locustfile_onnx.py --headless -u 1 -r 1 -t 8h 
             --csv results/onnx --loglevel WARNING
    "

Outputs under results/:
    onnx_stats.csv          - Locust aggregate stats
    onnx_stats_history.csv  - time-series throughput and latency
    onnx_failures.csv       - any failed requests
    onnx_detailed.csv       - per-request: gen_time_s, tokens_generated, tok/s
"""

import csv
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np

# ── resolve experiment directory ─────────────────────────────────────────────
_EXP_DIR = Path(__file__).resolve().parent.parent / "experiments" / "gemma4_e2b_onnx"
sys.path.insert(0, str(_EXP_DIR))

from run_inference_cpu import (  # noqa: E402
    _init_kv_cache,
    _num_logits_to_keep,
    build_session,
)
from transformers import AutoTokenizer  # noqa: E402

# Use an absolute path so the model is found regardless of CWD when locust runs
_MODEL_DIR = _EXP_DIR / "models" / "gemma-4-E2B-it-ONNX"

from locust import User, between, events, task  # noqa: E402

# ── auto-stop after N requests ────────────────────────────────────────────────
import os as _os  # noqa: E402
_TARGET_REQUESTS = int(_os.environ.get("LOCUST_TARGET_REQUESTS", "100"))
_req_count = 0
_req_count_lock = threading.Lock()


@events.init.add_listener
def _setup_stop_after_n(environment, **_kwargs) -> None:
    """Stop the runner automatically after _TARGET_REQUESTS completed requests."""
    @events.request.add_listener
    def _on_request(**kw) -> None:  # noqa: ANN001
        global _req_count
        with _req_count_lock:
            _req_count += 1
            if _req_count >= _TARGET_REQUESTS:
                environment.runner.quit()

# ── import prompt pool ────────────────────────────────────────────────────────
_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR))
from prompts import PROMPTS  # noqa: E402

# ── detailed per-request CSV writer ──────────────────────────────────────────
RESULTS_DIR = _TESTS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

_DETAILED_CSV = RESULTS_DIR / "onnx_detailed.csv"
_FIELDNAMES = [
    "seq",
    "prompt_snippet",
    "gen_time_s",
    "tokens_generated",
    "decode_speed_tok_s",
]

_csv_lock = threading.Lock()
_seq_counter = 0
_csv_fh = None
_csv_writer_obj = None


def _write_row(row: dict) -> None:
    global _seq_counter, _csv_fh, _csv_writer_obj
    with _csv_lock:
        if _csv_writer_obj is None:
            _csv_fh = open(_DETAILED_CSV, "w", newline="", encoding="utf-8")
            _csv_writer_obj = csv.DictWriter(_csv_fh, fieldnames=_FIELDNAMES)
            _csv_writer_obj.writeheader()
        _seq_counter += 1
        row["seq"] = _seq_counter
        _csv_writer_obj.writerow(row)
        _csv_fh.flush()


# ── Locust User ───────────────────────────────────────────────────────────────
class GemmaOnnxUser(User):
    """Single-user sequential load test — raw ONNX Runtime CPU backend."""

    host = ""          # not HTTP
    wait_time = between(0, 0)

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def on_start(self) -> None:
        """Load tokenizer and ONNX sessions once; reuse across all tasks."""
        model_dir = _MODEL_DIR
        if not model_dir.exists():
            raise FileNotFoundError(
                f"[onnx] Model directory not found: {model_dir}\n"
                "Run experiments/gemma4_e2b_onnx/download_model.py first."
            )

        embed_path = model_dir / "onnx" / "embed_tokens_q4.onnx"
        decoder_path = model_dir / "onnx" / "decoder_model_merged_q4.onnx"
        if not embed_path.exists() or not decoder_path.exists():
            raise FileNotFoundError(
                f"[onnx] ONNX files missing under {model_dir}/onnx/.\n"
                "Expected: embed_tokens_q4.onnx, decoder_model_merged_q4.onnx"
            )

        print(f"\n[onnx] Loading tokenizer from {model_dir} …")
        t0 = time.perf_counter()
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self._embed_sess = build_session(embed_path)
        self._decoder_sess = build_session(decoder_path)

        # Pre-compute session metadata (constant across requests)
        self._embed_out_names = [o.name for o in self._embed_sess.get_outputs()]
        self._dec_in_names: set[str] = {
            inp.name for inp in self._decoder_sess.get_inputs()
        }
        self._dec_out_names = [o.name for o in self._decoder_sess.get_outputs()]
        self._nlk = _num_logits_to_keep(self._decoder_sess)
        print(f"[onnx] Sessions ready — load time: {time.perf_counter() - t0:.2f}s\n")

    def on_stop(self) -> None:
        # ONNX sessions are closed by the GC; nothing to do explicitly
        pass

    # ── task ──────────────────────────────────────────────────────────────────
    @task
    def infer(self) -> None:
        """Pick a random prompt, run autoregressive decode, record metrics."""
        prompt = random.choice(PROMPTS)
        tokenizer = self._tokenizer
        embed_sess = self._embed_sess
        decoder_sess = self._decoder_sess
        embed_out_names = self._embed_out_names
        dec_in_names = self._dec_in_names
        dec_out_names = self._dec_out_names
        nlk = self._nlk

        gen_start = time.perf_counter()
        try:
            # ── tokenise ────────────────────────────────────────────────────
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="np")
            input_ids = inputs["input_ids"].astype(np.int64)
            seq_len = input_ids.shape[1]

            # ── embed ────────────────────────────────────────────────────────
            embed_outs = embed_sess.run(None, {"input_ids": input_ids})
            embed_map = dict(zip(embed_out_names, embed_outs))
            inputs_embeds = embed_map["inputs_embeds"]
            per_layer = {k: v for k, v in embed_map.items() if k != "inputs_embeds"}

            # ── prefill ──────────────────────────────────────────────────────
            past_kv = _init_kv_cache(decoder_sess)
            prefill_feed: dict = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": np.ones((1, seq_len), dtype=np.int64),
                "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, -1),
                "use_cache_branch": np.array([False], dtype=bool),
                "num_logits_to_keep": nlk,
            }
            prefill_feed.update(past_kv)
            prefill_feed.update(per_layer)
            prefill_feed = {k: v for k, v in prefill_feed.items() if k in dec_in_names}

            prefill_outs = decoder_sess.run(None, prefill_feed)
            prefill_map = dict(zip(dec_out_names, prefill_outs))
            for key, val in prefill_map.items():
                if key.startswith("present."):
                    pkey = key.replace("present.", "past_key_values.")
                    if pkey in dec_in_names:
                        past_kv[pkey] = val

            next_token = int(np.argmax(prefill_map["logits"][0, -1, :]))

            # ── autoregressive decode ────────────────────────────────────────
            decode_start = time.perf_counter()
            generated_ids = [next_token]
            total_seq = seq_len + 1
            max_new_tokens = 64  # keep tests fast; matches ONNX benchmark default

            for _ in range(max_new_tokens - 1):
                if next_token == tokenizer.eos_token_id:
                    break

                tok_ids = np.array([[next_token]], dtype=np.int64)
                tok_embed_outs = embed_sess.run(None, {"input_ids": tok_ids})
                tok_embed_map = dict(zip(embed_out_names, tok_embed_outs))
                step_embeds = tok_embed_map["inputs_embeds"]
                step_per_layer = {
                    k: v for k, v in tok_embed_map.items() if k != "inputs_embeds"
                }

                decode_feed: dict = {
                    "inputs_embeds": step_embeds,
                    "attention_mask": np.ones((1, total_seq), dtype=np.int64),
                    "position_ids": np.array([[total_seq - 1]], dtype=np.int64),
                    "use_cache_branch": np.array([True], dtype=bool),
                    "num_logits_to_keep": nlk,
                }
                decode_feed.update(past_kv)
                decode_feed.update(step_per_layer)
                decode_feed = {
                    k: v for k, v in decode_feed.items() if k in dec_in_names
                }

                decode_outs = decoder_sess.run(None, decode_feed)
                decode_map = dict(zip(dec_out_names, decode_outs))
                for key, val in decode_map.items():
                    if key.startswith("present."):
                        pkey = key.replace("present.", "past_key_values.")
                        if pkey in dec_in_names:
                            past_kv[pkey] = val

                next_token = int(np.argmax(decode_map["logits"][0, -1, :]))
                generated_ids.append(next_token)
                total_seq += 1

            gen_time_s = time.perf_counter() - decode_start
            total_ms = (time.perf_counter() - gen_start) * 1000
            tokens_generated = len(generated_ids)
            tps = tokens_generated / gen_time_s if gen_time_s > 0 else 0.0

            # Report to Locust (response_time covers full prefill + decode)
            self.environment.events.request.fire(
                request_type="INFERENCE",
                name="onnx_runtime",
                response_time=total_ms,
                response_length=tokens_generated,
                exception=None,
            )

            # Append row to detailed CSV
            _write_row(
                {
                    "prompt_snippet": prompt[:80],
                    "gen_time_s": round(gen_time_s, 4),
                    "tokens_generated": tokens_generated,
                    "decode_speed_tok_s": round(tps, 2),
                }
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - gen_start) * 1000
            self.environment.events.request.fire(
                request_type="INFERENCE",
                name="onnx_runtime",
                response_time=elapsed_ms,
                response_length=0,
                exception=exc,
            )
