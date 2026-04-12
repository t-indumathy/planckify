"""Locust load test — Gemma 4 E2B LiteRT-LM CPU flavour.

The model is loaded ONCE per virtual user in on_start() and reused across
all task iterations, matching real-world serving behaviour.

Stopping: the test stops automatically after TARGET_REQUESTS (default 100)
requests via an @events.request hook that calls runner.quit().  Pass the env
var LOCUST_TARGET_REQUESTS=N to override.

Run (100 sequential prompts, 1 virtual user — via run_locust_tests.py):
    python run_locust_tests.py --flavour cpu --iterations 100

Or directly:
    wsl bash -c "
      source .wsl-venv/bin/activate
      cd locust_tests
      locust -f locustfile_cpu.py --headless -u 1 -r 1 -t 8h
             --csv results/litertlm --loglevel WARNING
    "

Outputs under results/:
    litertlm_stats.csv          - Locust aggregate stats (avg, p50, p95, p99, RPS)
    litertlm_stats_history.csv  - time-series throughput and latency
    litertlm_failures.csv       - any failed requests
    litertlm_detailed.csv       - per-request: ttft_s, gen_time_s, tokens, tok/s
"""

import csv
import random
import sys
import threading
import time
from pathlib import Path

# ── resolve experiment directory ─────────────────────────────────────────────
_EXP_DIR = Path(__file__).resolve().parent.parent / "experiments" / "gemma4_e2b_litertlm"
sys.path.insert(0, str(_EXP_DIR))

import litert_lm  # noqa: E402
from run_inference_cpu import SYSTEM_PROMPT  # noqa: E402

# Use an absolute path so the model is found regardless of CWD when locust runs
_MODEL_PATH = _EXP_DIR / "models" / "gemma-4-E2B-it.litertlm"

from locust import User, between, events, task  # noqa: E402

# Match the ONNX flavour's max token cap for a fair apples-to-apples comparison
MAX_NEW_TOKENS = 64

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

_DETAILED_CSV = RESULTS_DIR / "litertlm_detailed.csv"
_FIELDNAMES = [
    "seq",
    "prompt_snippet",
    "ttft_s",
    "gen_time_s",
    "approx_tokens",
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
class GemmaCpuUser(User):
    """Single-user sequential load test — LiteRT-LM CPU backend."""

    host = ""          # not HTTP; value satisfies Locust's host check
    wait_time = between(0, 0)   # fire requests back-to-back

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def on_start(self) -> None:
        """Load the model once; keep the engine alive for all iterations."""
        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
        model_path = _MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(
                f"[cpu] Model not found: {model_path}\n"
                "Run experiments/gemma4_e2b_litertlm/download_model.py first."
            )
        print(f"\n[cpu] Loading model from {model_path} …")
        t0 = time.perf_counter()
        self._engine_ctx = litert_lm.Engine(
            str(model_path),
            backend=litert_lm.Backend.CPU,
            cache_dir="/tmp/planckify-litert-cache",
        )
        self._engine = self._engine_ctx.__enter__()
        print(f"[cpu] Model ready — load time: {time.perf_counter() - t0:.2f}s\n")

    def on_stop(self) -> None:
        try:
            self._engine_ctx.__exit__(None, None, None)
        except Exception:
            pass

    # ── task ──────────────────────────────────────────────────────────────────
    @task
    def infer(self) -> None:
        """Pick a random prompt, stream a response, record Locust + CSV metrics."""
        prompt = random.choice(PROMPTS)
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            }
        ]

        gen_start = time.perf_counter()
        chunks: list[str] = []
        first_chunk_time: float | None = None
        approx_tokens_so_far = 0

        try:
            with self._engine.create_conversation(messages=messages) as conv:
                for chunk in conv.send_message_async(prompt):
                    for item in chunk.get("content", []):
                        if item.get("type") == "text":
                            chunks.append(item["text"])
                            if first_chunk_time is None:
                                first_chunk_time = time.perf_counter()
                            approx_tokens_so_far += len(item["text"].split())
                    if approx_tokens_so_far >= MAX_NEW_TOKENS:
                        conv.cancel_process()  # signal LiteRT-LM to stop
                        break

            gen_end = time.perf_counter()
            gen_time_s = gen_end - gen_start
            total_ms = gen_time_s * 1000
            ttft_s = (
                (first_chunk_time - gen_start)
                if first_chunk_time is not None
                else gen_time_s
            )
            response = "".join(chunks)
            approx_tokens = len(response.split())
            tps = approx_tokens / gen_time_s if gen_time_s > 0 else 0.0

            # Report to Locust (response_time = generation time in ms)
            self.environment.events.request.fire(
                request_type="INFERENCE",
                name="cpu_litert_lm",
                response_time=total_ms,
                response_length=approx_tokens,
                exception=None,
            )

            # Append row to detailed CSV
            _write_row(
                {
                    "prompt_snippet": prompt[:80],
                    "ttft_s": round(ttft_s, 4),
                    "gen_time_s": round(gen_time_s, 4),
                    "approx_tokens": approx_tokens,
                    "decode_speed_tok_s": round(tps, 2),
                }
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - gen_start) * 1000
            self.environment.events.request.fire(
                request_type="INFERENCE",
                name="cpu_litert_lm",
                response_time=elapsed_ms,
                response_length=0,
                exception=exc,
            )
