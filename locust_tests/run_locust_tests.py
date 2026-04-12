r"""
Orchestrator: run Locust load tests for both inference flavours and print a
side-by-side comparison of all captured metrics.

Runtime routing
───────────────
  BOTH flavours run via WSL using .wsl-venv (Python 3.12 Linux).

  Why WSL for the ONNX flavour too?
  ----------------------------------
  onnxruntime itself has native Windows wheels and works on Python 3.13
  Windows without WSL (the ONNX inference script runs fine natively).
  However, Locust depends on gevent → greenlet whose C extension
  (_greenlet.cp313-win_amd64.pyd) requires msvcp140.dll from the
  Microsoft Visual C++ 2015-2022 Redistributable. If that runtime is not
  installed system-wide, greenlet fails with:
      ImportError: DLL load failed while importing _greenlet
  rather than a clean missing-package error — and there is no pure-Python
  fallback.

  .wsl-venv (Python 3.12, Linux) has gevent/greenlet working out of the
  box and already contains litert-lm, onnxruntime, transformers, and
  locust — making it the single execution environment for all load tests.

  To run ONNX inference natively on Windows (without Locust):
      .venv\Scripts\python experiments\gemma4_e2b_onnx\run_inference_cpu.py

Usage (from the repo root):
    # Run both flavours (100 iterations each — uses WSL for both)
    python locust_tests\run_locust_tests.py

    # Quick smoke test
    python locust_tests\run_locust_tests.py --iterations 10

    # Single flavour
    python locust_tests\run_locust_tests.py --flavour onnx
    python locust_tests\run_locust_tests.py --flavour cpu

    # Re-print comparison from existing CSVs (no re-run)
    python locust_tests\run_locust_tests.py --skip-run

Results are written to locust_tests/results/.
"""

import argparse
import csv
import json
import platform
import shutil
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ── path constants ────────────────────────────────────────────────────────────
TESTS_DIR  = Path(__file__).resolve().parent
REPO_ROOT  = TESTS_DIR.parent
RESULTS_DIR = TESTS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Windows .venv — used for ONNX (native)
_WIN_VENV_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

# WSL .wsl-venv — used for CPU/LiteRT-LM (Linux-only package)
# On Windows the path inside WSL is /mnt/<drive-letter>/...
def _win_to_wsl(p: Path) -> str:
    """Convert a Windows absolute path to its WSL /mnt/… equivalent."""
    drive, rest = p.drive.lower().rstrip(":"), p.as_posix()[2:]  # strip 'D:/'
    return f"/mnt/{drive}{rest}"

_WSL_VENV_PYTHON = _win_to_wsl(REPO_ROOT / ".wsl-venv" / "bin" / "python")
_WSL_TESTS_DIR   = _win_to_wsl(TESTS_DIR)
_WSL_RESULTS_DIR = _win_to_wsl(RESULTS_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Runtime detection
# ─────────────────────────────────────────────────────────────────────────────

def _is_windows() -> bool:
    return platform.system() == "Windows"


def _wsl_available() -> bool:
    return _is_windows() and shutil.which("wsl") is not None


# ─────────────────────────────────────────────────────────────────────────────
# Locust runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_locust_onnx(locustfile: str, csv_prefix: str, iterations: int) -> int:
    """Run ONNX locust test via WSL using .wsl-venv.

    Although onnxruntime has native Windows wheels and the inference script
    works on Windows with .venv, Locust's gevent backend requires msvcp140.dll
    (VC++ Redistributable) for its greenlet C extension.  If that DLL is
    missing (common on fresh Windows installs), greenlet raises an ImportError.
    Running via WSL+.wsl-venv uses Python 3.12 on Linux where gevent works
    without any native DLL dependencies.
    """
    if not _wsl_available():
        print(
            "[ERROR] WSL is not available.\n"
            "        The Locust load tests require WSL because Locust's gevent\n"
            "        backend needs msvcp140.dll (VC++ Redistributable) on\n"
            "        Windows Python 3.13, which is not installed here.\n"
            "        Install WSL + .wsl-venv deps and re-run.\n"
            "        Alternatively, install the VC++ 2015-2022 Redistributable\n"
            "        from https://aka.ms/vs/17/release/vc_redist.x64.exe and\n"
            "        use .venv directly."
        )
        return 1

    wsl_locustfile = f"{_WSL_TESTS_DIR}/{locustfile}"
    wsl_csv = f"{_WSL_RESULTS_DIR}/{csv_prefix}"
    inner_cmd = (
        f"LOCUST_TARGET_REQUESTS={iterations}"
        f" {_WSL_VENV_PYTHON} -m locust"
        f" -f {wsl_locustfile}"
        f" --headless --users 1 --spawn-rate 1"
        f" --run-time 8h"
        f" --csv {wsl_csv}"
        f" --loglevel WARNING"
        f" --host ''"
    )
    cmd = ["wsl", "bash", "-c", inner_cmd]
    _print_run_header("ONNX", locustfile, csv_prefix, iterations, _WSL_VENV_PYTHON, via_wsl=True)
    result = subprocess.run(cmd)
    return result.returncode


def _run_locust_cpu(locustfile: str, csv_prefix: str, iterations: int) -> int:
    """Run CPU/LiteRT-LM locust test via WSL using .wsl-venv."""
    if not _wsl_available():
        print(
            "[ERROR] WSL is not available on this machine.\n"
            "        The LiteRT-LM flavour requires Linux (litert-lm-api-nightly\n"
            "        has no Windows wheel). Install WSL and re-run, or use --flavour onnx."
        )
        return 1

    wsl_locustfile = f"{_WSL_TESTS_DIR}/{locustfile}"
    wsl_csv = f"{_WSL_RESULTS_DIR}/{csv_prefix}"
    # Build the python -m locust command that WSL will execute
    inner_cmd = (
        f"LOCUST_TARGET_REQUESTS={iterations}"
        f" {_WSL_VENV_PYTHON} -m locust"
        f" -f {wsl_locustfile}"
        f" --headless --users 1 --spawn-rate 1"
        f" --run-time 8h"
        f" --csv {wsl_csv}"
        f" --loglevel WARNING"
        f" --host ''"
    )
    cmd = ["wsl", "bash", "-c", inner_cmd]
    _print_run_header("CPU", locustfile, csv_prefix, iterations, _WSL_VENV_PYTHON, via_wsl=True)
    result = subprocess.run(cmd)
    return result.returncode


def _print_run_header(
    flavour: str, locustfile: str, csv_prefix: str,
    iterations: int, python: str, via_wsl: bool
) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Flavour   : {flavour}")
    print(f"  File      : {locustfile}")
    print(f"  Iterations: {iterations}")
    print(f"  CSV prefix: results/{csv_prefix}")
    print(f"  Python    : {python}")
    print(f"  Via WSL   : {'yes' if via_wsl else 'no (Windows native)'}")
    print(f"{sep}\n")


def _read_locust_stats(csv_prefix: str) -> dict:
    """Return the 'Aggregated' row from Locust's *_stats.csv file."""
    stats_file = RESULTS_DIR / f"{csv_prefix}_stats.csv"
    if not stats_file.exists():
        return {}
    with open(stats_file, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    # Locust appends an "Aggregated" row at the bottom
    for row in reversed(rows):
        if row.get("Name") == "Aggregated":
            return row
    return rows[-1] if rows else {}


def _read_detailed(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _floats(rows: list[dict], key: str) -> list[float]:
    vals = []
    for r in rows:
        try:
            vals.append(float(r[key]))
        except (KeyError, ValueError, TypeError):
            pass
    return vals


def _stat_block(values: list[float], unit: str = "") -> str:
    if not values:
        return "  n/a"
    avg = statistics.mean(values)
    p50 = statistics.median(values)
    p95 = sorted(values)[int(len(values) * 0.95)]
    mn, mx = min(values), max(values)
    u = f" {unit}" if unit else ""
    return (
        f"  avg={avg:.3f}{u}  p50={p50:.3f}{u}"
        f"  p95={p95:.3f}{u}  min={mn:.3f}{u}  max={mx:.3f}{u}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_locust_stats(label: str, stats: dict) -> None:
    print(f"\n  {'─'*60}")
    print(f"  {label} — Locust built-in stats")
    print(f"  {'─'*60}")
    if not stats:
        print("  (stats CSV not found — did the test run successfully?)")
        return
    req   = stats.get("Request Count", "?")
    fail  = stats.get("Failure Count", "?")
    avg   = stats.get("Average Response Time", "?")
    p50   = stats.get("50%", "?")
    p95   = stats.get("95%", "?")
    p99   = stats.get("99%", "?")
    p100  = stats.get("100%", "?")
    rps   = stats.get("Requests/s", "?")
    print(f"  Requests        : {req}")
    print(f"  Failures        : {fail}")
    print(f"  Avg latency (ms): {avg}")
    print(f"  p50  (ms)       : {p50}")
    print(f"  p95  (ms)       : {p95}")
    print(f"  p99  (ms)       : {p99}")
    print(f"  p100 (ms)       : {p100}")
    print(f"  Throughput (r/s): {rps}")


def _print_detailed_stats(label: str, rows: list[dict], has_ttft: bool) -> None:
    print(f"\n  {'─'*60}")
    print(f"  {label} — per-request detailed metrics ({len(rows)} rows)")
    print(f"  {'─'*60}")
    if not rows:
        print("  (detailed CSV not found)")
        return

    gen_times = _floats(rows, "gen_time_s")
    tps_vals  = _floats(rows, "decode_speed_tok_s")
    tokens    = _floats(rows, "approx_tokens") or _floats(rows, "tokens_generated")

    if gen_times:
        print(f"  Gen time (s)    :{_stat_block(gen_times, 's')}")
    if tps_vals:
        print(f"  Decode speed    :{_stat_block(tps_vals, 'tok/s')}")
    if tokens:
        print(f"  Tokens/response :{_stat_block(tokens, 'tok')}")

    if has_ttft:
        ttft_vals = _floats(rows, "ttft_s")
        if ttft_vals:
            print(f"  TTFT (s)        :{_stat_block(ttft_vals, 's')}")


# ─────────────────────────────────────────────────────────────────────────────
# Comparison JSON export
# ─────────────────────────────────────────────────────────────────────────────

def _export_comparison(cpu_stats: dict, onnx_stats: dict,
                        cpu_rows: list[dict], onnx_rows: list[dict]) -> Path:
    def _safe_float(d: dict, key: str) -> float | None:
        try:
            return float(d[key])
        except (KeyError, ValueError, TypeError):
            return None

    def _summary(stats: dict, rows: list[dict], ttft: bool) -> dict:
        gen_times = _floats(rows, "gen_time_s")
        tps_vals  = _floats(rows, "decode_speed_tok_s")
        tokens    = _floats(rows, "approx_tokens") or _floats(rows, "tokens_generated")
        out = {
            "locust_requests":   _safe_float(stats, "Request Count"),
            "locust_failures":   _safe_float(stats, "Failure Count"),
            "locust_avg_ms":     _safe_float(stats, "Average Response Time"),
            "locust_p50_ms":     _safe_float(stats, "50%"),
            "locust_p95_ms":     _safe_float(stats, "95%"),
            "locust_p99_ms":     _safe_float(stats, "99%"),
            "locust_rps":        _safe_float(stats, "Requests/s"),
            "gen_time_avg_s":    statistics.mean(gen_times) if gen_times else None,
            "gen_time_p50_s":    statistics.median(gen_times) if gen_times else None,
            "gen_time_p95_s":    sorted(gen_times)[int(len(gen_times)*0.95)] if gen_times else None,
            "decode_speed_avg":  statistics.mean(tps_vals) if tps_vals else None,
            "decode_speed_p50":  statistics.median(tps_vals) if tps_vals else None,
            "tokens_avg":        statistics.mean(tokens) if tokens else None,
        }
        if ttft:
            ttft_vals = _floats(rows, "ttft_s")
            out["ttft_avg_s"]  = statistics.mean(ttft_vals) if ttft_vals else None
            out["ttft_p50_s"]  = statistics.median(ttft_vals) if ttft_vals else None
            out["ttft_p95_s"]  = sorted(ttft_vals)[int(len(ttft_vals)*0.95)] if ttft_vals else None
        return out

    comparison = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "litertlm": _summary(cpu_stats, cpu_rows, ttft=True),
        "onnx_runtime":  _summary(onnx_stats, onnx_rows, ttft=False),
    }
    out_path = RESULTS_DIR / "comparison.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Locust load tests for both Gemma 4 E2B inference flavours."
    )
    parser.add_argument(
        "--iterations", "-i", type=int, default=100,
        help="Number of inference requests per flavour (default: 100)"
    )
    parser.add_argument(
        "--flavour", choices=["cpu", "onnx", "both"], default="both",
        help="Which flavour(s) to test (default: both)"
    )
    parser.add_argument(
        "--skip-run", action="store_true",
        help="Skip running Locust; just re-read existing CSVs and print comparison"
    )
    args = parser.parse_args()

    # ── run tests ─────────────────────────────────────────────────────────────
    if not args.skip_run:
        # CPU/LiteRT-LM: Linux-only — must run via WSL using .wsl-venv
        if args.flavour in ("cpu", "both"):
            rc = _run_locust_cpu("locustfile_cpu.py", "litertlm", args.iterations)
            if rc != 0:
                print(f"[WARN] CPU Locust test exited with code {rc}")

        # ONNX: Windows-native — run directly with .venv
        if args.flavour in ("onnx", "both"):
            rc = _run_locust_onnx("locustfile_onnx.py", "onnx", args.iterations)
            if rc != 0:
                print(f"[WARN] ONNX Locust test exited with code {rc}")

    # ── read results ──────────────────────────────────────────────────────────
    cpu_stats  = _read_locust_stats("litertlm")
    onnx_stats = _read_locust_stats("onnx")
    cpu_rows   = _read_detailed(RESULTS_DIR / "litertlm_detailed.csv")
    onnx_rows  = _read_detailed(RESULTS_DIR / "onnx_detailed.csv")

    # ── print comparison ──────────────────────────────────────────────────────
    sep = "=" * 70
    print(f"\n{sep}")
    print("  LOAD TEST COMPARISON  —  Gemma 4 E2B  (100 random prompts)")
    print(sep)

    if args.flavour in ("cpu", "both"):
        _print_locust_stats("CPU \u2014 LiteRT-LM (XNNPACK) via WSL", cpu_stats)
        _print_detailed_stats("CPU \u2014 LiteRT-LM (XNNPACK) via WSL", cpu_rows, has_ttft=True)

    if args.flavour in ("onnx", "both"):
        _print_locust_stats("ONNX \u2014 raw ONNX Runtime via WSL", onnx_stats)
        _print_detailed_stats("ONNX \u2014 raw ONNX Runtime via WSL", onnx_rows, has_ttft=False)

    # ── delta summary ─────────────────────────────────────────────────────────
    if args.flavour == "both" and cpu_rows and onnx_rows:
        cpu_gen  = _floats(cpu_rows, "gen_time_s")
        onnx_gen = _floats(onnx_rows, "gen_time_s")
        cpu_tps  = _floats(cpu_rows, "decode_speed_tok_s")
        onnx_tps = _floats(onnx_rows, "decode_speed_tok_s")
        if cpu_gen and onnx_gen:
            delta_gen = statistics.mean(cpu_gen) - statistics.mean(onnx_gen)
            faster = "ONNX" if delta_gen > 0 else "CPU"
            print(f"\n  {'─'*60}")
            print(f"  Delta summary")
            print(f"  {'─'*60}")
            print(
                f"  Avg gen time  : CPU={statistics.mean(cpu_gen):.3f}s"
                f"  ONNX={statistics.mean(onnx_gen):.3f}s"
                f"  → {faster} is faster by {abs(delta_gen):.3f}s"
            )
        if cpu_tps and onnx_tps:
            delta_tps = statistics.mean(cpu_tps) - statistics.mean(onnx_tps)
            faster_tps = "CPU" if delta_tps > 0 else "ONNX"
            print(
                f"  Avg decode spd: CPU={statistics.mean(cpu_tps):.1f} tok/s"
                f"  ONNX={statistics.mean(onnx_tps):.1f} tok/s"
                f"  → {faster_tps} is faster by {abs(delta_tps):.1f} tok/s"
            )

    # ── export JSON ───────────────────────────────────────────────────────────
    if cpu_rows or onnx_rows:
        out = _export_comparison(cpu_stats, onnx_stats, cpu_rows, onnx_rows)
        print(f"\n  Full comparison written to: {out}")

    print(f"\n  All result files: {RESULTS_DIR}/")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
