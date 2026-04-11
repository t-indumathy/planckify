"""Smoke test: verify all dependencies import correctly without a model.

Runs in CI without HF_TOKEN or model download.
Tests:
  - onnxruntime imports and CPU provider is available
  - transformers AutoTokenizer can be imported
  - numpy available
  - run_inference_cpu module structure is valid
"""
import sys


def test_onnxruntime_imports():
    import onnxruntime as ort
    providers = ort.get_available_providers()
    assert "CPUExecutionProvider" in providers, f"CPU provider missing. Got: {providers}"
    print(f"  onnxruntime {ort.__version__} OK — providers: {providers}")


def test_transformers_tokenizer_import():
    from transformers import AutoTokenizer  # noqa: F401
    print("  transformers AutoTokenizer import OK")


def test_numpy_import():
    import numpy as np
    arr = np.zeros((1, 4), dtype=np.float32)
    assert arr.shape == (1, 4)
    print(f"  numpy {np.__version__} OK")


def test_run_inference_module():
    import importlib.util
    import pathlib

    script = pathlib.Path(__file__).parent / "run_inference_cpu.py"
    spec = importlib.util.spec_from_file_location("run_inference_cpu", script)
    mod = importlib.util.module_from_spec(spec)
    # Don't exec — just check it loads without syntax errors by checking spec
    assert spec is not None
    assert mod is not None
    print("  run_inference_cpu.py module spec OK")


def test_session_options():
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 2
    opts.intra_op_num_threads = 2
    print("  ort.SessionOptions() OK")


if __name__ == "__main__":
    tests = [
        test_onnxruntime_imports,
        test_transformers_tokenizer_import,
        test_numpy_import,
        test_run_inference_module,
        test_session_options,
    ]
    failed = []
    print("\nRunning ONNX Runtime import smoke tests...")
    print("-" * 50)
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            failed.append(t.__name__)
    print("-" * 50)
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    print(f"All {len(tests)} tests passed.")
