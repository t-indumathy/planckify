"""Download Gemma 4 E2B ONNX model from HuggingFace.

Model: onnx-community/gemma-4-E2B-it-ONNX
Files: q4 quantized ONNX decoder + tokenizer (text-only CPU, raw onnxruntime)
Docs: https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX
"""
import os
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download, login

REPO_ID = "onnx-community/gemma-4-E2B-it-ONNX"
DEFAULT_MODEL_DIR = Path("./models/gemma-4-E2B-it-ONNX")

# Ignore large files we don't need for text-only CPU inference:
# - vision/audio encoders
# - non-q4 precision variants (fp16, fp32, int8, uint8)
# - web task files
IGNORE_PATTERNS = [
    "onnx/vision_encoder*",
    "onnx/audio_encoder*",
    "onnx/*_fp16*",
    "onnx/*_fp32*",
    "onnx/*_int8*",
    "onnx/*_uint8*",
    "onnx/*_q4f16*",
    "*.task",
    "flax_model*",
    "tf_model*",
    "pytorch_model*",
]


def download_model(model_dir: Path, hf_token: str | None = None) -> Path:
    """Download the q4 ONNX model files from HuggingFace Hub."""
    model_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded (look for any .onnx file)
    existing_onnx = list(model_dir.rglob("*.onnx"))
    if existing_onnx:
        print(f"Model already exists at {model_dir}, skipping download.")
        print(f"Found ONNX files: {[str(f) for f in existing_onnx]}")
        return model_dir

    if hf_token:
        login(token=hf_token, add_to_git_credential=False)

    print(f"Downloading {REPO_ID} (text-only q4 CPU subset) ...")
    local_dir = snapshot_download(
        repo_id=REPO_ID,
        ignore_patterns=IGNORE_PATTERNS,
        token=hf_token,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
    )
    print(f"Model saved to: {local_dir}")

    # Verify required files are present — tokenizer + at least one ONNX file
    onnx_files = list(Path(local_dir).rglob("*.onnx"))
    tokenizer_json = Path(local_dir) / "tokenizer.json"

    missing = []
    if not tokenizer_json.exists():
        missing.append("tokenizer.json")
    if not onnx_files:
        missing.append("*.onnx (no ONNX model files found)")

    if missing:
        raise FileNotFoundError(f"Missing required files after download: {missing}")

    print(f"Verified: tokenizer.json OK, ONNX files: {[str(f) for f in onnx_files]}")
    return Path(local_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Gemma 4 E2B ONNX model")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()
    path = download_model(args.model_dir, args.hf_token)
    print(f"Ready: {path}")
