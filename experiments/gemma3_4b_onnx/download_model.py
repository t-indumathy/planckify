"""Download Gemma 3 4B IT ONNX model (int8/quantized) from HuggingFace.

Model: onnx-community/gemma-3-4b-it-ONNX
Files: int8-quantized ONNX decoder + embed_tokens + tokenizer (text-only CPU)
Docs:  https://huggingface.co/onnx-community/gemma-3-4b-it-ONNX
"""
import os
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download, login

REPO_ID = "onnx-community/gemma-3-4b-it-ONNX"
DEFAULT_MODEL_DIR = Path("./models/gemma-3-4b-it-ONNX")

# Download only the int8 (quantized) text files; skip vision encoder and
# all other precision variants (fp32, fp16, q4, q4f16)
IGNORE_PATTERNS = [
    "onnx/vision_encoder*",
    "onnx/*_fp32*",
    "onnx/*_fp16*",
    "onnx/*_q4*",
    "onnx/embed_tokens.onnx",
    "onnx/embed_tokens.onnx_data",
    "onnx/decoder_model_merged.onnx",
    "onnx/decoder_model_merged.onnx_data*",
    "*.task",
    "flax_model*",
    "tf_model*",
    "pytorch_model*",
]


def download_model(model_dir: Path, hf_token: str | None = None) -> Path:
    """Download the int8-quantized ONNX model files from HuggingFace Hub."""
    model_dir.mkdir(parents=True, exist_ok=True)

    existing_onnx = list(model_dir.rglob("*.onnx"))
    if existing_onnx:
        print(f"Model already exists at {model_dir}, skipping download.")
        print(f"Found ONNX files: {[str(f) for f in existing_onnx]}")
        return model_dir

    if hf_token:
        login(token=hf_token, add_to_git_credential=False)

    print(f"Downloading {REPO_ID} (text-only int8/quantized CPU subset) ...")
    local_dir = snapshot_download(
        repo_id=REPO_ID,
        ignore_patterns=IGNORE_PATTERNS,
        token=hf_token,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
    )
    print(f"Model saved to: {local_dir}")

    onnx_files = list(Path(local_dir).rglob("*.onnx"))
    tokenizer_json = Path(local_dir) / "tokenizer.json"
    if not onnx_files:
        raise RuntimeError(f"No .onnx files found under {local_dir}")
    if not tokenizer_json.exists():
        raise RuntimeError(f"tokenizer.json not found under {local_dir}")

    print(f"ONNX files   : {[f.name for f in onnx_files]}")
    print(f"Tokenizer    : {tokenizer_json}")
    return Path(local_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Gemma 3 4B IT int8 ONNX model"
    )
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    path = download_model(args.model_dir, args.hf_token)
    print(f"Ready: {path}")
