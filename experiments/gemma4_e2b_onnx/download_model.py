"""Download Gemma 4 E2B ONNX model from HuggingFace.

Model: onnx-community/gemma-4-E2B-it-ONNX
Files: q4 quantized ONNX decoder + embedder + vision + audio encoders
Docs:  https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download, login

REPO_ID = "onnx-community/gemma-4-E2B-it-ONNX"
DEFAULT_MODEL_DIR = Path("./models/gemma-4-E2B-it-ONNX")

# q4-quantized ONNX files needed for text-only CPU inference
ALLOW_PATTERNS = [
    "onnx/decoder_model_merged_q4.onnx*",
    "onnx/embed_tokens_q4.onnx*",
    "tokenizer*",
    "*.json",
    "*.txt",
]


def download_model(model_dir: Path, hf_token: str | None = None) -> Path:
    """Download the q4 ONNX model files from HuggingFace Hub."""
    model_dir.mkdir(parents=True, exist_ok=True)

    decoder_path = model_dir / "onnx" / "decoder_model_merged_q4.onnx"
    if decoder_path.exists():
        print(f"Model already exists at {model_dir}, skipping download.")
        return model_dir

    if hf_token:
        login(token=hf_token, add_to_git_credential=False)

    print(f"Downloading {REPO_ID} (q4 CPU files only) ...")
    local_dir = snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=ALLOW_PATTERNS,
        token=hf_token,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
    )
    print(f"Model saved to: {local_dir}")
    return Path(local_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Gemma 4 E2B ONNX model")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    path = download_model(args.model_dir, args.hf_token)
    print(f"Ready: {path}")
