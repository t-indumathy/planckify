"""Download Gemma 4 E2B LiteRT-LM model from HuggingFace.

Model: litert-community/gemma-4-E2B-it-litert-lm
File:  gemma-4-E2B-it.litertlm (~2.58 GB, int4 quantized)
Docs:  https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download, login

REPO_ID = "litert-community/gemma-4-E2B-it-litert-lm"
MODEL_FILENAME = "gemma-4-E2B-it.litertlm"
DEFAULT_MODEL_DIR = Path("./models")


def download_model(model_dir: Path, hf_token: str | None = None) -> Path:
    """Download the .litertlm model file from HuggingFace Hub."""
    model_dir.mkdir(parents=True, exist_ok=True)
    dest = model_dir / MODEL_FILENAME

    if dest.exists():
        print(f"Model already exists at {dest}, skipping download.")
        return dest

    if hf_token:
        login(token=hf_token, add_to_git_credential=False)

    print(f"Downloading {MODEL_FILENAME} from {REPO_ID} ...")
    cached_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME,
        token=hf_token,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
    )
    print(f"Model saved to: {cached_path}")
    return Path(cached_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Gemma 4 E2B LiteRT-LM model")
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    path = download_model(args.model_dir, args.hf_token)
    print(f"Ready: {path}")
