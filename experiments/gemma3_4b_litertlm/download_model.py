"""Download Gemma 3 4B IT LiteRT-LM model from HuggingFace.

Model: litert-community/Gemma3-4B-IT
File:  gemma3-4b-it-int8.litertlm (int8 quantized)
Docs:  https://huggingface.co/litert-community/Gemma3-4B-IT
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download, login

REPO_ID = "litert-community/Gemma3-4B-IT"
MODEL_FILENAME = "gemma3-4b-it-int8.litertlm"
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
    parser = argparse.ArgumentParser(description="Download Gemma 3 4B IT int8 LiteRT-LM model")
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    path = download_model(args.model_dir, args.hf_token)
    print(f"Ready: {path}")
