"""Download Gemma 4 E2B LiteRT-LM model from HuggingFace.

Model: litert-community/gemma-4-E2B-it-litert-lm
Size:  ~2.6 GB (int4 quantized .litertlm file)
Docs:  https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download, login

REPO_ID = "litert-community/gemma-4-E2B-it-litert-lm"
MODEL_FILENAME = "gemma-4-E2B-it-litert-lm.litertlm"
DEFAULT_MODEL_DIR = Path("./models")


def download_model(model_dir: Path, hf_token: str | None = None) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    local_path = model_dir / MODEL_FILENAME

    if local_path.exists():
        print(f"Model already exists at: {local_path}")
        return local_path

    if hf_token:
        login(token=hf_token)
    elif os.environ.get("HF_TOKEN"):
        login(token=os.environ["HF_TOKEN"])
    else:
        print(
            "[WARNING] No HF_TOKEN found. Model access may require authentication.\n"
            "Set HF_TOKEN env var or pass --hf-token."
        )

    print(f"Downloading {MODEL_FILENAME} from {REPO_ID} ...")
    print("This is ~2.6 GB — please be patient.")

    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
    )

    print(f"\nModel downloaded to: {downloaded}")
    return Path(downloaded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Gemma 4 E2B LiteRT-LM model")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Directory to save the model (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace access token (or set HF_TOKEN env var)",
    )
    args = parser.parse_args()
    download_model(args.model_dir, args.hf_token)
