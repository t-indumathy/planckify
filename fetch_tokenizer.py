from huggingface_hub import hf_hub_download
import os

token = os.environ.get("HF_TOKEN")
model_dir = "experiments/gemma4_e2b_onnx/models/gemma-4-E2B-it-ONNX"

for f in ["onnx/embed_tokens_q4.onnx_data"]:
    try:
        hf_hub_download(
            "onnx-community/gemma-4-E2B-it-ONNX",
            f,
            local_dir=model_dir,
            token=token,
        )
        print(f"Downloaded {f}")
    except Exception as e:
        print(f"Failed {f}: {e}")
