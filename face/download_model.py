"""
face/download_model.py
======================
Script to ensure pre-trained ArcFace ONNX weights are downloaded
to `data/models/arcface.onnx` for real 512-d feature extraction.
"""

import os
import sys

MODEL_DIR = os.path.join("data", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "arcface.onnx")


def download_arcface():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1000000:
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"[SAFEPASS P1] Model already available: {MODEL_PATH} ({size_mb:.1f} MB)")
        return MODEL_PATH

    print("[SAFEPASS P1] Downloading ArcFace ONNX model from HuggingFace (onnx-community/arcface-onnx)...")
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="onnx-community/arcface-onnx",
            filename="arcface.onnx",
            local_dir=MODEL_DIR
        )
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"[SAFEPASS P1] Successfully downloaded to {path} ({size_mb:.1f} MB)")
        return path
    except Exception as e:
        print(f"[SAFEPASS P1] Failed to download model: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    download_arcface()
