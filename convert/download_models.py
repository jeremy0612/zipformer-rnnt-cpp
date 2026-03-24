"""
download_models.py
------------------
Download Zipformer-RNNT ONNX models from HuggingFace.

Usage:
    pip install huggingface_hub
    python convert/download_models.py

Downloads to: onnx_streaming/
  encoder.onnx  (from encoder-epoch-20-avg-10.int8.onnx — smaller, faster on NPU)
  decoder.onnx
  joiner.onnx
  ../models/bpe.model
"""

import os
import shutil
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    raise SystemExit(
        "ERROR: huggingface_hub not installed.\n"
        "  pip install huggingface_hub"
    )

REPO_ID  = "hynt/Zipformer-30M-RNNT-6000h"
ROOT     = Path(__file__).parent.parent
ONNX_DIR = ROOT / "onnx_streaming"
MDL_DIR  = ROOT / "models"

# Use int8 variants — smaller, and RKNN will re-quantize anyway if needed.
# Switch to the non-int8 names if you want FP32 inputs to the RKNN converter.
FILES = {
    "encoder-epoch-20-avg-10.int8.onnx": ONNX_DIR / "encoder.onnx",
    "decoder-epoch-20-avg-10.int8.onnx": ONNX_DIR / "decoder.onnx",
    "joiner-epoch-20-avg-10.int8.onnx":  ONNX_DIR / "joiner.onnx",
    "bpe.model":                          MDL_DIR  / "bpe.model",
}


def main():
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    MDL_DIR.mkdir(parents=True, exist_ok=True)

    for hf_name, dest in FILES.items():
        if dest.exists():
            print(f"[skip] {dest.name} already exists")
            continue
        print(f"[download] {hf_name} -> {dest}")
        cached = hf_hub_download(repo_id=REPO_ID, filename=hf_name)
        shutil.copy(cached, dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  OK  {dest}  ({size_mb:.1f} MB)")

    print("\nAll files ready.")
    print("Next: run convert_to_rknn.py inside Docker")


if __name__ == "__main__":
    main()
