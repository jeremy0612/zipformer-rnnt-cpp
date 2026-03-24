"""
convert_to_rknn.py
------------------
Convert streaming Zipformer-RNNT ONNX models to RKNN format.

Run this INSIDE the rknn-toolkit2 Docker container:

  # Step 1 — export streaming ONNX (run on host):
  python convert/export_onnx_streaming.py --model-dir /path/to/Zipformer-30M-RNNT-6000h

  # Step 2 — convert to RKNN (run inside Docker):
  docker run --platform linux/amd64 --rm \\
    -v $PWD:/work -w /work rknn-toolkit2:2.3.2 \\
    python convert/convert_to_rknn.py --no-quant

  # With INT8 quantization (needs calibration data first):
  python convert/gen_calibration.py
  docker run ... python convert/convert_to_rknn.py

Model I/O
---------
  encoder: input  [1, chunk+left+right, 80]  float32 + cache tensors
           output [1, chunk, encoder_dim]     float32 + updated caches
  decoder: input  [1]   int64  (last token id) + hidden/cell
           output [1, 1, decoder_dim]          float32 + new hidden/cell
  joiner:  input  [encoder_dim]  float32 + [decoder_dim] float32
           output [vocab_size]   float32
"""

import os
import sys
import argparse
import numpy as np

try:
    from rknn.api import RKNN
except ImportError:
    raise SystemExit(
        "ERROR: rknn-toolkit2 not installed.\n"
        "Run this script inside the rknn-toolkit2 Docker container:\n\n"
        "  docker run --platform linux/amd64 --rm \\\n"
        "    -v $PWD:/work -w /work rknn-toolkit2:2.3.2 \\\n"
        "    python convert/convert_to_rknn.py --no-quant"
    )

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = os.path.join(os.path.dirname(__file__), "..")
ONNX_DIR     = os.path.join(ROOT, "onnx_streaming")
MODELS_DIR   = os.path.join(ROOT, "models")
DATASET_TXT  = os.path.join(ROOT, "calib_dataset.txt")

MODELS = ["encoder", "decoder", "joiner"]

# Input shapes — adjust once the streaming ONNX shapes are confirmed
# encoder: [batch, chunk+left+right, mel_bins]  (without cache tensors listed separately)
# decoder: [batch, 1]  (single token)
# joiner:  [batch, encoder_dim], [batch, decoder_dim]
INPUT_SIZES = {
    "encoder": [[1, 100, 80], [1]],   # x: [1,100,80], x_lens: [1]
    "decoder": [[1, 2]],              # y: [1,2]  (int64, 2-gram context)
    "joiner":  [[1, 512], [1, 512]],  # encoder_out: [1,512], decoder_out: [1,512]
}

# Input names for each model (must match ONNX graph input names exactly)
INPUT_NAMES = {
    "encoder": ["x", "x_lens"],
    "decoder": ["y"],
    "joiner":  ["encoder_out", "decoder_out"],
}


def convert(name: str, do_quantization: bool):
    onnx_path = os.path.join(ONNX_DIR,   f"{name}.onnx")
    rknn_path = os.path.join(MODELS_DIR, f"{name}.rknn")

    if not os.path.exists(onnx_path):
        print(f"[SKIP] {onnx_path} not found — run export_onnx_streaming.py first")
        return

    rknn = RKNN(verbose=True)

    # ── 1. Configure ───────────────────────────────────────────────────────────
    print(f"\n=== {name} ===")
    print("--> Configuring")
    ret = rknn.config(
        target_platform     = "rk3588",
        quantized_dtype     = "asymmetric_quantized-8",
        quantized_method    = "channel",
        quantized_algorithm = "mmse",
        optimization_level  = 3,
    )
    assert ret == 0, f"rknn.config failed: {ret}"

    # ── 2. Load ONNX ───────────────────────────────────────────────────────────
    print(f"--> Loading {onnx_path}")
    ret = rknn.load_onnx(model=onnx_path, inputs=INPUT_NAMES[name], input_size_list=INPUT_SIZES[name])
    if ret != 0:
        print(f"load_onnx failed: {ret}")
        rknn.release()
        sys.exit(ret)

    # ── 3. Build ───────────────────────────────────────────────────────────────
    if do_quantization:
        if not os.path.exists(DATASET_TXT):
            print(f"[WARN] {DATASET_TXT} not found — falling back to FP16")
            do_quantization = False

    if do_quantization:
        print(f"--> Building INT8 (MMSE, per-channel), dataset: {DATASET_TXT}")
        ret = rknn.build(do_quantization=True, dataset=DATASET_TXT)
    else:
        print("--> Building FP16 (no quantization)")
        ret = rknn.build(do_quantization=False)

    if ret != 0:
        print(f"rknn.build failed: {ret}")
        rknn.release()
        sys.exit(ret)

    # ── 4. Export ──────────────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"--> Exporting to {rknn_path}")
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print(f"export_rknn failed: {ret}")
        rknn.release()
        sys.exit(ret)

    size_mb = os.path.getsize(rknn_path) / (1024 * 1024)
    print(f"OK -> {rknn_path}  ({size_mb:.1f} MB)")
    rknn.release()


def main():
    import faulthandler, traceback
    faulthandler.enable()

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-quant", action="store_true",
                        help="Build FP16 — no calibration data required (good first pass)")
    parser.add_argument("--model", choices=MODELS + ["all"], default="all",
                        help="Which model to convert (default: all)")
    args = parser.parse_args()

    targets = MODELS if args.model == "all" else [args.model]

    try:
        for name in targets:
            convert(name, do_quantization=not args.no_quant)
        print("\nAll done. Copy models/*.rknn + models/bpe.model to the target device.")
    except Exception:
        log = "/work/convert_crash.log"
        with open(log, "w") as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
