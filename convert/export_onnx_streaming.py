"""
export_onnx_streaming.py
------------------------
Prepare streaming-compatible ONNX models from the HuggingFace repo files.

The HF repo ships three separate ONNX models (encoder / decoder / joiner)
which already match the streaming RNNT pattern:
  - encoder:  processes a chunk of frames, outputs encoder_out
  - decoder:  stateful LSTM prediction network (one token at a time)
  - joiner:   combines encoder_out + decoder_out -> logits

This script:
  1. Verifies the downloaded ONNX files are valid
  2. Prints their I/O specs (shapes + names) so we can set INPUT_SIZES correctly
  3. Optionally runs onnxsim to clean up the graphs

Run on the host (no Docker needed):
    pip install onnx onnxruntime onnxsim huggingface_hub
    python convert/download_models.py   # download first
    python convert/export_onnx_streaming.py

Outputs: onnx_streaming/encoder.onnx, decoder.onnx, joiner.onnx  (in-place simplify)
"""

import os
import sys
from pathlib import Path

try:
    import onnx
except ImportError:
    raise SystemExit("pip install onnx")

ROOT     = Path(__file__).parent.parent
ONNX_DIR = ROOT / "onnx_streaming"
MODELS   = ["encoder", "decoder", "joiner"]


def type_str(elem_type: int) -> str:
    return {
        1: "float32", 2: "uint8", 3: "int8", 5: "int16",
        6: "int32", 7: "int64", 10: "float16", 11: "float64",
    }.get(elem_type, f"type({elem_type})")


def print_io(model: onnx.ModelProto, label: str):
    print(f"\n  {label}")
    print("  Inputs:")
    for inp in model.graph.input:
        t = inp.type.tensor_type
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param
                 for d in t.shape.dim] if t.HasField("shape") else ["?"]
        print(f"    {inp.name:30s}  {type_str(t.elem_type)}  {shape}")
    print("  Outputs:")
    for out in model.graph.output:
        t = out.type.tensor_type
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param
                 for d in t.shape.dim] if t.HasField("shape") else ["?"]
        print(f"    {out.name:30s}  {type_str(t.elem_type)}  {shape}")


def simplify_model(path: Path) -> bool:
    try:
        import onnxsim
    except ImportError:
        print("  [skip simplify] onnxsim not installed")
        return False

    model = onnx.load(str(path))
    simplified, ok = onnxsim.simplify(model)
    if ok:
        onnx.save(simplified, str(path))
        print(f"  [simplify] OK -> {path.name}")
    else:
        print(f"  [simplify] failed, keeping original")
    return ok


def main():
    missing = [m for m in MODELS if not (ONNX_DIR / f"{m}.onnx").exists()]
    if missing:
        raise SystemExit(
            f"Missing ONNX files: {missing}\n"
            "Run: python convert/download_models.py"
        )

    print("=" * 60)
    print("Zipformer-RNNT ONNX model inspection")
    print("=" * 60)

    for name in MODELS:
        path = ONNX_DIR / f"{name}.onnx"
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"\n[{name}]  {size_mb:.1f} MB  opset={model.opset_import[0].version}")
        print_io(model, "Before simplify")

        simplify_model(path)

        # Re-print after simplify to show final shapes
        model2 = onnx.load(str(path))
        print_io(model2, "After simplify")

    print("\n" + "=" * 60)
    print("Update INPUT_SIZES in convert_to_rknn.py to match the shapes above.")
    print("Then run: docker ... python convert/convert_to_rknn.py --no-quant")


if __name__ == "__main__":
    main()
