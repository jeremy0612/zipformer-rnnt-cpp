"""
fix_encoder_onnx.py
-------------------
Fix the encoder ONNX graph for RKNN compatibility.

Root cause:
  The If nodes' subgraphs reference outer-scope initializers
  (e.g. 'self.encoder.encoders.5.encoder.encoder_pos.embed_dim').
  RKNN's internal onnxruntime (older, stricter) does not resolve outer-scope
  references inside subgraphs and throws INVALID_GRAPH.

Fix:
  Copy any outer-scope initializer that is referenced inside an If subgraph
  directly into that subgraph as a local initializer.  No onnxsim needed.

Run on host OR inside Docker:
    python convert/fix_encoder_onnx.py
"""

import shutil
import sys
from pathlib import Path
from typing import Set

import numpy as np

try:
    import onnx
    from onnx import numpy_helper, shape_inference
except ImportError:
    raise SystemExit("pip install onnx")

ROOT     = Path(__file__).parent.parent
ONNX_DIR = ROOT / "onnx_streaming"
SRC      = ONNX_DIR / "encoder.onnx"
BACKUP   = ONNX_DIR / "encoder_orig.onnx"
DST      = ONNX_DIR / "encoder.onnx"


def collect_subgraph_defined(subgraph) -> Set[str]:
    """Names that are defined within a subgraph (inputs + node outputs + initializers)."""
    defined = set()
    for inp in subgraph.input:
        defined.add(inp.name)
    for init in subgraph.initializer:
        defined.add(init.name)
    for node in subgraph.node:
        for out in node.output:
            if out:
                defined.add(out)
    return defined


def collect_subgraph_references(subgraph) -> Set[str]:
    """All names consumed by nodes inside a subgraph."""
    refs = set()
    for node in subgraph.node:
        for inp in node.input:
            if inp:
                refs.add(inp)
    return refs


def fix_subgraph(subgraph, outer_init_map: dict, depth: int = 0) -> int:
    """
    Recursively fix a subgraph:
    - For each node input that is not defined locally but exists in
      outer_init_map, copy the initializer into this subgraph.
    Returns the number of initializers injected.
    """
    injected = 0
    local_inits = {init.name for init in subgraph.initializer}

    defined  = collect_subgraph_defined(subgraph)
    refs     = collect_subgraph_references(subgraph)
    missing  = refs - defined   # referenced but not defined locally

    for name in missing:
        if name in outer_init_map and name not in local_inits:
            subgraph.initializer.append(outer_init_map[name])
            local_inits.add(name)
            injected += 1
            indent = "  " * (depth + 1)
            print(f"{indent}injected initializer '{name}' into subgraph")

    # Recurse into nested subgraphs (If inside If, etc.)
    for node in subgraph.node:
        for attr in node.attribute:
            if attr.HasField("g"):
                # Build a merged outer map for the nested subgraph:
                # its outer scope includes both original outer + this subgraph's inits
                nested_outer = dict(outer_init_map)
                for init in subgraph.initializer:
                    nested_outer[init.name] = init
                injected += fix_subgraph(attr.g, nested_outer, depth + 1)

    return injected


def fix_model(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    graph = model.graph

    # Build outer initializer map
    outer_init_map = {init.name: init for init in graph.initializer}

    total_injected = 0
    for node in graph.node:
        if node.op_type == "If":
            for attr in node.attribute:
                if attr.HasField("g"):
                    n = fix_subgraph(attr.g, outer_init_map, depth=0)
                    total_injected += n

    return model, total_injected


def verify_with_ort(path: Path):
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [skip] onnxruntime not installed")
        return True

    print("  Verifying with onnxruntime ...")
    try:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = {}
        for inp in sess.get_inputs():
            shape = []
            for i, d in enumerate(inp.shape):
                if isinstance(d, int) and d > 0:
                    shape.append(d)
                elif i == 0:
                    shape.append(1)
                else:
                    shape.append(100)  # T
            dtype = np.float32 if "float" in inp.type else np.int64
            inputs[inp.name] = np.zeros(shape, dtype=dtype)
        outputs = sess.run(None, inputs)
        print(f"  OK — output shapes: {[list(o.shape) for o in outputs]}")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def main():
    # Always work from the original backup if it exists, so this script is idempotent
    src = BACKUP if BACKUP.exists() else SRC
    if not src.exists():
        raise SystemExit(
            f"encoder.onnx not found.\nRun: python convert/download_models.py"
        )

    print(f"Loading {src.name}  ({src.stat().st_size/1e6:.1f} MB) ...")
    model = onnx.load(str(src))
    if_count = sum(1 for n in model.graph.node if n.op_type == "If")
    print(f"  opset: {model.opset_import[0].version}   If nodes: {if_count}")

    if if_count == 0:
        print("No If nodes — model is already clean.")
        if not DST.exists() or src == BACKUP:
            onnx.save(model, str(DST))
        verify_with_ort(DST)
        return

    print(f"\nInjecting outer-scope initializers into {if_count} If subgraph(s) ...")
    model, total = fix_model(model)
    print(f"  Total initializers injected: {total}")

    # Shape inference (best-effort)
    print("  Running shape inference ...")
    try:
        model = shape_inference.infer_shapes(model, check_type=True, strict_mode=False)
    except Exception as e:
        print(f"  [WARN] shape inference: {e}")

    # Save
    if not BACKUP.exists():
        shutil.copy(SRC, BACKUP)
        print(f"  Backup -> {BACKUP.name}")

    onnx.save(model, str(DST))
    print(f"  Saved  -> {DST.name}  ({DST.stat().st_size/1e6:.1f} MB)")

    ok = verify_with_ort(DST)
    if ok:
        print("\nDone. Run convert_to_rknn.py inside Docker.")
    else:
        print("\n[FAIL] ORT still rejects the model. See error above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
