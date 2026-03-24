# Docker Guide

Two Docker images are used in this project. Each runs on a different platform and handles a different stage.

| Image | Platform | Purpose |
|-------|----------|---------|
| `rknn-toolkit2:2.3.2` | `linux/amd64` | Fix ONNX graphs + convert to RKNN |
| `zipformer-arm64` | `linux/arm64` | Cross-compile C++ binary for RK3588 |

All commands below are run from the **`playground/` directory** (one level above this repo).

---

## Stage 1 — Fix encoder ONNX graph

The encoder exported from icefall has `If` nodes with broken subgraph topology that RKNN rejects. This script injects missing initializers into the subgraphs.

**Run on host (no Docker needed):**

```bash
pip install onnx onnxruntime
python zipformer-rnnt-cpp/convert/fix_encoder_onnx.py
```

Expected output:
```
Loaded encoder.onnx  (92.0 MB)
  opset: 13   If nodes: 6
Injecting outer-scope initializers into 6 If subgraph(s) ...
  Total initializers injected: N
  Saved  -> encoder.onnx
  Verifying with onnxruntime ...
  OK — output shapes: [[1, 23, 512], [1]]
Done. Run convert_to_rknn.py inside Docker.
```

---

## Stage 2 — Convert ONNX → RKNN

`rknn-toolkit2` only runs on x86/amd64 Linux. Use the Docker image.

### 2a. Build the image (once)

Run from `playground/` so the RKNN wheel can be `COPY`-ed from `rknn-toolkit2/`:

```bash
docker build --platform linux/amd64 \
  -f zipformer-rnnt-cpp/docker/Dockerfile.rknn-toolkit2 \
  -t rknn-toolkit2:2.3.2 \
  rknn-toolkit2/rknn-toolkit2/docker/docker_file/ubuntu_20_04_cp38
```

> This takes ~15 min on first build (PyTorch download). Subsequent builds are cached.

### 2b. Convert encoder (needs ONNX fix from Stage 1 first)

```bash
docker run --platform linux/amd64 --rm \
  -v $PWD/zipformer-rnnt-cpp:/work -w /work rknn-toolkit2:2.3.2 \
  python convert/convert_to_rknn.py --no-quant --model encoder
```

### 2c. Convert decoder

```bash
docker run --platform linux/amd64 --rm \
  -v $PWD/zipformer-rnnt-cpp:/work -w /work rknn-toolkit2:2.3.2 \
  python convert/convert_to_rknn.py --no-quant --model decoder
```

### 2d. Convert joiner

```bash
docker run --platform linux/amd64 --rm \
  -v $PWD/zipformer-rnnt-cpp:/work -w /work rknn-toolkit2:2.3.2 \
  python convert/convert_to_rknn.py --no-quant --model joiner
```

### 2e. Convert all at once

```bash
docker run --platform linux/amd64 --rm \
  -v $PWD/zipformer-rnnt-cpp:/work -w /work rknn-toolkit2:2.3.2 \
  python convert/convert_to_rknn.py --no-quant
```

Output files: `zipformer-rnnt-cpp/models/encoder.rknn`, `decoder.rknn`, `joiner.rknn`

### 2f. INT8 quantization (optional, smaller + faster)

Generate calibration data first, then convert:

```bash
# Generate calibration data (run on host)
python zipformer-rnnt-cpp/convert/gen_calibration.py

# Convert with INT8
docker run --platform linux/amd64 --rm \
  -v $PWD/zipformer-rnnt-cpp:/work -w /work rknn-toolkit2:2.3.2 \
  python convert/convert_to_rknn.py
```

---

## Stage 3 — Build C++ binary for RK3588

Uses a native `linux/arm64` container (no QEMU emulation on Apple Silicon).
Compiles with `-march=armv8.2-a` to target the Cortex-A55/A76 cores in RK3588.

### 3a. Build the image (once)

Run from `playground/` so the RKNN runtime `.so` can be `COPY`-ed:

```bash
docker build --platform linux/arm64 \
  -f zipformer-rnnt-cpp/docker/Dockerfile.cross-compile \
  -t zipformer-arm64 \
  rknn-toolkit2
```

### 3b. Compile

```bash
# Clear stale CMake cache if it exists
rm -rf zipformer-rnnt-cpp/build_arm64

docker run --platform linux/arm64 --rm \
  -v $PWD/zipformer-rnnt-cpp:/src \
  zipformer-arm64
```

Outputs:
- `build_arm64/zipformer_asr` — the binary
- `build_arm64/lib/libsentencepiece.so*` — bundled sentencepiece
- `build_arm64/lib/librknnrt.so` — bundled RKNN runtime

> Re-run this step any time you change source files. The CMake cache is reused if `build_arm64/` already exists.

---

## Stage 4 — Deploy to device

```bash
DEVICE=root@<device-ip>

# First deploy (copy everything)
ssh $DEVICE "mkdir -p /opt/zipformer/models /opt/zipformer/lib"

scp zipformer-rnnt-cpp/build_arm64/zipformer_asr          $DEVICE:/opt/zipformer/
scp zipformer-rnnt-cpp/build_arm64/lib/*                   $DEVICE:/opt/zipformer/lib/
scp zipformer-rnnt-cpp/models/encoder.rknn \
    zipformer-rnnt-cpp/models/decoder.rknn \
    zipformer-rnnt-cpp/models/joiner.rknn \
    zipformer-rnnt-cpp/models/bpe.model                    $DEVICE:/opt/zipformer/models/
```

After recompile, only the binary needs updating:

```bash
scp zipformer-rnnt-cpp/build_arm64/zipformer_asr $DEVICE:/opt/zipformer/
```

---

## Stage 5 — Run on device

```bash
ssh root@<device-ip>
cd /opt/zipformer

# Basic usage
LD_LIBRARY_PATH=/opt/zipformer/lib ./zipformer_asr \
  --encoder models/encoder.rknn \
  --decoder models/decoder.rknn \
  --joiner  models/joiner.rknn \
  --bpe     models/bpe.model \
  speaker1.wav

# Streaming output (print partial results as they arrive)
LD_LIBRARY_PATH=/opt/zipformer/lib ./zipformer_asr \
  --encoder models/encoder.rknn \
  --decoder models/decoder.rknn \
  --joiner  models/joiner.rknn \
  --bpe     models/bpe.model \
  --streaming \
  speaker1.wav
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Illegal instruction` | Binary compiled for wrong ARM arch | Rebuild with `-march=armv8.2-a` (already set in Dockerfile) — make sure to delete `build_arm64/` first |
| `libsentencepiece.so.0: not found` | Missing lib on device | Copy `build_arm64/lib/libsentencepiece.so*` to device and set `LD_LIBRARY_PATH` |
| `INVALID_GRAPH` in RKNN convert | Encoder `If` nodes | Run `fix_encoder_onnx.py` before converting |
| `CMakeCache` path mismatch | Stale cache from different host path | `rm -rf build_arm64/` then rerun Docker |
| RKNN `rknn_init` returns non-zero | Wrong model path or corrupted `.rknn` file | Check model paths; reconvert if needed |
