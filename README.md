# Zipformer-RNNT RKNN C++ Streaming ASR

Real-time streaming speech recognition using Zipformer-RNNT on RK3588 NPU (RKNN).

Model source: [hynt/Zipformer-30M-RNNT-6000h](https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h)

## Architecture

```
Audio (16kHz PCM)
      │
      ▼
 Log-mel fbank (80 bins, 25ms/10ms)
      │
      ▼
 ZipformerEncoder.rknn  ←──── streaming cache (attention KV + conv state)
      │ encoder_out [T, 512]
      ▼
 RNNTJoiner.rknn  ◄──── RNNTDecoder.rknn (prediction network)
      │ logits [vocab]
      ▼
 Greedy / Beam search
      │
      ▼
 BPE decode → text
```

Three separate RKNN models run on the NPU:
- **encoder** — Zipformer streaming encoder (cache-based, chunk-by-chunk)
- **decoder** — LSTM prediction network
- **joiner**  — small MLP that combines encoder + decoder outputs

## Pipeline

### 1. Download + inspect ONNX models

```bash
pip install huggingface_hub onnx onnxruntime onnxsim

# Download encoder/decoder/joiner ONNX + bpe.model from HuggingFace
python convert/download_models.py

# Inspect I/O shapes and simplify graphs (updates INPUT_SIZES if needed)
python convert/export_onnx_streaming.py
```

### 2. Convert to RKNN

`rknn-toolkit2` is only available inside Docker (x86/amd64 host required).

```bash
# Build the Docker image once (run from playground/ so the wheel can be COPY-ed)
docker build --platform linux/amd64 \
  -f zipformer-rnnt-cpp/docker/Dockerfile.rknn-toolkit2 \
  -t rknn-toolkit2:2.3.2 \
  rknn-toolkit2/rknn-toolkit2/docker/docker_file/ubuntu_20_04_cp38

# FP16 (no calibration data needed — good first pass)
docker run --platform linux/amd64 --rm \
  -v $PWD/zipformer-rnnt-cpp:/work -w /work rknn-toolkit2:2.3.2 \
  python convert/convert_to_rknn.py --no-quant

# INT8 quantization (generate calibration data first)
docker run ... python convert/gen_calibration.py
docker run --platform linux/amd64 --rm \
  -v $PWD/zipformer-rnnt-cpp:/work -w /work rknn-toolkit2:2.3.2 \
  python convert/convert_to_rknn.py
```

### 3. Build (cross-compile for RK3588)

```bash
export RKNN_API_PATH=/path/to/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api
bash scripts/build_arm64.sh
```

### 4. Run on device

```bash
# Copy binary + models to device
scp build_arm64/zipformer_asr models/ root@<device>:/opt/zipformer/

# On device
./zipformer_asr \
    --encoder models/encoder.rknn \
    --decoder models/decoder.rknn \
    --joiner  models/joiner.rknn \
    --bpe     models/bpe.model \
    --streaming \
    test_audio/sample.wav
```

## Project Structure

```
zipformer-rnnt-cpp/
├── src/
│   ├── main.cpp               CLI entry point
│   ├── streaming_asr.hpp/.cpp High-level streaming ASR API
│   ├── zipformer_encoder.hpp  Encoder RKNN wrapper
│   ├── rnnt_decoder.hpp       Decoder RKNN wrapper
│   ├── rnnt_joiner.hpp        Joiner RKNN wrapper
│   ├── bpe_tokenizer.hpp      sentencepiece BPE wrapper
│   └── audio_utils.hpp        WAV I/O + log-mel fbank
├── convert/
│   ├── export_onnx_streaming.py   PyTorch → streaming ONNX
│   └── convert_to_rknn.py         ONNX → RKNN
├── cmake/
│   └── aarch64-toolchain.cmake
├── scripts/
│   └── build_arm64.sh
└── models/                    (place .rknn + bpe.model here)
```

## Dependencies

| Library | Purpose |
|---------|---------|
| librknnrt | RKNN NPU runtime |
| sentencepiece | BPE tokenization |
| libsndfile | WAV file I/O |

## Status

- [x] Project scaffold & headers
- [ ] `export_onnx_streaming.py` — requires icefall model loading code
- [ ] C++ RKNN wrapper implementations
- [ ] Greedy / beam search decoder
- [ ] Log-mel fbank (CPU or NPU)
- [ ] End-to-end test
