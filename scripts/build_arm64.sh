#!/usr/bin/env bash
# Cross-compile for RK3588 / aarch64-linux-gnu
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."
BUILD_DIR="$ROOT/build_arm64"

TOOLCHAIN=${TOOLCHAIN:-aarch64-linux-gnu}
RKNN_API_PATH=${RKNN_API_PATH:-"$ROOT/../rknn-toolkit2/rknpu2/runtime/Linux/librknn_api"}

cmake -S "$ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$ROOT/cmake/aarch64-toolchain.cmake" \
    -DBUILD_RKNN=ON \
    -DRKNN_API_PATH="$RKNN_API_PATH"

cmake --build "$BUILD_DIR" -j"$(nproc)"

echo "Binary: $BUILD_DIR/zipformer_asr"
