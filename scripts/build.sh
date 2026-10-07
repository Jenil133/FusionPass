#!/usr/bin/env bash
set -euo pipefail
: "${LLVM_PREFIX:?set LLVM_PREFIX to your LLVM/MLIR install}"
export FUSIONPASS_BUILD_DIR="${FUSIONPASS_BUILD_DIR:-build}"
# FileCheck (lit) lives in $LLVM_PREFIX/bin when LLVM was installed with -DLLVM_INSTALL_UTILS=ON.
export PATH="${LLVM_PREFIX}/bin:${PATH:-}"
cmake -S . -B "$FUSIONPASS_BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR="$LLVM_PREFIX/lib/cmake/mlir" -DLLVM_DIR="$LLVM_PREFIX/lib/cmake/llvm"
cmake --build "$FUSIONPASS_BUILD_DIR"
export FUSIONPASS_COMPILE="$PWD/$FUSIONPASS_BUILD_DIR/bin/fusionpass-compile"
lit -v test/
pytest -q
