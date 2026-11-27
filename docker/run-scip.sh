#!/usr/bin/env bash
# Runs all bundled SCIP indexers against a mounted /workspace and writes *.scip into /out.
set -uo pipefail

OUT_DIR="${SCIP_OUT_DIR:-/out}"
mkdir -p "${OUT_DIR}"

echo "=== [1/4] scip-clang (impulse-graph-core/impulse-cpp) ==="
if [ -d "/workspace/impulse-graph-core/impulse-cpp" ]; then
  cmake -S /workspace/impulse-graph-core/impulse-cpp -B /tmp/cpp-build \
    -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    && scip-clang --compdb-path=/tmp/cpp-build/compile_commands.json \
       --index-output-path="${OUT_DIR}/core_cpp.scip" \
    || echo "[WARN] scip-clang failed; continuing"
fi

echo "=== [2/4] rust-analyzer scip (impulse-graph-core/impulse-rust) ==="
if [ -d "/workspace/impulse-graph-core/impulse-rust" ]; then
  (cd /workspace/impulse-graph-core/impulse-rust && \
    rust-analyzer scip . --output "${OUT_DIR}/core_rust.scip") \
    || echo "[WARN] rust-analyzer scip failed; continuing"
fi

echo "=== [3/4] scip-java (impulse-graph-java) ==="
if [ -d "/workspace/impulse-graph-java" ]; then
  (cd /workspace/impulse-graph-java && \
    scip-java index --output "${OUT_DIR}/java.scip") \
    || echo "[WARN] scip-java failed; continuing"
fi

echo "=== [4/4] scip-python (impulse-verification) ==="
if [ -d "/workspace/impulse-verification" ]; then
  (cd /workspace/impulse-verification && \
    scip-python index . --project-name impulse-verification --output "${OUT_DIR}/verification_py.scip") \
    || echo "[WARN] scip-python failed; continuing"
fi

ls -lh "${OUT_DIR}"/*.scip 2>/dev/null || echo "No .scip files generated."
