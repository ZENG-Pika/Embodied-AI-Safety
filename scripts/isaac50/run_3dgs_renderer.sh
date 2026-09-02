#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${GSPLAT_PYTHON:-/home/zxw/miniconda3/envs/simgen/bin/python}"
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda-11.8}"

export CUDA_HOME="$CUDA_ROOT"
export PATH="$(dirname -- "$PYTHON_BIN"):$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/extras/CUPTI/lib64:$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT_DIR/third_party/isaac50_python${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export MAX_JOBS="${MAX_JOBS:-4}"

exec "$PYTHON_BIN" "$ROOT_DIR/scripts/isaac50/render_3dgs.py" "$@"
