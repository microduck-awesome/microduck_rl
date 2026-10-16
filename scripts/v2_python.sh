#!/usr/bin/env bash
# Process-local CUDA compatibility, matching backup/local-work-20261015.
set -euo pipefail
V2_REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$V2_REPO_DIR"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export TRITON_DEFAULT_FP_FUSION=${TRITON_DEFAULT_FP_FUSION:-0}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/home/l40-vm01/data2/experiments/microduck-v2-20261015/inductor-no-fma}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/home/l40-vm01/data2/experiments/microduck-v2-20261015/triton-no-fma}
export PYTHONPATH="$V2_REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
V2_COMPAT_DIR=${SC0090_CUDA_COMPAT_DIR:-$V2_REPO_DIR/../microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat}
V2_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
case "$V2_DRIVER" in
  535.*|570.*)
    if [[ ! -f "$V2_COMPAT_DIR/libcuda.so.1" ]]; then
      echo "CUDA graph compatibility library missing: $V2_COMPAT_DIR" >&2
      exit 2
    fi
    export LD_LIBRARY_PATH="$V2_COMPAT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ;;
esac
V2_PYTHON=${SC0090_V2_PYTHON:-/home/l40-vm01/data2/experiments/microduck-v2-20261015/venv-p04/bin/python}
exec "$V2_PYTHON" "$@"
