#!/usr/bin/env bash
set -euo pipefail
REPRO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPRO_ROOT"
export PYTHONPATH="$REPRO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$REPRO_ROOT/.repro-cache/inductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$REPRO_ROOT/.repro-cache/triton}
REPRO_PYTHON=${SC0090_REPRO_PYTHON:-$REPRO_ROOT/.venv-repro/bin/python}
if [[ ${REPRO_USE_COMPAT:-auto} != 0 ]]; then
  REPRO_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
  if [[ $REPRO_DRIVER == 535.* || $REPRO_DRIVER == 570.* || ${REPRO_USE_COMPAT:-auto} == 1 ]]; then
    if [[ ! -f ${SC0090_CUDA_COMPAT_DIR:-}/libcuda.so.1 ]]; then
      echo 'Set SC0090_CUDA_COMPAT_DIR to a CUDA 12.9 compatibility library directory containing libcuda.so.1.' >&2
      exit 2
    fi
    export LD_LIBRARY_PATH="$SC0090_CUDA_COMPAT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi
exec "$REPRO_PYTHON" "$@"
