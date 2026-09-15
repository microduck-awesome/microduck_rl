#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_DIR"
kind=${1:?Usage: train_sc0090_local.sh walk|recovery|walk_v2|recovery_v2|recovery_v3 [train arguments...]}
shift
case "$kind" in
  walk) task=Mjlab-Velocity-Flat-MicroDuck ;;
  recovery) task=Mjlab-StandUp-Flat-MicroDuck ;;
  walk_v2) task=Mjlab-Velocity-Flat-MicroDuck-SC0090-V2 ;;
  recovery_v2) task=Mjlab-StandUp-Flat-MicroDuck-SC0090-V2 ;;
  recovery_v3) task=Mjlab-StandUp-Flat-MicroDuck-SC0090-V3 ;;
  *) echo "Unknown task: $kind" >&2; exit 2 ;;
esac

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

# Local user-space driver libraries enable CUDA Graphs on this L40/R535 host.
# Only load them on NVIDIA's supported R535/R570 branches; newer drivers use
# their own libraries. No system driver or desktop configuration is changed.
SC0090_CUDA_COMPAT_DIR=${SC0090_CUDA_COMPAT_DIR:-$REPO_DIR/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat}
if [[ -f "$SC0090_CUDA_COMPAT_DIR/libcuda.so.1" ]]; then
  SC0090_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  case "$SC0090_DRIVER_VERSION" in
    535.*|570.*) export LD_LIBRARY_PATH="$SC0090_CUDA_COMPAT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
fi

exec .venv/bin/python -m mjlab_microduck.train_cli "$task" \
  --agent.logger tensorboard \
  --agent.experiment-name "sc0090_${kind}" \
  --agent.run-name "sc0090_12v_m6" \
  "$@"
