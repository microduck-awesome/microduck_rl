#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
if [[ ! -x .venv/bin/python ]]; then
  echo "Missing Python environment. Run uv sync in $repo_dir first." >&2
  exit 1
fi
if [[ ! -f simulator/app/dist/index.html ]]; then
  if ! command -v npm >/dev/null; then
    echo "Build the web app with Node.js 22+: cd simulator/app && npm ci && npm run build" >&2
    exit 1
  fi
  (cd simulator/app && npm ci --no-audit --no-fund && npm run build)
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec .venv/bin/python simulator/scripts/sc0090_server.py "$@"
