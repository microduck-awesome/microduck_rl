#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export SC0090_CUDA_COMPAT_DIR=/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat
M4_TASK=Mjlab-StandUp-Flat-MicroDuck-V2-Fresh-DirectionGate-M4-Repaired-S0-O0-R1
python3 - <<'CHECK'
import os,time,json
from pathlib import Path
assert json.loads(Path('runs-v2/fresh-p04-m4-smoke/report.json').read_text())['status']=='PASS'
deadline=time.monotonic()+60
while '1 passed' not in Path('runs-v2/fresh-p04-m4-tests-v2.log').read_text():
    if time.monotonic()>deadline:raise RuntimeError('Direction gate tests did not pass')
    time.sleep(1)
deadline=time.monotonic()+60
while os.path.exists('/proc/2565942'):
    if time.monotonic()>deadline:raise RuntimeError('Previous GPU0 allocation did not exit')
    time.sleep(1)
old=json.loads(Path('runs-v2/fresh-p04-direction-allocation-001/status.json').read_text())
assert old['status']=='STOPPED' and Path(old['resume_checkpoint']).exists()
CHECK
scripts/v2_python.sh scripts/smoke_v2_ppo.py --task "$M4_TASK" --device cuda:0 \
 --num-envs 32768 --iterations 16 --force-timeouts --compiled-measurement --fused-registers \
 --output-dir runs-v2/fresh-p04-m4-throughput-32768 > runs-v2/fresh-p04-m4-throughput-32768.log 2>&1
exec scripts/v2_python.sh scripts/train_v2_standup.py --task "$M4_TASK" --device cuda:0 \
 --num-envs 32768 --iterations 128 --eval-interval 64 --eval-samples 128 \
 --preflight runs-v2/fresh-p04-m4-throughput-32768/report.json \
 --adapt-from runs-v2/fresh-p04-overlap-allocation-001/resume_000768.pt \
 --output runs-v2/fresh-p04-m4-allocation-001 > runs-v2/fresh-p04-m4-allocation-001.log 2>&1
