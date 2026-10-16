"""Record the trained model's execution chain after the catch verification."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04')
SOURCE = ROOT / 'paired-head-accuracy-worktree'
CHECKPOINT = ROOT / 'paired-head-accuracy-allocation-003/checkpoint_004594.pt'
PROBE = ROOT / 'probe_v2_head_execution.py'
PLAN = ROOT / 'paired-head-execution-4594-plan.json'
OUTPUT = ROOT / 'paired-head-execution-4594'
WAIT = ROOT / 'paired-negative-pitch-accuracy-4274-4594-plan.json'
TASK = 'Mjlab-StandUp-Flat-MicroDuck-V2-PairedAdapt-Catch2sPitch10-LateEntropy005-HeadStdCap001-AccuracyBonus05-Repaired-S0-O0-R1'
assert not OUTPUT.exists() and not PLAN.exists()

def save(value):
    temp = PLAN.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(PLAN)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

plan = dict(status='WAITING_FOR_CATCH_VERIFICATION', created_unix=time.time(),
    source=str(SOURCE), checkpoint=str(CHECKPOINT), task=TASK, device='cuda:1',
    purpose='Record actual command, MLP input/mean, native goal and motor input/PWM/torque for the best current micro-head candidate; determine the next bounded experiment after observed combination-hold regression.',
    scope='Frozen diagnostic, 17 idle/small/current single-axis cases, 8 repeats, 2 seeds; measured real P04 S0/O0 simulation. No training or action replacement.',
    recipe='Use the checkpoint own Head5 metadata and original training source; no fresh Head2 recipe substitution.',
    limitation='Fresh micro1856 previous execution probe used nominal voltage; do not claim a matched cross-route causal comparison.',
    followup='No automatic extra training segment', queue_sha256=sha(Path(__file__)))
save(plan)
try:
    deadline = time.monotonic()+3600
    while True:
        try:
            status = json.loads(WAIT.read_text())['status']
        except (FileNotFoundError, json.JSONDecodeError):
            status = 'PENDING'
        if status == 'COMPLETE':
            break
        if status == 'FAILED':
            raise RuntimeError('Catch verification failed')
        if time.monotonic() > deadline:
            raise TimeoutError('Catch verification did not finish')
        time.sleep(5)
    assert sha(PROBE) == '449406aa54dfb971c24507e8322adf4fad5741484e6a0597d50f9225e5b5a91d'
    command = [str(SOURCE/'scripts/v2_python.sh'), str(PROBE), '--task', TASK,
        '--checkpoint', str(CHECKPOINT), '--output', str(OUTPUT), '--device', 'cuda:1',
        '--repeats', '8', '--randomization-stage', 'measured']
    env = dict(os.environ, SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat')
    started = time.time()
    with (ROOT/'paired-head-execution-4594.log').open('w') as log:
        process = subprocess.Popen(command, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT)
        plan.update(status='RUNNING', pid=process.pid, process_start_unix=started,
            checkpoint_sha256=sha(CHECKPOINT), probe_sha256=sha(PROBE), command=command)
        save(plan)
        code = process.wait()
    plan.update(status='COMPLETE' if code == 0 else 'FAILED', returncode=code,
        wall_seconds=time.time()-started)
    save(plan)
    raise SystemExit(code)
except Exception as error:
    plan.update(status='FAILED', error=repr(error), failure_unix=time.time())
    save(plan)
    raise
