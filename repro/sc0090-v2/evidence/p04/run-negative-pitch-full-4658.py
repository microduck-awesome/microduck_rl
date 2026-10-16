"""Bounded frozen catch validation, sequenced after the active training queue."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04')
SOURCE = ROOT / 'paired-frozen-sampling-worktree'
TRAIN_SOURCE = ROOT / 'paired-catch-full-pitch-worktree'
BEFORE = ROOT / 'paired-head-accuracy-allocation-003/checkpoint_004594.pt'
AFTER = ROOT / 'paired-catch-full-pitch-allocation-001/checkpoint_004658.pt'
MARKER = ROOT / 'paired-catch-full-pitch-4658-common-measured/fullpitch4658/standing-pitch-bins.json'
PREVIOUS = ROOT / 'paired-catch-full-pitch-plan.json'
OUTPUT = ROOT / 'paired-negative-pitch-full-4594-4658'
PLAN = ROOT / 'paired-negative-pitch-full-4594-4658-plan.json'
EVALUATOR = SOURCE / 'scripts/audit_negative_pitch_rollout_comparison.py'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(plan):
    tmp = PLAN.with_suffix('.tmp')
    tmp.write_text(json.dumps(plan, indent=2) + '\n')
    tmp.replace(PLAN)

assert not OUTPUT.exists() and not PLAN.exists()
plan = dict(status='WAITING_FOR_FULL_PITCH_TRAINING_AND_EVALUATIONS', created_unix=time.time(),
    before=str(BEFORE), after=str(AFTER), device='cuda:1', seeds=[2026101603,2026101604],
    samples_per_seed=512, pitch_deg=[-10,-5], roll_deg=[-10,10], duration_s=8,
    purpose='Test whether expanding targeted reset exposure to the full [-10,-5] interval improves direct catch, comparing the frozen 4594 parent with 4658 after 64 updates; retain all recovery/head comparisons.',
    scope='Frozen deterministic measured S0/O0; same captured initial states; fixed development seeds. No new optimizer updates or hardware claim.',
    score_boundary='Shared original Head5 evaluation reward, excluding the independent accuracy bonus in both candidates; not the training return.',
    followup='No automatic further training; inspect falls before/after first stable, final stability, head hold and time before deciding.',
    queue_sha256=sha(Path(__file__)))
save(plan)
try:
    deadline = time.monotonic() + 3600
    while True:
        try:
            status = json.loads(PREVIOUS.read_text())['status']
        except (FileNotFoundError, json.JSONDecodeError):
            status = 'PENDING'
        if status == 'ALLOCATION_AND_EVALUATIONS_COMPLETE' and MARKER.exists():
            break
        if 'FAILED' in status:
            raise RuntimeError(f'Previous allocation failed: {status}')
        if time.monotonic() > deadline:
            raise TimeoutError('Previous allocation or evaluation not completed')
        time.sleep(5)
    assert sha(EVALUATOR) == 'd4a142ebde8f37a3f60101a81d6fd7f309e09ffc62e80807fe0cff80668a8be0'
    physics_hashes = {}
    for folder in ['src/mjlab_microduck/sc0090_v2', 'src/mjlab_microduck/actuator']:
        for path in (TRAIN_SOURCE / folder).rglob('*.py'):
            relative = path.relative_to(TRAIN_SOURCE)
            assert sha(path) == sha(SOURCE / relative), str(relative)
            physics_hashes[str(relative)] = sha(path)
    plan.update(before_sha256=sha(BEFORE), after_sha256=sha(AFTER),
        evaluator_sha256=sha(EVALUATOR), physics_source_sha256=physics_hashes)
    command = [str(SOURCE / 'scripts/v2_python.sh'), str(EVALUATOR), '--device', 'cuda:1',
        '--before', str(BEFORE), '--after', str(AFTER), '--wait-for-after', str(MARKER), '--output', str(OUTPUT)]
    env = dict(os.environ, SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat')
    with (ROOT / 'paired-negative-pitch-full-4594-4658.log').open('w') as log:
        started = time.time()
        process = subprocess.Popen(command, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT)
        plan.update(status='RUNNING', pid=process.pid, command=command, process_start_unix=started)
        save(plan)
        code = process.wait()
    plan.update(status='COMPLETE' if code == 0 else 'FAILED', returncode=code,
        process_wall_seconds=time.time()-started, goal_complete=False)
    save(plan)
    raise SystemExit(code)
except Exception as error:
    plan.update(status='FAILED', error=repr(error), failure_unix=time.time())
    save(plan)
    raise
