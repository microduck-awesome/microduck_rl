"""Bounded same-recipe continuation, retaining all parent and comparison artifacts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04')
SOURCE = ROOT / 'paired-head-stdcap-worktree'
TASK = 'Mjlab-StandUp-Flat-MicroDuck-V2-PairedAdapt-Catch2sPitch10-LateEntropy005-HeadStdCap001-Repaired-S0-O0-R1'
OUTPUT = ROOT / 'paired-stdcap-allocation-002'
PLAN = ROOT / 'paired-stdcap-allocation-002-plan.json'
TIMING = ROOT / 'paired-stdcap-allocation-002-timing.json'
PARENT = ROOT / 'paired-stdcap-allocation-001/resume_004018.pt'
ENV = dict(os.environ, SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat')
HELPERS = Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/baseline/head-foundation-worktree/scripts')

def read(path):
    return json.loads(path.read_text())

def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def wait_file(path):
    deadline = time.monotonic() + 1800
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(path)
        time.sleep(5)

assert not OUTPUT.exists(), 'Do not overwrite an existing allocation'
plan = dict(status='WAITING_FOR_COMMON_EVALUATION', parent=str(PARENT), updates=128,
            num_envs=32768, steps_per_env=24, planned_controls=100663296,
            task=TASK, source=str(SOURCE), created_unix=time.time(),
            reason='First cap segment improved current 2-axis deterministic arrival/hold from 117/100 at 3968 to 134/121 at 4018; continue one bounded segment while retaining catch checks.',
            unchanged='same task/reward/commands/Pitch10 Catch2s/reset mixture/PPO/normalizer/measurement; exact resume, no LR reset',
            admission=dict(nominal_recovery=1792, measured_recovery=1792, faults=0,
                           nominal_standing_falls_max=5, measured_standing_falls_max=6,
                           current_two_axis_arrival_min=134, current_two_axis_hold_min=121),
            followup='No automatic third segment. Decide from full recovery, standing order/pitch and deterministic head/hold results; preserve all candidates.')
save(PLAN, plan)
wait_file(ROOT / 'paired-stdcap-4018-common-measured/stdcap4018/standing-pitch-bins.json')
common = read(ROOT / 'paired-stdcap-4018-common-measured/report.json')
assert common['source_unchanged'] and common['condition'] == 'measured'
candidate, = common['candidates']
assert candidate['wins'] == candidate['trials'] == 1792 and candidate['faults'] == 0
assert candidate['actor_and_normalizer_unchanged']
nominal = read(ROOT / 'paired-stdcap-allocation-001/evaluation_004018.json')
assert sum(g['wins'] for c in nominal['cases'] for g in c['groups'].values()) == 1792
assert sum(g['faults'] for c in nominal['cases'] for g in c['groups'].values()) == 0
nstanding = read(ROOT / 'paired-stdcap-allocation-001/evaluation_004018_evidence/standing-order.json')['totals']
mstanding = read(ROOT / 'paired-stdcap-4018-common-measured/stdcap4018/standing-order.json')['totals']
assert nstanding['any_fall_after_0p2'] <= 5 and mstanding['any_fall_after_0p2'] <= 6
head = read(ROOT / 'paired-stdcap-4018-head-summary/report.json')['reports'][0]
assert head['checkpoint_sha256'] == candidate['checkpoint_sha256']
pair = head['summary']['step/current/2-axis']
assert pair['simultaneous_pass'] >= 134 and pair['final_half_s_hold_pass'] >= 121
qualification = read(ROOT / 'paired-stdcap-qualification.json')
assert qualification['status'] == 'PASS'
for path, expected in qualification['source_sha256'].items():
    assert digest(SOURCE / path) == expected, path
plan.update(status='ADMITTED', parent_sha256=digest(PARENT),
            common_report_sha256=digest(ROOT / 'paired-stdcap-4018-common-measured/report.json'),
            head_report_sha256=digest(ROOT / 'paired-stdcap-4018-head-summary/report.json'),
            nominal_standing=nstanding, measured_standing=mstanding)
save(PLAN, plan)
timing = dict(pipeline_start_unix=time.time(), phases=[], scope='Continuation only; reuses qualified identical source; no repeated smoke or preflight. Previous experiment costs remain separate.')

def run(label, arguments, log):
    start = time.time()
    with log.open('w') as stream:
        process = subprocess.Popen([str(SOURCE / 'scripts/v2_python.sh'), *map(str, arguments)],
                                   cwd=SOURCE, env=ENV, stdout=stream, stderr=subprocess.STDOUT)
        phase = dict(label=label, pid=process.pid, start_unix=start, arguments=list(map(str, arguments)))
        timing['phases'].append(phase)
        save(TIMING, timing)
        print(label, process.pid, flush=True)
        code = process.wait()
    phase.update(end_unix=time.time(), returncode=code)
    phase['wall_seconds'] = phase['end_unix'] - start
    if label == 'formal':
        phase['controls'] = read(OUTPUT / 'status.json')['allocation_controls']
        phase['complete_process_interactions_per_s'] = phase['controls'] / phase['wall_seconds']
        save(OUTPUT / 'actual-wall-time.json', phase)
    save(TIMING, timing)
    if code:
        raise subprocess.CalledProcessError(code, arguments)

run('formal', ['scripts/train_v2_standup.py', '--task', TASK, '--device', 'cuda:1',
    '--num-envs', '32768', '--iterations', '128', '--eval-interval', '64', '--eval-samples', '128',
    '--head-precision', '--head-precision-stage', '5', '--freeze-normalizers',
    '--preflight', ROOT / 'paired-stdcap-throughput-32768/report.json', '--resume', PARENT,
    '--output', OUTPUT], ROOT / 'paired-stdcap-allocation-002.log')
assert read(OUTPUT / 'status.json')['completed_updates'] == 4146
for evidence in sorted(OUTPUT.glob('evaluation_*_evidence')):
    for script in ['analyze_v2_standing_order.py', 'analyze_standing_pitch_bins.py']:
        run(evidence.name + script, [SOURCE / 'scripts' / script, evidence],
            OUTPUT / (evidence.name + '-' + script + '.log'))
for iteration in [4096, 4146]:
    head_dir = ROOT / f'paired-stdcap-{iteration}-head-measured'
    run(f'head{iteration}', ['scripts/evaluate_v2_head_normalizer.py', '--task', TASK,
        '--checkpoint', OUTPUT / f'checkpoint_{iteration:06d}.pt', '--output', head_dir,
        '--device', 'cuda:1', '--repeats', '8', '--randomization-stage', 'measured'],
        ROOT / f'paired-stdcap-{iteration}-head-measured.log')
    run(f'head-summary{iteration}', [HELPERS / 'summarize_v2_head_saved.py', '--device', 'cpu',
        '--reports', head_dir / 'report.json', '--output', ROOT / f'paired-stdcap-{iteration}-head-summary'],
        ROOT / f'paired-stdcap-{iteration}-head-summary.log')
common_dir = ROOT / 'paired-stdcap-4146-common-measured'
run('common4146', [HELPERS / 'evaluate_common_p04_candidates.py', '--checkpoints',
    OUTPUT / 'checkpoint_004146.pt', '--labels', 'stdcap4146', '--output', common_dir,
    '--device', 'cuda:1', '--samples', '128'], ROOT / 'paired-stdcap-4146-common-measured.log')
for script in ['analyze_v2_standing_order.py', 'analyze_standing_pitch_bins.py']:
    run('common-' + script, [SOURCE / 'scripts' / script, common_dir / 'stdcap4146'],
        common_dir / (script + '.log'))
timing['pipeline_end_unix'] = time.time()
timing['pipeline_wall_seconds'] = timing['pipeline_end_unix'] - timing['pipeline_start_unix']
save(TIMING, timing)
plan['status'] = 'ALLOCATION_AND_EVALUATIONS_COMPLETE'
plan['goal_complete'] = False
save(PLAN, plan)
