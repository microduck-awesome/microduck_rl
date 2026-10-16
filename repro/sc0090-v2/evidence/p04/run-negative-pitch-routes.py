"""Read-only negative-pitch validation after the current training/evaluation queue."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04')
source=root/'paired-frozen-sampling-worktree'
current_source=root/'paired-head-accuracy-worktree'
before=root/'paired-stdcap-allocation-003/checkpoint_004274.pt'
after=root/'head2-micro-exposure-allocation-001/checkpoint_001856.pt'
marker=root/'paired-head-accuracy-4338-common-measured/accuracy4338/standing-pitch-bins.json'
output=root/'paired-negative-pitch-routes-4274-1856'
plan_file=root/'paired-negative-pitch-routes-4274-1856-plan.json'
assert not output.exists()
plan=dict(status='WAITING_FOR_FORMAL_AND_STANDARD_EVALUATIONS',created_unix=time.time(),
    before=str(before),after=str(after),device='cuda:1',seeds=[2026101603,2026101604],
    samples_per_seed=512,pitch_deg=[-10,-5],roll_deg=[-10,10],duration_s=8,
    purpose='Compare preserved baseline-rooted T24-4274 and independent-from-zero micro1856 under the same complete [-10,-5] pitch bucket and captured initial conditions. Different lineages/training budgets/rewards; behavioral route comparison, not a single-factor causal training experiment.',
    scope='Frozen deterministic real measured S0/O0 simulation; fixed development seeds reused from T24/T12, not hardware or a held-out acceptance claim. No new optimization steps.')
plan_file.write_text(json.dumps(plan,indent=2)+'\n')
deadline=time.monotonic()+3600
while not (marker.exists() and json.loads((root/'paired-head-accuracy-plan.json').read_text())['status']=='ALLOCATION_AND_EVALUATIONS_COMPLETE'):
    if time.monotonic()>deadline:raise TimeoutError(marker)
    time.sleep(5)
physics_hashes={}
fresh_source=root/'rise-head-micro-exposure-worktree'
for folder in ['src/mjlab_microduck/sc0090_v2','src/mjlab_microduck/actuator']:
    for path in (current_source/folder).rglob('*.py'):
        other=source/path.relative_to(current_source)
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        assert other.exists() and hashlib.sha256(other.read_bytes()).hexdigest()==sha,path
        fresh=fresh_source/path.relative_to(current_source)
        assert fresh.exists() and hashlib.sha256(fresh.read_bytes()).hexdigest()==sha,fresh
        physics_hashes[str(path.relative_to(current_source))]=sha
plan.update(physics_sources_identical=True, common_full_reward='Head5 shared evaluation score only; do not call this the Head2 fresh training return', fresh_source=str(fresh_source),physics_source_sha256=physics_hashes,
    before_sha256=hashlib.sha256(before.read_bytes()).hexdigest(),
    after_sha256=hashlib.sha256(after.read_bytes()).hexdigest(),
    evaluator_sha256=hashlib.sha256((source/'scripts/audit_negative_pitch_rollout_comparison.py').read_bytes()).hexdigest())
command=[str(source/'scripts/v2_python.sh'),'scripts/audit_negative_pitch_rollout_comparison.py','--device','cuda:1',
    '--before',str(before),'--after',str(after),'--wait-for-after',str(marker),'--output',str(output)]
env=dict(os.environ,SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat')
with (root/'paired-negative-pitch-routes-4274-1856.log').open('w') as log:
    plan['process_start_unix']=time.time()
    process=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
    plan.update(status='RUNNING',pid=process.pid,command=command)
    plan_file.write_text(json.dumps(plan,indent=2)+'\n');print('frontier',process.pid,flush=True)
    code=process.wait()
plan.update(status='COMPLETE' if code==0 else 'FAILED',returncode=code,
    process_wall_seconds=time.time()-plan['process_start_unix'],goal_complete=False)
plan_file.write_text(json.dumps(plan,indent=2)+'\n')
raise SystemExit(code)
