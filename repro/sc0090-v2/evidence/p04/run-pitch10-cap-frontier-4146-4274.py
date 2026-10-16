"""Read-only negative-pitch validation after the current training/evaluation queue."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04')
source=root/'paired-frozen-sampling-worktree'
current_source=root/'paired-head-stdcap-worktree'
before=root/'paired-stdcap-allocation-002/checkpoint_004146.pt'
after=root/'paired-stdcap-allocation-003/checkpoint_004274.pt'
marker=root/'paired-stdcap-4274-common-measured/stdcap4274/standing-pitch-bins.json'
output=root/'paired-cap-pitch10-frontier-4146-4274'
plan_file=root/'paired-cap-pitch10-frontier-4146-4274-plan.json'
assert not output.exists()
plan=dict(status='WAITING_FOR_FORMAL_AND_STANDARD_EVALUATIONS',created_unix=time.time(),
    before=str(before),after=str(after),device='cuda:1',seeds=[2026101601,2026101602],
    samples_per_seed=512,pitch_deg=[-10,-7],roll_deg=[-10,10],duration_s=8,
    purpose='Check residual negative-pitch catch regression/improvement with paired initial qpos/qvel/actor observation/origins; do not infer robust no-fall behavior from 1/256.',
    scope='Frozen deterministic real measured S0/O0 simulation; new development seeds, not hardware or a final held-out acceptance claim. No new optimization steps.')
plan_file.write_text(json.dumps(plan,indent=2)+'\n')
deadline=time.monotonic()+3600
while not marker.exists():
    if time.monotonic()>deadline:raise TimeoutError(marker)
    time.sleep(5)
physics_hashes={}
for folder in ['src/mjlab_microduck/sc0090_v2','src/mjlab_microduck/actuator']:
    for path in (current_source/folder).rglob('*.py'):
        other=source/path.relative_to(current_source)
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        assert other.exists() and hashlib.sha256(other.read_bytes()).hexdigest()==sha,path
        physics_hashes[str(path.relative_to(current_source))]=sha
plan.update(physics_sources_identical=True,physics_source_sha256=physics_hashes,
    before_sha256=hashlib.sha256(before.read_bytes()).hexdigest(),
    after_sha256=hashlib.sha256(after.read_bytes()).hexdigest(),
    evaluator_sha256=hashlib.sha256((source/'scripts/audit_pitch10_cap_frontier.py').read_bytes()).hexdigest())
command=[str(source/'scripts/v2_python.sh'),'scripts/audit_pitch10_cap_frontier.py','--device','cuda:1',
    '--before',str(before),'--after',str(after),'--wait-for-after',str(marker),'--output',str(output)]
env=dict(os.environ,SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat')
with (root/'paired-cap-pitch10-frontier-4146-4274.log').open('w') as log:
    plan['process_start_unix']=time.time()
    process=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
    plan.update(status='RUNNING',pid=process.pid,command=command)
    plan_file.write_text(json.dumps(plan,indent=2)+'\n');print('frontier',process.pid,flush=True)
    code=process.wait()
plan.update(status='COMPLETE' if code==0 else 'FAILED',returncode=code,
    process_wall_seconds=time.time()-plan['process_start_unix'],goal_complete=False)
plan_file.write_text(json.dumps(plan,indent=2)+'\n')
raise SystemExit(code)
