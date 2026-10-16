"""Read-only head/leg compatibility probe after the current candidate's full evaluation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04')
source = root / 'paired-head-stdcap-worktree'
body = root / 'paired-stdcap-allocation-002/checkpoint_004146.pt'
teacher = root.parent / 'baseline/paired-feedback-allocation-001/checkpoint_003762.pt'
out = root / 'paired-frozen-head-splice-4146-measured'
plan_file = root / 'paired-frozen-head-splice-4146-plan.json'
assert not out.exists()
plan = dict(status='WAITING_FOR_CURRENT_COMMON_EVALUATION', created_unix=time.time(),
    scope='Frozen diagnostic only; no optimizer, no policy weight change, no deployment artifact.',
    reason='Body catch improves but full head precision still below retained 3762; test whether its head means remain useful with current learned leg behavior before choosing a new training objective.',
    body_checkpoint=str(body), head_teacher=str(teacher), source=str(source),
    actor_inputs='Both unchanged actor normalizers consume the same real measured 61D actor observation. Four named head means from teacher, ten leg means from body.',
    limitations='Two-network composition is not a single trained policy; standing head battery does not prove all-recovery or hardware compatibility.', phases=[])
def write():
    plan_file.write_text(json.dumps(plan,indent=2)+'\n')
write()
marker = root / 'paired-stdcap-4146-common-measured/stdcap4146/standing-pitch-bins.json'
deadline = time.monotonic()+1800
while not marker.exists():
    if time.monotonic()>deadline:
        raise TimeoutError(marker)
    time.sleep(5)
for name, path in [('body',body),('teacher',teacher),('evaluator',source/'scripts/evaluate_v2_head_frozen_splice.py')]:
    plan[name+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
env = dict(os.environ,SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat')
def run(label, args, log):
    plan['status']='RUNNING_'+label
    phase=dict(label=label,start_unix=time.time(),arguments=list(map(str,args)))
    plan['phases'].append(phase)
    with log.open('w') as stream:
        child=subprocess.Popen([str(source/'scripts/v2_python.sh'),*map(str,args)],cwd=source,env=env,stdout=stream,stderr=subprocess.STDOUT)
        phase['pid']=child.pid;write();print(label,child.pid,flush=True)
        phase['returncode']=child.wait()
    phase['wall_seconds']=time.time()-phase['start_unix'];write()
    if phase['returncode']:
        raise subprocess.CalledProcessError(phase['returncode'],args)
task='Mjlab-StandUp-Flat-MicroDuck-V2-PairedAdapt-Catch2sPitch10-LateEntropy005-HeadStdCap001-Repaired-S0-O0-R1'
run('head', ['scripts/evaluate_v2_head_frozen_splice.py','--task',task,'--checkpoint',body,
    '--head-teacher',teacher,'--output',out,'--device','cuda:1','--repeats','8','--randomization-stage','measured'],
    root/'paired-frozen-head-splice-4146-measured.log')
run('summary',[root.parent/'baseline/head-foundation-worktree/scripts/summarize_v2_head_saved.py',
    '--device','cpu','--reports',out/'report.json','--output',root/'paired-frozen-head-splice-4146-summary'],
    root/'paired-frozen-head-splice-4146-summary.log')
run('absolute',[root/'summarize-head-absolute-precision.py','--device','cuda:1','--reports',out/'report.json',
    '--output',root/'paired-frozen-head-splice-4146-absolute.json'],root/'paired-frozen-head-splice-4146-absolute.log')
plan['status']='DIAGNOSTIC_COMPLETE';plan['goal_complete']=False;write()
