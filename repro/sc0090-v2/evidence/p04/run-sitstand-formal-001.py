import json,os,subprocess,time
from pathlib import Path
root=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');work=root/'sitstand-repaired-worktree'
env=dict(os.environ,SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat')
qualification=root/'sitstand-training-qualification-001.json'
while not qualification.exists():
 log=root/'run-sitstand-admission-002.log'
 if log.exists() and 'Traceback (most recent call last)' in log.read_text():raise RuntimeError('Admission blocked; formal not started')
 time.sleep(5)
q=json.loads(qualification.read_text());assert q['status']=='PASS'
started=time.monotonic();out=root/'sitstand-allocation-001'
cmd=['scripts/v2_python.sh','scripts/train_v2_sitstand.py','--output',str(out),'--device','cuda:0','--num-envs','32768','--iterations','128','--qualification',str(qualification)]
with (root/'sitstand-allocation-001.log').open('w') as log:
 child=subprocess.Popen(cmd,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT)
 print(json.dumps(dict(event='FORMAL_PROCESS_STARTED',pid=child.pid,unix=time.time(),output=str(out))),flush=True)
 rc=child.wait()
 process_wall=time.monotonic()-started
 (out/'outer-process-wall.json').write_text(json.dumps(dict(returncode=rc,wall_seconds_including_imports=process_wall),indent=2)+'\n')
 assert rc==0
r=json.loads((out/'report.json').read_text());assert r['completed_updates']==128 and r['inherited_controls']==0
print(json.dumps(dict(event='FORMAL_DONE',controls=r['allocation_controls'],outer_process_wall_seconds=process_wall,outer_control_per_second=r['allocation_controls']/process_wall)),flush=True)
for stage in ('nominal','measured'):
 target=root/('sitstand-0128-'+stage)
 with (root/('sitstand-0128-'+stage+'.log')).open('w') as log:
  child=subprocess.Popen(['scripts/v2_python.sh','scripts/evaluate_v2_sitstand.py','--checkpoint',str(out/'checkpoint_000128.pt'),'--output',str(target),'--device','cuda:0','--stage',stage,'--num-envs','256','--seed','2603'],cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT)
  print('evaluation',stage,child.pid,flush=True);assert child.wait()==0
print('BOUNDED_ALLOCATION_AND_EVALUATION_COMPLETE; no automatic continuation',flush=True)
