import os,time,json,pathlib,subprocess
r=pathlib.Path(__file__).resolve().parent;src=r/'walking-repaired-prep-worktree'
plan=dict(status='WAITING_EVALUATOR',created_unix=time.time(),phases=[])
p=r/'walking-resume-qualification-plan.json'
def save():
 q=p.with_suffix('.tmp');q.write_text(json.dumps(plan,indent=2)+'\n');q.replace(p)
save()
try:
 while True:
  d=json.loads((r/'walking-evaluator-qualification-plan.json').read_text())
  if d['status'] in ('PASS','FAILED'):break
  time.sleep(2)
 assert d['status']=='PASS',d
 env=dict(os.environ,SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat')
 jobs=[('resume',[str(src/'scripts/v2_python.sh'),'scripts/train_v2_walking.py','--device','cuda:1','--num-envs','64','--iterations','2','--validation-only','--resume',str(r/'walking-controller-smoke-64/checkpoint_000005.pt'),'--output',str(r/'walking-controller-resume-64')]),('export',[str(src/'scripts/v2_python.sh'),str(r/'verify-walking-export.py')])]
 for name,cmd in jobs:
  phase=dict(name=name,arguments=cmd,start_unix=time.time());plan['phases'].append(phase);plan['status']='RUNNING';save()
  with (r/f'walking-qualification-{name}.log').open('w') as log:
   proc=subprocess.Popen(cmd,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT);phase['pid']=proc.pid;save();phase['returncode']=proc.wait()
  phase['wall_seconds']=time.time()-phase['start_unix'];save();assert phase['returncode']==0,phase
 d=json.loads((r/'walking-controller-resume-64/report.json').read_text());assert d['completed_updates']==7 and d['allocation_controls']==3072 and d['common_step_counter']==168
 plan['status']='PASS';save()
except BaseException as e:
 plan['status']='FAILED';plan['error']=repr(e);save();raise
