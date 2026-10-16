import os,time,json,pathlib,subprocess
r=pathlib.Path(__file__).resolve().parent
p=r/'walking-evaluator-qualification-plan.json'
src=r/'walking-repaired-prep-worktree'
plan=dict(status='WAITING_CONTROLLER_ACTUAL_N',created_unix=time.time(),wait_pid=3603760,scope='Diagnostic 5-update policy only; software/evaluator validation, not a learned walker',phases=[])
def save():
 q=p.with_suffix('.tmp');q.write_text(json.dumps(plan,indent=2)+'\n');q.replace(p)
save()
while True:
 f=pathlib.Path('/proc/3603760/stat')
 if not f.exists() or f.read_text().split()[2]=='Z':break
 time.sleep(2)
try:
 d=json.loads((r/'walking-controller-throughput-32768/report.json').read_text());assert d['status']=='ALLOCATION_COMPLETE'
 env=dict(os.environ,SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat')
 for stage in ('nominal','measured'):
  out=r/('walking-evaluator-qualification-'+stage)
  cmd=[str(src/'scripts/v2_python.sh'),'scripts/evaluate_v2_walking.py','--checkpoint',str(r/'walking-controller-smoke-64/checkpoint_000005.pt'),'--output',str(out),'--device','cuda:1','--stage',stage,'--repeats','2','--seeds','2026101614']
  phase=dict(stage=stage,arguments=cmd,start_unix=time.time());plan['phases'].append(phase);plan['status']='RUNNING';save()
  with (out.with_suffix('.log')).open('w') as log:
   proc=subprocess.Popen(cmd,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT);phase['pid']=proc.pid;save();phase['returncode']=proc.wait()
  phase['wall_seconds']=time.time()-phase['start_unix'];save();assert phase['returncode']==0
  report=json.loads((out/'report.json').read_text());assert report['status']=='EVALUATION_COMPLETE'
  assert report['checks'][0]['oracle_calls']==400
 plan['status']='PASS';save()
except BaseException as e:
 plan['status']='FAILED';plan['error']=repr(e);save();raise
