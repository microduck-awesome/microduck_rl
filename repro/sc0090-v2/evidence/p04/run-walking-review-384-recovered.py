import hashlib,json,os,pathlib,subprocess,time
r=pathlib.Path(__file__).resolve().parent;src=r/'walking-repaired-prep-worktree';p=r/'walking-review-384-recovered-plan.json'
plan=dict(status='WAITING_STANDARD_384_EVALUATIONS',created_unix=time.time(),phases=[],scope='Separate diagnostic trajectory capture and saved-state visualization; no policy/learning changes')
def save():
 q=p.with_suffix('.tmp');q.write_text(json.dumps(plan,indent=2)+'\n');q.replace(p)
save()
try:
 recovery=json.loads((r/'walking-yaw-fine-evaluation-recovery.json').read_text());assert recovery['status']=='PASS_CLEAN_INDEPENDENT_REEVALUATION'
 plan['recovery']=recovery;save()
 while int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<4500:time.sleep(3)
 env=dict(os.environ,PYTHONFAULTHANDLER='1',PYTHONUNBUFFERED='1',SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='1')
 capture=r/'walking-review-000384-capture';render=r/'walking-review-000384-video'
 jobs=[('capture',[str(src/'scripts/v2_python.sh'),str(r/'capture-walking-review.py'),'--checkpoint',str(r/'walking-yaw-fine-allocation-001/checkpoint_000384.pt'),'--output',str(capture),'--stage','measured','--repeats','2','--seeds','2026101614','--device','cuda:1']),('render',[str(src/'scripts/v2_python.sh'),str(r/'render-walking-review-padded.py'),'--capture',str(capture),'--output',str(render)])]
 for name,cmd in jobs:
  phase=dict(name=name,arguments=cmd,start_unix=time.time());plan['phases'].append(phase);plan['status']='RUNNING';save()
  with (r/f'walking-review-384-{name}-recovered.log').open('w') as log:
   proc=subprocess.Popen(cmd,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT);phase['pid']=proc.pid;save();phase['returncode']=proc.wait()
  phase['wall_seconds']=time.time()-phase['start_unix'];save();assert phase['returncode']==0,phase
 plan['status']='REVIEW_ARTIFACTS_COMPLETE';save()
except BaseException as e:
 plan['status']='FAILED';plan['error']=repr(e);save();raise
