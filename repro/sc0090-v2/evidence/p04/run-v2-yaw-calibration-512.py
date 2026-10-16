import hashlib,json,os,pathlib,subprocess,time
r=pathlib.Path(__file__).resolve().parent;src=r/'walking-repaired-prep-worktree';p=r/'walking-yaw-calibration-512-plan.json';assert not p.exists()
plan=dict(status='WAITING_512_COMMON_EVALUATIONS',created_unix=time.time(),planned_control_samples=55*4*400*2,phases=[],scope='One nominal teacher response grid with raw states; measured verification only if a usable inverse is supported. No student reward/request remap and no formal PPO controls.')
def save():
 q=p.with_suffix('.tmp');q.write_text(json.dumps(plan,indent=2)+'\n');q.replace(p)
save()
try:
 while True:
  ep=json.loads((r/'walking-yaw-fine-evaluation-plan-002.json').read_text())
  if ep['status']=='FAILED':raise RuntimeError(ep.get('error'))
  if ep['status']=='ALL_SCHEDULED_EVALUATIONS_COMPLETE':break
  time.sleep(2)
 parent=r/'walking-yaw-fine-allocation-002/checkpoint_000512.pt';plan['checkpoint_sha256']=hashlib.sha256(parent.read_bytes()).hexdigest()
 out=r/'walking-yaw-calibration-512-nominal';cmd=[str(src/'scripts/v2_python.sh'),str(r/'calibrate-v2-walking-yaw.py'),'--checkpoint',str(parent),'--output',str(out),'--device','cuda:1','--stage','nominal','--repeats','4','--seeds','2026101615','2026101616']
 env=dict(os.environ,SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat',PYTHONFAULTHANDLER='1')
 phase=dict(name='nominal_response_grid',arguments=cmd,start_unix=time.time());plan['phases'].append(phase)
 with (r/'walking-yaw-calibration-512-nominal.log').open('w') as log:
  proc=subprocess.Popen(cmd,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT);phase['pid']=proc.pid;plan['status']='RUNNING';save();phase['returncode']=proc.wait()
 phase['wall_seconds']=time.time()-phase['start_unix'];save();assert phase['returncode']==0,phase
 report=json.loads((out/'report.json').read_text());assert report['status']=='EVALUATION_COMPLETE'
 plan.update(status='CALIBRATION_DATA_COMPLETE_NOT_A_VALIDATED_MAP',trials=sum(x['trials'] for x in report['records']),survived=sum(x['survived_trials'] for x in report['records']),faults=sum(x['feedback_invalid_trials'] for x in report['records']),report_sha256=hashlib.sha256((out/'report.json').read_bytes()).hexdigest());save()
except BaseException as error:
 plan.update(status='FAILED',error=repr(error));save();raise
