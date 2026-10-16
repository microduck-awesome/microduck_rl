import json,os,subprocess,time
from pathlib import Path
r=Path(__file__).resolve().parent;status=r/'velstand-camera-preflight-status.json'
def state(**kw):
 p=status.with_suffix('.tmp');p.write_text(json.dumps(dict(updated_unix=time.time(),**kw),indent=2)+'\n');p.replace(status)
try:
 assert json.loads((r/'velstand-camera-cpu-002.json').read_text())['status']=='PASS_CPU_ONLY_NOT_GPU_QUALIFIED'
 marker=r/'roulade-endpoint-pipeline-status.json';state(status='WAITING_GPU0_ROULADE_FINAL_EVALUATION',marker=str(marker),formal_controls=0)
 start=time.monotonic()
 while True:
  q=json.loads(marker.read_text())
  if q['status']=='STAGE_BUDGET_COMPLETE_NO_FURTHER_TRAINING':break
  if 'STOP' in q['status'] or 'FAIL' in q['status']:raise RuntimeError(q)
  if time.monotonic()-start>1800:raise TimeoutError('Roulade still not terminal; no GPU overlap or restart')
  time.sleep(2)
 w=r/'velstand-camera-worktree';env=dict(os.environ,SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat');env.pop('MICRODUCK_WARM_START',None)
 log=r/'velstand-camera-physical-001.log';assert not log.exists();started=time.monotonic()
 with log.open('w') as f:
  p=subprocess.Popen(['scripts/v2_python.sh','tests/validate_v2_camera_physics.py','--output',str(r/'velstand-camera-physical-001')],cwd=w,env=env,stdout=f,stderr=subprocess.STDOUT);state(status='RUNNING_GPU0_PHYSICAL_ADMISSION',pid=p.pid,log=str(log),formal_controls=0);p.wait();assert p.returncode==0,p.returncode
 a=json.loads((r/'velstand-camera-physical-001/report.json').read_text());assert a['status']=='PASS_PHYSICS_SAMPLING_NOT_PPO_QUALIFIED'
 state(status='PHYSICAL_ADMISSION_COMPLETE_NO_FORMAL',outer_wall_seconds=time.monotonic()-started,report=str(r/'velstand-camera-physical-001/report.json'),formal_controls=0)
except BaseException as exc:
 state(status='STOPPED_REQUIRES_REVIEW',error=repr(exc),formal_controls=0);raise
