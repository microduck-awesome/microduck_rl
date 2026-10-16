import json,os,runpy,sys,time,hashlib
from pathlib import Path
r=Path(__file__).resolve().parent;w=r/'velstand-camera-worktree';status=r/'velstand-camera-preflight-status.json'
def state(**kw):
 p=status.with_suffix('.tmp');p.write_text(json.dumps(dict(updated_unix=time.time(),supervisor_pid=os.getpid(),**kw),indent=2)+'\n');p.replace(status)
try:
 assert json.loads((r/'velstand-camera-cpu-002.json').read_text())['status']=='PASS_CPU_ONLY_NOT_GPU_QUALIFIED'
 source=w/'tests/validate_v2_camera_physics.py';digest=hashlib.sha256(source.read_bytes()).hexdigest();start=time.monotonic()
 state(status='PRELOADING_CPU_IMPORTS_DURING_ROULADE_EVALUATION',formal_controls=0)
 sys.path.insert(0,str(w/'scripts'));namespace=runpy.run_path(str(source),run_name='camera_physics_preload')
 preload=time.monotonic()-start;marker=r/'roulade-endpoint-pipeline-status.json';state(status='WAITING_GPU0_ROULADE_FINAL_EVALUATION',cpu_preload_seconds=preload,marker=str(marker),formal_controls=0)
 while True:
  q=json.loads(marker.read_text())
  if q['status']=='STAGE_BUDGET_COMPLETE_NO_FURTHER_TRAINING':break
  if 'STOP' in q['status'] or 'FAIL' in q['status']:raise RuntimeError(q)
  if time.monotonic()-start>1800:raise TimeoutError('Roulade not terminal; no overlap or restart')
  time.sleep(2)
 assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
 state(status='RUNNING_GPU0_PHYSICAL_ADMISSION',pid=os.getpid(),cpu_preload_seconds=preload,formal_controls=0)
 sys.argv=[str(source),'--output',str(r/'velstand-camera-physical-001')];begin=time.monotonic();namespace['main']()
 a=json.loads((r/'velstand-camera-physical-001/report.json').read_text());assert a['status']=='PASS_PHYSICS_SAMPLING_NOT_PPO_QUALIFIED'
 state(status='PHYSICAL_ADMISSION_COMPLETE_NO_FORMAL',cpu_preload_seconds=preload,gpu_environment_seconds=time.monotonic()-begin,report=str(r/'velstand-camera-physical-001/report.json'),formal_controls=0)
except BaseException as exc:
 state(status='STOPPED_REQUIRES_REVIEW',error=repr(exc),formal_controls=0);raise
