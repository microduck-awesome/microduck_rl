"""Preload while head learning finishes, then one bounded teacher response grid."""
from pathlib import Path
import gc,hashlib,json,os,runpy,sys,time
R=Path(__file__).resolve().parent;W=R/'velstand-head-bias-worktree'
STATUS=R/'velstand-teacher-yaw-status-001.json';assert not STATUS.exists()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def state(**kw):
 p=STATUS.with_suffix('.tmp');p.write_text(json.dumps(dict(pid=os.getpid(),updated_unix=time.time(),formal_controls=0,diagnostic_control_cap=940800,**kw),indent=2)+'\n');p.replace(STATUS)
state(status='PRELOADING')
sys.path.insert(0,str(W/'scripts'));ns=runpy.run_path(str(W/'scripts/train_v2_velstand.py'),run_name='teacher_yaw_imports')
import torch
plan=read(R/'velstand-teacher-yaw-plan-001.json');script=R/'evaluate-velstand-teacher-yaw-001.py'
assert sha(script)==plan['evaluator_sha256']
try:
 state(status='WAITING_FOR_FIXED_HEAD_PROGRAM')
 deadline=time.monotonic()+1800
 while True:
  marker=read(R/'velstand-head-bias-learning-status-002.json')
  assert marker['status']!='STOPPED_REQUIRES_DIAGNOSIS',marker
  if marker['status']=='FIXED_RECIPE_256_UPDATES_COMPLETE_REVIEW_CURVE':
   p=Path(f"/proc/{marker['pid']}/cmdline");cmd=p.read_bytes() if p.exists() else b''
   if b'run-velstand-head-bias-learning-002.py' not in cmd:break
  assert time.monotonic()<deadline,'Head program did not finish within30min'
  time.sleep(2)
 torch.cuda.set_device('cuda:0');torch.set_num_threads(2)
 q=read(R/'velstand-head-bias-qualification-001.json');assert ns['source_hashes'](W)==q['source_sha256']
 for name,digest in q['assets_sha256'].items():assert sha(W/name)==digest
 checkpoint=Path(plan['teacher']);assert sha(checkpoint)==plan['teacher_sha256']
 reports=[]
 for stage in ('nominal','measured'):
  out=R/f'velstand-teacher-yaw-{stage}-001';state(status='MEASURING_FROZEN_TEACHER',stage=stage,output=str(out))
  sys.argv=[str(script),'--checkpoint',str(checkpoint),'--output',str(out),'--device','cuda:0','--stage',stage]
  runpy.run_path(str(script),run_name='__main__')
  report=read(out/'report.json');assert report['status']=='EVALUATION_COMPLETE';assert sum(x['trials'] for x in report['records'])==784
  reports.append(str(out/'report.json'));gc.collect();torch.cuda.empty_cache()
 state(status='TEACHER_RESPONSE_DATA_COMPLETE_NOT_A_VALIDATED_MAP',diagnostic_controls=940800,reports=reports)
except BaseException as exc:
 state(status='STOPPED_REQUIRES_REVIEW',error=repr(exc));raise
