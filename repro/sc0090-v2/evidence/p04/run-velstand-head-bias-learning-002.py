"""Continue the one fixed head-bias recipe to its declared learning budget.

The first64 is an observation point; preserve qualification, both optimizer
states, clocks and numerical source. No efficacy-based early stop or sweep.
"""
from pathlib import Path
import gc,hashlib,json,os,runpy,sys,time
R=Path(__file__).resolve().parent;W=R/'velstand-head-bias-worktree'
STATUS=R/'velstand-head-bias-learning-status-002.json'
assert not STATUS.exists()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def state(**kw):
 p=STATUS.with_suffix('.tmp');p.write_text(json.dumps(dict(pid=os.getpid(),updated_unix=time.time(),total_program_control_cap=150994944,additional_worker_control_cap=113246208,**kw),indent=2)+'\n');p.replace(STATUS)
state(status='PRELOADING_FOR_FIXED_RECIPE_CONTINUATION',additional_formal_controls=0)
sys.path.insert(0,str(W/'scripts'))
ns=runpy.run_path(str(W/'scripts/train_v2_velstand.py'),run_name='head_bias_continuation_trainer')
import torch
torch.set_num_threads(2)
plan=read(R/'velstand-head-bias-learning-plan-002.json')
assert plan['remaining_allocations']==[64,64,64] and plan['total_formal_controls']==150994944
try:
 state(status='WAITING_FOR_FIRST64_AND_EVALUATION',additional_formal_controls=0)
 deadline=time.monotonic()+1800
 while True:
  marker=read(R/'velstand-head-bias-pipeline-status-001.json')
  assert marker['status']!='STOPPED_REQUIRES_REVIEW',marker
  if marker['status']=='BOUNDED_HEAD_BIAS_STAGE_COMPLETE_REVIEW_REQUIRED':
   # All original reports saved; wait for actual worker termination before GPU use.
   p=Path(f"/proc/{marker['pid']}/cmdline")
   cmd=p.read_bytes() if p.exists() else b''
   if b'run-velstand-head-bias-001.py' not in cmd:break
  assert time.monotonic()<deadline,'First-stage worker did not finish within30min'
  time.sleep(2)
 qpath=R/'velstand-head-bias-qualification-001.json';q=read(qpath)
 assert q['status']=='PASS'
 assert ns['source_hashes'](W)==q['source_sha256']
 for name,digest in q['assets_sha256'].items():assert sha(W/name)==digest
 torch.cuda.set_device('cuda:0')
 stand=R/'head2-micro-exposure-allocation-001/checkpoint_001856.pt'
 walk=R/'walking-yaw-fine-allocation-002/checkpoint_000512.pt'
 parent=R/'velstand-head-bias-allocation-001/checkpoint_000384.pt'
 assert read(parent.parent/'report.json')['allocation_controls']==37748736
 protocol=read(R/'velstand-camera-eval-protocol.json')
 assert sha(R/'evaluate_v2_velstand_commands.py')==protocol['commands_sha256']
 assert sha(R/'evaluate_v2_velstand_survey_device.py')==protocol['survey_device_sha256']
 done=0;reports=[]
 for index,end in ((2,448),(3,512),(4,576)):
  assert ns['source_hashes'](W)==q['source_sha256']
  for name,digest in q['assets_sha256'].items():assert sha(W/name)==digest
  out=R/f'velstand-head-bias-allocation-{index:03d}'
  state(status='FORMAL_TRAINING',allocation=out.name,additional_formal_controls=done,parent=str(parent))
  started=time.monotonic()
  sys.argv=[str(W/'scripts/train_v2_velstand.py'),'--output',str(out),'--device','cuda:0','--num-envs','24576','--iterations','64','--camera-weight','-.01','--standing-teacher',str(stand),'--walking-teacher',str(walk),'--native-range-budget','16','--resume',str(parent),'--qualification',str(qpath)]
  ns['main']()
  d=read(out/'report.json');assert d['status']=='ALLOCATION_COMPLETE' and d['completed_updates']==end and d['allocation_controls']==37748736
  assert d['feedback_audit']['unexplained_invalid_lanes']==0 and d['camera_audit']['raw_gated_fault_nonfinite_totals'][3]==0
  done+=d['allocation_controls']
  (out/'outer-phase-wall.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-started,controls=d['allocation_controls'],scope='Formal phase excluding qualification/evaluation'),indent=2)+'\n')
  parent=out/f'checkpoint_{end:06d}.pt';gc.collect();torch.cuda.empty_cache()
  for stage in ('nominal','measured'):
   for kind,script in [('commands','evaluate_v2_velstand_commands.py'),('survey','evaluate_v2_velstand_survey_device.py')]:
    dest=R/f'velstand-head-bias-{end:04d}-{kind}-{stage}'
    state(status='INTERMEDIATE_EVALUATION',phase=dest.name,additional_formal_controls=done)
    sys.argv=[str(R/script),'--checkpoint',str(parent),'--output',str(dest),'--device','cuda:0','--stage',stage]
    runpy.run_path(str(R/script),run_name='__main__')
    result=read(dest/'report.json');assert result['status'] in ('EVALUATION_COMPLETE','STUDENT_SURVEY_COMPLETE')
    reports.append(str(dest/'report.json'));gc.collect();torch.cuda.empty_cache()
 assert done==113246208
 state(status='FIXED_RECIPE_256_UPDATES_COMPLETE_REVIEW_CURVE',additional_formal_controls=done,total_program_formal_controls=150994944,reports=reports,checkpoint=str(parent))
except BaseException as exc:
 state(status='STOPPED_REQUIRES_DIAGNOSIS',error=repr(exc));raise
