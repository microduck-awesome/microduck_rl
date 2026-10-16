"""Preload, wait for both arms' qualification, one64 teacher allocation, original eval."""
from pathlib import Path
import gc,hashlib,json,os,runpy,sys,time
r=Path(__file__).resolve().parent;w=r/'roulade-landing-teacher-worktree';started=time.monotonic()
status=r/'roulade-teacher-formal-pipeline-status.json';assert not status.exists()
def read(p):return json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def state(**kw):
 p=status.with_suffix('.tmp');save(p,dict(pid=os.getpid(),updated_unix=time.time(),whole_roulade_budget=150994944,automatic_further_allocation=False,**kw));p.replace(status)
try:
 state(status='IMPORTING',formal_controls=0)
 sys.path.insert(0,str(w/'scripts'));ns=runpy.run_path(str(w/'scripts/train_v2_roulade.py'),run_name='roulade_teacher_formal')
 import torch
 imported=time.monotonic();gate=r/'roulade-pair-ready-001.json'
 state(status='WAITING_FOR_BOTH_QUALIFICATIONS',formal_controls=0,import_seconds=imported-started)
 while not gate.exists():time.sleep(2)
 pair=read(gate);assert pair['status']=='PASS_BOTH_ARMS' and pair['additional_controls']==50331648
 qpath=r/'roulade-teacher-qualification-001.json';assert pair['teacher_qualification_sha256']==sha(qpath)
 q=read(qpath);assert q['source_sha256']==ns['source_hashes'](w)
 for name,digest in q['model_asset_sha256'].items():assert sha(w/name)==digest,name
 evalpath=w/'scripts/evaluate_v2_roulade.py';evalsha=sha(evalpath)
 out=r/'roulade-teacher-allocation-001';training_start=time.monotonic();state(status='FORMAL_TRAINING',formal_controls=0)
 sys.argv=[str(w/'scripts/train_v2_roulade.py'),'--output',str(out),'--device','cuda:0','--num-envs','16384','--iterations','64','--resume',str(r/'roulade-endpoint-allocation-001/checkpoint_000256.pt'),'--source-transition',str(r/'roulade-teacher-source-transition-002.json'),'--standing-teacher',str(r/'head2-micro-exposure-allocation-001/checkpoint_001856.pt'),'--roll-anchor',str(r/'roulade-endpoint-allocation-001/checkpoint_000256.pt'),'--qualification',str(qpath)]
 ns['main']();trained=read(out/'report.json');assert trained['status']=='ALLOCATION_COMPLETE' and trained['completed_updates']==320 and trained['allocation_controls']==25165824
 now=time.monotonic();save(out/'outer-process-wall.json',dict(import_seconds=imported-started,waiting_seconds=training_start-imported,formal_call_seconds=now-training_start,worker_elapsed_through_formal_seconds=now-started,controls=25165824,import_plus_formal_control_per_second=25165824/((imported-started)+(now-training_start)),scope='Full worker elapsed includes explicit qualification queue wait; separately report import+formal work, neither is steady throughput.'))
 checkpoint=out/'checkpoint_000320.pt';gc.collect();torch.cuda.empty_cache();reports=[]
 for stage in ('nominal','measured'):
  assert sha(evalpath)==evalsha
  dest=r/f'roulade-teacher-0320-{stage}';state(status='EVALUATING',phase=dest.name,formal_controls=25165824)
  sys.argv=[str(evalpath),'--checkpoint',str(checkpoint),'--output',str(dest),'--device','cuda:0','--stage',stage,'--num-envs','256']
  runpy.run_path(str(evalpath),run_name='__main__');assert read(dest/'report.json')['status']=='EVALUATED_NOT_HARDWARE_ACCEPTANCE'
  reports.append(str(dest/'report.json'));gc.collect();torch.cuda.empty_cache()
 state(status='TEACHER_ARM_BUDGET_COMPLETE_EVALUATIONS_REVIEW_READY',formal_controls=25165824,evaluations=reports)
except BaseException as exc:
 out=r/'roulade-teacher-allocation-001';state(status='STOPPED_REQUIRES_REVIEW',error=repr(exc),formal_controls=read(out/'status.json')['allocation_controls'] if (out/'status.json').exists() else 0)
 raise
