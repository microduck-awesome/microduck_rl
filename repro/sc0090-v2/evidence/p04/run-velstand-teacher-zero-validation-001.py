"""One fixed mapped/unmapped teacher test. No PPO or inference deployment change."""
from pathlib import Path
import gc,hashlib,json,os,runpy,sys,time,importlib.util
R=Path(__file__).resolve().parent;W=R/'velstand-head-bias-worktree';STATUS=R/'velstand-teacher-zero-validation-status-001.json'
assert not STATUS.exists()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def state(**kw):
 p=STATUS.with_suffix('.tmp');p.write_text(json.dumps(dict(pid=os.getpid(),updated_unix=time.time(),formal_controls=0,diagnostic_control_cap=537600,**kw),indent=2)+'\n');p.replace(STATUS)
state(status='IMPORTING')
sys.path.insert(0,str(W/'scripts'));ns=runpy.run_path(str(W/'scripts/train_v2_velstand.py'),run_name='teacher_zero_validation_imports')
import torch
torch.set_num_threads(2);torch.cuda.set_device('cuda:1')
q=read(R/'velstand-head-bias-qualification-001.json');assert ns['source_hashes'](W)==q['source_sha256']
for name,digest in q['assets_sha256'].items():assert sha(W/name)==digest
analysis=read(R/'velstand-teacher-yaw-analysis-001.json');candidate={float(x['vx']):float(x['fed_yaw_for_zero']) for x in analysis['candidates'] if x['status']=='CANDIDATE_REQUIRES_CLOSED_LOOP_VALIDATION'}
assert len(candidate)==7
script=R/'evaluate-velstand-teacher-yaw-001.py';spec=importlib.util.spec_from_file_location('bounded_zero_yaw',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
checkpoint=R/'walking-yaw-fine-allocation-002/checkpoint_000512.pt'
assert sha(checkpoint)==analysis['teacher_sha256']
try:
 reports=[]
 for stage in ('nominal','measured'):
  for arm in ('unmapped','mapped'):
   def cases_for():
    return [dict(name=f'requested_vx{v:+.2f}_yaw0',twist=[v,0.,candidate[v] if arm=='mapped' else 0.],head=[0.]*4,requested_twist=[v,0.,0.],diagnostic_arm=arm) for v in sorted(candidate)]
   module.cases_for=cases_for
   out=R/f'velstand-teacher-zero-{arm}-{stage}-001';state(status='FROZEN_CLOSED_LOOP_VALIDATION',stage=stage,arm=arm,output=str(out))
   sys.argv=[str(script),'--checkpoint',str(checkpoint),'--output',str(out),'--device','cuda:1','--stage',stage,'--repeats','16','--seeds','2026101613','2026101614']
   module.main();d=read(out/'report.json');assert d['status']=='EVALUATION_COMPLETE';assert sum(x['trials'] for x in d['records'])==224
   ns['write_json'](out/'external-request-contract.json',dict(status='DIAGNOSTIC_ONLY',desired_yaw=0.,fed_yaw_table=candidate,arm=arm,evaluator_source_sha256=sha(script),candidate_source_sha256=sha(R/'velstand-teacher-yaw-analysis-001.json'),report_scope='Native velocity error fields inreport are relative tofed command; evaluate straightness independently against externalrequested yaw0. No teacher orstudent training.'))
   reports.append(str(out/'report.json'));gc.collect();torch.cuda.empty_cache()
 initial=[]
 for stage in ('nominal','measured'):
  for seed in (2026101613,2026101614):
   data=[torch.load(R/f'velstand-teacher-zero-{arm}-{stage}-001'/f'trace-{seed}.pt',map_location='cpu',weights_only=False)['initial'] for arm in ('unmapped','mapped')]
   equal={k:torch.equal(data[0][k],data[1][k]) for k in data[0]};assert all(equal.values()),(stage,seed,equal);initial.append(dict(stage=stage,seed=seed,equal=equal))
 state(status='ZERO_YAW_CLOSED_LOOP_COMPLETE',diagnostic_controls=537600,reports=reports,initial_equality=initial)
except BaseException as exc:
 state(status='STOPPED_REQUIRES_REVIEW',error=repr(exc));raise
