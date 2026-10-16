"""Bounded longer motion evaluation, both frozen models and both conditions."""
from pathlib import Path
import gc,hashlib,json,os,runpy,sys,time
R=Path(__file__).resolve().parent;W=R/'velstand-head-free-bc-worktree'
STATUS=R/'velstand-head-bc-longhold-status-001.json'
assert not STATUS.exists()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def state(**kw):
    t=STATUS.with_suffix('.tmp');t.write_text(json.dumps(dict(pid=os.getpid(),updated_unix=time.time(),formal_controls=0,diagnostic_control_cap=1254400,**kw),indent=2)+'\n');t.replace(STATUS)
state(status='IMPORTING')
sys.path.insert(0,str(W/'scripts'))
ns=runpy.run_path(str(W/'scripts/train_v2_velstand.py'),run_name='longhold_imports')
import torch
torch.cuda.set_device('cuda:0')
q=read(R/'velstand-head-free-bc-qualification-001.json')
assert ns['source_hashes'](W)==q['source_sha256']
script=R/'evaluate_v2_velstand_longhold_001.py'
plan=read(R/'velstand-head-bc-longhold-plan-001.json');assert sha(script)==plan['evaluator_sha256']
checkpoints=dict(full_bc=R/'velstand-camera-mild-allocation-001/checkpoint_000320.pt',leg_bc=R/'velstand-head-free-bc-allocation-001/checkpoint_000320.pt')
try:
    reports=[]
    for stage in ('nominal','measured'):
        for arm,checkpoint in checkpoints.items():
            out=R/f'velstand-head-bc-longhold-{arm}-{stage}-001'
            state(status='EVALUATING',arm=arm,stage=stage,output=str(out))
            sys.argv=[str(script),'--checkpoint',str(checkpoint),'--output',str(out),'--device','cuda:0','--stage',stage]
            runpy.run_path(str(script),run_name='__main__')
            d=read(out/'report.json');assert d['status']=='EVALUATION_COMPLETE'
            assert sum(x['trials'] for x in d['records'])==224
            reports.append(str(out/'report.json'));gc.collect();torch.cuda.empty_cache()
    for stage in ('nominal','measured'):
        for seed in (2026101612,2026101613):
            initial=[torch.load(R/f'velstand-head-bc-longhold-{arm}-{stage}-001'/f'trace-{seed}.pt',map_location='cpu',weights_only=False)['initial'] for arm in checkpoints]
            for k in initial[0]:assert torch.equal(initial[0][k],initial[1][k]),(stage,seed,k)
    state(status='LONGHOLD_COMPLETE_REVIEW_REQUIRED',diagnostic_controls=1254400,reports=reports,initial_equality=True)
except BaseException as exc:
    state(status='STOPPED_REQUIRES_REVIEW',error=repr(exc));raise
