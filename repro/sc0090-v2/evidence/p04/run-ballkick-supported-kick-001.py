"""Single bounded two-condition physics probe, no training allocation."""
from pathlib import Path
import gc,json,os,runpy,sys,time
R=Path(__file__).resolve().parent;W=R/'ballkick-finish-worktree'
STATUS=R/'ballkick-supported-kick-status-001.json'
assert not STATUS.exists()
def state(**kw):
    t=STATUS.with_suffix('.tmp');t.write_text(json.dumps(dict(pid=os.getpid(),updated_unix=time.time(),formal_controls=0,diagnostic_control_cap=128000,**kw),indent=2)+'\n');t.replace(STATUS)
state(status='IMPORTING')
sys.path.insert(0,str(W/'scripts'))
runpy.run_path(str(W/'scripts/train_v2_ballkick.py'),run_name='handoff_imports')
import torch
torch.cuda.set_device('cuda:0')
try:
    reports=[]
    for stage in ('nominal','measured'):
        out=R/f'ballkick-supported-kick-{stage}-001'
        state(status='PHYSICAL_DIAGNOSTIC',stage=stage,output=str(out))
        script=R/'probe-ballkick-supported-kick-001.py'
        sys.argv=[str(script),'--checkpoint',str(R/'ballkick-finish-allocation-001/checkpoint_000170.pt'),
            '--output',str(out),'--device','cuda:0','--stage',stage]
        runpy.run_path(str(script),run_name='__main__')
        d=json.loads((out/'report.json').read_text());assert d['status']=='EVALUATED_NOT_AUTOMATIC_SKILL_ACCEPTANCE'
        reports.append(str(out/'report.json'));gc.collect();torch.cuda.empty_cache()
    state(status='SUPPORTED_KICK_PROBE_COMPLETE_REVIEW_REQUIRED',diagnostic_controls=128000,reports=reports)
except BaseException as exc:
    state(status='FAILED_REQUIRES_REVIEW',error=repr(exc));raise
