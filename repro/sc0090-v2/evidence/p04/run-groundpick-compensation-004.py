"""Use GPU1 only after the existing SitStand worker has terminated successfully."""
from pathlib import Path
import gc,hashlib,json,os,runpy,sys,time,traceback

R=Path(__file__).resolve().parent
STATUS=R/'groundpick-compensation-status-004.json'
assert not STATUS.exists()
def read(p):return json.loads(p.read_text())
def state(**kw):
    tmp=STATUS.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(pid=os.getpid(),updated_unix=time.time(),formal_controls=0,**kw),indent=2)+'\n')
    tmp.replace(STATUS)
marker=R/'sitstand-history-hold320-pipeline-status.json'
original=read(marker);assert original['pid']==356135
state(status='QUEUED_WAITING_VERIFIED_SITSTAND_PROCESS_EXIT',waiting_pid=356135)
try:
    while Path('/proc/356135/cmdline').exists():
        argv=Path('/proc/356135/cmdline').read_bytes()
        assert b'run-sitstand-history-hold320-001.py' in argv,'PID identity changed'
        time.sleep(2)
    terminal=read(marker)
    assert terminal['status']=='SITSTAND_HOLD320_COMPLETE_NO_AUTOMATIC_THIRD_STAGE',terminal
    state(status='IMPORTING_AFTER_SITSTAND_COMPLETE',released_from=terminal)
    import torch
    torch.set_num_threads(2);torch.cuda.set_device('cuda:1')
    script=R/'probe_groundpick_compensated_target_004.py'
    reports=[]
    for arm in ['uncorrected','corrected']:
        out=R/f'groundpick-compensation-{arm}-004'
        state(status='DIAGNOSTIC_RUNNING',arm=arm,output=str(out),diagnostic_control_cap=19200)
        sys.argv=[str(script),'--output',str(out),'--device','cuda:1','--arm',arm]
        runpy.run_path(str(script),run_name='__main__')
        report=read(out/'report.json')
        assert report['status']=='ONE_STATIC_CANDIDATE_DYNAMIC_DIAGNOSTIC_COMPLETE'
        assert report['diagnostic_controls']==9600 and report['original_dynamic_gate_unchanged']
        reports.append(str(out/'report.json'));gc.collect();torch.cuda.empty_cache()
    initials=[torch.load(R/f'groundpick-compensation-{arm}-004/traces.pt',map_location='cpu',weights_only=False)['initial'] for arm in ['uncorrected','corrected']]
    equal={k:torch.equal(initials[0][k],initials[1][k]) for k in initials[0]}
    state(status='FIXED_COMPENSATION_PAIR_COMPLETE_REVIEW_REQUIRED',diagnostic_controls=19200,reports=reports,initial_exact=equal)
except BaseException as exc:
    state(status='STOPPED_REQUIRES_REVIEW',error=repr(exc));traceback.print_exc();raise
