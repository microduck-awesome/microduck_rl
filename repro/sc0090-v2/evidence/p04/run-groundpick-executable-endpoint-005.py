"""Bounded endpoint qualification after the current SitStand GPU1 worker exits."""
from pathlib import Path
import hashlib,json,os,runpy,sys,time,traceback
R=Path(__file__).resolve().parent;STATUS=R/'groundpick-executable-endpoint-status-005.json'
assert not STATUS.exists()
def read(p):return json.loads(p.read_text())
def state(**kw):
    t=STATUS.with_suffix('.tmp');t.write_text(json.dumps(dict(pid=os.getpid(),updated_unix=time.time(),formal_controls=0,**kw),indent=2)+'\n');t.replace(STATUS)
state(status='QUEUED_WAITING_VERIFIED_SITSTAND_EXIT',waiting_pid=356135)
try:
    while Path('/proc/356135/cmdline').exists():
        assert b'run-sitstand-history-hold320-001.py' in Path('/proc/356135/cmdline').read_bytes()
        time.sleep(2)
    terminal=read(R/'sitstand-history-hold320-pipeline-status.json')
    assert terminal['status']=='SITSTAND_HOLD320_COMPLETE_NO_AUTOMATIC_THIRD_STAGE',terminal
    plan=read(R/'groundpick-executable-endpoint-plan-005.json');script=Path(plan['script'])
    assert hashlib.sha256(script.read_bytes()).hexdigest()==plan['script_sha256']
    assert hashlib.sha256(Path(plan['candidate']).read_bytes()).hexdigest()==plan['candidate_sha256']
    out=R/'groundpick-executable-endpoint-hold-005'
    state(status='DIAGNOSTIC_INITIALIZING',output=str(out),diagnostic_control_cap=9600)
    sys.argv=[str(script),'--output',str(out),'--device','cuda:1'];runpy.run_path(str(script),run_name='__main__')
    report=read(out/'report.json');assert report['diagnostic_controls']==9600 and report['original_dynamic_gate_unchanged']
    state(status='ONE_EXECUTABLE_ENDPOINT_DIAGNOSTIC_COMPLETE',diagnostic_controls=9600,report=str(out/'report.json'),results=report['results'])
except BaseException as exc:
    state(status='STOPPED_REQUIRES_REVIEW',error=repr(exc));traceback.print_exc();raise
