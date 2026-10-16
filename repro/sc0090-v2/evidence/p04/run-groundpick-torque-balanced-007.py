"""One small concurrent GPU0 endpoint diagnostic; formal jobs keep running."""
from pathlib import Path
import hashlib,json,os,runpy,sys,time,traceback
R=Path(__file__).resolve().parent;S=R/'groundpick-torque-balanced-hold-status-007.json';assert not S.exists()
def read(p):return json.loads(p.read_text())
def state(**kw):
    t=S.with_suffix('.tmp');t.write_text(json.dumps(dict(pid=os.getpid(),updated_unix=time.time(),formal_controls=0,**kw),indent=2)+'\n');t.replace(S)
plan=read(R/'groundpick-torque-balanced-hold-plan-007.json');script=Path(plan['source_script'])
before=read(R/'roulade-history-formal-001/status.json');assert before['status']=='RUNNING'
assert b'run-roulade-history-001.py' in Path(f'/proc/{before["pid"]}/cmdline').read_bytes()
assert hashlib.sha256(script.read_bytes()).hexdigest()==plan['source_sha256']
assert hashlib.sha256(Path(plan['candidate']).read_bytes()).hexdigest()==plan['candidate_sha256']
state(status='INITIALIZING_SMALL_CONCURRENT_PROBE',shared_gpu0_roulade_before=before,diagnostic_control_cap=9600)
started=time.monotonic()
try:
    out=R/'groundpick-torque-balanced-hold-007'
    sys.argv=[str(script),'--output',str(out),'--device','cuda:0'];runpy.run_path(str(script),run_name='__main__')
    report=read(out/'report.json');assert report['diagnostic_controls']==9600 and report['original_dynamic_gate_unchanged']
    after=read(R/'roulade-history-formal-001/status.json')
    state(status='ONE_TORQUE_BALANCED_HOLD_DIAGNOSTIC_COMPLETE',diagnostic_controls=9600,wall_seconds=time.monotonic()-started,
        report=str(out/'report.json'),results=report['results'],shared_gpu0_roulade_after=after,
        resource_note='FormalRoulade stayedlive; this interval is shared-GPU performance, not exclusive benchmark.')
except BaseException as exc:
    state(status='STOPPED_REQUIRES_REVIEW',error=repr(exc),wall_seconds=time.monotonic()-started);traceback.print_exc();raise
