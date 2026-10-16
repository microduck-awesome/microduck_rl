"""One margin-qualified candidate in nominal/measured P04, no policy updates."""
from pathlib import Path
import gc,hashlib,json,os,runpy,sys,time,traceback
R=Path(__file__).resolve().parent;S=R/'groundpick-torque-margin-hold-status-008.json';assert not S.exists()
def read(p):return json.loads(p.read_text())
def state(**kw):
    t=S.with_suffix('.tmp');t.write_text(json.dumps(dict(pid=os.getpid(),updated_unix=time.time(),formal_controls=0,**kw),indent=2)+'\n');t.replace(S)
plan=read(R/'groundpick-torque-margin-hold-plan-008.json');script=Path(plan['source_script'])
assert hashlib.sha256(script.read_bytes()).hexdigest()==plan['source_sha256']
assert hashlib.sha256(Path(plan['candidate']).read_bytes()).hexdigest()==plan['candidate_sha256']
before=read(R/'roulade-history-formal-001/status.json');assert before['status']=='RUNNING'
assert b'run-roulade-history-001.py' in Path(f'/proc/{before["pid"]}/cmdline').read_bytes()
started=time.monotonic();state(status='IMPORTING_SMALL_CONCURRENT_PROBE',shared_gpu0_before=before,diagnostic_control_cap=19200)
try:
    import torch
    reports=[]
    for stage in plan['stages']:
        out=R/f'groundpick-torque-margin-hold-{stage}-008';state(status='DIAGNOSTIC_RUNNING',stage=stage,output=str(out),diagnostic_control_cap=19200)
        sys.argv=[str(script),'--output',str(out),'--device','cuda:0','--stage',stage];runpy.run_path(str(script),run_name='__main__')
        report=read(out/'report.json');assert report['diagnostic_controls']==9600 and report['original_dynamic_gate_unchanged'];reports.append(str(out/'report.json'))
        gc.collect();torch.cuda.empty_cache()
    state(status='ONE_MARGIN_CANDIDATE_TWO_CONDITIONS_COMPLETE',diagnostic_controls=19200,reports=reports,wall_seconds=time.monotonic()-started,
        shared_gpu0_after=read(R/'roulade-history-formal-001/status.json'),resource_note='Roulade continued; performance during this window includes smallGPU0diagnostic contention.')
except BaseException as exc:
    state(status='STOPPED_REQUIRES_REVIEW',error=repr(exc),wall_seconds=time.monotonic()-started);traceback.print_exc();raise
