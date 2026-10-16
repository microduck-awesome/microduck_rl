"""Original384 checkpoint, fixed command clock, two original12s conditions."""
from pathlib import Path
import gc,json,os,runpy,sys,time
r=Path(__file__).resolve().parent;script=r/'evaluate_v2_sitstand_command_clock_fixed_001.py';status=r/'sitstand-clock-reeval-status.json';assert not status.exists()
def state(**kw):
 p=status.with_suffix('.tmp');p.write_text(json.dumps(dict(pid=os.getpid(),updated_unix=time.time(),formal_controls=0,**kw),indent=2)+'\n');p.replace(status)
try:
 state(status='IMPORTING');ns=runpy.run_path(str(script),run_name='sitstand_clock_eval')
 import torch
 reports=[]
 for stage in ('nominal','measured'):
  out=r/f'sitstand-0384-clock-fixed-{stage}';state(status='EVALUATING',phase=out.name)
  sys.argv=[str(script),'--checkpoint',str(r/'sitstand-lesson4-allocation-001/checkpoint_000384.pt'),'--output',str(out),'--device','cuda:0','--stage',stage,'--num-envs','256']
  start=time.monotonic();ns['main']();d=json.loads((out/'report.json').read_text());assert d['status']=='EVALUATED_NOT_SKILL_ACCEPTANCE'
  (out/'phase-wall.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-start,scope='Per-condition after shared imports, no training'),indent=2)+'\n')
  reports.append(str(out/'report.json'));gc.collect();torch.cuda.empty_cache()
 state(status='COMMAND_CLOCK_REEVALUATION_COMPLETE',reports=reports)
except BaseException as exc:
 state(status='STOPPED_REQUIRES_REVIEW',error=repr(exc));raise
