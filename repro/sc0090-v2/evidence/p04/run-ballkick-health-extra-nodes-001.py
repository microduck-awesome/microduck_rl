from pathlib import Path
import os,sys,json,time,hashlib,gc,importlib.util
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'ballkick-health-history-worktree';os.chdir(W);sys.path.insert(0,str(W/'scripts'))
state=R/'ballkick-health-extra-nodes-status-001.json';assert not state.exists()
def write(**kw):
 p=state.with_suffix('.tmp');p.write_text(json.dumps(dict(pid=os.getpid(),updated_unix=time.time(),formal_controls_added=0,diagnostic_cap=256000,**kw),indent=2)+'\n');p.replace(state)
write(status='IMPORTING',diagnostic_controls=0)
assert json.loads((R/'ballkick-health-history-pipeline-status.json').read_text())['status']=='BALLKICK_HEALTH_HISTORY_COMPLETE_NO_AUTOMATIC_CONTINUATION'
assert not Path('/proc/397198').exists()
import torch
import evaluate_v2_ballkick as evaluator
controls=0;reports=[]
try:
 for node in (192,320):
  for stage in ('nominal','measured'):
   out=R/f'ballkick-health-history-{node:04d}-{stage}';cp=R/f'ballkick-health-history-allocation-001/checkpoint_{node:06d}.pt';assert cp.exists()and not out.exists()
   write(status='EVALUATING',node=node,stage=stage,output=str(out),diagnostic_controls=controls)
   sys.argv=['evaluate','--checkpoint',str(cp),'--output',str(out),'--device','cuda:1','--stage',stage,'--num-envs','256','--seed','2026101633'];evaluator.main()
   controls+=256*250;reports.append(str(out/'report.json'));gc.collect();torch.cuda.empty_cache()
 assert controls==256000
 write(status='BALLKICK_EXTRA_NODES_EVALUATED',diagnostic_controls=controls,reports=reports,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
except BaseException as e:write(status='FAILED_REQUIRES_DIAGNOSIS',error=repr(e),diagnostic_controls=controls);raise
