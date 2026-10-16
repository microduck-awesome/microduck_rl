"""Two fixed-parent reference evaluations after the precision worker releases GPU0."""
import os,sys,time,json,hashlib,gc,traceback,subprocess
from pathlib import Path
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'groundpick-health-history-worktree';sys.path.insert(0,str(W/'scripts'));os.chdir(W)
state=R/'groundpick-precision-reference-status-001.json';assert not state.exists();started=time.time();completed=0
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(**kw):
 p=state.with_suffix('.tmp');p.write_text(json.dumps(dict(pid=os.getpid(),updated_unix=time.time(),diagnostic_cap=307200,completed_diagnostic_controls=completed,new_formal_controls=0,**kw),indent=2)+'\n');p.replace(state)
write(status='WAITING_PRECISION_TRAINING_AND_EVALUATION')
try:
 deadline=time.time()+7200
 while True:
  marker=json.loads((R/'groundpick-target-precision-pipeline-status-002.json').read_text())
  if marker['status']=='GROUNDPICK_TARGET_PRECISION_COMPLETE_NO_AUTOMATIC_CONTINUATION' and not Path('/proc/'+str(marker['pid'])).exists():break
  if marker['status'].startswith('FAILED'):raise RuntimeError('Predecessor failed; no automatic GPU takeover')
  assert time.time()<deadline,'Bounded queue expired'
  time.sleep(2)
 import torch
 import evaluate_v2_ground_pick as evaluator
 torch.set_num_threads(2);torch.cuda.set_device('cuda:0');parent=R/'groundpick-health-history-allocation-001/checkpoint_000512.pt'
 for stage in ('nominal','measured'):
  dest=R/f'groundpick-precision-reference-0512-{stage}';write(status='EVALUATING_FIXED_PARENT',stage=stage,output=str(dest))
  sys.argv=['evaluate','--checkpoint',str(parent),'--output',str(dest),'--device','cuda:0','--stage',stage,'--seed','2026101633','--num-envs','256'];evaluator.main();completed+=153600;gc.collect();torch.cuda.empty_cache()
 write(status='PAIRED_REFERENCE_COMPLETE',parent_sha256=sha(parent),wall_s=time.time()-started)
except BaseException:
 write(status='STOPPED_REQUIRES_REVIEW',exception=traceback.format_exc());raise
