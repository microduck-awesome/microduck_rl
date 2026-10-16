import json,os,subprocess,time,signal
from pathlib import Path
r=Path(__file__).resolve().parent;w=r/'ballkick-repaired-worktree'
env=dict(os.environ,SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat');env.pop('MICRODUCK_WARM_START',None)
patterns=('ccd overflow','contact overflow','constraint buffer full','broadphase overflow','narrowphase overflow','height field collision overflow','nefc overflow','njmax_nnz overflow')
def state(**kw):
 p=r/'ballkick-admission-status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(updated_unix=time.time(),**kw),indent=2)+'\n');t.replace(p)
def run(args,name):
 log=r/(name+'.log');assert not log.exists();start=time.monotonic()
 with log.open('w') as f:
  child=subprocess.Popen(['scripts/v2_python.sh',*args],cwd=w,env=env,stdout=f,stderr=subprocess.STDOUT);state(status='VALIDATING',phase=name,pid=child.pid,log=str(log));print(name,child.pid,flush=True);bad=False
  while child.poll() is None:
   if not bad and any(x in log.read_text().lower() for x in patterns):bad=True;child.send_signal(signal.SIGTERM)
   time.sleep(2)
  assert child.returncode==0 and not bad,(name,child.returncode,bad)
 report=json.loads((r/name/'report.json').read_text());(r/name/'outer-process-wall.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-start,allocation_kind='validation'),indent=2)+'\n');return report
try:
 smoke=json.loads((r/'ballkick-smoke64-001/report.json').read_text());assert smoke['status']=='ALLOCATION_COMPLETE' and smoke['completed_updates']==5 and smoke['terminal_bootstrap_samples']>=32 and smoke['onnx_max_abs_error']<2e-5
 assert json.loads((r/'ballkick-terminal-001.json').read_text())['status']=='PASS'
 assert smoke['training_metrics']['metrics']['feedback_not_ready_fraction']==smoke['training_metrics']['metrics']['nonfinite_fraction']==0
 actual=run(['scripts/train_v2_ballkick.py','--output',str(r/'ballkick-actual16384-001'),'--num-envs','16384','--iterations','16','--validation-only','--force-timeouts'],'ballkick-actual16384-001')
 assert actual['completed_updates']==16 and actual['terminal_bootstrap_samples']>=8192 and actual['source_sha256']==smoke['source_sha256']
 assert actual['training_metrics']['metrics']['feedback_not_ready_fraction']==actual['training_metrics']['metrics']['nonfinite_fraction']==0
 assert max(actual['training_metrics']['oracle_max_abs'])<2e-5
 for stage in ('nominal','measured'):
  name='ballkick-evaluator-smoke-0005-'+stage
  a=run(['scripts/evaluate_v2_ballkick.py','--checkpoint',str(r/'ballkick-smoke64-001/checkpoint_000005.pt'),'--output',str(r/name),'--stage',stage,'--num-envs','32'],name)
  assert a['actor_normalizer_unchanged'] and a['trials']==32 and a['nonfinite_trials']==a['feedback_fault_trials']==0 and max(a['oracle_max_abs'])<2e-5
 state(status='QUALIFICATION_REVIEW_READY',num_envs=16384,throughput=actual['throughput'],formal_controls=0)
except BaseException as exc:
 state(status='ADMISSION_STOPPED_REQUIRES_REVIEW',error=repr(exc),formal_controls=0);raise
