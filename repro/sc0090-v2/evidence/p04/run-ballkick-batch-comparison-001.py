import json,os,subprocess,time,signal
from pathlib import Path
r=Path(__file__).resolve().parent;w=r/'ballkick-repaired-worktree';status=r/'ballkick-batch-comparison-status.json'
def state(**kw):
 p=status.with_suffix('.tmp');p.write_text(json.dumps(dict(updated_unix=time.time(),**kw),indent=2)+'\n');p.replace(status)
state(status='WAITING_SERIAL_EVALUATOR_ADMISSION',formal_controls=0)
start=time.monotonic()
while True:
 a=json.loads((r/'ballkick-admission-status.json').read_text())
 if a['status']=='QUALIFICATION_REVIEW_READY':break
 if a['status']=='ADMISSION_STOPPED_REQUIRES_REVIEW':raise RuntimeError(a)
 if time.monotonic()-start>1200:raise TimeoutError('Existing evaluation admission did not finish; no GPU overlap allowed')
 time.sleep(2)
old=json.loads((r/'ballkick-actual16384-001/report.json').read_text());assert old['throughput']['aggregate_control_per_second']<100000
name='ballkick-actual24576-001';log=r/(name+'.log');assert not log.exists();started=time.monotonic();env=dict(os.environ,SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat');env.pop('MICRODUCK_WARM_START',None)
with log.open('w') as f:
 child=subprocess.Popen(['scripts/v2_python.sh','scripts/train_v2_ballkick.py','--output',str(r/name),'--num-envs','24576','--iterations','16','--validation-only','--force-timeouts'],cwd=w,env=env,stdout=f,stderr=subprocess.STDOUT);state(status='VALIDATING_SINGLE_BATCH_COMPARISON',pid=child.pid,num_envs=24576);print(name,child.pid,flush=True);bad=False
 while child.poll() is None:
  if not bad and any(t in log.read_text().lower() for t in ('ccd overflow','contact overflow','constraint buffer full','broadphase overflow','narrowphase overflow','nefc overflow','njmax_nnz overflow')):bad=True;child.send_signal(signal.SIGTERM)
  time.sleep(2)
 assert child.returncode==0 and not bad,(child.returncode,bad)
a=json.loads((r/name/'report.json').read_text());(r/name/'outer-process-wall.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-started,allocation_kind='validation'),indent=2)+'\n')
assert a['completed_updates']==16 and a['terminal_bootstrap_samples']>=12288 and a['source_sha256']==old['source_sha256']
assert a['training_metrics']['metrics']['feedback_not_ready_fraction']==a['training_metrics']['metrics']['nonfinite_fraction']==0
assert max(a['training_metrics']['oracle_max_abs'])<2e-5
state(status='QUALIFICATION_REVIEW_READY',num_envs=24576,throughput=a['throughput'],formal_controls=0,budget='same total100663296; first85 updates at24576 or128 at16384, no further batch search')
