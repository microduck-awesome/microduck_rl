import json,os,subprocess,time,signal
from pathlib import Path
r=Path(__file__).resolve().parent;w=r/'roulade-repaired-prep-worktree';env=dict(os.environ,SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat');env.pop('MICRODUCK_WARM_START',None)
patterns=('ccd overflow','contact overflow','constraint buffer full','broadphase overflow','narrowphase overflow','height field collision overflow','nefc overflow','njmax_nnz overflow')
def state(**kw):
 p=r/'roulade-pipeline-status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(updated_unix=time.time(),**kw),indent=2)+'\n');t.replace(p)
def run(args,name):
 log=r/(name+'.log');start=time.monotonic()
 with log.open('w') as f:
  child=subprocess.Popen(['scripts/v2_python.sh',*args],cwd=w,env=env,stdout=f,stderr=subprocess.STDOUT);state(status='VALIDATING',phase=name,pid=child.pid,log=str(log));print(name,child.pid,flush=True);bad=False
  while child.poll() is None:
   if not bad and any(x in log.read_text().lower() for x in patterns):bad=True;child.send_signal(signal.SIGTERM)
   time.sleep(2)
  assert child.returncode==0 and not bad,(name,child.returncode,bad)
 a=json.loads((r/name/'report.json').read_text());(r/name/'outer-process-wall.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-start,allocation_kind='validation'),indent=2)+'\n');return a
smoke=run(['scripts/train_v2_roulade.py','--output',str(r/'roulade-smoke-003'),'--device','cuda:0','--num-envs','64','--iterations','5','--validation-only','--force-timeouts'],'roulade-smoke-003')
assert smoke['completed_updates']==5 and smoke['terminal_bootstrap_samples']>=32 and smoke['terminal_validation']['status']=='PASS' and smoke['onnx_max_abs_error']<2e-5
# Parent BallKick received the explicit small-N window. Never overlap its probe with benchmark.
marker=r/'ballkick-physical-001-process.json';state(status='WAITING_PARENT_PROBE_COMPLETION',phase='before_actual16384',marker=str(marker))
while not marker.exists() or Path('/proc/4081362').exists():time.sleep(2)
actual=run(['scripts/train_v2_roulade.py','--output',str(r/'roulade-actual16384-002'),'--device','cuda:0','--num-envs','16384','--iterations','16','--validation-only'],'roulade-actual16384-002')
assert actual['completed_updates']==16 and actual['terminal_bootstrap_samples']>0
for stage in ('nominal','measured'):
 name='roulade-evaluator-smoke-0005b-'+stage
 a=run(['scripts/evaluate_v2_roulade.py','--checkpoint',str(r/'roulade-smoke-003/checkpoint_000005.pt'),'--output',str(r/name),'--device','cuda:0','--stage',stage,'--num-envs','32'],name)
 assert a['actor_normalizer_unchanged'] and a['num_envs']==32 and a['seconds']==5
state(status='QUALIFICATION_REVIEW_READY',num_envs=16384,throughput=actual['throughput'],formal_updates=0,scope='Corrected episodic command sampling source; no automatic long training without qualification and budget record')
