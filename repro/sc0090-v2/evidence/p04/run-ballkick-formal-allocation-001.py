import json,hashlib,os,subprocess,time,signal
from pathlib import Path
r=Path(__file__).resolve().parent;w=r/'ballkick-repaired-worktree';qp=r/'ballkick-qualification-001.json';q=json.loads(qp.read_text());assert q['status']=='PASS'
env=dict(os.environ,SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat');env.pop('MICRODUCK_WARM_START',None)
patterns=('ccd overflow','contact overflow','constraint buffer full','broadphase overflow','narrowphase overflow','height field collision overflow','nefc overflow','njmax_nnz overflow')
def state(**kw):
 p=r/'ballkick-formal-status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(updated_unix=time.time(),**kw),indent=2)+'\n');t.replace(p)
def verify():
 for rel,sha in {**q['source_sha256'],**q['model_asset_sha256']}.items():assert hashlib.sha256((w/rel).read_bytes()).hexdigest()==sha,rel
 assert hashlib.sha256((w/'scripts/evaluate_v2_ballkick.py').read_bytes()).hexdigest()==q['evaluator_script_sha256']
def run(args,name,formal=False):
 log=r/(name+'.log');assert not log.exists();start=time.monotonic()
 with log.open('w') as f:
  child=subprocess.Popen(['scripts/v2_python.sh',*args],cwd=w,env=env,stdout=f,stderr=subprocess.STDOUT);state(status='RUNNING',phase=name,pid=child.pid,log=str(log));print(name,child.pid,flush=True);bad=False
  while child.poll() is None:
   if not bad and any(x in log.read_text().lower() for x in patterns):bad=True;child.send_signal(signal.SIGTERM)
   time.sleep(2)
  assert child.returncode==0 and not bad,(name,child.returncode,bad)
 a=json.loads((r/name/'report.json').read_text());wall=time.monotonic()-start
 (r/name/'outer-process-wall.json').write_text(json.dumps(dict(wall_seconds=wall,allocation_kind='formal' if formal else 'evaluation',controls=a.get('allocation_controls',0),control_per_second=a.get('allocation_controls',0)/wall),indent=2)+'\n');return a
try:
 verify();n=q['num_envs'];updates=q['formal_allocation_updates'];out=r/'ballkick-right-allocation-001'
 assert n*24*updates==q['formal_allocation_controls']<=50331648 and q['stage_max_controls']==100663296 and q['resume'] is None
 a=run(['scripts/train_v2_ballkick.py','--output',str(out),'--num-envs',str(n),'--iterations',str(updates),'--qualification',str(qp)],out.name,True)
 assert a['completed_updates']==updates and a['allocation_controls']==q['formal_allocation_controls'] and a['fresh_initialization']
 for stage in ('nominal','measured'):
  name=f'ballkick-right-{updates:04d}-{stage}'
  run(['scripts/evaluate_v2_ballkick.py','--checkpoint',str(out/f'checkpoint_{updates:06d}.pt'),'--output',str(r/name),'--stage',stage],name)
 verify();state(status='FIRST_ALLOCATION_EVALUATED_REVIEW_REQUIRED',formal_controls=a['allocation_controls'],stage_max_controls=q['stage_max_controls'],scope='No automatic second right/left allocation; real kick, body recovery and camera stability decide remaining shared budget')
except BaseException as exc:
 state(status='STOPPED_REQUIRES_REVIEW',error=repr(exc));raise
