import json,hashlib,os,subprocess,time,signal
from pathlib import Path
r=Path(__file__).resolve().parent;w=r/'roulade-repaired-prep-worktree';env=dict(os.environ,SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat');env.pop('MICRODUCK_WARM_START',None)
patterns=('ccd overflow','contact overflow','constraint buffer full','broadphase overflow','narrowphase overflow','height field collision overflow','nefc overflow','njmax_nnz overflow')
qualification=r/'roulade-qualification-001.json'
def state(**kw):
 p=r/'roulade-formal-status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(updated_unix=time.time(),**kw),indent=2)+'\n');t.replace(p)
def run(args,name,formal=False):
 log=r/(name+'.log');start=time.monotonic()
 with log.open('w') as f:
  child=subprocess.Popen(['scripts/v2_python.sh',*args],cwd=w,env=env,stdout=f,stderr=subprocess.STDOUT);state(status='RUNNING',phase=name,pid=child.pid,log=str(log));print(name,child.pid,flush=True);stop=None
  while child.poll() is None:
   text=log.read_text().lower()
   if not stop and any(x in text for x in patterns):stop='capacity_log'
   s=r/name/'status.json'
   if formal and not stop and s.exists():
    try:
     m=json.loads(s.read_text())['training_metrics']['metrics']
     if m['feedback_invalid_fraction']>0:stop='new_feedback_event_requires_attribution'
     elif m['nonfinite_fraction']>0:stop='nonfinite_physics'
    except (KeyError,json.JSONDecodeError):pass
   if stop and child.poll() is None:
    child.send_signal(signal.SIGTERM);state(status='SAFE_STOP_REQUESTED',phase=name,pid=child.pid,reason=stop)
    child.wait();break
   time.sleep(2)
  assert child.returncode==0 and stop is None,(name,child.returncode,stop)
 report=json.loads((r/name/'report.json').read_text());wall=time.monotonic()-start
 (r/name/'outer-process-wall.json').write_text(json.dumps(dict(wall_seconds=wall,allocation_kind='formal' if formal else 'evaluation',controls=report.get('allocation_controls',0),control_per_second=report.get('allocation_controls',0)/wall),indent=2)+'\n');return report
state(status='WAITING_SIGNED_QUALIFICATION',qualification=str(qualification),formal_updates=0)
while not qualification.exists():time.sleep(2)
q=json.loads(qualification.read_text());assert q['status']=='PASS' and q['num_envs']==16384 and q['formal_allocation_updates']==128 and q['stage_max_controls']==100663296
assert (r/'ballkick-physical-001-process.json').exists() and not Path('/proc/4081362').exists()
for rel,sha in q['source_sha256'].items():assert hashlib.sha256((w/rel).read_bytes()).hexdigest()==sha,rel
for rel,sha in q.get('model_asset_sha256',{}).items():assert hashlib.sha256((w/rel).read_bytes()).hexdigest()==sha,rel
out=r/'roulade-allocation-001'
a=run(['scripts/train_v2_roulade.py','--output',str(out),'--device','cuda:0','--num-envs','16384','--iterations','128','--qualification',str(qualification)],'roulade-allocation-001',True)
assert a['completed_updates']==128 and a['allocation_controls']==50331648 and a['fresh_initialization']
for stage in ('nominal','measured'):
 name='roulade-0128-'+stage
 run(['scripts/evaluate_v2_roulade.py','--checkpoint',str(out/'checkpoint_000128.pt'),'--output',str(r/name),'--device','cuda:0','--stage',stage],name)
for rel,sha in q.get('model_asset_sha256',{}).items():assert hashlib.sha256((w/rel).read_bytes()).hexdigest()==sha,rel
state(status='EVALUATION_COMPLETE_DECISION_REQUIRED',formal_updates=128,formal_controls=50331648,stage_max_controls=100663296,scope='No automatic second recipe or allocation; standing-origin full roll, quiet landing/head, and measured condition decide within total budget')
