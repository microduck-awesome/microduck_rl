import json,os,time,subprocess,signal
from pathlib import Path
r=Path(__file__).resolve().parent;w=r/'roulade-repaired-prep-worktree';env=dict(os.environ,SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat');env.pop('MICRODUCK_WARM_START',None)
patterns=('ccd overflow','contact overflow','constraint buffer full','broadphase overflow','narrowphase overflow','height field collision overflow','nefc overflow','njmax_nnz overflow')
def state(**kwargs):
 p=r/'roulade-tail-status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(updated_unix=time.time(),**kwargs),indent=2)+'\n');t.replace(p)
def run(args,name):
 log=r/(name+'.log');start=time.monotonic()
 with log.open('w') as f:
  p=subprocess.Popen(['scripts/v2_python.sh',*args],cwd=w,env=env,stdout=f,stderr=subprocess.STDOUT);state(status='VALIDATING',phase=name,pid=p.pid,log=str(log));print(name,p.pid,flush=True);bad=False
  while p.poll() is None:
   if not bad and any(x in log.read_text().lower() for x in patterns):bad=True;p.send_signal(signal.SIGTERM)
   time.sleep(2)
  assert p.returncode==0 and not bad,(name,p.returncode,bad)
 data=json.loads((r/name/'report.json').read_text());(r/name/'outer-process-wall.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-start,allocation_kind='validation'),indent=2)+'\n');return data
state(status='WAITING_ACTUAL16384',pid=4069350)
while Path('/proc/4069350').exists():time.sleep(2)
a=json.loads((r/'roulade-actual16384-001/report.json').read_text());assert a['status']=='ALLOCATION_COMPLETE' and a['completed_updates']==16 and a['stop_reason'] is None and a['terminal_bootstrap_samples']>0
candidates=[dict(num_envs=16384,path=str(r/'roulade-actual16384-001/report.json'),throughput=a['throughput'])]
if a['throughput']['median_control_per_second']<100000:
 b=run(['scripts/train_v2_roulade.py','--output',str(r/'roulade-actual24576-001'),'--device','cuda:0','--num-envs','24576','--iterations','16','--validation-only'],'roulade-actual24576-001');assert b['status']=='ALLOCATION_COMPLETE' and b['completed_updates']==16 and b['terminal_bootstrap_samples']>0
 candidates.append(dict(num_envs=24576,path=str(r/'roulade-actual24576-001/report.json'),throughput=b['throughput']))
(r/'roulade-throughput-candidates.json').write_text(json.dumps(candidates,indent=2)+'\n')
for stage in ('nominal','measured'):
 name='roulade-evaluator-smoke-0005-'+stage
 report=run(['scripts/evaluate_v2_roulade.py','--checkpoint',str(r/'roulade-smoke-002/checkpoint_000005.pt'),'--output',str(r/name),'--device','cuda:0','--stage',stage,'--num-envs','32'],name)
 assert report['actor_normalizer_unchanged'] and report['num_envs']==32 and report['seconds']==5
state(status='QUALIFICATION_REVIEW_READY',candidates=candidates,formal_updates=0,scope='No extra N sweeps. Independent evaluation entry exercised;5updates are not skill qualification. Reviewer must select fixed N/budget and record actual source before formal.')
