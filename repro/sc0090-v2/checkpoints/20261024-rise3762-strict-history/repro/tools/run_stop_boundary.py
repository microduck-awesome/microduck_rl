"""Exercise the reviewed original update-12 stop in an isolated 64-env GPU run."""
from pathlib import Path
import fcntl, hashlib, json, os, subprocess, time
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 review=json.loads((ROOT/'document-review.json').read_text())
 assert review['status']=='PASS_FOR_QUALIFICATION_ONLY'
 for name,h in review['locked_files'].items():assert sha(ROOT/name)==h,name
 plan=json.loads((ROOT/'execution-plan.json').read_text());row=plan['phases'][0]
 q=Path(plan['qualification_root']);out=q/'update12-boundary';assert not out.exists()
 lock=open(q/'process.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 report=q/row['name']/'smoke64/report.json';smoke=json.loads(report.read_text())
 assert smoke['status']=='PASS' and smoke['iterations']==5 and smoke['envs']==64
 assert smoke['robot_xml_sha256']=='8ae0d89041950d565bbcd6272ab87317f46132ff1eef99b922782ecbe2b95308'
 args=row['launch_argv'].copy()
 for key,value in [('--output',str(out)),('--preflight',str(report)),('--event-report',str(out/'historical-stop-event.json'))]:args[args.index(key)+1]=value
 args.extend(['--num-envs','64'])
 args.append('--validation-only')
 assert ('--iterations' not in args or args[args.index('--iterations')+1]=='1000') and '--resume' not in args
 env=os.environ.copy();contract=plan['environment_contract']
 for key in contract['unset']:env.pop(key,None)
 for key in ('SC0090_CUDA_COMPAT_DIR','SC0090_V2_PYTHON'):env[key]=contract[key]
 argv=['bash',row['wrapper'],*args]
 receipt=dict(status='RUNNING',pid=os.getpid(),argv=argv,started_unix=time.time(),formal_weights_used=False)
 path=q/'update12-boundary-process.json';path.write_text(json.dumps(receipt,indent=2)+'\n')
 with (q/'update12-boundary.log').open('w') as log:
  result=subprocess.run(argv,cwd=row['source'],env=env,stdout=log,stderr=subprocess.STDOUT)
 receipt.update(status='EXITED',returncode=result.returncode,ended_unix=time.time());path.write_text(json.dumps(receipt,indent=2)+'\n')
 assert result.returncode==0
 status=json.loads((out/'status.json').read_text());event=json.loads((out/'historical-stop-event.json').read_text())
 assert status['status']=='STOPPED' and status['completed_updates']==12
 assert event['before_completed_update']==12 and not list(out.glob('evaluation*'))
 import torch
 checkpoint=out/'checkpoint_000012.pt';d=torch.load(checkpoint,map_location='cpu',weights_only=False)
 assert d['infos']['v2']['completed_updates']==12 and d['infos']['env_state']['common_step_counter']==288
 assert len(d['infos']['v2']['cuda_rng'])==2
 assert not any(int(p.stem.rsplit('_',1)[1])>12 for p in out.glob('checkpoint_*.pt'))
 receipt.update(status='PASS',completed_updates=12,common_step_counter=288,no_evaluation=True,no_update13=True,checkpoint_sha256=sha(checkpoint),historical_entry_sha256=sha(ROOT/'tools/historical_entry.py'),document_sha256=sha(ROOT/'STRICT_3762_SINGLE_GPU.md'))
 path.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
