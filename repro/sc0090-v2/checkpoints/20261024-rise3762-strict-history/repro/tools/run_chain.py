"""Wait for the existing qualification process, then run the reviewed ten allocations."""
from pathlib import Path
import argparse, fcntl, hashlib, json, os, signal, subprocess, time
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def identity(pid):
 try:
  data=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
  return None if data[0]=='Z' else data[19]
 except FileNotFoundError:return None

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--qualification-pid',type=int,required=True);parser.add_argument('--qualification-start-ticks',required=True);a=parser.parse_args()
 plan=read(ROOT/'execution-plan.json');base=Path(plan['formal_root']).parent
 lock=open(base/'chain.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 path=base/'chain-status.json';assert not path.exists(),'Never overwrite an earlier chain attempt'
 state=dict(status='WAITING_FOR_EXISTING_QUALIFICATION',pid=os.getpid(),qualification_pid=a.qualification_pid,qualification_start_ticks=a.qualification_start_ticks,started_unix=time.time(),completed_phases=[])
 def save():
  state['updated_unix']=time.time();tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(path)
 child=None;stopping=False
 def stop(*_):
  nonlocal stopping
  stopping=True
  if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 try:
  assert identity(a.qualification_pid)==a.qualification_start_ticks,'Wait target is no longer the identified qualification process'
  save()
  while identity(a.qualification_pid)==a.qualification_start_ticks:
   if stopping:raise InterruptedError('Stop requested while awaiting qualification')
   time.sleep(10)
  review=read(ROOT/'document-review.json');assert review['status']=='PASS_FOR_QUALIFICATION_ONLY'
  assert str(Path(__file__).relative_to(ROOT)) in review['locked_files']
  for name,h in review['locked_files'].items():assert sha(ROOT/name)==h,name
  q=Path(plan['qualification_root'])
  for row in plan['phases']:
   for kind,directory in [('smoke','smoke64'),('batch','batch')]:
    record=read(q/row['name']/f'{kind}-process.json');report=q/row['name']/directory/'report.json'
    assert record['status']=='PASS' and record['report_sha256']==sha(report),(row['name'],kind)
    assert read(report)['status']=='PASS'
  assert read(q/'update12-boundary-process.json')['status']=='PASS'
  assert not Path(plan['formal_root']).exists() or not list(Path(plan['formal_root']).glob('*/status.json')),'Unexpected existing formal state'
  state.update(status='ALL_QUALIFICATIONS_PASSED',document_review_sha256=sha(ROOT/'document-review.json'));save()
  python=plan['environment_contract']['SC0090_V2_PYTHON']
  for row in plan['phases']:
   if stopping:raise InterruptedError('Stop requested before next phase')
   state.update(status='FORMAL_PHASE_RUNNING',current_phase=row['name']);save()
   print('FORMAL_PHASE_START',row['index'],row['name'],flush=True)
   child=subprocess.Popen([python,str(ROOT/'tools/run_formal_phase.py'),'--phase',str(row['index'])])
   code=child.wait();child=None
   assert code==0,('Formal phase failed; no automatic retry',row['name'],code)
   record=read(Path(plan['formal_root'])/f'{row["name"]}-process.json');assert record['status']=='PASS'
   state['completed_phases'].append(dict(index=row['index'],name=row['name'],updates=record['completed_updates']));save()
  state.update(status='TRAINING_CHAIN_FINISHED_PENDING_FINAL_AUDIT',completed_updates=3762);save()
 except BaseException as error:
  state.update(status='STOPPED' if stopping else 'FAILED',error=repr(error));save();raise
if __name__=='__main__':main()
