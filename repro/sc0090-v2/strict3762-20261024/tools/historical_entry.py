"""Replay the original update-12 SIGTERM boundary without modifying archived training code."""
from pathlib import Path
import argparse,hashlib,json,os,runpy,signal,sys,functools
ROOT=Path(__file__).resolve().parents[1]

def boundary_learn(original,callback,at_update):
 count=0
 @functools.wraps(original)
 def learn(self,*args,**kwargs):
  previous=self.alg.update
  def update(*a,**kw):
   nonlocal count
   count+=1
   if count==at_update:callback(count)
   return previous(*a,**kw)
  self.alg.update=update
  try:return original(self,*args,**kwargs)
  finally:self.alg.update=previous
 return learn

def main():
 p=argparse.ArgumentParser();p.add_argument('--script',type=Path,required=True);p.add_argument('--event-report',type=Path,required=True);p.add_argument('--stop-after',type=int,required=True);p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args();script=a.script.resolve()
 phase=json.loads((ROOT/'history-inventory.json').read_text())['phases'][0]
 assert a.stop_after==12 and hashlib.sha256(script.read_bytes()).hexdigest()==phase['source_files']['scripts/train_v2_standup.py']
 original_signal=signal.signal;installed=[]
 def registered(sig,handler):
  result=original_signal(sig,handler)
  code=getattr(handler,'__code__',None)
  if sig==signal.SIGTERM and code and Path(code.co_filename).resolve()==script and handler.__name__=='stop':
   assert not installed
   from rsl_rl.runners import OnPolicyRunner
   old=OnPolicyRunner.learn
   def stop_at_boundary(count):
    a.event_report.parent.mkdir(parents=True,exist_ok=True)
    a.event_report.write_text(json.dumps(dict(event='historical_SIGTERM_boundary',before_completed_update=count,original_script_sha256=phase['source_files']['scripts/train_v2_standup.py'],entry_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),pid=os.getpid()),indent=2)+'\n')
    os.kill(os.getpid(),signal.SIGTERM)
   OnPolicyRunner.learn=boundary_learn(old,stop_at_boundary,a.stop_after);installed.append((OnPolicyRunner,old));signal.signal=original_signal
  return result
 signal.signal=registered
 try:
  args=a.arguments[1:]if a.arguments[:1]==['--']else a.arguments
  sys.argv=[str(script),*args];sys.path[0]=str(script.parent)
  runpy.run_path(str(script),run_name='__main__')
  assert installed and a.event_report.exists(),'The historical stop boundary was not reached'
 finally:
  signal.signal=original_signal
  for cls,method in installed:cls.learn=method
if __name__=='__main__':main()
