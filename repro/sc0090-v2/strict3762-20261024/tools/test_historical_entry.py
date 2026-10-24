"""CPU-only control-flow proof: update 12 completes, no update13 or evaluation is run."""
from pathlib import Path
import json,random,types
from historical_entry import boundary_learn
class StopRequested(Exception):pass
class Runner:
 def __init__(self):
  self.stopping=False;self.completed=0;self.values=[];self.rng=random.Random(20261015);self.alg=types.SimpleNamespace(update=self.checked_update)
 def checked_update(self):
  # Mirror original trainer order: optimizer -> counter -> stop check.
  self.values.append(self.rng.random());self.completed+=1
  if self.stopping:raise StopRequested()
 def learn(self,amount):
  for _ in range(amount):self.alg.update()
r=Runner();events=[]
def stop(count):events.append(count);r.stopping=True
learn=boundary_learn(Runner.learn,stop,12)
try:learn(r,500)
except StopRequested:pass
else:raise AssertionError('Missing stop')
reference=random.Random(20261015);assert r.values==[reference.random()for _ in range(12)]
assert r.completed==12 and events==[12] and r.alg.update==r.checked_update
report=dict(status='PASS',cpu_control_flow_only=True,completed_updates=12,update13_executed=False,optimizer_call_sequence_and_rng_preserved=True,original_checked_update_restored=True,real_environment_boundary_qualification_pending=True)
root=Path(__file__).resolve().parents[1];(root/'stop-boundary-unit-check.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
