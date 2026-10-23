import json,sys,torch
from pathlib import Path
root=Path(sys.argv[1]); states=[]
for rank in [0,1]:
 p=root/f'rank{rank}'
 status=json.loads((p/'status.json').read_text())
 assert status['status']=='ALLOCATION_COMPLETE' and status['completed_updates']==506,status
 restore=json.loads((p/'resume-verification.json').read_text());assert restore['actor_critic_normalizers_optimizer_exact']
 for i in (502,504,506):assert (p/f'evaluation_{i:06d}.json').exists()
 states.append(torch.load(p/'resume_000506.pt',weights_only=False,map_location='cpu'))
def equal(a,b):
 if isinstance(a,torch.Tensor):assert torch.equal(a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(tuple,list)):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b
for key in ('actor_state_dict','critic_state_dict','optimizer_state_dict'):equal(states[0][key],states[1][key])
assert states[0]['infos']['v2']['recovery']['stage']==states[1]['infos']['v2']['recovery']['stage']
report=dict(status='PASS',restored_update=500,final_update=506,assessment_boundaries=[502,504,506],both_ranks_exact_restoration=True,post_training_actor_critic_normalizers_optimizer_equal=True,validation_only=True)
(root/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
