import sys,torch,json
from pathlib import Path
p=Path(sys.argv[1]);a=torch.load(p/'rank0/resume_000005.pt',weights_only=False,map_location='cpu');b=torch.load(p/'rank1/resume_000005.pt',weights_only=False,map_location='cpu')
for key in ['actor_state_dict','critic_state_dict']:
 for name,x in a[key].items():
  assert torch.equal(x,b[key][name]),(key,name,(x-b[key][name]).abs().max())
assert a['infos']['v2']['recovery']['stage']==b['infos']['v2']['recovery']['stage']
print(json.dumps(dict(status='PASS',actor_critic_and_normalizers_exact=True,shared_curriculum=True)))
