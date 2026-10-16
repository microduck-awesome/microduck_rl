"""Read-only, same reward formula across saved Head2 1664/1792 measured trajectories."""
from pathlib import Path
import hashlib,json,numpy as np
base=Path(__file__).resolve().parent
sources={'1664':base/'frozen-head2-1664-head-measured','1792':base/'frozen-head2-1792-head-measured'}
raw={};hashes={}
for arm,p in sources.items():
 report=json.loads((p/'report.json').read_text());hashes[str(p/'report.json')]=hashlib.sha256((p/'report.json').read_bytes()).hexdigest();rows=[]
 for seed in sorted({x['seed'] for x in report['records']}):
  f=p/f'traces-{seed}.npz';hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest();z=np.load(f);records=[x for x in report['records'] if x['seed']==seed];offset=0
  for record in records:
   n=record['trials'];sl=slice(offset,offset+n);offset+=n;cmd=z['command_rad'][150:200,sl];q=z['physical_head_rad'][150:200,sl]
   np.testing.assert_allclose(cmd[0,0],record['target'],atol=1e-7,rtol=0)
   active=np.abs(cmd[0,0])>0;assert np.all((np.abs(cmd)>0)==active)
   if not active.any():continue
   rows.append(dict(name=record['name'],range=record['range'],seed=seed,trials=n,active=active,err=q-cmd,group=f"{record['range']}/{int(active.sum())}-axis",invalid=bool(z['invalid'][150:200,sl].any())))
 raw[arm]=rows
assert [(x['name'],x['seed'],x['trials']) for x in raw['1664']]==[(x['name'],x['seed'],x['trials']) for x in raw['1792']]
outputs={}
for fine in [.08,.04]:
 outputs[str(fine)]={}
 for arm,rows in raw.items():
  grouped={};bycase={}
  for x in rows:
   reward=.5*np.exp(-(x['err']/.5)**2)+.5*np.exp(-(x['err']/fine)**2)
   weighted=reward.copy();active=x['active'];nactive=int(active.sum())
   if nactive==1:weighted[...,active]*=2.;weighted[...,~active]*=2./3.
   metrics=dict(overall=float(reward.mean()),active=float(reward[...,active].mean()),inactive=float(reward[...,~active].mean()) if nactive<4 else None,balanced_candidate=float(weighted.mean()),invalid=x['invalid'])
   grouped.setdefault(x['group'],[]).append(metrics);bycase[f"{x['name']}/{x['seed']}"]=metrics
  outputs[str(fine)][arm]=dict(groups={g:{k:float(np.mean([x[k] for x in xs])) if xs[0][k] is not None else None for k in ['overall','active','inactive','balanced_candidate']} for g,xs in grouped.items()},cases=bycase)
r=dict(schema=1,status='PASS',paired_case_order_and_command_values=True,initial_states_pairing='same protocol/seeds; exact pre-action initial-state tensors were not saved, so no strict initial-state pairing claim',window='last0.5s of the2s commanded step',reward='same broad Gaussian std.5 + narrow Gaussian fine std.08 or.04,50/50 blend,mean4joints; no other training rewards',limitation='read-only standing-response trajectories, not the full PPO reward/gradient population; candidate balancing is counterfactual only',source_sha256=hashes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results=outputs)
(base/'frozen-head2-continuation-active-inactive-audit.json').write_text(json.dumps(r,indent=2)+'\n')
for fine,arms in outputs.items():
 print('same fine',fine)
 for group in ['small/1-axis','current/1-axis','current/2-axis','current/4-axis']:
  print(group,{a:x['groups'][group] for a,x in arms.items()})
