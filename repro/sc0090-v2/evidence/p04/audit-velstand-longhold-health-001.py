"""Offline audit of all/healthy200Hz camera windows and first walking failures."""
from pathlib import Path
import json,math,runpy,hashlib
import torch
R=Path(__file__).resolve().parent;torch.set_num_threads(2)
cm=runpy.run_path(str(R/'velstand-camera-worktree/src/mjlab_microduck/custom_v2/head_metrics.py'))['camera_metrics']
phases=[('idle',0,50),('move',50,550),('stop',550,700),('reverse',700,1200),('return',1200,1400)]
groups=dict(idle=lambda n,p:n=='idle',forward=lambda n,p:n.startswith('forward') and p in ('move','reverse'),turn=lambda n,p:n.startswith('yaw') and p in ('move','reverse'),walking_head=lambda n,p:n.startswith('walking_head') and p in ('move','reverse'),stop=lambda n,p:n!='idle' and p=='stop',return_to_idle=lambda n,p:n!='idle' and p=='return')
result=dict(status='LONGHOLD_HEALTH_WINDOWS_AND_FIRST_FAILURE_AUDITED',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results={},criteria='Eligible entire phase:height>=.102m,tilt<=20deg,finite,feedback and no prior walking failure; double support not required in motion. Report all and eligible denominators, not a changed survival gate.',timing='First failure at after-reward control boundary; derived geometry retains native5ms lag. Raw camera motion includes intentional turning/head commands.')
for stage in ('nominal','measured'):
 result['results'][stage]={}
 for arm in ('full_bc','leg_bc'):
  d=R/f'velstand-head-bc-longhold-{arm}-{stage}-001'
  report=json.loads((d/'report.json').read_text());items={g:[] for g in groups};failures=[]
  for seed in (2026101612,2026101613):
   tr=torch.load(d/f'trace-{seed}.pt',map_location='cpu',weights_only=False);x=tr['measurements']
   expected=(x['height']<.08)|(x['tilt']>math.pi/4)|~x['finite']|~x['feedback_ready']
   assert torch.equal(expected,x['failed'])
   rows=[row for row in report['records'] if row['seed']==seed]
   for i,row in enumerate(rows):
    sl=slice(i*16,(i+1)*16)
    for phase,a,b in phases:
     mask=((x['height'][a:b,sl]>=.102)&(x['tilt'][a:b,sl]<=math.radians(20))&x['finite'][a:b,sl]&x['feedback_ready'][a:b,sl]).all(0)&~expected[:b,sl].any(0)
     vals=cm(tr['camera'][a*4:b*4+1,sl])
     for name,select in groups.items():
      if select(row['name'],phase):items[name].append((mask,vals))
    for local in torch.where(expected[:,sl].any(0))[0].tolist():
     world=i*16+local;step=int(torch.where(expected[:,world])[0][0]);phase=next(n for n,a,b in phases if a<=step<b)
     failures.append(dict(seed=seed,case=row['name'],trial_in_case=local,control_index=step,boundary_time_s=(step+1)*.02,phase=phase,height=float(x['height'][step,world]),tilt_deg=float(x['tilt'][step,world])*180/math.pi,finite=bool(x['finite'][step,world]),feedback_ready=bool(x['feedback_ready'][step,world]),twist=x['command'][step,world].tolist(),head_command=x['head_command'][step,world].tolist(),linear=x['linear'][step,world].tolist(),angular=x['angular'][step,world].tolist()))
  out={}
  for name,parts in items.items():
   mask=torch.cat([v[0] for v in parts]);values={k:torch.cat([v[1][k] for v in parts]) for k in parts[0][1]}
   out[name]=dict(all_trial_phases=len(mask),eligible_trial_phases=int(mask.sum()),all_metrics={k:float(v.mean()) for k,v in values.items()},eligible_metrics={k:float(v[mask].mean()) for k,v in values.items()} if mask.any() else None)
  result['results'][stage][arm]=dict(groups=out,first_failures=failures,report_sha256=hashlib.sha256((d/'report.json').read_bytes()).hexdigest())
p=R/'velstand-head-bc-longhold-health-001.json';assert not p.exists();p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({s:{a:dict(failures=x['first_failures'],eligible={k:(v['eligible_trial_phases'],v['all_trial_phases']) for k,v in x['groups'].items()}) for a,x in y.items()} for s,y in result['results'].items()},indent=2))
