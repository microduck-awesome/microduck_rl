"""Inspect original checkpoint metadata on CPU without constructing an environment."""
from pathlib import Path
import json,hashlib,torch,importlib.metadata
ROOT=Path(__file__).resolve().parents[1]
inv=json.loads((ROOT/'history-inventory.json').read_text());rows=[]
for phase in inv['phases']:
 name=phase['name'];n=phase['completed_updates'];filename=f'checkpoint_{n:06d}.pt' if phase['index']<2 else f'resume_{n:06d}.pt'
 p=Path(phase['checkpoints'][filename]['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==phase['checkpoints'][filename]['sha256']
 d=torch.load(p,map_location='cpu',weights_only=False);v=d['infos']['v2'];env=d['infos']['env_state']
 rows.append(dict(phase=name,file=filename,sha256=phase['checkpoints'][filename]['sha256'],checkpoint_iter=d.get('iter'),completed_updates=v['completed_updates'],common_step_counter=env.get('common_step_counter'),recovery=v['recovery'],eval_history=v.get('eval_history'),head_precision_recipe=v.get('head_precision_recipe'),physics_resume=v.get('physics_resume'),torch_rng_sha256=hashlib.sha256(v['torch_rng'].cpu().numpy().tobytes()).hexdigest(),cuda_rng_states=len(v['cuda_rng']),cuda_rng_sha256=[hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()for x in v['cuda_rng']],optimizer_learning_rates=[g['lr']for g in d['optimizer_state_dict']['param_groups']],terminal_bootstrap=d.get('terminal_critic_bootstrap')))
 actual={k:importlib.metadata.version(k)for k in phase['dependencies']};assert actual==phase['dependencies'],(name,actual,phase['dependencies'])
(ROOT/'checkpoint-history.json').write_text(json.dumps(dict(status='PASS',current_dependency_versions=actual,checkpoints=rows),indent=2,default=str)+'\n')
for x in rows:print(x['phase'],x['file'],x['completed_updates'],x['common_step_counter'],x['recovery']['stage'],x['recovery']['polish'],x['cuda_rng_states'])
