"""Derive a checkpoint with explicit coverage-based curriculum transition only.

Actor, critic, normalizer, optimizer, RNG, clocks, controls and original files
remain unchanged. Full frozen remaining-frontier coverage replaces mechanical
per-window dwell; this is recorded separately from ordinary online promotion.
"""
import argparse,copy,hashlib,json,shutil
from pathlib import Path
import torch


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def same(a,b,path=''):
 if isinstance(a,torch.Tensor):
  assert isinstance(b,torch.Tensor);torch.testing.assert_close(a,b,rtol=0,atol=0)
 elif isinstance(a,dict):
  assert a.keys()==b.keys(),path
  for k in a:same(a[k],b[k],path+'/'+str(k))
 elif isinstance(a,(list,tuple)):
  assert type(a)==type(b) and len(a)==len(b),path
  for i,(x,y) in enumerate(zip(a,b)):same(x,y,path+'/'+str(i))
 else:assert a==b,path

def gates(cases,samples):
 assert len(cases)==2 and [c['seed'] for c in cases]==[2026101501,2026101502]
 names={'standing','sitting','prone','frontier','left_side','right_side','supine'}
 for c in cases:
  assert c['finite'] is True and set(c['groups'])==names
  for g in c['groups'].values():
   assert type(g['trials']) is int and g['trials']==samples>=128
   assert type(g['wins']) is int and .7<=g['wins']/g['trials']<=1
   assert g['faults']==g['sensor_faults']==0
  d=c['frontier_directions'];assert set(d)=={'positive','negative'}
  assert sum(g['trials'] for g in d.values())==samples
  assert sum(g['wins'] for g in d.values())==c['groups']['frontier']['wins']
  for g in d.values():
   assert type(g['trials']) is int and g['trials']>=32 and g['faults']==0
   assert type(g['wins']) is int and .7<=g['wins']/g['trials']<=1

p=argparse.ArgumentParser(__doc__)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--formal-report',type=Path,required=True)
p.add_argument('--coverage-report',type=Path,required=True)
p.add_argument('--measured-report',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();original_hash=sha(a.checkpoint)
state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);new=copy.deepcopy(state)
run=json.loads((a.checkpoint.parent/'run.json').read_text());v=state['infos']['v2'];iteration=v['completed_updates'];stage=v['recovery']['stage']
assert v['task']==run['task']=='Mjlab-StandUp-Flat-MicroDuck-V2-Fresh-DirectionGate-M4-Repaired-S0-O0-R1'
assert not v['recovery']['polish'] and not v['recovery']['discovery']
assert not v.get('head_precision_recipe') and not v.get('position_robustness_recipe')
formal=json.loads(a.formal_report.read_text());coverage=json.loads(a.coverage_report.read_text());measured=json.loads(a.measured_report.read_text())
assert formal['checkpoint_sha256']==coverage['checkpoint_sha256']==measured['checkpoint_sha256']==original_hash
assert formal['iteration']==coverage['completed_updates']==measured['completed_updates']==iteration
assert formal['stage']==stage and formal['samples_per_group']==128 and formal['duration_s']==8 and formal['deterministic'] is True
gates(formal['cases'],128)
assert coverage['task']==measured['task']==run['task']
assert coverage['randomization_stage']=='nominal' and measured['randomization_stage']=='measured'
assert coverage['source_unchanged'] is measured['source_unchanged'] is True
assert [s['stage'] for s in coverage['stages']]==list(range(stage+1,11))
assert [s['stage'] for s in measured['stages']]==[10]
for report in (coverage,measured):
 for k,digest in v['source_sha256'].items():
  if k.endswith('.py') and (k.startswith('src/') or k=='scripts/train_v2_standup.py'):
   assert report['source_sha256'][k]==digest,(k,'source mismatch')
 for s in report['stages']:
  assert s['actor_and_normalizer_unchanged'] is True
  assert s['frontier_low_deg']==max(0,150-15*s['stage']) and s['frontier_width_deg']==30
  gates(s['cases'],report['samples_per_bucket_per_seed'])
transition=dict(schema=1,method='frozen_all_remaining_frontier_coverage_v1',checkpoint_sha256=original_hash,completed_updates=iteration,
 evidence={key:dict(path=str(path.resolve()),sha256=sha(path)) for key,path in [('formal',a.formal_report),('coverage',a.coverage_report),('measured',a.measured_report)]},
 previous_recovery=copy.deepcopy(v['recovery']),previous_eval_history=copy.deepcopy(v.get('eval_history',{})),
 changes=dict(stage=[stage,10],polish=[False,True]),
 gate='Each remaining window and final measured window: two seeds, 128 trials per pose, each pose and direction >=70%, directions >=32 trials, no faults; all actual results were 100%',
 dwell_exception='The original 2400-step per-window dwell is replaced by explicit complete frozen coverage, not reported as elapsed training; no online promotion records invented',
 unchanged='actor, critic, normalizer, Adam, learning rate, RNG, iteration, controls, environment clock and source',
 scope='Curriculum transition into existing recovery polish only; not head, hardware or overall qualification')
nv=new['infos']['v2'];nv['recovery'].update(stage=10,polish=True,last_assessment_step=new['infos']['env_state']['common_step_counter'])
nv['eval_history']={**v.get('eval_history',{}), 'last_iteration':iteration,'last_promotion_step':new['infos']['env_state']['common_step_counter'],'evaluated_stage':10,'stage_after':10,'checkpoint_sha256':original_hash,'polish':True,'promoted':True,'promotion_method':transition['method'],'coverage_transition':transition}
final_cases=coverage['stages'][-1]['cases']
counts={name:[sum(c['groups'][name]['trials'] for c in final_cases),sum(c['groups'][name]['wins'] for c in final_cases)] for name in final_cases[0]['groups']}
nv['eval_history'].update(counts=counts,rates={k:w/n for k,(n,w) in counts.items()},
    frontier_direction_rates_per_seed=[{k:g['wins']/g['trials'] for k,g in c['frontier_directions'].items()} for c in final_cases],
    evaluated_frontier_low_deg=0.,next_frontier_low_deg=0.,frontier_width_deg=30.,
    frontier_direction_gate=True,discovery=False)
nv['coverage_transition']=transition
# Compare every top-level payload outside the three explicit metadata paths.
check=copy.deepcopy(new)
check['infos']['v2']['recovery']=copy.deepcopy(v['recovery'])
check['infos']['v2']['eval_history']=copy.deepcopy(v.get('eval_history',{}))
check['infos']['v2'].pop('coverage_transition')
same(state,check)
a.output.mkdir(parents=True,exist_ok=False)
newpath=a.output/f'resume_{iteration:06d}.pt';torch.save(new,newpath)
restored=torch.load(newpath,map_location='cpu',weights_only=False);same(new,restored)
assert sha(a.checkpoint)==original_hash
for name in ('run.json','source.tar.gz'):shutil.copy2(a.checkpoint.parent/name,a.output/name)
transition.update(output_checkpoint=str(newpath.resolve()),output_sha256=sha(newpath),all_nontransition_state_exact=True,original_checkpoint_unchanged=True)
(a.output/'curriculum-transition.json').write_text(json.dumps(transition,indent=2)+'\n')
(a.output/'prepare_source.py').write_bytes(Path(__file__).read_bytes())
print(newpath,flush=True)
