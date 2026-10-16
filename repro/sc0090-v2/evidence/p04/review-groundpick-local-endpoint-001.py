"""CPU review of the completed bounded endpoint action intervention."""
import json,hashlib,math
from pathlib import Path
import torch
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');torch.set_num_threads(2)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
records=[];loaded={}
for arm in ['policy','local_endpoint']:
 for stage in ['nominal','measured']:
  d=R/f'groundpick-local-endpoint-{arm}-{stage}-001';report=json.loads((d/'report.json').read_text());x=torch.load(d/'traces.pt',map_location='cpu',weights_only=False);m=x['measurements'];loaded[arm,stage]=x
  mask=x['endpoint_active_mask'];phase=torch.remainder(torch.atan2(x['actor_obs_preaction'][:,:,49],x['actor_obs_preaction'][:,:,48])/(2*torch.pi),1.)
  assert bool((~mask|((phase>=.25)&(phase<.425))).all())
  expected=torch.where(mask[:,:,None],x['endpoint_action'],x['policy_actions']) if arm=='local_endpoint' else x['policy_actions'];assert torch.equal(expected,x['executed_actions'])
  assert all(bool(torch.isfinite(v).all())for v in [x['actor_obs_preaction'],x['policy_actions'],x['executed_actions'],x['camera_quat_w']])
  assert x['head_self_found_200Hz'].shape==(2400,64,2) and x['camera_quat_w'].shape==(2401,64,4)
  finite=m['finite']&m['feedback_ready'];clean=finite&~m['head_contact']&~m['self_contact'];feet=m['feet'].all(-1);healthy=clean&feet
  body=healthy&(m['linear'].norm(dim=-1)<.05)&(m['omega']<.5)
  low=body&(m['mouth_height']>=0)&(m['mouth_height']<=.025)&(m['mouth_down_cos']>=.95)&(m['tilt']<math.radians(60))
  stand=body&((m['height']-.112).abs()<.012)&(m['tilt']<math.radians(15))&(m['physical_q'][:,:,[0,1,2,3,4,9,10,11,12,13]].abs().mean(-1)<.2)&(m['physical_q'][:,:,5:9].abs().amax(-1)<.1)
  assert torch.equal(low,m['low_goal'])and torch.equal(stand,m['standing_goal'])
  cycles=[]
  for c in range(3):
   lo,hi=c*200,(c+1)*200;p=m['phase'][lo:hi];at=(low[lo:hi]&(p>=.375)&(p<.425)).unfold(0,5,1).all(-1).any(0);ret=stand[hi-25:hi].all(0);original_clean=healthy[lo:hi].all(0)
   contact200=x['head_self_found_200Hz'][lo*4:hi*4].any(0).any(-1)
   actual=dict(low_goal_sustained_100ms=int(at.sum()),return_stand_tail_500ms=int(ret.sum()),clean_cycle=int(original_clean.sum()),complete_cycle=int((at&ret&original_clean).sum()))
   for k,v in actual.items():assert v==sum(g['cycles'][c][k]for g in report['results']),(arm,stage,c,k)
   actual.update(phase_cycle=c+1,head_self_200Hz_any_trials=int(contact200.sum()),declared_no_head_self_fault_50Hz_trials=int(clean[lo:hi].all(0).sum()),finite_ready_entire_trials=int(finite[lo:hi].all(0).sum()),any_missing_foot_trials=int((~feet[lo:hi]).any(0).sum()),missing_foot_control_frames=int((~feet[lo:hi]).sum()),active_trials=int(mask[lo:hi].any(0).sum()),active_control_frames=int(mask[lo:hi].sum()),lowphase_height_mean_m=float(m['mouth_height'][lo:hi][(p>=.375)&(p<.425)].mean()),additional_complete_no_bad_contact200Hz=int((at&ret&clean[lo:hi].all(0)&~contact200).sum()))
   cycles.append(actual)
  first_support=[]
  for e in range(64):
   where=torch.nonzero(feet[:,e],as_tuple=False).flatten();first_support.append(None if len(where)==0 else int(where[0])*.02)
  records.append(dict(arm=arm,stage=stage,cycles=cycles,first_support_time_s=first_support,first_step_supported=int(feet[0].sum()),first_100ms_missing_foot_frames=int((~feet[:5]).sum()),after_100ms_cycle1_missing_foot_frames=int((~feet[5:200]).sum()),actor_normalizer_unchanged=report['actor_normalizer_unchanged'],reward_oracle_max=report['oracle_max_abs'],report=str(d/'report.json'),report_sha256=sha(d/'report.json')))
pairs=[]
for stage in ['nominal','measured']:
 a=loaded['policy',stage];b=loaded['local_endpoint',stage];init={k:bool(torch.equal(a['initial'][k],b['initial'][k]))for k in a['initial']};assert all(init.values())
 mask=b['endpoint_active_mask'];starts=mask.any(-1).nonzero().flatten();first=int(starts[0]);pairs.append(dict(stage=stage,initial_exact=init,first_intervention_step=first,prefix_max_abs_qpos=float((a['preaction_qpos'][:first]-b['preaction_qpos'][:first]).abs().max()),prefix_max_abs_actor_obs=float((a['actor_obs_preaction'][:first]-b['actor_obs_preaction'][:first]).abs().max())))
result=dict(status='CPU_REVIEW_PASS_LOCAL_CONSTANT_NOT_SUCCESSFUL',diagnostic_controls=153600,new_formal_controls=0,source_actor_frozen=True,action_route_exact=True,original_goal_formula_parity=True,cases=records,pairs=pairs,conclusions=['Original learned policy and localconstant both have0 sustainedlowgoals across3cycles inbothconditions.','008staticendpoint qualification doesnotestablish dynamics from current learned state; do not use it as validated transitionteacher.','Original clean_cycle includes bothfeet throughout and resets startwithoutfeet; reportprose omits this constraint. Preserve originalcounts and recordadditionaldeclaredclean count. Lowgoal failure is independent of this discrepancy.','Separate runs may differ before intervention; exact initialconditions do not imply pairedidentical trajectories.','All original reportcamera metrics retained with separatephase/healthy denominators; not pooled.'],script_sha256=sha(Path(__file__)))
(R/'groundpick-local-endpoint-closeout-001.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'cases':[{k:v for k,v in y.items()if k in ['arm','stage','cycles','first_step_supported','after_100ms_cycle1_missing_foot_frames']}for y in records],'pairs':pairs},indent=2))
