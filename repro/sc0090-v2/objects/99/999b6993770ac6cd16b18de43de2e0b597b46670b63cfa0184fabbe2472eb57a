"""Frozen three-cycle GroundPick battery: four loads, unpushed/pushed, real P04."""
import argparse,hashlib,json,math
from dataclasses import asdict
from pathlib import Path
import torch
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
from rsl_rl.utils import resolve_callable,resolve_obs_groups
from mjlab_microduck.custom_v2.ground_pick import GROUND_PICK_TASK,initialize_ground_pick_payload
from mjlab_microduck.custom_v2.ground_pick_metrics import GroundPickMeasurements,BOUNDARY
from mjlab_microduck.custom_v2.ground_pick_history import HISTORY_TASK
from mjlab_microduck.custom_v2.ground_pick_precision import PRECISION_TASK
from mjlab_microduck.custom_v2.head_metrics import camera_metrics
from train_v2_ground_pick import source_hashes
LOADS=(0.,.01,.025,.04)
def fixed_load(env,env_ids,**kwargs):
 ids=env_ids if env_ids is not None else torch.arange(env.num_envs,device=env.device)
 if not hasattr(env,'_mouth_payload_kg'):env._mouth_payload_kg=torch.zeros(env.num_envs,device=env.device)
 env._mouth_payload_kg[ids]=torch.tensor(LOADS,device=env.device)[ids%4]
def selected_push(env,env_ids,**kwargs):
 from mjlab.envs.mdp.events import push_by_setting_velocity
 ids=env_ids if env_ids is not None else torch.arange(env.num_envs,device=env.device)
 ids=ids[(ids//4)%2==1]
 if len(ids):push_by_setting_velocity(env,ids,**kwargs)
def camera_summary(q,healthy):
 values=camera_metrics(q,dt=.005,exposure_s=.02);valid=healthy.all(0)
 def aggregate(mask):
  if not bool(mask.any()):return None
  return {k:{'mean_over_trials':float(v[mask].mean()),'p95_over_trials':float(torch.quantile(v[mask],.95)),'max_over_trials':float(v[mask].max())} for k,v in values.items()}
 return dict(trials=q.shape[1],healthy_entire_window_trials=int(valid.sum()),all_trials=aggregate(torch.ones_like(valid)),healthy_entire_window=aggregate(valid))
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--stage',choices=('nominal','measured'),default='nominal');p.add_argument('--seed',type=int,default=2607);p.add_argument('--num-envs',type=int,default=256);a=p.parse_args()
 assert a.num_envs%8==0;a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.cuda.set_device(a.device)
 checkpoint=torch.load(a.checkpoint,map_location=a.device,weights_only=False);meta=checkpoint['infos']['v2'];task=meta['task'];assert task in (GROUND_PICK_TASK,HISTORY_TASK,PRECISION_TASK)
 hashes=source_hashes(Path(__file__).resolve().parents[1]);assert hashes==meta['source_sha256'],'Numerical evaluation source differs from checkpoint'
 cfg=load_env_cfg(task,play=True);cfg.seed=a.seed;cfg.scene.num_envs=a.num_envs;cfg.actions['joint_pos'].randomization_stage=a.stage
 cfg.curriculum={};cfg.terminations={};cfg.auto_reset=False;cfg.observations['actor'].enable_corruption=False
 cfg.commands['twist'].randomize_phase=False
 cfg.events['sample_mouth_payload'].func=fixed_load
 cfg.events['push_robot'].func=selected_push
 # Use the training push interval, recorded as a separate subgroup; phase is native.
 cfg.events['push_robot'].interval_range_s=(3.,6.)
 (a.output/'executed-script.py').write_bytes(Path(__file__).read_bytes());(a.output/'config.json').write_text(json.dumps(asdict(cfg),indent=2,default=str)+'\n')
 env=ManagerBasedRlEnv(cfg,device=a.device)
 try:
  obs,_=env.reset();initialize_ground_pick_payload(env);robot=env.scene['robot'];reader=GroundPickMeasurements(env,oracle=True)
  initial={k:getattr(env.sim.data,k).clone().cpu() for k in ('qpos','qvel')};initial.update(actor_obs=obs['actor'].clone().cpu(),origins=env.scene.env_origins.clone().cpu())
  agent=asdict(load_rl_cfg(task));td=TensorDict(obs,batch_size=[env.num_envs]);groups=resolve_obs_groups(td,agent['obs_groups'],['actor','critic']);ac=agent['actor']
  for key in ('cnn_cfg','distribution_cfg'):
   if ac.get(key) is None:ac.pop(key,None)
  if ac.get('rnn_type') is None:
   for key in ('rnn_type','rnn_hidden_dim','rnn_num_layers'):ac.pop(key,None)
  actor=resolve_callable(ac.pop('class_name'))(td,groups,'actor',14,**ac).to(a.device);actor.load_state_dict(checkpoint['actor_state_dict'],strict=True);actor.eval();frozen={k:v.clone() for k,v in actor.state_dict().items()}
  data={};control=[0];native_reward=env.reward_manager.compute
  def record(dt):
   reward=native_reward(dt);x=reader.read()
   if not data:data.update({k:torch.empty((600,*v.shape),device=v.device,dtype=v.dtype) for k,v in x.items()})
   for k,v in x.items():data[k][control[0]].copy_(v)
   return reward
  env.reward_manager.compute=record
  camera_site=robot.find_sites('head_camera')[0][0];camera=torch.empty(2401,env.num_envs,4,device=a.device);contact200=torch.empty(2400,env.num_envs,2,device=a.device,dtype=torch.bool);sub=[0];native_sub=env.metrics_manager.compute_substep
  def capture():
   native_sub();camera[sub[0]].copy_(robot.data.site_quat_w[:,camera_site]);contact200[sub[0]].copy_(torch.stack((env.scene['head_impact_contact'].data.found.reshape(env.num_envs,-1).any(-1),env.scene['self_collision'].data.found.reshape(env.num_envs,-1).any(-1)),-1));sub[0]+=1
  env.metrics_manager.compute_substep=capture
  with torch.inference_mode():
   for step in range(600):
    control[0]=step;torch._assert_async((obs['actor'][:,48:51]==env.command_manager.get_command('twist')).all(),'Actor phase command differs from actual control command')
    action=actor(TensorDict(obs,batch_size=[env.num_envs]),stochastic_output=False)
    obs,_,_,_,_=env.step(action)
   camera[-1].copy_(robot.data.site_quat_w[:,camera_site])
  assert sub[0]==2400 and reader.calls==600
  assert bool(torch.isfinite(camera).all());torch.testing.assert_close(camera.norm(dim=-1),torch.ones_like(camera[...,0]),rtol=0,atol=2e-5)
  for k,v in frozen.items():assert torch.equal(actor.state_dict()[k],v)
  torch.save({'measurements':{k:v.cpu() for k,v in data.items()},'camera_quat_w':camera.cpu(),'head_self_found_200Hz':contact200.cpu(),'initial':initial,'load_cases':LOADS,'subgroup_index':'idx%4=load; idx//4%2=pushed'},a.output/'traces.pt')
  healthy=data['finite']&data['feedback_ready']&(~data['head_contact'])&(~data['self_contact'])&data['feet'].all(-1)
  motion_healthy=healthy&(data['tilt']<math.radians(60))
  quiet=motion_healthy&(data['linear'].norm(dim=-1)<.05)&(data['omega']<.5)
  stand_body=quiet&((data['height']-.112).abs()<.012)&(data['tilt']<math.radians(15))
  idx=torch.arange(env.num_envs,device=a.device);results=[]
  for group in range(8):
   select=idx%8==group;cycles=[]
   for cycle in range(3):
    lo,hi=cycle*200,(cycle+1)*200;phase=data['phase'][lo:hi,select];torch._assert_async((phase==phase[:,:1]).all(),'Native phase should be common in frozen battery')
    low_phase=(phase>=.375)&(phase<.425);low=data['low_goal'][lo:hi,select]&low_phase
    attained=low.unfold(0,5,1).all(-1).any(0);standing=data['standing_goal'][hi-25:hi,select].all(0)
    clean=healthy[lo:hi,select].all(0);fault=(~data['feedback_ready'][lo:hi,select]).any(0)
    cams={}
    for name,p0,p1,hm in [('descent',0.,.375,motion_healthy),('low_hold',.375,.425,quiet),('rise',.425,.8,motion_healthy),('stand_hold',.8,1.,stand_body)]:
     steps=torch.nonzero((phase[:,0]>=p0)&(phase[:,0]<p1),as_tuple=False).flatten()
     if len(steps)<2:cams[name]=None;continue
     first,last=lo+int(steps[0]),lo+int(steps[-1])+1
     cams[name]=camera_summary(camera[first*4:last*4+1,select],hm[first:last,select])
    t=data['head_torque'][lo:hi,select];headq=data['physical_q'][hi-25:hi,select,5:9]
    cycles.append(dict(trials=int(select.sum()),low_goal_sustained_100ms=int(attained.sum()),return_stand_tail_500ms=int(standing.sum()),clean_cycle=int(clean.sum()),complete_cycle=int((attained&standing&clean&(~fault)).sum()),feedback_fault_trials=int(fault.sum()),head_contact_trials=int(data['head_contact'][lo:hi,select].any(0).sum()),self_contact_trials=int(data['self_contact'][lo:hi,select].any(0).sum()),mouth_min_height_mean_m=float(data['mouth_height'][lo:hi,select].amin(0).mean()),mouth_low_height_mean_m=float(data['mouth_height'][lo:hi,select][low_phase].mean()),mouth_low_down_cos_mean=float(data['mouth_down_cos'][lo:hi,select][low_phase].mean()),body_tilt_p95_deg=float(torch.rad2deg(torch.quantile(data['tilt'][lo:hi,select],.95))),head_return_mae_deg=(torch.rad2deg(headq.abs().mean((0,1)))).tolist(),head_return_all_axes_1deg_hold=int(((headq.abs()<torch.pi/180).all(-1)&stand_body[hi-25:hi,select]).all(0).sum()),head_torque_abs_p95_Nm=float(torch.quantile(t,.95)),head_torque_abs_max_Nm=float(t.max()),camera=cams))
   results.append(dict(payload_kg=LOADS[group%4],training_interval_pushes=bool(group//4),cycles=cycles))
  report=dict(status='EVALUATED_NOT_SKILL_ACCEPTANCE',task=task,checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),stage=a.stage,seed=a.seed,num_envs=env.num_envs,protocol='3 native4s cycles from phase0 HOME; four payloads; unpushed and training3–6s interval-push subgroups; actor corruption off but real S0/O0 and startup DR retained',boundary=BOUNDARY,actor_normalizer_unchanged=True,source_sha256=hashes,oracle_max_abs=reader.oracle_max.cpu().tolist(),head_semantics='Head intentionally follows mouth task, not an independent head command. Report low/standing stabilization separately from descent/rise camera motion.',camera_protocol={'site':'head_camera CAD placeholder, angular motion only, not calibrated pixel blur','hz':200,'exposure_s':.02,'health':'motion: ready finite no head/selfcontact bothfeet tilt<60deg; low: additionally linear<.05m/s and angular<.5rad/s; stand: additionally height.112±.012m and tilt<15deg','samples':2401},development_task_gates={'low':'mouth0–25mm/downcos>=.95, quiet healthy body, sustained100ms during native low phase','return':'HOME legsMAE<.2rad/headmax<.1rad; stable height/tilt/speed/bothfeet and no contacts for last500ms','complete':'low and return gates plus no head/selfcontact or feedback failure in entirecycle'},results=results,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
  (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
 finally:env.close()
if __name__=='__main__':main()
