"""One bounded physical teacher survey on actual V2 VelStand contacts/rewards."""
from pathlib import Path
from dataclasses import asdict
import json,hashlib,torch,math
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
from mjlab_microduck.custom_v2.velstand import TASK,reset_velstand_states
from mjlab_microduck.custom_v2.walking_metrics import WalkingMeasurements
from mjlab_microduck.custom_v2.head_metrics import camera_metrics
import argparse
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=('nominal','measured'),required=True);p.add_argument('--seed',type=int,default=2026101621);args=p.parse_args()
r=Path(__file__).resolve().parent;out=args.output.resolve();out.mkdir(exist_ok=False)
torch.set_num_threads(2)
names=('idle','forward_01','forward_03','yaw_negative','yaw_positive','prone','crouch','sit');repeats=16
cfg=load_env_cfg(TASK);cfg.seed=args.seed;cfg.actions['joint_pos'].randomization_stage=args.stage;cfg.scene.num_envs=len(names)*repeats
cfg.auto_reset=False;cfg.terminations={};cfg.curriculum={}
cfg.events={k:v for k,v in cfg.events.items() if v.mode!='interval'}
cfg.observations['actor'].enable_corruption=False
# Occurs at the same reset position as the actual task's event, before joint
# guards and feedback/action reset. Never edit qpos behind the actor afterwards.
def spawns(env,env_ids,**kwargs):
 ids=torch.as_tensor(env_ids,device=env.device,dtype=torch.long) if env_ids is not None else torch.arange(env.num_envs,device=env.device)
 for group,kind in ((5,'prone_prob'),(6,'crouch_prob'),(7,'sit_prob')):
  part=ids[ids//repeats==group]
  if len(part):reset_velstand_states(env,part,prone_prob=float(kind=='prone_prob'),crouch_prob=float(kind=='crouch_prob'),sit_prob=float(kind=='sit_prob'))
cfg.events['random_prone_init'].func=spawns
(out/'config.json').write_text(json.dumps(asdict(cfg),default=str,indent=2)+'\n')
env=ManagerBasedRlEnv(cfg,device='cuda:1')
try:
 agent=load_rl_cfg(TASK);agent.logger='tensorboard';agent.upload_model=False
 agent.algorithm.bc_cfg.update(checkpoint_path=str(r/'head2-micro-exposure-allocation-001/checkpoint_001856.pt'),anchor_checkpoint_path=str(r/'walking-yaw-fine-allocation-002/checkpoint_000512.pt'))
 wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
 runner=load_runner_cls(TASK)(wrapped,asdict(agent),None,'cuda:1')
 runner.load(str(args.checkpoint),map_location='cuda:1')
 teacher=anchor=runner.alg.get_policy().eval()
 frozen=[{k:v.clone() for k,v in model.state_dict().items()} for model in (teacher,anchor)]
 target=torch.tensor([[0,0,0],[.1,0,0],[.3,0,0],[0,0,-.5],[0,0,.5],[0,0,0],[0,0,0],[0,0,0]],device=env.device,dtype=torch.float32).repeat_interleave(repeats,0)
 walking=target.abs().amax(-1)>0;original=env.command_manager.compute
 def commands(dt):
  original(dt)
  env.command_manager.get_command('twist').copy_(target)
  env.command_manager.get_term('twist').is_standing_env.copy_(~walking)
  env.command_manager.get_term('body_pose')._raw.zero_()
  env.command_manager.get_command('head_pose').zero_();env.command_manager.get_command('body_pose').zero_()
 env.command_manager.compute=commands
 obs,_=env.reset();initial=dict(qpos=env.sim.data.qpos.clone(),qvel=env.sim.data.qvel.clone(),actor=obs['actor'].clone())
 reader=WalkingMeasurements(env,oracle=True);values={};reward_terms=torch.empty(300,env.num_envs,len(env.reward_manager.active_terms),device=env.device)
 sites,_=env.scene['robot'].find_sites('head_camera');assert len(sites)==1
 camera=torch.empty(1201,env.num_envs,4,device=env.device);sub=[0];control=[0]
 origsub=env.metrics_manager.compute_substep
 def substep():
  origsub();camera[sub[0]].copy_(env.scene['robot'].data.site_quat_w[:,sites[0]]);sub[0]+=1
 env.metrics_manager.compute_substep=substep
 model=env.sim.mj_model
 by_name={model.actuator(i).name.rsplit('/',1)[-1]:i for i in range(model.nu)}
 cols=[by_name[n] for n in env.cfg.actions['joint_pos'].actuator_names]
 dofs=[int(model.jnt_dofadr[model.actuator(i).trnid[0]]) for i in cols]
 assert len(set(cols))==len(set(dofs))==14
 cols=torch.tensor(cols,device=env.device);dofs=torch.tensor(dofs,device=env.device)
 origreward=env.reward_manager.compute
 def record(dt):
  reward=origreward(dt);x=reader.read()
  if not values:values.update({k:torch.empty((300,*v.shape),device=v.device,dtype=v.dtype) for k,v in x.items()})
  for k,v in x.items():values[k][control[0]].copy_(v)
  reward_terms[control[0]].copy_(env.reward_manager._step_reward)
  # Independent MuJoCo actuator-to-joint transmission mapping, not the cached
  # reward helper's selected columns. Includes no separately controlled mouth.
  term=env.reward_manager.get_term_cfg('servo_stall')
  expected=((env.sim.data.actuator_force[:,cols].abs()>term.params['torque_thresh']) & (env.sim.data.qvel[:,dofs].abs()<term.params.get('vel_thresh',.5))).float().sum(-1)*term.weight
  actual=env.reward_manager._step_reward[:,env.reward_manager.active_terms.index('servo_stall')]
  torch._assert_async(torch.isclose(expected,actual,atol=1e-7,rtol=1e-6).all(),'Named servo stall independent oracle mismatch')
  quat=env.scene['robot'].data.root_link_quat_w;qw,qx,qy,qz=quat.unbind(-1)
  roll=torch.atan2(2*(qw*qx+qy*qz),1-2*(qx.square()+qy.square()))
  pitch=torch.asin((2*(qw*qy-qz*qx)).clamp(-1,1))
  body=env.reward_manager.get_term_cfg('body_pose_tracking')
  assert body.params['axis_weights']==(0.,0.,1.,1.,1.,0.)
  wanted=(torch.exp(-((x['height']-.112)/body.params['z_std']).square())+torch.exp(-(roll/body.params['angle_std']).square())+torch.exp(-(pitch/body.params['angle_std']).square()))/3*body.weight*(~walking)
  actual=env.reward_manager._step_reward[:,env.reward_manager.active_terms.index('body_pose_tracking')]
  torch._assert_async(torch.isclose(wanted,actual,atol=3e-7,rtol=1e-6).all(),'Physical body reward or forced-command gate mismatch')
  return reward
 env.reward_manager.compute=record
 contact=wp.to_torch(env.sim.wp_data.nacon);peak=torch.zeros_like(contact);original_step=env.sim.step
 def checked():
  original_step();peak.copy_(torch.maximum(peak,contact));torch._assert_async((contact<=env.sim.wp_data.naconmax).all(),'Contact capacity exceeded')
 env.sim.step=checked
 with torch.inference_mode():
  for step in range(300):
   control[0]=step
   torch._assert_async((obs['actor'][:,48:51]==target).all(),'Teacher requested command misalignment')
   torch._assert_async((obs['actor'][:,51:61]==0).all(),'Head/body zero command misalignment')
   action=torch.where(walking[:,None],anchor(obs),teacher(obs))
   obs,*_=wrapped.step(action)
  camera[-1].copy_(env.scene['robot'].data.site_quat_w[:,sites[0]])
 assert reader.calls==300 and sub[0]==1200
 for before,model in zip(frozen,(teacher,anchor)):
  for k,v in before.items():assert torch.equal(v,model.state_dict()[k])
 healthy=(values['height']>=.102)&(values['tilt']<=math.radians(20))&(values['linear'].norm(dim=-1)<.15)&(values['angular'].norm(dim=-1)<1.5)&values['contact'].all(-1)&values['finite']&values['feedback_ready']
 rows=[]
 for i,name in enumerate(names):
  sl=slice(i*repeats,(i+1)*repeats);valid=healthy[-50:,sl].all(0)
  cam=camera_metrics(camera[-201:,sl]);allcam={k:float(v.mean()) for k,v in cam.items()}
  goodcam={k:float(v[valid].mean()) if bool(valid.any()) else None for k,v in cam.items()}
  # Moving samples use the walking health boundary, not a zero-speed/double-
  # support requirement that would reject every intended walking gait.
  upright=(~values['failed'][-50:,sl]).all(0)
  uprightcam={k:float(v[upright].mean()) if bool(upright.any()) else None for k,v in cam.items()}
  terms=reward_terms[:,sl].mean((0,1));tailterms=reward_terms[-50:,sl].mean((0,1))
  row=dict(case=name,trials=repeats,tail_healthy_hold_trials=int(valid.sum()),feedback_invalid_trials=int((~values['feedback_ready'][:,sl]).any(0).sum()),nonfinite_trials=int((~values['finite'][:,sl]).any(0).sum()),tail_linear_mean=values['linear'][-50:,sl].mean((0,1)).cpu().tolist(),tail_angular_mean=values['angular'][-50:,sl].mean((0,1)).cpu().tolist(),tail_height_mean_m=float(values['height'][-50:,sl].mean()),tail_tilt_mean_deg=float(values['tilt'][-50:,sl].mean()*180/math.pi),weighted_reward_rates=dict(zip(env.reward_manager.active_terms,terms.cpu().tolist())),tail_weighted_reward_rates=dict(zip(env.reward_manager.active_terms,tailterms.cpu().tolist())),camera_tail_per_trial_mean=allcam,camera_tail_healthy_trials=int(valid.sum()),camera_tail_healthy_per_trial_mean=goodcam,camera_tail_upright_finite_trials=int(upright.sum()),camera_tail_upright_finite_per_trial_mean=uprightcam)
  rows.append(row)
 torch.save(dict(initial=initial,values={k:v.cpu() for k,v in values.items()},camera=camera.cpu(),reward_terms=reward_terms.cpu(),reward_names=env.reward_manager.active_terms),out/'trace.pt')
 report=dict(status='STUDENT_SURVEY_COMPLETE',task=TASK,checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),stage=args.stage,seed=args.seed,rows=rows,teacher_contract=runner.alg.teacher_contract,oracle_calls=reader.calls,oracle_max_abs=reader.oracle_max.cpu().tolist(),contact_peak=int(peak.max()),contact_capacity=env.sim.wp_data.naconmax,actor_normalizers_unchanged=True,additional_oracles=['14_named_torque_velocity_stall','physical_z_roll_pitch_body_reward_and_explicit_command_gate'],camera_timing='200Hz actual physical substeps; last1s full vs continuously healthy subset, per-trial statistics averaged',scope='One16-repeat student diagnostic; no body commands or pushes, no curriculum/auto reset. Moving cases not judged by stationary hold. Not real-world or full skill acceptance.')
 (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps([dict(case=x['case'],healthy=x['tail_healthy_hold_trials'],fault=x['feedback_invalid_trials'],tilt=x['tail_tilt_mean_deg']) for x in rows]),flush=True)
finally:env.close()
