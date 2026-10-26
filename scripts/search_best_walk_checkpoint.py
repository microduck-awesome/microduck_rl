"""Evaluate saved checkpoints with paired conditions; never call PPO learn/update."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from tensordict import TensorDict
from torch.func import functional_call,vmap
import mjlab.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.buffers import DelayBuffer
from mjlab_microduck.tasks.mdp import quat_from_angle_axis
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
from mjlab_microduck.custom_v2.microduck_v2 import TASK_ID,POLICY_JOINTS

CASES=[('idle',(0.,0.,0.),'idle'),('fwd010',(.10,0.,0.),'forward'),('fwd015',(.15,0.,0.),'forward'),('fwd020',(.20,0.,0.),'forward'),('fwd030',(.30,0.,0.),'forward'),('fwd040',(.40,0.,0.),'forward'),('back010',(-.10,0.,0.),'backward'),('left010',(0.,.10,0.),'lateral'),('right010',(0.,-.10,0.),'lateral'),('turn_left',(0.,0.,.50),'turn'),('turn_right',(0.,0.,-.50),'turn'),('walk_left',(.15,0.,.50),'combined'),('walk_right',(.15,0.,-.50),'combined')]
p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,nargs='*');p.add_argument('--seconds',type=float,default=8.);p.add_argument('--replicas',type=int,default=2);p.add_argument('--voltages',type=float,nargs='+',default=[11.1]);p.add_argument('--seed',type=int,default=20261026);p.add_argument('--official-onnx',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
files=([a.official_onnx] if a.official_onnx else sorted(a.run.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1])));files=[p for p in files if not a.iterations or int(p.stem.split('_')[-1]) in a.iterations];assert files
C=len(files);K=len(CASES);R=a.replicas;B=K*R;N=C*B;device='cuda:0';torch.set_num_threads(2)
# Each checkpoint sees the same commands, nuisance draws, voltage and delay schedule.
commands=torch.tensor([cmd for _,cmd,_ in CASES],device=device).repeat_interleave(R,dim=0);fixed=commands.repeat(C,1)
volts=torch.tensor([a.voltages[i%len(a.voltages)] for i in range(R)],device=device).repeat(K).reshape(B,1).repeat(C,1)
task_id='Mjlab-Velocity-Flat-MicroDuck' if a.official_onnx else TASK_ID
cfg=load_env_cfg(task_id,play=True);cfg.scene.num_envs=N;cfg.scene.env_spacing=0.;cfg.seed=a.seed;cfg.episode_length_s=a.seconds+5
cfg.curriculum={};cfg.rewards={};cfg.terminations={};cfg.events={n:t for n,t in cfg.events.items() if n in {'expand_bam_friction_fields','reset_base','reset_robot_joints','reset_action_history','mouth_hold','mouth_reset'}}
cfg.events['reset_base'].params['pose_range']={n:(0.,0.) for n in ('x','y','z','yaw','roll','pitch')}
if a.official_onnx:cfg.events['reset_base'].params['pose_range']['z']=(.12,.12)
for g in cfg.observations.values():g.enable_corruption=False
for cmd in cfg.commands.values():cmd.resampling_time_range=(10000.,10000.);cmd.debug_vis=False
for motor in cfg.scene.entities['robot'].articulation.actuators:motor.vin=(7.4 if a.official_onnx else 11.1);motor.vin_range=None;motor.vin_drop_gain_range=None
agent=load_rl_cfg(task_id);agent.logger='tensorboard';agent.upload_model=False
start=time.monotonic();env=ManagerBasedRlEnv(cfg,device=device)
try:
 wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions);runner=load_runner_cls(task_id)(wrapped,asdict(agent),device=device);actor=runner.get_inference_policy(device=device)
 if a.official_onnx:
  import onnx,onnxruntime as ort
  model=onnx.load(a.official_onnx);onnx.checker.check_model(model);metadata={m.key:m.value for m in model.metadata_props}
  options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
  session=ort.InferenceSession(str(a.official_onnx),sess_options=options,providers=['CPUExecutionProvider'])
  assert session.get_inputs()[0].shape==[1,61];assert session.get_outputs()[0].shape==[1,14]
  assert metadata['joint_names'].split(',')==list(POLICY_JOINTS)
  np.testing.assert_allclose(env.scene['robot'].data.default_joint_pos[0].cpu().numpy(),np.fromstring(metadata['default_joint_pos'],sep=','),atol=5e-4,rtol=0.)
  assert float(metadata['action_scale'])==float(cfg.actions['joint_pos'].scale)==1.
  def batch_forward(state,x):
   array=x.reshape(-1,61).cpu().numpy()
   outputs=np.concatenate([session.run(None,{session.get_inputs()[0].name:r[None]})[0] for r in array],axis=0)
   return torch.from_numpy(outputs).to(device).reshape(C,B,14)
  stacked={};hashes=[hashlib.sha256(a.official_onnx.read_bytes()).hexdigest()];parity_max=None
 else:
  # Run the unmodified official actor through torch.func, including each checkpoint's normalizer.
  states=[];hashes=[]
  for path in files:
   saved=torch.load(path,map_location='cpu',weights_only=False);states.append(saved['actor_state_dict']);hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
  stacked={k:torch.stack([s[k] for s in states]).to(device) for k in states[0]};del saved
  def forward(state,x):return functional_call(actor,state,(TensorDict({'actor':x},batch_size=[x.shape[0]]),),strict=True)
  batch_forward=vmap(forward,in_dims=(0,0))
  with torch.inference_mode():
   sample=torch.randn(C,3,61,device=device)*.15;batched=batch_forward(stacked,sample);parity_max=0.
   for i,state in enumerate(states):
    actor.load_state_dict(state,strict=True);expected=actor(TensorDict({'actor':sample[i]},batch_size=[3]));parity_max=max(parity_max,float((expected-batched[i]).abs().max()));torch.testing.assert_close(expected,batched[i],atol=1e-4,rtol=1e-4)
  del states
 twist=env.command_manager.get_term('twist');head=env.command_manager.get_term('head_pose');body=env.command_manager.get_term('body_pose')
 twist._update_command=lambda:twist.command.copy_(fixed);head._update_command=lambda:head.command.zero_();body._update_command=lambda:body.command.zero_()
 stream_index=[0]
 def pair_buffer(buffer):
  # A private B-world native sampler makes the nuisance schedule independent
  # of how many checkpoints fit in this evaluation batch.
  generator=torch.Generator(device=device).manual_seed(a.seed+1000+stream_index[0]);stream_index[0]+=1
  sampler=DelayBuffer(min_lag=buffer.min_lag,max_lag=buffer.max_lag,batch_size=B,device=device,per_env=buffer.per_env,hold_prob=buffer.hold_prob,update_period=buffer.update_period,per_env_phase=buffer.per_env_phase,generator=generator)
  def sample_lags(mask):
   sampler._current_lags=buffer._current_lags[:B]
   return sampler._sample_lags(mask[:B]).repeat(C)
  buffer._sample_lags=sample_lags
  buffer._current_lags.zero_();buffer._step_count.zero_();buffer._phase_offsets.copy_(sampler._phase_offsets.repeat(C))
 for motor in env.scene['robot'].actuators:
  motor.vin_tensor.copy_(volts)
  if motor._delay_buffer is not None:pair_buffer(motor._delay_buffer)
 for group in env.observation_manager._group_obs_term_delay_buffer.values():
  for buffer in group.values():pair_buffer(buffer)
 imu_generator=torch.Generator(device=device).manual_seed(a.seed+700)
 axis=torch.randn(B,3,device=device,generator=imu_generator);axis=axis/(torch.linalg.vector_norm(axis,dim=-1,keepdim=True)+1e-8)
 angle=torch.rand(B,device=device,generator=imu_generator)*(np.pi/30.)
 imu=quat_from_angle_axis(angle,axis);env._imu_misalign_quat=imu.repeat(C,1)
 nuisance_hash=hashlib.sha256(imu.cpu().numpy().tobytes()+volts[:B].cpu().numpy().tobytes()).hexdigest()
 for term in (twist,head,body):term.time_left.fill_(10000.);term._update_command()
 obs=wrapped.get_observations();robot=env.scene['robot'];ids,_=robot.find_joints(POLICY_JOINTS,preserve_order=True);limits=robot.data.joint_pos_limits[:,ids];initial=robot.data.root_link_pos_w.clone()
 fail=torch.zeros(N,dtype=torch.bool,device=device);max_tilt=torch.zeros(N,device=device);limit_count=torch.zeros(N,14,device=device);head_abs=torch.zeros(N,4,device=device);records=[];warm=100;steps=round(a.seconds/.02)
 with torch.inference_mode():
  for step in range(steps):
   torch._assert_async(torch.isclose(twist.command,fixed).all(),'Command changed')
   actions=batch_forward(stacked,obs['actor'].reshape(C,B,61)).reshape(N,14)
   obs,_,_,_=wrapped.step(actions)
   torch._assert_async(torch.isclose(twist.command,fixed).all(),'Command changed after stepping')
   finite=torch.isfinite(env.sim.data.qpos).all(-1);tilt=torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:,2]).clamp(-1,1)));fail|=(~finite)|(tilt>60.)|(robot.data.root_link_pos_w[:,2]<.055);max_tilt=torch.maximum(max_tilt,tilt)
   if step>=warm:
    velocity=torch.cat((robot.data.root_link_lin_vel_b[:,:2],robot.data.root_link_ang_vel_b[:,2:3]),dim=-1);records.append(velocity.clone())
    q=robot.data.joint_pos[:,ids];margin=torch.minimum(q-limits[:,:,0],limits[:,:,1]-q)/(limits[:,:,1]-limits[:,:,0]);limit_count+=(margin<.02);head_abs+=q[:,5:9].abs()
   if (step+1)%100==0:print(json.dumps({'step':step+1,'steps':steps,'checkpoints':C,'environments':N,'elapsed_s':round(time.monotonic()-start,1)}),flush=True)
 velocity=torch.stack(records).reshape(steps-warm,C,K,R,3).cpu().numpy();failed=fail.reshape(C,K,R).cpu().numpy();tilts=max_tilt.reshape(C,K,R).cpu().numpy();occupancy=(limit_count/(steps-warm)).reshape(C,K,R,14).cpu().numpy();heading=(head_abs/(steps-warm)).reshape(C,K,R,4).cpu().numpy();mean=velocity.mean(0)
 results=[]
 for c,path in enumerate(files):
  cases=[];groups={};responsive=0;stable_trials=int((~failed[c]).sum());limit_trials=int((occupancy[c].max(-1)>.10).sum())
  for k,(name,cmd,group) in enumerate(CASES):
   target=np.asarray(cmd);v=mean[c,k];speed=float(np.linalg.norm(target[:2]));xy_error=np.linalg.norm(v[:,:2]-target[:2],axis=-1)/max(speed,.10);yaw_error=np.abs(v[:,2]-target[2])/max(abs(target[2]),.50);trial_error=.5*(xy_error+yaw_error)
   # Score bounded errors so one catastrophic trial cannot hide broad command coverage.
   errors=np.minimum(trial_error,3.)+failed[c,k]*3.+(occupancy[c,k].max(-1)>.10)*1.
   groups.setdefault(group,[]).extend(errors.tolist())
   if group!='idle':
    progress=(v[:,:2]@target[:2]/speed**2 if speed>0 else v[:,2]/target[2]);respond=(progress>=.25)&(~failed[c,k]);responsive+=int(respond.sum())
   case={'case':name,'command':cmd,'mean_vx':float(v[:,0].mean()),'mean_vy':float(v[:,1].mean()),'mean_yaw':float(v[:,2].mean()),'std_vx_across_trials':float(v[:,0].std()),'std_yaw_across_trials':float(v[:,2].std()),'trial_velocities':v.tolist(),'failed_trials':int(failed[c,k].sum()),'max_tilt_deg':float(tilts[c,k].max()),'near_limit_occupancy_max':float(occupancy[c,k].max()),'normalized_error':float(errors.mean())};cases.append(case)
  balanced=float(np.mean([np.mean(e) for e in groups.values()]));motion_total=(K-1)*R;coverage=responsive/motion_total;score=balanced+.5*(1.-coverage)
  results.append({'iteration':(3750 if a.official_onnx else int(path.stem.split('_')[-1])),'checkpoint':str(path.resolve()),'sha256':hashes[c],'selection_score_lower_is_better':score,'balanced_tracking_error':balanced,'responsive_motion_fraction':coverage,'stable_trials':stable_trials,'total_trials':K*R,'near_limit_trials':limit_trials,'group_errors':{g:float(np.mean(e)) for g,e in groups.items()},'cases':cases})
 results.sort(key=lambda x:(x['total_trials']-x['stable_trials'],x['near_limit_trials'],x['selection_score_lower_is_better']))
 report={'status':'COMPLETE','training_status':'PAUSED_BY_USER','robot':('official XL330' if a.official_onnx else 'V2 SC0090'),'inference':('unmodified published ONNX, ORT CPU' if a.official_onnx else 'native MLPModel via vmap'),'ppo_updates':0,'device':device,'seed':a.seed,'seconds_per_trial':a.seconds,'replicas':R,'voltages':a.voltages,'checkpoints_evaluated':C,'command_cases':CASES,'paired_conditions':True,'protocol_version':'private-rng-v2','paired_nuisance_sha256':nuisance_hash,'actor_vmap_max_abs_error':parity_max,'elapsed_s':time.monotonic()-start,'selection_rule':'stable trials first, near-limit trials second, then equal-weight six behavior-group normalized tracking error + 0.5*(1-responsive fraction); response threshold 25% used only for ranking, not hardware acceptance','results':results,'hardware_validated':False}
 (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'ranking.csv').write_text('iteration,score,coverage,stable,total,near_limit\n'+''.join(f"{x['iteration']},{x['selection_score_lower_is_better']},{x['responsive_motion_fraction']},{x['stable_trials']},{x['total_trials']},{x['near_limit_trials']}\n" for x in results));print(json.dumps({'top':[{k:x[k] for k in ('iteration','selection_score_lower_is_better','responsive_motion_fraction','stable_trials','near_limit_trials')} for x in results[:12]],'elapsed_s':report['elapsed_s']},indent=2))
finally:env.close()
