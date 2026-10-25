"""Controlled fixed-command CPU playback; does not modify the training run."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw
import torch
import mjlab.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
from mjlab_microduck.custom_v2.microduck_v2 import TASK_ID,POLICY_JOINTS

p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
CASES=[('idle',(0.,0.,0.)),('forward_015',(.15,0.,0.)),('forward_030',(.30,0.,0.)),('turn_050',(0.,0.,.50)),('walk_turn',(.15,0.,.50))]
fixed=torch.tensor([c for _,c in CASES]);torch.set_num_threads(2)
cfg=load_env_cfg(TASK_ID,play=True);cfg.scene.num_envs=len(CASES);cfg.seed=20261025
cfg.episode_length_s=20.;cfg.curriculum={};cfg.rewards={};cfg.terminations={}
cfg.events={n:t for n,t in cfg.events.items() if n in {'expand_bam_friction_fields','reset_base','reset_robot_joints','reset_action_history','mouth_hold','mouth_reset'}}
cfg.events['reset_base'].params['pose_range']={k:(0.,0.) for k in ('x','y','z','yaw','roll','pitch')}
for group in cfg.observations.values():group.enable_corruption=False
for cmd in cfg.commands.values():cmd.resampling_time_range=(10000.,10000.);cmd.debug_vis=False
for motor in cfg.scene.entities['robot'].articulation.actuators:
 motor.vin=11.1;motor.vin_range=None;motor.vin_drop_gain_range=None
cfg.viewer.width=480;cfg.viewer.height=320;cfg.viewer.distance=.75;cfg.viewer.elevation=-15.;cfg.viewer.azimuth=135.;cfg.viewer.max_extra_envs=0
agent=load_rl_cfg(TASK_ID);agent.logger='tensorboard';agent.upload_model=False
started=time.monotonic();env=ManagerBasedRlEnv(cfg,device='cpu',render_mode='rgb_array');writer=None
try:
 wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
 runner=load_runner_cls(TASK_ID)(wrapped,asdict(agent),device='cpu')
 runner.load(str(a.checkpoint),load_cfg={'actor':True},strict=True,map_location='cpu');policy=runner.get_inference_policy(device='cpu')
 twist=env.command_manager.get_term('twist');head=env.command_manager.get_term('head_pose');body=env.command_manager.get_term('body_pose')
 def hold_twist():twist.command.copy_(fixed)
 def hold_head():head.command.zero_()
 def hold_body():body.command.zero_()
 twist._update_command=hold_twist;head._update_command=hold_head;body._update_command=hold_body
 hold_twist();hold_head();hold_body()
 for term in (twist,head,body):term.time_left.fill_(10000.)
 obs=wrapped.get_observations();robot=env.scene['robot'];initial=robot.data.root_link_pos_w.clone()
 ids,_=robot.find_joints(POLICY_JOINTS,preserve_order=True);limits=robot.data.joint_pos_limits[:,ids]
 logs=[];screens=[];sample_steps={25,50,100,200,300,399}
 writer=imageio.get_writer(a.output/'fixed-commands.mp4',fps=25,codec='libx264',quality=7,macro_block_size=16)
 with torch.inference_mode():
  for step in range(400):
   assert torch.equal(twist.command,fixed),'Held command changed before policy observation'
   obs,_,_,_=wrapped.step(policy(obs))
   assert torch.equal(twist.command,fixed),'Held command changed during environment step'
   assert torch.isfinite(env.sim.data.qpos).all()
   tilt=torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:,2]).clamp(-1,1)))
   pos=robot.data.root_link_pos_w.clone();vel=robot.data.root_link_lin_vel_b.clone();ang=robot.data.root_link_ang_vel_b.clone();q=robot.data.joint_pos[:,ids]
   clearance=torch.minimum(q-limits[:,:,0],limits[:,:,1]-q)/(limits[:,:,1]-limits[:,:,0])
   logs.append(dict(t=(step+1)*.02,pos=pos.numpy().tolist(),vel_b=vel.numpy().tolist(),ang_b=ang.numpy().tolist(),tilt_deg=tilt.numpy().tolist(),limit_fraction=clearance.numpy().tolist()))
   if step%2==0 or step in sample_steps:
    board=Image.new('RGB',(1440,640),(235,238,241));draw=ImageDraw.Draw(board)
    for i,(name,cmd) in enumerate(CASES):
     env._offline_renderer._cfg.env_idx=i
     frame=env.render();tile=Image.fromarray(frame)
     td=ImageDraw.Draw(tile);td.rectangle((0,0,480,38),fill=(245,245,245));td.text((8,5),f'{name}  cmd={cmd}  t={(step+1)*.02:.2f}s',fill=(10,10,10));td.text((8,21),f'vx={vel[i,0]:+.3f}  wz={ang[i,2]:+.3f}  tilt={tilt[i]:.1f}',fill=(10,10,10));board.paste(tile,((i%3)*480,(i//3)*320))
    if step%2==0:writer.append_data(np.asarray(board))
    if step in sample_steps:
     path=a.output/f'frame-{step:03d}.jpg';board.save(path);screens.append(path)
   if step%100==99:print('Playback steps:',step+1,flush=True)
 writer.close();writer=None
 rows=[]
 for i,(name,cmd) in enumerate(CASES):
  settled=logs[100:];v=np.asarray([x['vel_b'][i] for x in settled]);w=np.asarray([x['ang_b'][i] for x in settled]);tilt=np.asarray([x['tilt_deg'][i] for x in logs]);qclear=np.asarray([x['limit_fraction'][i] for x in settled]);end=np.asarray(logs[-1]['pos'][i]);delta=end-initial[i].numpy()
  rows.append(dict(case=name,command=list(cmd),duration_s=8.,mean_vx_after_2s=float(v[:,0].mean()),mean_vy_after_2s=float(v[:,1].mean()),mean_yaw_rate_after_2s=float(w[:,2].mean()),displacement_xyz_m=delta.tolist(),max_tilt_deg=float(tilt.max()),final_tilt_deg=float(tilt[-1]),fall_detected=bool((tilt>60).any()),joint_limit_occupancy={n:float((qclear[:,j]<.05).mean()) for j,n in enumerate(POLICY_JOINTS)}))
 report=dict(checkpoint=str(a.checkpoint.resolve()),sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),device='cpu',simulation='official task and V2 SC0090 V5; nominal 11.1V; observation corruption off; no pushes or body DR; official command delay retained',held_command_verified_every_step=True,seconds=8.,cases=rows,elapsed_s=time.monotonic()-started,hardware_validated=False)
 (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'trajectory.json').write_text(json.dumps(logs)+'\n')
 print(json.dumps(report,indent=2))
finally:
 if writer is not None:writer.close()
 env.close()
