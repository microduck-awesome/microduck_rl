"""Synchronized four-policy playback on the V2 and official robot models."""
from collections import deque
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import imageio.v2 as imageio
import numpy as np
import onnxruntime as ort
from PIL import Image,ImageDraw,ImageFont
import torch
from tensordict import TensorDict
import mjlab.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
from mjlab.utils.buffers import DelayBuffer
from mjlab_microduck.custom_v2.microduck_v2 import TASK_ID,POLICY_JOINTS

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs/evidence/official-walk/corrected/best-search/four-policy-comparison';OUT.mkdir(parents=True,exist_ok=True)
RUN=next((ROOT/'logs/rsl_rl/microduck_v2_official_baseline').glob('*dr_fresh'));OFFICIAL=ROOT/'docs/evidence/official-walk/corrected/best-search/official-reference/alpha_walking.onnx'
TIMELINE=[('静止',3,(0.,0.,0.)),('极低速前进',4,(.10,0.,0.)),('低速前进',6,(.15,0.,0.)),('正常前进',6,(.30,0.,0.)),('原地左转',5,(0.,0.,.50)),('原地右转',5,(0.,0.,-.50)),('边走边左转',6,(.15,0.,.50)),('边走边右转',6,(.15,0.,-.50)),('后退',4,(-.10,0.,0.)),('左侧移',4,(0.,.10,0.)),('右侧移',4,(0.,-.10,0.)),('停止',3,(0.,0.,0.))]
TOTAL=sum(n for _,n,_ in TIMELINE);DEVICE='cuda:0';torch.set_num_threads(2)
fontpath='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';F24=ImageFont.truetype(fontpath,24);F20=ImageFont.truetype(fontpath,20);F17=ImageFont.truetype(fontpath,17)

class Playback:
 def __init__(self,official=False):
  self.official=official;self.count=1 if official else 3;task='Mjlab-Velocity-Flat-MicroDuck' if official else TASK_ID
  cfg=load_env_cfg(task,play=True);cfg.scene.num_envs=self.count;cfg.scene.env_spacing=0.;cfg.seed=20261026;cfg.episode_length_s=TOTAL+5;cfg.curriculum={};cfg.rewards={};cfg.terminations={}
  cfg.events={n:t for n,t in cfg.events.items() if n in {'expand_bam_friction_fields','reset_base','reset_robot_joints','reset_action_history','mouth_hold','mouth_reset'}}
  cfg.events['reset_base'].params['pose_range']={n:(0.,0.) for n in ('x','y','z','yaw','roll','pitch')}
  if official:cfg.events['reset_base'].params['pose_range']['z']=(.12,.12)
  for group in cfg.observations.values():group.enable_corruption=False
  for term in cfg.commands.values():term.resampling_time_range=(10000.,10000.);term.debug_vis=False
  for motor in cfg.scene.entities['robot'].articulation.actuators:motor.vin=7.4 if official else 11.1;motor.vin_range=None;motor.vin_drop_gain_range=None
  cfg.viewer.width=640;cfg.viewer.height=336;cfg.viewer.distance=.72;cfg.viewer.elevation=-15.;cfg.viewer.azimuth=135.;cfg.viewer.max_extra_envs=0
  self.env=ManagerBasedRlEnv(cfg,device=DEVICE,render_mode='rgb_array');self.wrapped=RslRlVecEnvWrapper(self.env);self.robot=self.env.scene['robot'];self.command=torch.zeros(self.count,3,device=DEVICE)
  if official:
   opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1;self.session=ort.InferenceSession(str(OFFICIAL),sess_options=opt,providers=['CPUExecutionProvider']);meta=self.session.get_modelmeta().custom_metadata_map
   assert meta['joint_names'].split(',')==list(POLICY_JOINTS);np.testing.assert_allclose(self.robot.data.default_joint_pos[0].cpu().numpy(),np.fromstring(meta['default_joint_pos'],sep=','),atol=5e-4,rtol=0.)
  else:
   agent=load_rl_cfg(task);agent.logger='tensorboard';agent.upload_model=False;runner=load_runner_cls(task)(self.wrapped,asdict(agent),device=DEVICE);self.policies=[]
   for it in (10250,2000,7500):
    runner.load(str(RUN/f'model_{it}.pt'),load_cfg={'actor':True},strict=True,map_location=DEVICE);self.policies.append(deepcopy(runner.get_inference_policy(device=DEVICE)))
  self.twist=self.env.command_manager.get_term('twist');head=self.env.command_manager.get_term('head_pose');body=self.env.command_manager.get_term('body_pose')
  self.twist._update_command=lambda:self.twist.command.copy_(self.command);head._update_command=lambda:head.command.zero_();body._update_command=lambda:body.command.zero_()
  for term in (self.twist,head,body):term.time_left.fill_(10000.);term._update_command()
  # Nominal zero IMU mounting error in this illustrative continuous playback.
  self.env._imu_misalign_quat=torch.zeros(self.count,4,device=DEVICE);self.env._imu_misalign_quat[:,0]=1.
  for motor in self.robot.actuators:
   if motor._delay_buffer is None:continue
   buffer=motor._delay_buffer;sampler=DelayBuffer(min_lag=buffer.min_lag,max_lag=buffer.max_lag,batch_size=1,device=DEVICE,generator=torch.Generator(device=DEVICE).manual_seed(2026102604));buffer._step_count.zero_();buffer._current_lags.zero_();buffer._phase_offsets.zero_()
   def sample(mask,buffer=buffer,sampler=sampler):
    sampler._current_lags=buffer._current_lags[:1]
    return sampler._sample_lags(mask[:1]).repeat(self.count)
   buffer._sample_lags=sample
  self.obs=self.wrapped.get_observations()
 def step(self,command):
  self.command[:]=torch.tensor(command,device=DEVICE);self.twist._update_command();self.obs['actor'][:,48:51].copy_(self.command)
  torch._assert_async(torch.isclose(self.obs['actor'][:,48:51],self.command).all(),'Policy command mismatch')
  if self.official:
   output=self.session.run(None,{self.session.get_inputs()[0].name:self.obs['actor'].cpu().numpy()})[0];actions=torch.from_numpy(output).to(DEVICE)
  else:actions=torch.cat([p(TensorDict({'actor':self.obs['actor'][i:i+1]},batch_size=[1])) for i,p in enumerate(self.policies)],dim=0)
  self.obs,_,_,_=self.wrapped.step(actions);torch._assert_async(torch.isclose(self.twist.command,self.command).all(),'Manager changed held command')
  tilt=torch.rad2deg(torch.acos((-self.robot.data.projected_gravity_b[:,2]).clamp(-1,1)))
  return torch.cat((self.robot.data.root_link_lin_vel_b[:,:2],self.robot.data.root_link_ang_vel_b[:,2:3],tilt[:,None],self.robot.data.root_link_pos_w[:,2:3]),dim=-1).cpu().numpy()
 def frame(self,index):
  self.env._offline_renderer._cfg.env_idx=index
  return self.env.render()
 def close(self):self.env.close()

started=time.monotonic();v2=official=None;writer=None
try:
 v2=Playback();official=Playback(True);writer=imageio.get_writer(OUT/'comparison.mp4',fps=25,codec='libx264',quality=8,macro_block_size=16)
 labels=[('V2 #10250 · 本轮首选','SC0090  |  11.1 V',(39,190,185)),('V2 #2000 · 低速/后退备选','SC0090  |  11.1 V',(235,166,65)),('V2 #7500 · 此前首选','SC0090  |  11.1 V',(160,126,225)),('官方 alpha_walking · #3750','原版结构 / XL330  |  7.4 V',(190,204,220))]
 history=deque(maxlen=50);logs=[];snapshots=[];phase_start=0
 with torch.inference_mode():
  for phase,(name,seconds,command) in enumerate(TIMELINE):
   steps=round(seconds/.02)
   for local_step in range(steps):
    step=phase_start+local_step;values=np.concatenate((v2.step(command),official.step(command)),axis=0);assert np.isfinite(values).all();history.append(values[:,:3]);mean=np.mean(history,axis=0);logs.append({'t':(step+1)*.02,'phase':name,'command':command,'values':values.tolist()})
    if step%2==0:
     board=Image.new('RGB',(1280,896),(18,25,36));draw=ImageDraw.Draw(board);draw.text((18,6),f'同步指令对比  |  {name}  |  {(step+1)*.02:05.2f} / {TOTAL:.0f} 秒',font=F24,fill=(245,248,252));draw.text((18,43),f'目标 vx {command[0]:+.2f} m/s   vy {command[1]:+.2f} m/s   转速 {command[2]:+.2f} rad/s',font=F20,fill=(212,227,244));draw.rectangle((18,86,18+int(1244*(step+1)/(TOTAL*50)),90),fill=(54,191,180))
     for i,(title,subtitle,color) in enumerate(labels):
      x=(i%2)*640;y=96+(i//2)*400;draw.rectangle((x,y,x+639,y+63),fill=(31,40,53));draw.rectangle((x,y,x+5,y+399),fill=color);draw.text((x+14,y+3),title,font=F20,fill=color);draw.text((x+14,y+34),subtitle,font=F17,fill=(203,215,227));draw.text((x+466,y+8),f'倾角 {values[i,3]:.1f}°',font=F17,fill=(235,240,245));frame=v2.frame(i) if i<3 else official.frame(0);board.paste(Image.fromarray(frame),(x,y+64));draw=ImageDraw.Draw(board);draw.rectangle((x+8,y+361,x+632,y+397),fill=(20,29,42));draw.text((x+14,y+365),f'实测近1s：vx {mean[i,0]:+.3f}  vy {mean[i,1]:+.3f}  转速 {mean[i,2]:+.3f}',font=F17,fill=(240,245,250))
      if values[i,3]>60 or values[i,4]<.055:draw.text((x+440,y+36),'倾倒',font=F20,fill=(255,80,80))
     writer.append_data(np.asarray(board))
     if local_step==steps//2:
      image=OUT/f'phase-{phase:02d}.jpg';board.save(image,quality=90);snapshots.append(str(image))
   phase_start+=steps;print(json.dumps({'completed_phase':name,'simulation_seconds':phase_start*.02,'total_seconds':TOTAL,'elapsed_s':round(time.monotonic()-started,1)}),flush=True)
 writer.close();writer=None
 report={'training_status':'PAUSED_BY_USER','ppo_updates':0,'fps':25,'width':1280,'height':896,'seconds':TOTAL,'continuous_playback_no_resets':True,'official_model':str(OFFICIAL),'official_sha256':hashlib.sha256(OFFICIAL.read_bytes()).hexdigest(),'models':[{'iteration':it,'sha256':hashlib.sha256((RUN/f'model_{it}.pt').read_bytes()).hexdigest()} for it in (10250,2000,7500)],'timeline':TIMELINE,'voltages':{'V2':11.1,'official':7.4},'conditions':'nominal zero IMU mounting error; no body DR or pushes; native command delays retained; separate robot structures; illustrative continuous switching, distinct from fixed-command benchmark','held_command_verified_each_step':True,'elapsed_s':time.monotonic()-started,'snapshots':snapshots,'hardware_validated':False}
 (OUT/'video-provenance.json').write_text(json.dumps(report,indent=2)+'\n');(OUT/'trajectory.json').write_text(json.dumps(logs)+'\n');print(json.dumps(report,indent=2))
finally:
 if writer is not None:writer.close()
 if v2 is not None:v2.close()
 if official is not None:official.close()
