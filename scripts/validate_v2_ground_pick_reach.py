"""One GPU batch: HOME and three geometric reaches, empty/30g mouth loads."""
import argparse,hashlib,json,math,time
from pathlib import Path
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import EventTermCfg
from mjlab_microduck.custom_v2.ground_pick import make_ground_pick_env_cfg,mouth_payload_step
from mjlab_microduck.custom_v2.microduck_v2 import POLICY_JOINTS,HOME_HEIGHT
from mjlab_microduck.custom_v2.head_metrics import camera_metrics
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
root=Path('runs-v2');report=root/'sitstand-lesson4-0384-measured/report.json'
while not report.exists():
 if not Path('/proc/3879785').exists():raise RuntimeError('SitStand final measured ended without report')
 time.sleep(2)
screen=json.loads((root/'groundpick-reach-screen-001.json').read_text());rows=screen['results']
poses=[dict(pose={},root_pitch_rad=0.,root_height_m=HOME_HEIGHT,target_mouth_height_m=None),rows[0],rows[2],rows[4]]
assert all(x.get('geometric_candidate',True) for x in poses)
def reset_pose(env,env_ids,**kwargs):
 robot=env.scene['robot'];ids=env_ids if env_ids is not None else torch.arange(env.num_envs,device=env.device)
 q=robot.data.default_joint_pos[ids].clone();v=torch.zeros_like(q);r=robot.data.default_root_state[ids].clone();r[:,:3]+=env.scene.env_origins[ids];r[:,7:]=0
 groups=ids%8//2
 for i,pose in enumerate(poses):
  select=groups==i
  for name,value in pose['pose'].items():
   jid,_=robot.find_joints('^'+name+'$');q[select,jid[0]]=value
  angle=pose['root_pitch_rad'];r[select,2]=env.scene.env_origins[ids[select],2]+pose['root_height_m']+.002
  r[select,3:7]=torch.tensor([math.cos(angle/2),0,math.sin(angle/2),0],device=env.device)
 policy,_=robot.find_joints(POLICY_JOINTS,preserve_order=True)
 q[:,policy]+=(torch.rand(len(ids),14,device=env.device)-.5)*.006
 robot.write_root_state_to_sim(r,env_ids=ids);robot.write_joint_state_to_sim(q,v,env_ids=ids)
def payload(env,env_ids,**kwargs):
 if not hasattr(env,'_mouth_payload_kg'):env._mouth_payload_kg=torch.zeros(env.num_envs,device=env.device)
 ids=env_ids if env_ids is not None else torch.arange(env.num_envs,device=env.device)
 env._mouth_payload_kg[ids]=(ids%2)*.03
cfg=make_ground_pick_env_cfg(play=True);cfg.seed=2605;cfg.scene.num_envs=128;cfg.curriculum={};cfg.terminations={};cfg.auto_reset=False
cfg.events={n:t for n,t in cfg.events.items() if t.mode=='startup' or n in {'reset_base','reset_robot_joints','v2_reset_joint_guards','mouth_reset','sensor_error_reset','mouth_payload','sample_mouth_payload'}}
cfg.events['reset_robot_joints']=EventTermCfg(func=reset_pose,mode='reset')
cfg.events['sample_mouth_payload']=EventTermCfg(func=payload,mode='reset')
# Keep runtime/physical sensors, but do not call any reward twice.
self_term=cfg.rewards['self_collisions'];impact_term=cfg.rewards['head_impact_penalty']
(out_config:=a.output/'diagnostic-config.json').write_text(json.dumps({'pose_candidates':poses,'loads_kg':[0,.03],'phase':.6,'seconds':3,'repeats':16,'source_model_modified':False,'scope':'Constant physical target hold, not a required learned action trajectory'},indent=2)+'\n')
torch.set_num_threads(2);torch.cuda.set_device('cuda:0');env=ManagerBasedRlEnv(cfg,device='cuda:0')
try:
 robot=env.scene['robot'];native=env.command_manager.compute
 def held(dt):
  env.command_manager.get_term('twist')._gp_phase.fill_(.6);native(0.)
 env.command_manager.compute=held;env.reset()
 for t in env.event_manager._mode_term_cfgs['step']:
  if t.func is mouth_payload_step:t.func(env,None,**t.params)
 ids,names=robot.find_joints(POLICY_JOINTS,preserve_order=True);assert tuple(names)==POLICY_JOINTS
 mouth,_=robot.find_sites('mouth_tip');camera_site,_=robot.find_sites('head_camera')
 action=torch.zeros(128,14,device='cuda:0');group=torch.arange(128,device='cuda:0')%8
 for i,pose in enumerate(poses):
  for name,value in pose['pose'].items():action[group//2==i,POLICY_JOINTS.index(name)]=value
 camera=torch.empty(601,128,4,device='cuda:0');sub=[0];old_sub=env.metrics_manager.compute_substep
 def capture():
  old_sub();camera[sub[0]].copy_(robot.data.site_quat_w[:,camera_site[0]]);sub[0]+=1
 env.metrics_manager.compute_substep=capture
 trace=[];qtrace=[]
 for step in range(150):
  env.step(action);pos=robot.data.site_pos_w[:,mouth[0]];q=robot.data.site_quat_w[:,mouth[0]]
  down=-2*(q[:,1]*q[:,3]-q[:,0]*q[:,2]);bodyq=robot.data.root_link_quat_w
  tilt=torch.acos((1-2*(bodyq[:,1]**2+bodyq[:,2]**2)).clamp(-1,1))
  headtorque=robot.data.qfrc_actuator[:,ids[5:9]].abs().amax(-1)
  contact=env.scene['feet_ground_contact'].data.found.reshape(128,-1).all(-1)
  trace.append(torch.stack((pos[:,2]-env.scene.env_origins[:,2],down,tilt,robot.data.root_link_lin_vel_w.norm(dim=-1),robot.data.root_link_ang_vel_b.norm(dim=-1),self_term.func(env,**self_term.params),impact_term.func(env,**impact_term.params),(~env._v2_joint_feedback.ready).float(),contact.float(),headtorque),-1))
  qtrace.append(robot.data.joint_pos[:,ids].clone())
 camera[-1].copy_(robot.data.site_quat_w[:,camera_site[0]]);assert sub[0]==600
 x=torch.stack(trace).cpu();qs=torch.stack(qtrace).cpu();torch.save({'trace':x,'physical_q':qs,'policy_targets':action.cpu(),'camera_quat_w':camera.cpu(),'group':group.cpu(),'columns':['mouth_height_m','mouth_down_cos','body_tilt_rad','linear_speed_mps','angular_speed_radps','self_contact_count','head_impact_excess_N','feedback_fault','both_feet_contact','head_torque_abs_max_Nm']},a.output/'traces.pt')
 results=[]
 for i in range(8):
  mask=group.cpu()==i;y=x[:,mask];tail=y[100:];settled=y[25:];cam=camera_metrics(camera[400:,group==i]);error=(qs[100:,mask]-action.cpu()[mask]).abs()
  results.append(dict(pose_index=i//2,payload_kg=(i%2)*.03,trials=int(mask.sum()),mouth_tail_height_mean_m=float(tail[:,:,0].mean()),mouth_tail_height_range_m=[float(tail[:,:,0].min()),float(tail[:,:,0].max())],mouth_tail_down_cos_min=float(tail[:,:,1].min()),body_tail_tilt_mean_deg=float(torch.rad2deg(tail[:,:,2]).mean()),linear_tail_max_mps=float(tail[:,:,3].max()),angular_tail_max_radps=float(tail[:,:,4].max()),self_contact_settled_trials=int((settled[:,:,5]>0).any(0).sum()),head_impact_settled_trials=int((settled[:,:,6]>0).any(0).sum()),feedback_fault_trials=int(y[:,:,7].bool().any(0).sum()),both_feet_tail_fraction=float(tail[:,:,8].mean()),head_torque_tail_abs_p95_Nm=float(torch.quantile(tail[:,:,9],.95)),joint_target_tail_mae_rad=float(error.mean()),joint_target_tail_max_error_rad=float(error.max()),camera_tail={k:float(v.mean()) for k,v in cam.items()}))
 result=dict(status='PHYSICAL_HOLD_DIAGNOSTIC_COMPLETE_NOT_SKILL',seconds=3,candidates=poses,results=results,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),physics_model_modified=False,scope='Settled .5–3s contacts; last1s kinematics/camera/target tracking; whole3s faults. Direct targets are feasibility probes, not mandatory transition templates.')
 (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
finally:env.close()
