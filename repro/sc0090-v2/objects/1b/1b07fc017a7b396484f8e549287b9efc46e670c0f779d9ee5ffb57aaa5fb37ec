"""Read actual settled GPU self-contact pairs; no policy or reward mutation."""
import json,argparse
from pathlib import Path
import torch,warp as wp
from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.custom_v2.sitstand import make_sitstand_env_cfg
from mjlab_microduck.custom_v2.training import SIT_CANDIDATE,SIT_HEIGHT
from mjlab_microduck.tasks import mdp
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(2);torch.cuda.set_device('cuda:0');cfg=make_sitstand_env_cfg(play=True);cfg.scene.num_envs=128;cfg.seed=2603
cfg.curriculum={};cfg.terminations={};cfg.auto_reset=False
cfg.events={n:t for n,t in cfg.events.items() if t.mode=='startup' or n in {'reset_base','reset_robot_joints','set_ground_state','v2_reset_joint_guards','mouth_reset','sensor_error_reset'}}
g=cfg.events['set_ground_state'].params;g.update(face_down_prob=0.,face_up_prob=0.,sitting_prob=1.,standing_prob=0.,sitting_joint_noise_std=.01,sitting_tilt_max=.035)
env=ManagerBasedRlEnv(cfg,device='cuda:0')
try:
 env.reset();action=torch.zeros(128,14,device='cuda:0')
 for i,v in SIT_CANDIDATE.items():action[:,i]=v
 faults=torch.zeros(128,dtype=torch.bool,device='cuda:0')
 for step in range(150):
  env.step(action);faults|=mdp.v2_feedback_fault(env)
 data=env.sim.wp_data;num=int(wp.to_torch(data.nacon)[0]);c=data.contact
 geom=wp.to_torch(c.geom)[:num].cpu();world=wp.to_torch(c.worldid)[:num].cpu();dist=wp.to_torch(c.dist)[:num].cpu();addr=wp.to_torch(c.efc_address)[:num].cpu();forces=wp.to_torch(data.efc.force).cpu()
 torch.save({'qpos':env.sim.data.qpos.cpu(),'qvel':env.sim.data.qvel.cpu(),'geom':geom,'world':world,'dist':dist,'efc_address':addr,'constraint_force':forces},a.output/'raw-contact-state.pt')
 model=env.sim.mj_model;pairs={}
 for i,(g1,g2) in enumerate(geom.tolist()):
  names=[model.geom(g1).name,model.geom(g2).name]
  if not all(n.startswith('robot/') for n in names):continue
  key=' | '.join(sorted(names));body=[model.body(int(model.geom_bodyid[k])).name for k in (g1,g2)]
  x=pairs.setdefault(key,dict(bodies=body,contacts=0,worlds=set(),min_dist_m=1.,max_dist_m=-1.,max_abs_constraint_row=0.))
  x['contacts']+=1;x['worlds'].add(int(world[i]));x['min_dist_m']=min(x['min_dist_m'],float(dist[i]));x['max_dist_m']=max(x['max_dist_m'],float(dist[i]));j=addr[i][addr[i]>=0].long()
  if len(j):x['max_abs_constraint_row']=max(x['max_abs_constraint_row'],float(forces[world[i],j].abs().max()))
 for x in pairs.values():x['worlds']=len(x['worlds'])
 robot=env.scene['robot'];tilt=torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:,2]).clamp(-1,1)))
 r=dict(status='CONTACT_IDENTIFICATION_COMPLETE',contacts=num,pairs=pairs,tilt_range_deg=[float(tilt.min()),float(tilt.max())],height_mean=float((robot.data.root_link_pos_w[:,2]-env.scene.env_origins[:,2]).mean()),faults=int(faults.sum()),cone=int(model.opt.cone),note='Raw constraint rows are reported, not asserted to be resultant normal force.')
 torch.save({'qpos':env.sim.data.qpos.cpu(),'qvel':env.sim.data.qvel.cpu(),'geom':geom,'world':world,'dist':dist,'efc_address':addr},a.output/'state.pt')
 (a.output/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
finally:env.close()
