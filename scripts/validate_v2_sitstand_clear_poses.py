"""One bounded batch of CPU-screened, non-intersecting V2 seated targets."""
import argparse,json
from pathlib import Path
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.custom_v2.sitstand import make_sitstand_env_cfg
from mjlab_microduck.tasks import mdp
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
r=json.loads(Path('runs-v2/sitstand-clearance-candidates-v2.json').read_text())
candidates=sorted([c for c in r['results'] if c['self_contacts']==0 and .05<c['foot_on_ground_height']<.1 and -.12<c['hip']-c['knee']+c['ankle']<.28],key=lambda c:c['foot_on_ground_height'])
assert len(candidates)==4

def ground(env,env_ids,**params):
 for i,c in enumerate(candidates):
  ids=env_ids[env_ids%4==i];q={1:.0873,2:c['hip'],3:c['knee'],4:c['ankle'],10:-.0873,11:c['hip'],12:c['knee'],13:c['ankle']}
  params=dict(params);params.update(face_down_prob=0.,face_up_prob=0.,sitting_prob=1.,standing_prob=0.,sitting_joint_overrides=q,sitting_z_min=c['foot_on_ground_height'],sitting_z_max=c['foot_on_ground_height']+.002,sitting_joint_noise_std=.01,sitting_tilt_max=.035)
  mdp.set_random_ground_state(env,ids,**params)
torch.set_num_threads(2);torch.cuda.set_device('cuda:0');cfg=make_sitstand_env_cfg(play=True);cfg.scene.num_envs=128;cfg.seed=2603;cfg.curriculum={};cfg.rewards={};cfg.terminations={};cfg.auto_reset=False
cfg.events={n:t for n,t in cfg.events.items() if t.mode=='startup' or n in {'reset_base','reset_robot_joints','set_ground_state','v2_reset_joint_guards','mouth_reset','sensor_error_reset'}}
cfg.events['set_ground_state'].func=ground
env=ManagerBasedRlEnv(cfg,device='cuda:0')
try:
 env.reset();robot=env.scene['robot'];group=torch.arange(128,device='cuda:0')%4;action=torch.zeros(128,14,device='cuda:0')
 for i,c in enumerate(candidates):
  for j,v in {1:.0873,2:c['hip'],3:c['knee'],4:c['ankle'],10:-.0873,11:c['hip'],12:c['knee'],13:c['ankle']}.items():action[group==i,j]=v
 rows=[]
 for _ in range(150):
  env.step(action)
  rows.append(torch.stack((robot.data.root_link_pos_w[:,2]-env.scene.env_origins[:,2],torch.acos((-robot.data.projected_gravity_b[:,2]).clamp(-1,1)),robot.data.root_link_lin_vel_w.norm(dim=-1),robot.data.root_link_ang_vel_b.norm(dim=-1),mdp.self_collision_cost(env,sensor_name='self_collision'),mdp.v2_feedback_fault(env).float()),-1))
 x=torch.stack(rows).cpu();torch.save({'trace':x,'group':group.cpu(),'columns':['height','tilt_rad','linear_speed','angular_speed','self_contact_count','feedback_fault'],'candidates':candidates},a.output/'traces.pt')
 result=[]
 for i,c in enumerate(candidates):
  z=x[:,group.cpu()==i];tail=z[100:];settled=z[25:]
  result.append(dict(candidate=c,worlds=z.shape[1],height_mean=float(tail[:,:,0].mean()),height_range=[float(tail[:,:,0].min()),float(tail[:,:,0].max())],tilt_max_deg=float(torch.rad2deg(settled[:,:,1]).max()),tilt_tail_mean_deg=float(torch.rad2deg(tail[:,:,1]).mean()),linear_max=float(tail[:,:,2].max()),angular_max=float(tail[:,:,3].max()),self_contact_worlds=int((settled[:,:,4]>0).any(0).sum()),fault_worlds=int(z[:,:,5].bool().any(0).sum()),qualified=bool((settled[:,:,1]<torch.pi/18).all() and (tail[:,:,2]<.03).all() and (tail[:,:,3]<.5).all() and (settled[:,:,4]==0).all() and (z[:,:,5]==0).all())))
 report=dict(status='BOUNDED_POSE_HOLD_COMPLETED',seconds=3,num_envs=128,candidates=result,scope='Physical target qualification only; no transition or learned skill claim')
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
finally:env.close()
