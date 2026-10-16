"""Actual P04 open-loop pose feasibility; this is not a learned-policy test."""
import argparse,json,time,hashlib
from pathlib import Path
import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.custom_v2.sitstand import make_sitstand_env_cfg,TASK
from mjlab_microduck.custom_v2.training import SIT_CANDIDATE,SIT_HEIGHT,STAND_HEIGHT
from mjlab_microduck.tasks import mdp

def ground_groups(env,env_ids,**params):
    for group in range(4):
        ids=env_ids[env_ids%4==group]
        p=dict(params)
        p.update(face_down_prob=0.,face_up_prob=0.,sitting_prob=float(group in (0,1)),
                 standing_prob=float(group in (2,3)),sitting_tilt_max=0. if group%2==0 else .035,
                 sitting_joint_noise_std=0. if group%2==0 else .01,
                 sitting_z_min=SIT_HEIGHT,sitting_z_max=SIT_HEIGHT+.002,
                 standing_z_min=STAND_HEIGHT,standing_z_max=STAND_HEIGHT+.002)
        mdp.set_random_ground_state(env,ids,**p)

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=128);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.cuda.set_device(a.device)
    torch.manual_seed(619);start=time.monotonic()
    cfg=make_sitstand_env_cfg(play=True);cfg.scene.num_envs=a.num_envs;cfg.seed=619
    cfg.curriculum={};cfg.rewards={};cfg.terminations={};cfg.auto_reset=False
    cfg.events={n:t for n,t in cfg.events.items() if n in {'expand_bam_friction_fields','v2_runtime','mouth_reset','reset_base','reset_robot_joints','set_ground_state','v2_reset_joint_guards'}}
    cfg.events['set_ground_state'].func=ground_groups
    cfg.commands['head_pose'].ranges=((0.,0.),)*4
    env=ManagerBasedRlEnv(cfg,device=a.device)
    try:
        obs,_=env.reset();assert obs['actor'].shape==(a.num_envs,61)
        robot=env.scene['robot'];feedback=env._v2_joint_feedback
        base=torch.zeros(a.num_envs,14,device=a.device);sit=base.clone()
        for i,v in SIT_CANDIDATE.items():sit[:,i]=v
        sitting=torch.arange(a.num_envs,device=a.device)%4<2
        flags=sitting.float();rows=[];fault=torch.zeros(a.num_envs,dtype=torch.bool,device=a.device)
        peak=0;capacity=env.sim.wp_data.naconmax
        for step in range(500):
            # 0–3s hold, 3–5s linear target change, 5–8s opposite hold, 8–10s return.
            t=step*env.step_dt
            blend=0. if t<3 else min((t-3)/2,1.) if t<8 else 1.-min((t-8)/2,1.)
            alpha=flags*(1-blend)+(1-flags)*blend
            action=sit*alpha[:,None]
            env.command_manager.get_term('twist').vel_command_b[:,0]=alpha>.5
            obs,*_=env.step(action)
            fault|=mdp.v2_feedback_fault(env)
            tilt=torch.acos((-robot.data.projected_gravity_b[:,2]).clamp(-1,1))
            z=robot.data.root_link_pos_w[:,2]-env.scene.env_origins[:,2]
            rows.append(torch.stack((z,tilt,robot.data.root_link_lin_vel_w[:,2],robot.data.root_link_ang_vel_b.norm(dim=-1),robot.data.joint_vel.abs().amax(-1),fault.float()),-1))
            assert all(torch.isfinite(x).all() for x in obs.values())
            if step%50==0:
                contacts=int(wp.to_torch(env.sim.wp_data.nacon).max());peak=max(peak,contacts)
                assert contacts<=capacity
                print(json.dumps({'step':step,'wall':time.monotonic()-start,'faults':int(fault.sum())}),flush=True)
        traces=torch.stack(rows).cpu();torch.save({'trace':traces,'columns':['height','tilt_rad','vz','omega','max_joint_speed','fault'],'group':torch.arange(a.num_envs)%4},a.output/'traces.pt')
        report={'status':'PHYSICAL_DIAGNOSTIC_COMPLETE','task':TASK,'allocation_kind':'validation','num_envs':a.num_envs,
          'controls':500*a.num_envs,'seconds':10,'no_reset':True,'pose_hold_seconds':3,'ramp_seconds':2,
          'actor_width':61,'action_width':14,'feedback':'actual S0/O0 valid raw native counts v2','no_extra_body_DR':True,
          'contacts_peak':peak,'contact_capacity':capacity,'wall_seconds':time.monotonic()-start,
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'groups':{}}
        for group,name in enumerate(('sit_exact','sit_noisy','stand_exact','stand_noisy')):
            r=traces[:,torch.arange(a.num_envs)%4==group]
            report['groups'][name]={}
            for phase,(lo,hi) in {'initial_hold':(50,150),'opposite_hold':(300,400),'whole':(0,500)}.items():
                x=r[lo:hi]
                report['groups'][name][phase]={'height_mean':float(x[:,:,0].mean()),'height_range':[float(x[:,:,0].min()),float(x[:,:,0].max())],
                  'tilt_mean_deg':float(torch.rad2deg(x[:,:,1]).mean()),'tilt_max_deg':float(torch.rad2deg(x[:,:,1]).max()),
                  'episodes_tilt_over15':int((x[:,:,1].amax(0)>torch.deg2rad(torch.tensor(15.))).sum()),
                  'vz_abs_max':float(x[:,:,2].abs().max()),'joint_speed_max':float(x[:,:,4].max()),'fault_episodes':int(x[-1,:,5].sum())}
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    finally:env.close()
if __name__=='__main__':main()
