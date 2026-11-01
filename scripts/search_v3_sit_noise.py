"""Matched-noise screening of the best exact V3 seated candidates, GPU 0."""
import json
import math
from pathlib import Path
import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.custom_v2.microduck_v2 import POLICY_JOINTS, HOME_JOINT_POSITIONS


def main():
    root=Path(__file__).resolve().parents[1]
    candidate_list=json.loads((root/'docs/v3-sit-grid-probe.json').read_text())['candidates'][:9]
    model=mujoco.MjModel.from_xml_path(str(root/'src/mjlab_microduck/custom_v2/assets/scene.xml'))
    data=mujoco.MjData(model); floor=model.geom('floor').id
    trials=[]
    for cidx,c in enumerate(candidate_list):
        for seed in range(16):
            rng=np.random.default_rng(20261101+seed)
            mujoco.mj_resetDataKeyframe(model,data,model.key('home').id)
            targets=np.array([HOME_JOINT_POSITIONS[n] for n in POLICY_JOINTS])
            for side,roll in [('left',-.07),('right',.07)]:
                for suffix,val in [('hip_roll',roll),('hip_pitch',c['hip']),('knee',c['knee']),('ankle',c['ankle'])]:
                    targets[POLICY_JOINTS.index(f'{side}_{suffix}')]=val
            q=targets+rng.normal(0,.12,14)
            for i,name in enumerate(POLICY_JOINTS):
                joint=model.joint(name);q[i]=np.clip(q[i],joint.range[0]+.002,joint.range[1]-.002)
                data.qpos[joint.qposadr[0]]=q[i]
            roll,pitch=rng.uniform(-math.radians(10),math.radians(10),2)
            data.qpos[3:7]=[math.cos(roll/2)*math.cos(pitch/2),math.sin(roll/2)*math.cos(pitch/2),
                            math.cos(roll/2)*math.sin(pitch/2),-math.sin(roll/2)*math.sin(pitch/2)]
            data.qpos[2]=.02
            mujoco.mj_forward(model,data)
            floor_depth=max([max(0.,-x.dist) for x in data.contact if floor in (x.geom1,x.geom2)] or [0.])
            self_depth=max([max(0.,-x.dist) for x in data.contact if floor not in (x.geom1,x.geom2)] or [0.])
            desired_z=rng.uniform(.0802,.1002)
            z=max(desired_z,.02+floor_depth+.002)
            trials.append({'candidate':cidx,'seed':seed,'target':targets.tolist(),'q':q.tolist(),
                           'quat':data.qpos[3:7].tolist(),'z':z,'initial_self_m':self_depth,
                           'voltage':(10.2,11.1,12.4)[seed%3]})
    cfg=load_env_cfg('Mjlab-StandUp-Flat-MicroDuck-V2-PairedError-Baseline',play=True)
    cfg.scene.num_envs,cfg.seed=len(trials),20261101
    cfg.curriculum,cfg.terminations={},{}
    cfg.events={k:v for k,v in cfg.events.items() if k in {'expand_bam_friction_fields','mouth_hold'}}
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        env.reset();robot=env.scene['robot'];names=list(robot.joint_names)
        q=robot.data.default_joint_pos.clone();roots=torch.zeros((len(trials),13),device=env.device)
        roots[:,:3]=env.scene.env_origins
        action=torch.zeros((len(trials),14),device=env.device)
        for i,t in enumerate(trials):
            roots[i,2]+=t['z'];roots[i,3:7]=torch.tensor(t['quat'],device=env.device)
            for j,name in enumerate(POLICY_JOINTS):q[i,names.index(name)]=t['q'][j]
            action[i]=torch.tensor(t['target'],device=env.device)-torch.tensor([HOME_JOINT_POSITIONS[n] for n in POLICY_JOINTS],device=env.device)
        robot.write_root_state_to_sim(roots);robot.write_joint_state_to_sim(q,torch.zeros_like(q));env.scene.write_data_to_sim();env.sim.forward()
        for a in robot.actuators:a.vin_tensor[:,0]=torch.tensor([t['voltage'] for t in trials],device=env.device)
        max_tilt=torch.zeros(len(trials),device=env.device)
        for step in range(150):
            obs,*_=env.step(action);assert torch.isfinite(obs['actor']).all()
            quat=robot.data.root_link_quat_w
            tilt=torch.rad2deg(torch.acos((1-2*(quat[:,1]**2+quat[:,2]**2)).clamp(-1,1)))
            max_tilt=torch.maximum(max_tilt,tilt)
        z=(robot.data.root_link_pos_w[:,2]-env.scene.env_origins[:,2]).tolist()
        for i,t in enumerate(trials):t.update(final_z=z[i],final_tilt_deg=float(tilt[i]),max_tilt_deg=float(max_tilt[i]))
    finally:env.close()
    ranking=[]
    for i,c in enumerate(candidate_list):
        group=[t for t in trials if t['candidate']==i]
        ranking.append({'candidate':i,'hip':c['hip'],'knee':c['knee'],'ankle':c['ankle'],
                        'within_15deg':sum(t['max_tilt_deg']<15 for t in group),
                        'max_initial_self_m':max(t['initial_self_m'] for t in group),
                        'max_tilt_deg':max(t['max_tilt_deg'] for t in group),
                        'median_final_height_m':float(np.median([t['final_z'] for t in group]))})
    ranking.sort(key=lambda x:(-x['within_15deg'],x['max_initial_self_m'],x['max_tilt_deg']))
    report={'protocol':'9 exact-grid candidates x 16 matched random q sigma .12 and root tilt ±10 deg, fixed targets, 3 s, 10.2/11.1/12.4 V; CPU geometry lift at reset; no pushes',
            'device':'physical GPU 0','limitations':'diagnostic holds, not policy or hardware qualification',
            'ranking':ranking,'trials':trials}
    (root/'docs/v3-sit-noise-grid-probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(ranking,indent=2))


if __name__=='__main__':main()
