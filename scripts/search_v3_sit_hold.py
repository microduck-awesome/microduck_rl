"""Bounded GPU screen for physically supported V3 seated reset candidates."""
import itertools
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.custom_v2.microduck_v2 import POLICY_JOINTS, HOME_JOINT_POSITIONS


def main():
    root = Path(__file__).resolve().parents[1]
    m = mujoco.MjModel.from_xml_path(str(root/'src/mjlab_microduck/custom_v2/assets/scene.xml'))
    d = mujoco.MjData(m)
    floor = m.geom('floor').id
    candidates = []
    for hip,knee,ankle in itertools.product((-.9,-.6,-.3,0.,.3),(-1.4,-1.2,-1.,-.8,-.6),(-1.2,-.8,-.4,0.,.4,.8)):
        mujoco.mj_resetDataKeyframe(m,d,m.key('home').id)
        d.qpos[2] = .02
        for side,roll in [('left',-.07),('right',.07)]:
            for name,val in [('hip_roll',roll),('hip_pitch',hip),('knee',knee),('ankle',ankle)]:
                d.qpos[m.joint(f'{side}_{name}').qposadr[0]] = val
        mujoco.mj_forward(m,d)
        max_self = max([max(0.,-c.dist) for c in d.contact if floor not in (c.geom1,c.geom2)] or [0.])
        max_floor = max([max(0.,-c.dist) for c in d.contact if floor in (c.geom1,c.geom2)] or [0.])
        z = .02 + max_floor + .002
        if max_self <= .0005 and z < .12:
            candidates.append({'hip':hip,'knee':knee,'ankle':ankle,'spawn_z':z,'initial_self_m':max_self})
    assert candidates
    cfg = load_env_cfg('Mjlab-StandUp-Flat-MicroDuck-V2-PairedError-Baseline',play=True)
    cfg.scene.num_envs, cfg.seed = len(candidates), 20261101
    cfg.curriculum, cfg.terminations = {}, {}
    cfg.events = {k:v for k,v in cfg.events.items() if k in {'expand_bam_friction_fields','mouth_hold'}}
    env = ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        env.reset()
        robot = env.scene['robot']
        base = robot.data.default_joint_pos.clone()
        names = list(robot.joint_names)
        q = base.clone()
        act = torch.zeros((len(candidates),14),device=env.device)
        roots = torch.zeros((len(candidates),13),device=env.device)
        roots[:,:3] = env.scene.env_origins
        roots[:,3] = 1.
        for i,c in enumerate(candidates):
            roots[i,2] += c['spawn_z']
            for side,roll in [('left',-.07),('right',.07)]:
                for suffix,val in [('hip_roll',roll),('hip_pitch',c['hip']),('knee',c['knee']),('ankle',c['ankle'])]:
                    name=f'{side}_{suffix}'
                    q[i,names.index(name)] = val
                    act[i,POLICY_JOINTS.index(name)] = val-HOME_JOINT_POSITIONS[name]
        robot.write_root_state_to_sim(roots)
        robot.write_joint_state_to_sim(q,torch.zeros_like(q))
        env.scene.write_data_to_sim()
        env.sim.forward()
        for a in robot.actuators:
            a.vin_tensor.fill_(11.1)
        max_tilt = torch.zeros(len(candidates),device=env.device)
        for step in range(150):
            obs,*_=env.step(act)
            assert torch.isfinite(obs['actor']).all()
            quat=robot.data.root_link_quat_w
            tilt=torch.rad2deg(torch.acos((1-2*(quat[:,1]**2+quat[:,2]**2)).clamp(-1,1)))
            max_tilt=torch.maximum(max_tilt,tilt)
        z=(robot.data.root_link_pos_w[:,2]-env.scene.env_origins[:,2]).cpu().numpy()
        tilt=tilt.cpu().numpy(); max_tilt=max_tilt.cpu().numpy()
        for i,c in enumerate(candidates):
            c.update(final_z=float(z[i]),final_tilt_deg=float(tilt[i]),max_tilt_deg=float(max_tilt[i]))
    finally:
        env.close()
    candidates.sort(key=lambda x:(x['max_tilt_deg'],abs(x['final_z']-.09)))
    report={'protocol':'one exact symmetric pose per grid candidate; no pushes/DR, 11.1 V SC0090 V5, 3 s fixed target, starts 2 mm above floor; self-intersecting candidates excluded',
            'device':'physical GPU 0','not_qualification':'noise, other voltages and recovery untested',
            'count':len(candidates),'candidates':candidates}
    (root/'docs/v3-sit-grid-probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'count':len(candidates),'top':candidates[:10]},indent=2))


if __name__ == '__main__':
    main()
