"""Measure V3 full-contact HOME hold using the actual SC0090 V5 actuator."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.custom_v2.training import make_standup_env_cfg
from mjlab_microduck.custom_v2.microduck_v2 import ASSETS, HOME_HEIGHT, HOME_JOINT_POSITIONS


def main():
    cfg = make_standup_env_cfg(play=True)
    cfg.scene.num_envs = 3
    cfg.seed = 42
    cfg.events = {k:v for k,v in cfg.events.items() if k == 'expand_bam_friction_fields'}
    cfg.curriculum, cfg.terminations = {}, {}
    cfg.episode_length_s = 10.
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    trials = []
    try:
        env.reset()
        robot = env.scene['robot']
        base = robot.data.default_joint_pos.clone()
        names = list(robot.joint_names)
        assert np.allclose(base[0].cpu(), [HOME_JOINT_POSITIONS[n] for n in names])
        root = torch.zeros((3,13),device=env.device)
        root[:,2] = HOME_HEIGHT
        root[:,3] = 1.
        adr = int(env.sim.mj_model.joint('robot/trunk_base_freejoint').qposadr[0])
        for voltage in (10.2,11.1,12.4):
            env.reset()
            q = base.clone()
            for seed in (1,2):
                q[seed] += torch.tensor(np.random.default_rng(seed).normal(0,.003,len(names)),device=env.device,dtype=q.dtype)
            robot.write_root_state_to_sim(root)
            robot.write_joint_state_to_sim(q,torch.zeros_like(q))
            robot.set_joint_position_target(base)
            for actuator in robot.actuators:
                actuator.vin_tensor.fill_(voltage)
            env.scene.write_data_to_sim()
            env.sim.forward()
            maximum = torch.zeros(3,device=env.device)
            for step in range(150):
                obs,*_ = env.step(torch.zeros((3,14),device=env.device))
                assert torch.isfinite(obs['actor']).all() and torch.isfinite(env.sim.data.qpos).all()
                quat = env.sim.data.qpos[:,adr+3:adr+7]
                tilt = torch.rad2deg(torch.acos((1-2*(quat[:,1]**2+quat[:,2]**2)).clamp(-1,1)))
                maximum = torch.maximum(maximum,tilt)
            for seed in range(3):
                trials.append({'voltage_v':voltage,'seed':seed,'maximum_tilt_deg':float(maximum[seed]),
                               'final_tilt_deg':float(tilt[seed]),
                               'final_root_height_m':float(env.sim.data.qpos[seed,adr+2]),
                               'final_joint_error_max_rad':float((robot.data.joint_pos[seed]-base[seed]).abs().max())})
    finally:
        env.close()
    report = {'status':'PASS_NO_FALL' if all(x['maximum_tilt_deg'] < 15 for x in trials) else 'HOLD_FAILURE',
              'model_sha256':hashlib.sha256((ASSETS/'robot.xml').read_bytes()).hexdigest(),
              'duration_s':3.,'actuator':'SC0090 V5 P15/D30','device':'physical GPU 0',
              'protocol':'3 s fixed HOME target; no DR or pushes; two perturbed and one exact initial joint state',
              'limitations':'Not recovery training or hardware qualification', 'trials':trials}
    path = Path(__file__).resolve().parents[1]/'docs/v3-full-contact-home-hold.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
