"""Probe the provisional V3 seated reset with its actual full-contact model."""
import json
from pathlib import Path
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.custom_v2 import recovery
from mjlab_microduck.custom_v2.microduck_v2 import POLICY_JOINTS, HOME_JOINT_POSITIONS
from mjlab_microduck.custom_v2.training import SIT_CANDIDATE


def main():
    cfg = load_env_cfg('Mjlab-StandUp-Flat-MicroDuck-V2-PairedError-Baseline',play=True)
    cfg.scene.num_envs, cfg.seed = 7*16, 20261101
    cfg.curriculum, cfg.terminations = {}, {}
    cfg.events = {k:v for k,v in cfg.events.items() if k in {'expand_bam_friction_fields','v2_runtime',
        'mouth_reset','mouth_hold','reset_base','set_ground_state','v2_reset_joint_guards',
        'v3_reset_ground_clearance','paired_error_reset'}}
    cfg.events['set_ground_state'].params['evaluation_buckets'] = True
    env = ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        recovery._sc0090_recovery_state(env).stage = 0
        env.reset()
        ids = torch.arange(16,32,device=env.device)
        action = torch.zeros((env.num_envs,14),device=env.device)
        for idx,target in SIT_CANDIDATE.items():
            action[:,idx] = target-HOME_JOINT_POSITIONS[POLICY_JOINTS[idx]]
        robot = env.scene['robot']
        max_tilt = torch.zeros(16,device=env.device)
        for step in range(150):
            obs,*_ = env.step(action)
            assert torch.isfinite(obs['actor']).all()
            quat = robot.data.root_link_quat_w[ids]
            tilt = torch.rad2deg(torch.acos((1-2*(quat[:,1]**2+quat[:,2]**2)).clamp(-1,1)))
            max_tilt = torch.maximum(max_tilt,tilt)
        z = robot.data.root_link_pos_w[ids,2] - env.scene.env_origins[ids,2]
        report = {'protocol':'16 noisy actual sitting resets, fixed candidate target, 3 s, no pushes; SC0090 V5 with paired voltage selection',
                  'device':'physical GPU 0','duration_s':3.,'candidate_overrides':SIT_CANDIDATE,
                  'final_heights_m':z.tolist(),'final_tilt_deg':tilt.tolist(),
                  'max_tilt_deg':max_tilt.tolist(),
                  'limitation':'Probe only; not proof of seated equilibrium, recovery or hardware qualification'}
        path=Path(__file__).resolve().parents[1]/'docs/v3-sit-hold-probe.json'
        path.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
