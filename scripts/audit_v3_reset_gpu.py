"""Inspect actual manager reset contacts, by recovery bucket, on GPU 0."""
import argparse
import json
from pathlib import Path
import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.custom_v2 import recovery
from mjlab_microduck.custom_v2.training import guard_ground_clearance
from mjlab_microduck.custom_v2.microduck_v2 import POLICY_JOINTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples-per-bucket',type=int,default=16)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'docs/v3-reset-gpu-audit.json')
    args = parser.parse_args()
    task = 'Mjlab-StandUp-Flat-MicroDuck-V2-PairedError-Baseline'
    cfg = load_env_cfg(task,play=True)
    cfg.scene.num_envs = 7 * args.samples_per_bucket
    cfg.seed = 20261101
    cfg.curriculum = {}
    cfg.events['set_ground_state'].params['evaluation_buckets'] = True
    env = ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        m = env.sim.mj_model
        terrains = [i for i in range(m.ngeom) if m.geom(i).name.startswith('terrain/')]
        if not terrains:
            terrains = [i for i in range(m.ngeom) if m.geom_type[i] == 0]
        assert terrains, 'Could not identify terrain geometry'
        rows = []
        initial_home_checks=[]
        for stage in range(6):
            recovery._sc0090_recovery_state(env).stage = stage
            env.reset()
            robot=env.scene['robot']
            home_ids,home_names=robot.find_joints(POLICY_JOINTS+('mouth',),preserve_order=True)
            assert tuple(home_names)==POLICY_JOINTS+('mouth',)
            delta=(robot.data.joint_pos[:,home_ids]-robot.data.default_joint_pos[:,home_ids]).abs()
            non_sit=recovery._sc0090_recovery_state(env).bucket!=1
            home_max=float(delta[non_sit].max())
            mouth_max=float(delta[:,-1].max())
            assert home_max<1e-6 and mouth_max<1e-6
            initial_home_checks.append({'stage':stage,'non_sitting_15_joint_max_error_rad':home_max,
                                        'all_buckets_mouth_max_error_rad':mouth_max})
            c = env.sim.wp_data.contact
            n = int(wp.to_torch(env.sim.wp_data.nacon).item())
            geoms = wp.to_torch(c.geom)[:n].long()
            world = wp.to_torch(c.worldid)[:n].long()
            depth = (-wp.to_torch(c.dist)[:n]).clamp_min(0)
            terrain = torch.zeros(n,dtype=torch.bool,device=env.device)
            for t in terrains:
                terrain |= (geoms[:,0] == t) | (geoms[:,1] == t)
            for name,mask in [('floor',terrain),('self',~terrain)]:
                per_world = torch.zeros(env.num_envs,device=env.device)
                per_world.scatter_reduce_(0,world[mask],depth[mask],reduce='amax',include_self=True)
                for group,bucket in enumerate(recovery.SC0090_RECOVERY_BUCKETS):
                    values = per_world[group*args.samples_per_bucket:(group+1)*args.samples_per_bucket]
                    rows.append({'stage':stage,'bucket':bucket,'kind':name,'samples':args.samples_per_bucket,
                                 'over_5mm':int((values>.005).sum()),'max_m':float(values.max())})
        # The guard's extra full-world forward must not change integrator state
        # of environments outside the subset being reset.
        env.step(torch.zeros((env.num_envs,14),device=env.device))
        fields = ('qpos','qvel','qacc_warmstart','ctrl')
        before = {name:getattr(env.sim.data,name).clone() for name in fields}
        subset = torch.arange(0,env.num_envs,2,device=env.device)
        untouched = torch.arange(1,env.num_envs,2,device=env.device)
        guard_ground_clearance(env,subset)
        unaffected = {name:float((getattr(env.sim.data,name)[untouched]-value[untouched]).abs().max())
                      for name,value in before.items()}
        path = args.output
        path.write_text(json.dumps({'task':task,'device':'physical GPU 0','terrain_geom_ids':terrains,
                                   'terrain_geom_names':[m.geom(i).name for i in terrains],
                                   'protocol':f'six stages, {args.samples_per_bucket} actual reset worlds per bucket, contact after forward; no physics step',
                                   'initial_home_checks':initial_home_checks,
                                   'unreset_world_state_max_abs_diff':unaffected,
                                   'results':rows},indent=2)+'\n')
        print(f'Wrote {path}')
        for row in rows:
            if row['over_5mm']:
                print(row)
    finally:
        env.close()


if __name__ == '__main__':
    main()
