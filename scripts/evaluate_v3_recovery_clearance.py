"""Add 200 Hz limit/contact evidence to the existing seven-bucket evaluator.

Run only after a training allocation has finished; this does not alter the
policy, environment, reward, evaluation seeds or success criterion. Head and
body contacts are measured, not automatically labelled failures during a rise.
"""
import argparse
import json
import math
from pathlib import Path
import runpy

import torch
import warp as wp
import train_v2_standup as evaluator
from mjlab_microduck.custom_v2 import recovery
from mjlab_microduck.custom_v2.microduck_v2 import POLICY_JOINTS, HOME_JOINT_POSITIONS


def summarize_by_bucket(values, group_size):
    """One JSON-friendly summary for each of the seven equal-sized buckets."""
    result = {}
    for group, name in enumerate(recovery.SC0090_RECOVERY_BUCKETS):
        sample = values[group*group_size:(group+1)*group_size]
        result[name] = {
            'samples': len(sample),
            'count_any': int((sample > 0).sum()),
            'max': float(sample.max()),
        }
    return result


def update_joint_metrics(q, limits, smallest, streak, longest, violation, threshold):
    """Update per-world/joint limit evidence at one physics boundary."""
    clearance = torch.minimum(q-limits[:,:,0],limits[:,:,1]-q)
    smallest = torch.minimum(smallest,clearance)
    streak = torch.where(clearance <= threshold,streak+1,0)
    longest = torch.maximum(longest,streak)
    violation = violation + (clearance < 0).to(torch.int32)
    return smallest,streak,longest,violation


def update_home_metrics(q, home, absolute_sum, signed_sum, absolute_max):
    """Accumulate physical joint error relative to the model's HOME."""
    error=q-home
    return absolute_sum+error.abs(), signed_sum+error, torch.maximum(absolute_max,error.abs())


def reduce_contacts(world,pair,dist,count,n,terrain,head,feet):
    """Reduce packed MuJoCo contacts into per-world flags and depths."""
    device=world.device
    valid = ((torch.arange(len(world),device=device)<count)
             & (world>=0) & (world<n) & (dist<=0))
    ground = (pair[:,0]==terrain) | (pair[:,1]==terrain)
    is_head = torch.zeros_like(valid)
    is_foot = torch.zeros_like(valid)
    for geom in head:
        is_head |= (pair[:,0]==geom) | (pair[:,1]==geom)
    for geom in feet:
        is_foot |= (pair[:,0]==geom) | (pair[:,1]==geom)
    masks = {'head_ground':valid & ground & is_head,
             'feet_ground':valid & ground & is_foot,
             'other_body_ground':valid & ground & ~(is_head | is_foot),
             'self':valid & ~ground}
    observations={}
    for name,mask in masks.items():
        detected = torch.zeros(n,device=device,dtype=torch.int32)
        depth = torch.zeros(n,device=device)
        wid = world.clamp(0,n-1)
        detected.scatter_reduce_(0,wid,mask.to(torch.int32),reduce='amax',include_self=True)
        depth.scatter_reduce_(0,wid,torch.where(mask,(-dist).clamp_min(0),0.),reduce='amax',include_self=True)
        observations[name]=(detected,depth)
    return observations


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output',type=Path,required=True)
    args,_ = parser.parse_known_args()
    real_env = evaluator.ManagerBasedRlEnv

    def instrumented_env(cfg,device):
        env = real_env(cfg,device=device)
        robot = env.scene['robot']
        ids,names = robot.find_joints(POLICY_JOINTS,preserve_order=True)
        assert tuple(names) == POLICY_JOINTS
        home_names=POLICY_JOINTS+('mouth',)
        home_ids,found=robot.find_joints(home_names,preserve_order=True)
        assert tuple(found)==home_names
        expected_home=torch.tensor([HOME_JOINT_POSITIONS[name] for name in home_names],device=device)
        torch.testing.assert_close(robot.data.default_joint_pos[0,home_ids],expected_home,rtol=0,atol=1e-7)
        model = env.sim.mj_model
        terrain = [i for i in range(model.ngeom) if model.geom(i).name == 'terrain']
        assert len(terrain) == 1
        def named(suffix):
            return [i for i in range(model.ngeom) if model.geom(i).name.removeprefix('robot/') in suffix]
        head = named({'ground_head_shell_collision','ground_jaw_collision'})
        feet = named({'left_foot_collision','right_foot_collision'})
        assert len(head) == 2 and len(feet) == 2
        n = env.num_envs
        assert n % len(recovery.SC0090_RECOVERY_BUCKETS) == 0
        group_size = n // len(recovery.SC0090_RECOVERY_BUCKETS)
        threshold = math.radians(5.)
        smallest = torch.full((n,14),float('inf'),device=device)
        streak = torch.zeros((n,14),device=device,dtype=torch.int32)
        longest = torch.zeros_like(streak)
        violation = torch.zeros_like(streak)
        home_abs_sum=torch.zeros((n,15),device=device)
        home_signed_sum=torch.zeros((n,15),device=device)
        home_abs_max=torch.zeros((n,15),device=device)
        counts = {name:torch.zeros(n,device=device,dtype=torch.int32)
                  for name in ('head_ground','other_body_ground','feet_ground','self')}
        terminal = {name:torch.zeros(n,device=device,dtype=torch.int32) for name in counts}
        contact_depth = {name:torch.zeros(n,device=device) for name in counts}
        num_substeps = 0
        original_substep = env.metrics_manager.compute_substep
        original_close = env.close

        def capture():
            nonlocal num_substeps,smallest,streak,longest,violation,home_abs_sum,home_signed_sum,home_abs_max
            original_substep()
            q = robot.data.joint_pos[:,ids]
            limits = robot.data.joint_pos_limits[:,ids,:]
            smallest,streak,longest,violation = update_joint_metrics(
                q,limits,smallest,streak,longest,violation,threshold)
            if num_substeps >= 1500:
                home_abs_sum,home_signed_sum,home_abs_max=update_home_metrics(
                    robot.data.joint_pos[:,home_ids],robot.data.default_joint_pos[:,home_ids],
                    home_abs_sum,home_signed_sum,home_abs_max)
            c = env.sim.wp_data.contact
            world = wp.to_torch(c.worldid).long()
            pair = wp.to_torch(c.geom).long()
            dist = wp.to_torch(c.dist)
            observations = reduce_contacts(world,pair,dist,wp.to_torch(env.sim.wp_data.nacon),
                                           n,terrain[0],head,feet)
            for name,(detected,depth) in observations.items():
                counts[name] += detected
                if num_substeps >= 1500:
                    terminal[name] += detected
                contact_depth[name] = torch.maximum(contact_depth[name],depth)
            num_substeps += 1

        def close():
            try:
                if num_substeps != 1600:
                    (args.output/f'clearance-{cfg.seed}-incomplete.json').write_text(
                        json.dumps({'seed':cfg.seed,'recorded_substeps':num_substeps,'expected_substeps':1600})+'\n')
                    return
                joint_results = {}
                home_results = {}
                for group,bucket in enumerate(recovery.SC0090_RECOVERY_BUCKETS):
                    sl = slice(group*group_size,(group+1)*group_size)
                    joint_results[bucket] = {}
                    for index,name in enumerate(POLICY_JOINTS):
                        joint_results[bucket][name] = {
                            'minimum_clearance_rad':float(smallest[sl,index].min()),
                            'samples_with_limit_violation':int((violation[sl,index]>0).sum()),
                            'samples_within_5deg':int((longest[sl,index]>0).sum()),
                            'samples_within_5deg_over_1s':int((longest[sl,index]>200).sum()),
                            'max_continuous_within_5deg_s':float(longest[sl,index].max())*.005,
                        }
                    home_results[bucket] = {}
                    for index,name in enumerate(home_names):
                        home_results[bucket][name] = {
                            'final_half_second_mean_abs_error_deg':float(home_abs_sum[sl,index].mean())/100*180/math.pi,
                            'final_half_second_mean_signed_error_deg':float(home_signed_sum[sl,index].mean())/100*180/math.pi,
                            'final_half_second_max_abs_error_deg':float(home_abs_max[sl,index].max())*180/math.pi,
                            'samples_with_all_final_errors_within_5deg':int((home_abs_max[sl,index]<=math.radians(5)).sum()),
                        }
                report = {
                    'seed':cfg.seed,'samples_per_bucket':group_size,'physics_dt_s':env.physics_dt,
                    'duration_s':num_substeps*env.physics_dt,
                    'joint_threshold_deg':5.,'joint_results':joint_results,
                    'home_joint_positions_rad':HOME_JOINT_POSITIONS,
                    'home_reference':'assembly V3 default joint positions, including independent mouth',
                    'final_half_second_home_results':home_results,
                    'contact_substeps':{name:summarize_by_bucket(value,group_size) for name,value in counts.items()},
                    'final_half_second_contact_substeps':{name:summarize_by_bucket(value,group_size) for name,value in terminal.items()},
                    'max_contact_penetration_m':{name:summarize_by_bucket(value,group_size) for name,value in contact_depth.items()},
                    'head_geoms':[model.geom(i).name for i in head],
                    'foot_geoms':[model.geom(i).name for i in feet],
                    'interpretation':'All episodes included; contact is descriptive, including necessary ground support during recovery. Limits are model limits, not measured robot stops.',
                }
                (args.output/f'clearance-{cfg.seed}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            finally:
                original_close()
        env.metrics_manager.compute_substep = capture
        env.close = close
        return env

    evaluator.ManagerBasedRlEnv = instrumented_env
    runpy.run_path(str(Path(__file__).with_name('evaluate3762_recovery.py')),run_name='__main__')


if __name__ == '__main__':
    main()
