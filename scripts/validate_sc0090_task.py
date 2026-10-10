"""Check actual SC0090 actor isolation, physical reward, actions and resets."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.sc0090_env_cfg import make_sc0090_env_cfg, MAIN_SETTINGS, MAIN_MASS, M6_PATH
from mjlab_microduck.sc0090.actions import ActionController


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();folder=args.output.resolve();folder.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);cfg=make_sc0090_env_cfg();cfg.scene.num_envs=64;cfg.seed=42
    source=[Path(__file__),Path('src/mjlab_microduck/tasks/sc0090_env_cfg.py'),Path('src/mjlab_microduck/tasks/mdp.py'),
            *Path('src/mjlab_microduck/sc0090').glob('*.py')]
    pre={'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
        'settings':MAIN_SETTINGS,'mass':MAIN_MASS,'tests':['actual_clock_shape','count_scalar_tensor_equivalence',
        'actor_sensor_isolation','physical_reward_isolation','unrelated_world_reset','fault_recovery','nonfinite_no_command'],
        'maximum_action_count_difference':0}
    (folder/'pre-run.json').write_text(json.dumps(pre,indent=2)+'\n')
    with torch.inference_mode():
        env=ManagerBasedRlEnv(cfg,device='cuda:0');obs,_=env.reset();action=env.action_manager.get_term('position')
        checks={}
        checks['actual_clock_shape']=obs['actor'].shape==(64,4) and action.action_dim==1 and env.physics_dt==.005 and env.step_dt==.02
        # Ported vector conversion must equal computer runner's count controller.
        controllers=[ActionController(cfg.actions['position'].contract) for _ in range(64)]
        for c in controllers:c.reset(action.zero_goal)
        maximum=0
        for scale in [-100.,-1.,0.,1.,100.,-.1,.1]:
            raw=torch.full((64,1),scale,device=env.device);action.process_actions(raw)
            expected=np.array([c.step(scale)['goal_count'] for c in controllers])
            maximum=max(maximum,int(abs(action.goal_count.cpu().numpy()-expected).max()))
            action.clock+=20000  # Completed command evidence advances monotonically.
        checks['count_scalar_tensor_equivalence']=maximum==0
        env.reset()
        before=mdp.sc0090_actor(env).clone();critic=mdp.sc0090_critic(env).clone();reward=mdp.sc0090_tracking(env).clone()
        saved=action.pipeline.stream.output_values.clone()
        action.pipeline.stream.output_values[:,:,0]+=.05
        action.pipeline.stream.output_values[:,:,1]+=.1
        changed=mdp.sc0090_actor(env)
        checks['actor_sensor_isolation']=bool(torch.allclose(changed[:,:2]-before[:,:2],torch.tensor([.05,.1],device=env.device),atol=1e-6))
        checks['physical_reward_isolation']=torch.equal(critic,mdp.sc0090_critic(env)) and torch.equal(reward,mdp.sc0090_tracking(env))
        action.pipeline.stream.output_values.copy_(saved)
        # Both a normal boundary/reset and an invalid source must rebuild the
        # sensor history without disturbing unrelated worlds.
        for i in range(80):
            obs,r,terminated,timeouts,_=env.step(torch.full((64,1),.4*np.sin(i*.03),device=env.device))
        buffers={name:value[1:].clone() for name,value in vars(action.pipeline.stream).items()
                 if isinstance(value,torch.Tensor) and value.ndim and value.shape[0]==64 and name!='_rows'}
        q=env.scene['bench'].data.joint_pos[1:].clone();v=env.scene['bench'].data.joint_vel[1:].clone()
        env.reset(env_ids=torch.tensor([0],device=env.device))
        unchanged=all(torch.equal(old,getattr(action.pipeline.stream,name)[1:]) for name,old in buffers.items())
        checks['unrelated_world_reset']=unchanged and torch.equal(q,env.scene['bench'].data.joint_pos[1:]) and torch.equal(v,env.scene['bench'].data.joint_vel[1:])
        action.pipeline.stream.reset(torch.tensor([0],device=env.device))
        flags=mdp.sc0090_fault(env)
        checks['fault_detected']=bool(flags[0]) and not bool(flags[1:].any())
        obs,r,terminated,timeouts,_=env.step(torch.zeros((64,1),device=env.device))
        checks['fault_recovery']=bool(terminated[0]) and bool(torch.isfinite(obs['actor']).all())
        command_before=action.pipeline.stream.command_t.clone()
        try:action.process_actions(torch.full((64,1),float('nan'),device=env.device));rejected=False
        except FloatingPointError:rejected=True
        checks['nonfinite_no_command']=rejected and torch.equal(command_before,action.pipeline.stream.command_t)
        target=env.command_manager.get_term('target')
        target.kind[:4]=torch.arange(4,device=env.device);target.amplitude[:4]=.7;target.frequency[:4]=.25
        curves=torch.stack([target.value_at(torch.full((64,),i*.02,device=env.device))[:4] for i in range(401)])
        checks['target_bounds_rate']=bool(curves.abs().max()<=.8) and bool((curves.diff(dim=0).abs()/.02).max()<=2.)
        target.amplitude[:4]=0.;checks['exact_zero_holds']=bool((target.value_at(torch.ones(64,device=env.device))[:4]==0).all())
        report={'passed':all(checks.values()),'checks':checks,'maximum_action_count_difference':maximum,
                'target_maximum_rad':float(curves.abs().max()),'target_maximum_rate_rad_s':float((curves.diff(dim=0).abs()/.02).max()),
                'scope':'Actual GPU environment runtime/task/action checks; trained-policy long-run and physical clearance remain separate'}
        (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n');env.close()
        print(json.dumps(report),flush=True)
        if not report['passed']:raise SystemExit('Task contract check failed')


if __name__=='__main__':main()
