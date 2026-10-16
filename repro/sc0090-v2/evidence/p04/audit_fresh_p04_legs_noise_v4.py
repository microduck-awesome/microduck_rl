"""Frozen P04 Gaussian versus mean diagnostic; not policy qualification.
Independent standing head-command response, return and camera-motion battery.

Full actual Head2/polish reward audit in controlled standing head-command phases. No learning or checkpoint writes. Current-training and final command ranges are
reported separately. Numerical evaluation is not a declaration of real-robot
qualification, and standing results do not cover walking/turning contexts.
"""
import argparse
from dataclasses import asdict
import hashlib
import inspect
import itertools
import json
import math
import re
import tarfile
from pathlib import Path

import numpy as np
import torch
import mjlab.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import RewardTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
from mjlab_microduck.custom_v2 import recovery
from mjlab_microduck.custom_v2.microduck_v2 import POLICY_JOINTS
from mjlab_microduck.custom_v2.head_metrics import response_metrics,summarize_response,camera_metrics
from mjlab_microduck.tasks import mdp

JOINTS=('neck_pitch','head_pitch','head_yaw','head_roll')


def cases_for(current,final):
    cases=[dict(name='idle',range='zero',target=[0.,0.,0.,0.])]
    for i,name in enumerate(JOINTS):
        for level,ranges in (('small',current),('current',current),('final',final)):
            for sign in (-1,1):
                edge=ranges[i][0 if sign<0 else 1]
                value=math.copysign(min(.03,abs(edge)*.5),edge) if level=='small' else edge
                target=[0.,0.,0.,0.];target[i]=value
                cases.append(dict(name=f'{name}_{level}_{sign:+}',range=level,target=target))
    for level,ranges in (('current',current),('final',final)):
        for pair in itertools.combinations(range(4),2):
            for signs in itertools.product((-1,1),repeat=2):
                target=[0.,0.,0.,0.]
                for i,sign in zip(pair,signs):target[i]=.5*ranges[i][0 if sign<0 else 1]
                cases.append(dict(name=f'pair_{pair}_{signs}_{level}',range=level,target=target))
        for signs in ((1,1,1,1),(-1,-1,-1,-1),(1,-1,1,-1),(-1,1,-1,1)):
            target=[.5*ranges[i][0 if sign<0 else 1] for i,sign in enumerate(signs)]
            cases.append(dict(name=f'four_{signs}_{level}',range=level,target=target))
    return cases


def evaluate_seed(args,seed,cases,output):
    cfg=load_env_cfg(args.task,play=True)
    cfg.actions['joint_pos'].compile_measurement=True
    cfg.actions['joint_pos'].randomization_stage=args.randomization_stage
    if args.position_recipe:
        from mjlab_microduck.custom_v2.training import configure_position_robustness_learning
        applied=configure_position_robustness_learning(cfg,args.position_recipe['stage'],require_corruption=False)
        assert applied==args.position_recipe
    if args.head_recipe:
        from mjlab_microduck.custom_v2.training import configure_head_precision_learning
        applied=configure_head_precision_learning(cfg,args.head_recipe.get("stage",1))
        assert applied["id"]==args.head_recipe["id"]
        assert tuple(map(tuple,applied["ranges"]))==tuple(map(tuple,args.head_recipe["ranges"]))
    cfg.seed,cfg.scene.num_envs=seed,len(cases)*args.repeats
    cfg.curriculum,cfg.terminations={},{}
    head_reward=cfg.rewards['head_pose_tracking']
    # Exact mature polish weights from the formal trainer, not stable-only eval.
    for reward_name,reward_weight in [('action_rate_l2',-.5),('arrival_damping',-.025),
            ('joint_torque_rate_l2',-.0005),('head_pose_bias',.5),('body_pose_tracking',.5)]:
        cfg.rewards[reward_name].weight=reward_weight
    cfg.events={n:t for n,t in cfg.events.items() if n in {
        'expand_bam_friction_fields','v2_runtime','mouth_reset','mouth_hold',
        'reset_base','set_ground_state','v2_reset_joint_guards','sensor_error_reset','reset_action_history'}}
    ground=cfg.events['set_ground_state']
    allowed=inspect.signature(mdp.set_random_ground_state).parameters
    ground.params={k:v for k,v in ground.params.items() if k in allowed}
    ground.func=mdp.set_random_ground_state
    ground.params.update(standing_prob=1.,sitting_prob=0.,face_down_prob=0.,face_up_prob=0.)
    cfg.observations['actor'].enable_corruption=False
    for name in ('head_pose','body_pose'):
        cfg.commands[name].ranges=tuple((0.,0.) for _ in cfg.commands[name].ranges)
    # This battery supplies commands explicitly. Keep unused sampler RNG
    # consumption independent of the checkpoint's training-exposure recipe.
    cfg.commands['head_pose'].zero_command_prob=.6
    if hasattr(cfg.commands['head_pose'],'precision_probability'):
        cfg.commands['head_pose'].precision_probability=0.
    twist=cfg.commands['twist'];twist.rel_standing_envs=1.;twist.rel_forward_envs=0.
    for name in ('lin_vel_x','lin_vel_y','ang_vel_z'):setattr(twist.ranges,name,(0.,0.))
    if args.reward_reference_run is not None:
        reference_run=json.loads(args.reward_reference_run.read_text())
        expected_rewards=reference_run['environment']['rewards']
        runtime_path=args.reward_reference_run.parent/'resume-environment.json'
        if runtime_path.exists():
            runtime_weights=json.loads(runtime_path.read_text())['reward_weights']
            for reward_name,reward_weight in runtime_weights.items():expected_rewards[reward_name]['weight']=reward_weight
        actual_rewards=json.loads(json.dumps(asdict(cfg)['rewards'],default=str))
        def canonical_rewards(value):return re.sub(r'0x[0-9a-fA-F]+','0xADDR',json.dumps(value,sort_keys=True))
        differences=[key for key in expected_rewards.keys()|actual_rewards.keys()
                     if canonical_rewards(expected_rewards.get(key))!=canonical_rewards(actual_rewards.get(key))]
        assert not differences,'Full reward stack differs from actual formal reference: '+str(differences)
        (output/f'reward-reference-{seed}.json').write_text(json.dumps(dict(status='PASS',
            reference=str(args.reward_reference_run.resolve()),reference_sha256=hashlib.sha256(args.reward_reference_run.read_bytes()).hexdigest(),
            terms=len(actual_rewards),functions_parameters_weights_equal=True),indent=2)+'\n')
    (output/f'eval-config-{seed}.json').write_text(json.dumps(asdict(cfg),indent=2,default=str))
    env=ManagerBasedRlEnv(cfg,device=args.device)
    assert math.isclose(env.step_dt,.02) and math.isclose(env.physics_dt,.005)
    try:
        robot=env.scene['robot'];ids,names=robot.find_joints(JOINTS,preserve_order=True)
        assert tuple(names)==JOINTS
        site_ids,_=robot.find_sites('head_camera');assert len(site_ids)==1
        n=env.num_envs;count=300
        desired=torch.zeros(n,4,device=args.device)
        target=torch.tensor([c['target'] for c in cases],device=args.device).repeat_interleave(args.repeats,0)
        native_command_compute=env.command_manager.compute
        def commanded_compute(dt):
            native_command_compute(dt)
            env.command_manager.get_command('head_pose').copy_(desired)
            env.command_manager.get_command('twist').zero_()
            env.command_manager.get_command('body_pose').zero_()
        env.command_manager.compute=commanded_compute
        agent=load_rl_cfg(args.task);agent.upload_model=False;agent.logger='tensorboard'
        wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
        runner=load_runner_cls(args.task)(wrapped,asdict(agent),None,args.device)
        runner.load(str(args.checkpoint),map_location=args.device)
        policy=runner.alg.get_policy().eval()
        initial_actor={name:value.detach().clone() for name,value in policy.state_dict().items()}
        sample_generator=torch.Generator(device=args.device).manual_seed(seed+730000)
        raw_std=policy.distribution.std_param.detach().clone()
        sample_std=(policy.distribution.effective_std().detach().clone()
                    if hasattr(policy.distribution,'effective_std') else raw_std.clone())
        head_actions=[POLICY_JOINTS.index(name) for name in JOINTS]
        noise_mask=torch.ones_like(sample_std)
        if args.noise_mode=='legs':noise_mask[head_actions]=0
        injected_std=sample_std*noise_mask if args.sampled_actions else torch.zeros_like(sample_std)
        (output/f'action-distribution-{seed}.json').write_text(json.dumps(dict(
            joint_order=POLICY_JOINTS,raw_std_parameter=raw_std.tolist(),
            effective_model_std=sample_std.tolist(),injected_std=injected_std.tolist(),
            noise_mode=args.noise_mode,sampled_actions=args.sampled_actions,
            head_indices=head_actions,head_zero_external_noise_checked_every_control=bool(args.sampled_actions and args.noise_mode=='legs'),
            interpretation='Frozen diagnostic; no PPO collection, log probability or update'),indent=2)+'\n')
        assert policy.distribution.std_type=='scalar' and bool((sample_std>0).all())
        obs,_=env.reset(seed=seed)
        initial_state={'qpos':env.sim.data.qpos[...].detach().cpu().clone(),
                       'qvel':env.sim.data.qvel[...].detach().cpu().clone(),
                       'actor_obs':obs['actor'].detach().cpu().clone(),
                       'torch_rng_cpu':torch.get_rng_state().clone(),
                       'torch_rng_cuda':torch.cuda.get_rng_state(args.device).clone()}
        seen=set()
        def snapshot_feedback(obj,path,depth=0):
            if isinstance(obj,torch.Tensor):
                initial_state[path]=obj.detach().cpu().clone();return
            if depth>7 or id(obj) in seen:return
            seen.add(id(obj))
            if isinstance(obj,dict):items=obj.items()
            elif isinstance(obj,(list,tuple)):items=enumerate(obj)
            elif type(obj).__module__.startswith(('mjlab_microduck.sc0090_v2','mjlab_microduck.custom_v2')):
                items=vars(obj).items()
            else:return
            for key,value in items:
                if str(key) not in ('env','_env'):snapshot_feedback(value,path+'.'+str(key),depth+1)
        snapshot_feedback(env._v2_joint_feedback,'feedback')
        if args.initial_state_reference is not None:
            assert env._v2_joint_feedback.pipeline.stream.model.mode=='O0'
            reference_path=args.initial_state_reference/f'initial-state-{seed}.pt'
            reference=torch.load(reference_path,map_location='cpu',weights_only=False)
            assert initial_state.keys()==reference.keys(), 'Initial tensor coverage differs'
            replay_key='feedback.pipeline.stream.raw_registers'
            def byte_equal(a,b):
                return a.shape==b.shape and a.dtype==b.dtype and torch.equal(
                    a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))
            mismatch=[key for key in reference if key!=replay_key and not byte_equal(initial_state[key],reference[key])]
            assert not mismatch, 'Non-replayed initial tensors differ: '+str(mismatch)
            before=initial_state[replay_key]
            raw_target=env._v2_joint_feedback.pipeline.stream.raw_registers
            raw_target.copy_(reference[replay_key].to(raw_target.device))
            initial_state[replay_key]=raw_target.detach().cpu().clone()
            assert all(byte_equal(initial_state[key],reference[key]) for key in reference)
            def tensor_sha(value):return hashlib.sha256(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
            replay=dict(schema=1,reference=str(reference_path.resolve()),reference_file_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                replayed_fields=[replay_key],before_sha256=tensor_sha(before),after_sha256=tensor_sha(initial_state[replay_key]),
                before_numeric_max_difference=float((before-reference[replay_key]).abs().max()),
                all_captured_initial_tensors_bitwise=True,tensor_count=len(reference),
                semantics='Explicit controlled cross-checkpoint initial feedback history replay only; not production reset or a learned-policy repair',
                limitation='Coverage is the 90 captured tensors including physical qpos/qvel, actor observation, global RNG and reachable S0 tensors; not arbitrary simulator internals or private generator object states')
            (output/f'initial-replay-{seed}.json').write_text(json.dumps(replay,indent=2)+'\n')
        torch.save(initial_state,output/f'initial-state-{seed}.pt')
        head_actions=[POLICY_JOINTS.index(name) for name in JOINTS]
        if args.direct_head_target:
            assert args.task.endswith('Official-Baseline')
            action_cfg=cfg.actions['joint_pos']
            assert action_cfg.scale==1. and action_cfg.use_default_offset
            assert tuple(action_cfg.actuator_names)==tuple(POLICY_JOINTS)
        actor_terms=env.observation_manager.active_terms['actor']
        actor_dims=env.observation_manager.group_obs_term_dim['actor']
        head_index=actor_terms.index('head_command')
        head_offset=sum(math.prod(dim) for dim in actor_dims[:head_index])
        assert head_offset==51 and actor_dims[head_index]==(4,)
        initial=robot.data.joint_pos[:,ids]-robot.data.default_joint_pos[:,ids]
        reward_names=list(env.reward_manager.active_terms)
        assert env.reward_manager._scale_by_dt
        head_term_index=reward_names.index('head_pose_tracking')
        full_rewards=torch.empty(count,n,len(reward_names),device=args.device)
        returned_rewards=torch.empty(count,n,device=args.device)
        positions=torch.empty(count,n,4,device=args.device)
        commands=torch.empty_like(positions)
        health=torch.empty(count,n,4,device=args.device)
        invalid=torch.zeros(count,n,dtype=torch.bool,device=args.device)
        camera=torch.empty(count*4+1,n,4,device=args.device)
        qpos=torch.empty(count,n,env.sim.mj_model.nq,device=args.device)
        control=[0];substep=[0]
        native_reward=env.reward_manager.compute
        fine_scale = head_reward.params.get('fine_std')
        if fine_scale is not None:
            fine_scale = torch.as_tensor(fine_scale, dtype=torch.float32, device=args.device)
        def recorded_reward(dt):
            result=native_reward(dt)
            index=control[0]
            full_rewards[index].copy_(env.reward_manager._step_reward*dt)
            returned_rewards[index].copy_(result)
            torch._assert_async(torch.isclose(full_rewards[index].sum(-1),result,rtol=1e-5,atol=2e-5).all(),
                                'Cached weighted reward terms do not match returned reward')
            q=robot.data.joint_pos[:,ids]-robot.data.default_joint_pos[:,ids]
            command=env.command_manager.get_command('head_pose')
            positions[index].copy_(q);commands[index].copy_(command)
            root=robot.data.root_link_quat_w
            tilt=torch.acos((1-2*(root[:,1].square()+root[:,2].square())).clamp(-1,1))
            height=robot.data.root_link_pos_w[:,2]-env.scene.env_origins[:,2]
            stable=recovery._sc0090_recovery_state(env).hold>=25
            health[index].copy_(torch.stack((height,tilt,robot.data.root_link_ang_vel_b.norm(dim=1),stable.float()),-1))
            invalid[index].copy_(~torch.isfinite(env.sim.data.qpos[...]).all(-1))
            feedback=getattr(env,'_v2_joint_feedback',None)
            if feedback is not None:invalid[index] |= ~feedback.ready
            # Independent per-axis errors must reproduce the task's actual
            # physical head reward at the same pre-reset boundary.
            params=head_reward.params;error=q-command
            expected=torch.exp(-(error/params.get('std',.5)).square())
            if params.get('fine_std') is not None:
                weight=params.get('fine_weight',.5)
                expected=(1-weight)*expected+weight*torch.exp(-(error/fine_scale).square())
            if params.get('precision_scale') is not None:
                expected *= .5 + .5/(1 + error.abs()/params['precision_scale'])
            actual=env.reward_manager._step_reward[:,head_term_index]/head_reward.weight
            expected_scalar=expected.mean(-1)
            if params.get('balance_active_axes',False):
                active_mask=(command!=0).to(expected.dtype)
                active_count=active_mask.sum(-1)
                active_group=(expected*active_mask).sum(-1)/active_count.clamp_min(1)
                inactive_group=(expected*(1-active_mask)).sum(-1)/(4-active_count).clamp_min(1)
                expected_scalar=torch.where((active_count==1)|(active_count==3),.5*(active_group+inactive_group),expected_scalar)
            torch._assert_async(torch.isclose(expected_scalar,actual,rtol=1e-6,atol=1e-7).all(),
                                'Head metric and training reward disagree')
            return result
        env.reward_manager.compute=recorded_reward
        native_substep=env.metrics_manager.compute_substep
        def recorded_substep():
            native_substep()
            camera[substep[0]].copy_(robot.data.site_quat_w[:,site_ids[0]])
            substep[0]+=1
        env.metrics_manager.compute_substep=recorded_substep
        with torch.inference_mode():
            for step in range(count):
                control[0]=step
                torch._assert_async((obs['actor'][:,head_offset:head_offset+4]
                    == env.command_manager.get_command('head_pose')).all(),
                    'Actor head command does not match the command used for this action')
                # Set the command that will populate the NEXT observation.
                # The action being applied now uses the existing observation.
                desired.copy_(target if 100<=step+1<200 else torch.zeros_like(target))
                action=policy(obs)
                if args.sampled_actions:
                    policy.distribution.update(action)
                    torch._assert_async((policy.distribution.std==sample_std[None]).all(),
                                        'Frozen Gaussian scale mismatch')
                    noise=torch.randn(action.shape,device=action.device,dtype=action.dtype,
                        generator=sample_generator)*injected_std
                    original_head=action[:,head_actions].clone()
                    action=action+noise
                    if args.noise_mode=='legs':
                        torch._assert_async((noise[:,head_actions]==0).all(),'Head external noise is nonzero')
                        torch._assert_async((action[:,head_actions]==original_head).all(),'Head action changed by external noise')
                if args.direct_head_target:
                    action=action.clone()
                    action[:,head_actions]=env.command_manager.get_command('head_pose')
                obs,_,_,_=wrapped.step(action)
                qpos[step].copy_(env.sim.data.qpos[...])
            camera[-1].copy_(robot.data.site_quat_w[:,site_ids[0]])
        torch.cuda.synchronize(args.device)
        assert substep[0]==1200
        torch.testing.assert_close(commands[:100],torch.zeros_like(commands[:100]),rtol=0,atol=0)
        torch.testing.assert_close(commands[100:200],target[None].expand(100,-1,-1),rtol=0,atol=0)
        torch.testing.assert_close(commands[200:],torch.zeros_like(commands[200:]),rtol=0,atol=0)
        responses={
            'step':response_metrics(positions[100:200],positions[99],target,target),
            'return':response_metrics(positions[200:],positions[199],torch.zeros_like(target),-target)}
        cameras={name:camera_metrics(camera[a*4:b*4+1]) for name,a,b in
                 (('idle',0,100),('step',100,200),('return',200,300))}
        results=[]
        for i,case in enumerate(cases):
            rows=slice(i*args.repeats,(i+1)*args.repeats)
            item={**case,'seed':seed,'trials':args.repeats,
                  'invalid_trials':int(invalid[:,rows].any(0).sum()),
                  'initial_head_error_deg':(initial[rows].mean(0)*180/math.pi).tolist()}
            for phase,a,b in (('step',100,200),('return',200,300)):
                h=health[a:b,rows]
                item[phase]={**summarize_response({k:v[rows] for k,v in responses[phase].items()}),
                    'stable_fraction':float(h[...,3].mean()),
                    'height_min_m':float(h[...,0].min()),
                    'tilt_p95_deg':float(torch.quantile(h[...,1],.95)*180/math.pi)}
            item['full_reward']={}
            for reward_phase,ra,rb in [('idle',0,100),('step',100,200),('return',200,300)]:
                phase_terms=full_rewards[ra:rb,rows]
                local_discount=(.99**torch.arange(rb-ra,device=args.device))[:,None,None]
                item['full_reward'][reward_phase]=dict(
                    return_mean=float(phase_terms.sum((0,2)).mean()),
                    discounted_return_mean=float((phase_terms*local_discount).sum((0,2)).mean()),
                    tail_half_s_return_mean=float(phase_terms[-25:].sum((0,2)).mean()),
                    terms={reward_name:float(phase_terms[:,:,j].sum(0).mean()) for j,reward_name in enumerate(reward_names)},
                    tail_half_s_terms={reward_name:float(phase_terms[-25:,:,j].sum(0).mean()) for j,reward_name in enumerate(reward_names)})
            item['camera']={phase:{k:float(v[rows].mean()) for k,v in metrics.items()}
                            for phase,metrics in cameras.items()}
            results.append(item)
        np.savez_compressed(output/f'traces-{seed}.npz',
            full_weighted_reward_dt=full_rewards.cpu().numpy(),returned_reward=returned_rewards.cpu().numpy(),reward_names=np.array(reward_names),
            physical_head_rad=positions.cpu().numpy(),command_rad=commands.cpu().numpy(),
            camera_quat_w=camera.cpu().numpy(),health=health.cpu().numpy(),
            invalid=invalid.cpu().numpy(),qpos=qpos.cpu().numpy(),env_origins=env.scene.env_origins.cpu().numpy())
        for name,value in initial_actor.items():
            torch.testing.assert_close(policy.state_dict()[name],value,rtol=0,atol=0)
        return results
    finally:
        env.close()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--task',required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--repeats',type=int,default=8)
    p.add_argument('--sampled-actions',action='store_true')
    p.add_argument('--noise-mode',choices=('all','legs'),default='legs')
    p.add_argument('--reward-stage',type=int,help='Explicit common head reward stage for all checkpoints; actor checkpoint remains unchanged')
    p.add_argument('--reward-reference-run',type=Path,help='Formal run.json whose full reward stack plus saved runtime polish weights must match')
    p.add_argument('--initial-state-reference',type=Path,help='Diagnostic only: copy mean-arm initial raw speed/load/voltage ring after exact match of every other captured tensor')
    p.add_argument('--randomization-stage',choices=('nominal','measured'),default='measured')
    p.add_argument('--reference-report',type=Path,
                   help='Also evaluate the exact current-range cases of an earlier report')
    p.add_argument('--direct-head-target',action='store_true',
                   help='Diagnostic only: replace head actions by requested motor position targets')
    args=p.parse_args();assert not (args.sampled_actions and args.direct_head_target);args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.cuda.set_device(args.device)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    args.head_recipe=checkpoint['infos']['v2'].get('head_precision_recipe')
    args.checkpoint_head_recipe=args.head_recipe
    if args.reward_stage is not None:
        from mjlab_microduck.custom_v2.training import configure_head_precision_learning
        reward_cfg=load_env_cfg(args.task)
        args.head_recipe=configure_head_precision_learning(reward_cfg,args.reward_stage)
    args.position_recipe=checkpoint['infos']['v2'].get('position_robustness_recipe')
    with tarfile.open(args.output/'source.tar.gz','w:gz') as archive:
        for source in [Path(__file__),*Path('src').glob('**/*.py')]:
            archive.add(source,arcname=str(('diagnostic/'+source.name) if source.is_absolute() else source))
    cfg=load_env_cfg(args.task)
    steps=checkpoint['infos']['env_state']['common_step_counter']
    stages=cfg.curriculum['head_pose_range'].params['range_stages']
    # A stage introduced at the checkpoint boundary has had no complete
    # rollout of exposure yet. Assess the last complete rollout's range.
    trained_step=max(0,steps-load_rl_cfg(args.task).num_steps_per_env)
    current=next(s['ranges'] for s in reversed(stages) if trained_step>=s['step'])
    if args.head_recipe:
        current=args.head_recipe['ranges']
    final=stages[-1]['ranges'];cases=cases_for(current,final)
    reference=None
    if args.reference_report is not None:
        previous=json.loads(args.reference_report.read_text())
        reference=dict(path=str(args.reference_report.resolve()),
            sha256=hashlib.sha256(args.reference_report.read_bytes()).hexdigest(),
            ranges=previous['current_ranges'])
        for case in cases_for(reference['ranges'],final):
            if case['range']=='current':
                cases.append({**case,'name':case['name'].replace('current','reference'),
                              'range':'reference'})
    records=[]
    for seed in (2026101503,2026101504):records.extend(evaluate_seed(args,seed,cases,args.output))
    report=dict(schema=1,task=args.task,checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        completed_updates=checkpoint['infos']['v2']['completed_updates'],
        randomization_stage=args.randomization_stage,execution_version='p04-o0-valid-raw-native-counts-v2',
        actor_and_normalizer_unchanged=True,
        checkpoint_head_recipe=args.checkpoint_head_recipe,common_reward_stage=args.reward_stage,
        full_mature_reward_stack=True,reward_computation='Exactly one native manager call per step; read cached weighted terms times dt, per-step returned-sum parity, no duplicate head-bias EMA calls',
        initial_state_reference=str(args.initial_state_reference.resolve()) if args.initial_state_reference else None,
        sampled_actions=args.sampled_actions,noise_mode=args.noise_mode,sampling_rng='independent torch.Generator seed+730000; same initial-state artifacts must be compared',
        action_mode=('diagnostic_frozen_'+args.noise_mode+'_gaussian' if args.sampled_actions else 'diagnostic_direct_head_motor_target_with_policy_legs' if args.direct_head_target
                     else 'checkpoint_policy'),
        direct_target_limitation=('No gravity/friction compensation; failure does not establish physical '
            'unreachability and results do not qualify the policy' if args.direct_head_target else None),
        protocol='standing 2s idle + 2s step + 2s return; physical q at reward boundary; camera 200 Hz',
        camera_timing='substep forward geometry at t=0,.005,...,5.995; final forward at 6.0',
        current_ranges=current,final_ranges=final,joint_order=JOINTS,head_precision_recipe=args.head_recipe,
        position_observation_recipe=args.position_recipe,
        camera_site='head_camera CAD placeholder; angular metrics only, not calibrated pixel blur',
        head_reward_boundary_parity_checked=True,actor_command_slot_checked=True,
        reference_range_source=reference,records=records,
        limitation='Diagnostic standing battery; no walking/turning, disturbance, hardware or overall qualification claim',
        source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            (str(Path(__file__).resolve()),'src/mjlab_microduck/custom_v2/head_metrics.py')})
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(dict(output=str(args.output),cases=len(cases),seeds=2,records=len(records))))


if __name__=='__main__':main()
