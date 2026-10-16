"""Frozen-policy reward ordering audit; no formal training changes."""
import argparse,hashlib,json,random
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
import mjlab.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
from rsl_rl.utils import resolve_callable,resolve_obs_groups
from mjlab_microduck.custom_v2 import recovery
from train_v2_standup import write_json
def evaluate_reward(task, policy, device, stage, samples, *, compiled_measurement=False,
             randomization_stage=None, evidence_dir=None,proxy_strength=None):
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    was_training = policy.training
    cases = []
    policy.eval()
    try:
        for seed in (2026101501, 2026101502):
            cfg = load_env_cfg(task, play=True)
            if compiled_measurement:
                cfg.actions['joint_pos'].compile_measurement = True
                cfg.actions['joint_pos'].fuse_registers = True
            if randomization_stage is not None:
                cfg.actions['joint_pos'].randomization_stage = randomization_stage
            if proxy_strength is not None:
                from mjlab_microduck.custom_v2.error_proxy import proxy_joint_pos,proxy_joint_vel
                for name,func in (('joint_pos',proxy_joint_pos),('joint_vel',proxy_joint_vel)):
                    term=cfg.observations['actor'].terms[name]
                    assert term.func is func
                    term.params['strength']=proxy_strength
            cfg.seed, cfg.scene.num_envs = seed, 7 * samples
            cfg.episode_length_s = 8.
            from mjlab.managers import RewardTermCfg
            cfg.curriculum, cfg.terminations = {}, {}
            # Retain the actual training reward stack for this audit.
            cfg.events = {n:t for n,t in cfg.events.items() if n in {
                'expand_bam_friction_fields','v2_runtime','mouth_reset','mouth_hold',
                'reset_base','set_ground_state','v2_reset_joint_guards','sensor_error_reset','reset_action_history'}}
            cfg.events['set_ground_state'].params['evaluation_buckets'] = True
            cfg.observations['actor'].enable_corruption = False
            for name in ('head_pose','body_pose'):
                cfg.commands[name].ranges = tuple((0.,0.) for _ in cfg.commands[name].ranges)
            cmd = cfg.commands['twist']
            cmd.rel_standing_envs = 1.
            for name in ('lin_vel_x','lin_vel_y','ang_vel_z'):
                setattr(cmd.ranges, name, (0.,0.))
            if evidence_dir is not None:
                write_json(evidence_dir/f'eval-config-{seed}.json',asdict(cfg))
            env = ManagerBasedRlEnv(cfg, device=device)
            try:
                state = recovery._sc0090_recovery_state(env)
                state.stage = stage
                obs, _ = env.reset()
                invalid = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
                first = torch.full((env.num_envs,), float('inf'), device=device)
                sensor_fault = torch.zeros_like(invalid)
                recorder=None
                if evidence_dir is not None and hasattr(env,'_v2_joint_feedback'):
                    from mjlab_microduck.custom_v2.feedback_diagnostics import FeedbackFaultRecorder
                    recorder=FeedbackFaultRecorder(env._v2_joint_feedback)
                traces = []
                assert env.reward_manager._scale_by_dt
                names=env.reward_manager.active_terms
                totals=torch.zeros(env.num_envs,len(names),dtype=torch.float64,device=device)
                discounted=torch.zeros_like(totals);first_second=torch.zeros_like(totals)
                total_reward=torch.zeros(env.num_envs,dtype=torch.float64,device=device)
                for step in range(400):
                    obs, reward, _, _, _ = env.step(policy(obs))
                    increments=env.reward_manager._step_reward.double()*env.step_dt
                    totals+=increments;discounted+=(.99**step)*increments
                    if step<50:first_second+=increments
                    total_reward+=reward.double()
                    # Read the exact reward-time metric; do not sample it again
                    # after forward/commands at a different boundary.
                    stable = state.hold >= 25
                    first = torch.where(stable & torch.isinf(first), (step+1)*.02, first)
                    invalid |= ~torch.isfinite(env.sim.data.qpos).all(-1)
                    feedback = getattr(env, '_v2_joint_feedback', None)
                    if feedback is not None:
                        invalid |= ~feedback.ready
                        sensor_fault |= ~feedback.ready
                    if recorder is not None:recorder.control_boundary(step)
                    if evidence_dir is not None:
                        traces.append(torch.cat((env.sim.data.qpos[...],env.sim.data.qvel[...],
                            stable[:,None],invalid[:,None]),dim=-1).clone())
                success = state.success & stable & ~invalid
                torch.testing.assert_close(totals.sum(-1),total_reward,rtol=1e-5,atol=2e-4)
                groups = {}
                for i, name in enumerate(recovery.SC0090_RECOVERY_BUCKETS):
                    mask = state.bucket == i
                    passing = mask & success
                    groups[name] = dict(trials=int(mask.sum()), wins=int(passing.sum()),
                        any_stable=int((mask & state.success).sum()), faults=int((mask & invalid).sum()),
                        sensor_faults=int((mask & sensor_fault).sum()),
                        first_stable_mean_s=float(first[passing].mean()) if passing.any() else None)
                for i,name in enumerate(recovery.SC0090_RECOVERY_BUCKETS):
                    mask=state.bucket==i
                    groups[name].update(total_reward_mean=float(total_reward[mask].mean()),
                        discounted_reward_mean=float(discounted[mask].sum(-1).mean()),
                        terms_mean={term:float(totals[mask,j].mean()) for j,term in enumerate(names)},
                        discounted_terms_mean={term:float(discounted[mask,j].mean()) for j,term in enumerate(names)},
                        first_second_terms_mean={term:float(first_second[mask,j].mean()) for j,term in enumerate(names)})
                if evidence_dir is not None:
                    np.savez_compressed(evidence_dir/f'traces-{seed}.npz',
                        states=torch.stack(traces).cpu().numpy(),bucket=state.bucket.cpu().numpy(),
                        env_origins=env.scene.env_origins.cpu().numpy(),
                        nq=env.sim.mj_model.nq,nv=env.sim.mj_model.nv)
                    if recorder is not None:
                        recorder.save(evidence_dir/f'faults-{seed}.npz',state.bucket)
                        recorder.close()
                frontier_directions={}
                for label,sign in (('positive',1),('negative',-1)):
                    mask=(state.bucket==3)&(state.frontier_direction==sign)
                    frontier_directions[label]=dict(trials=int(mask.sum()),wins=int((mask&success).sum()),
                        faults=int((mask&invalid).sum()))
                cases.append(dict(seed=seed, finite=bool(torch.isfinite(env.sim.data.qpos).all()),
                    groups=groups,frontier_directions=frontier_directions))
            finally:
                env.close()
    finally:
        policy.train(was_training)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)
    return cases

p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
args=p.parse_args();args.output.mkdir(exist_ok=False,parents=True)
torch.set_num_threads(2);torch.cuda.set_device(args.device)
task='Mjlab-StandUp-Flat-MicroDuck-V2-Fresh-DirectionGate-M4-Repaired-S0-O0-R1'
paths={768:Path('runs-v2/fresh-p04-overlap-allocation-001/checkpoint_000768.pt'),832:Path('runs-v2/fresh-p04-direction-allocation-001/checkpoint_000832.pt')}
report=dict(schema=1,scope='frozen policy reward ordering on zero-command nominal recovery; not a formal training or deployment qualification',
 task=task,stage=2,frontier_deg=[120,150],dt=.02,gamma=.99,samples_per_bucket_per_seed=32,
 term_boundary='actual reward_manager once-per-step values; no resampling stateful rewards',arms=[])
for iteration,path in paths.items():
 state=torch.load(path,map_location=args.device,weights_only=False)['actor_state_dict']
 cfg=asdict(load_rl_cfg(task));obs=TensorDict({'actor':torch.zeros(1,61,device=args.device),'critic':torch.zeros(1,74,device=args.device)},batch_size=[1])
 groups=resolve_obs_groups(obs,cfg['obs_groups'],['actor','critic'])
 for option in ('cnn_cfg','distribution_cfg'):
  if cfg['actor'].get(option) is None:cfg['actor'].pop(option,None)
 if cfg['actor'].get('rnn_type') is None:
  for option in ('rnn_type','rnn_hidden_dim','rnn_num_layers'):cfg['actor'].pop(option,None)
 actor=resolve_callable(cfg['actor'].pop('class_name'))(obs,groups,'actor',14,**cfg['actor']).to(args.device)
 actor.load_state_dict(state,strict=True);actor.eval()
 with torch.inference_mode():cases=evaluate_reward(task,actor,args.device,2,32,compiled_measurement=True,randomization_stage='nominal')
 for name,value in state.items():torch.testing.assert_close(actor.state_dict()[name],value,rtol=0,atol=0)
 report['arms'].append(dict(iteration=iteration,checkpoint=str(path.resolve()),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),actor_normalizer_unchanged=True,cases=cases))
 write_json(args.output/'progress.json',report)
 print('reward arm',iteration,[(c['seed'],c['groups']['right_side']['wins'],c['groups']['right_side']['total_reward_mean']) for c in cases],flush=True)
 del actor
report['source_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__),*Path('src').glob('**/*.py')]}
write_json(args.output/'report.json',report)
