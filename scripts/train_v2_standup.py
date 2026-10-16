"""Bounded V2 standup allocation with independent, two-seed recovery evaluation."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import tarfile
import time

import numpy as np
import torch
import warp as wp
import mjlab.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab_microduck.custom_v2 import recovery


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def evaluate(task, policy, device, stage, samples, *, compiled_measurement=False,
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
            cfg.rewards = {'stable_standing': RewardTermCfg(func=recovery.sc0090_stable_standing, weight=1.)}
            cfg.events = {n:t for n,t in cfg.events.items() if n in {
                'expand_bam_friction_fields','v2_runtime','mouth_reset','mouth_hold',
                'reset_base','set_ground_state','v2_reset_joint_guards','sensor_error_reset'}}
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
                for step in range(400):
                    obs, _, _, _, _ = env.step(policy(obs))
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
                groups = {}
                for i, name in enumerate(recovery.SC0090_RECOVERY_BUCKETS):
                    mask = state.bucket == i
                    passing = mask & success
                    groups[name] = dict(trials=int(mask.sum()), wins=int(passing.sum()),
                        any_stable=int((mask & state.success).sum()), faults=int((mask & invalid).sum()),
                        sensor_faults=int((mask & sensor_fault).sum()),
                        first_stable_mean_s=float(first[passing].mean()) if passing.any() else None)
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


def validate_rng_preflight_reuse(smoke, restored, *, checkpoint_sha256, parent_task, task, is_adaptation):
    """Qualify RNG implementation/config; a later checkpoint has a new valid stream."""
    expected = smoke['rng_restore']
    for field in ('source_sha256', 'source_device', 'target_device'):
        assert expected[field] == restored[field], 'RNG preflight differs: ' + field
    assert restored['status'] == 'PASS'
    assert restored['source_active_cuda_rng_sha256'] == restored['target_restored_cuda_rng_sha256'], 'Actual parent active stream was not restored'
    assert restored['active_stream_mapped'] == (restored['source_device'] != restored['target_device'])
    preflight_parent = smoke['initialization']['sha256']
    assert preflight_parent and checkpoint_sha256
    same_parent = preflight_parent == checkpoint_sha256
    if same_parent:
        assert expected['source_active_cuda_rng_sha256'] == restored['source_active_cuda_rng_sha256'], 'Same-parent RNG differs from preflight'
    else:
        assert not is_adaptation and parent_task == task == smoke['task'], 'Different preflight parent requires explicit same-task continuation'
    return dict(status='PASS', mode='same-parent-exact' if same_parent else 'same-task-later-parent',
        preflight_checkpoint_sha256=preflight_parent, actual_checkpoint_sha256=checkpoint_sha256,
        preflight_active_cuda_rng_sha256=expected['source_active_cuda_rng_sha256'],
        actual_parent_active_cuda_rng_sha256=restored['source_active_cuda_rng_sha256'],
        actual_restored_cuda_rng_sha256=restored['target_restored_cuda_rng_sha256'],
        source_device=restored['source_device'], target_device=restored['target_device'],
        same_source_and_target_devices_as_preflight=True,
        scope='RNG qualification reuse only; existing source, task, distribution, PPO, head and normalization guards remain mandatory')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--task', required=True)
    p.add_argument('--device', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=8192)
    p.add_argument('--iterations', type=int, default=1000)
    p.add_argument('--eval-interval', type=int, default=500)
    p.add_argument('--eval-samples', type=int, default=128)
    p.add_argument('--preflight', type=Path, required=True)
    parents=p.add_mutually_exclusive_group()
    parents.add_argument('--resume', type=Path)
    parents.add_argument('--adapt-from', type=Path,
                         help='Explicit cross-task parent; retain weights, optimizer and mature clocks')
    p.add_argument('--validation-only', action='store_true')
    p.add_argument('--head-precision', action='store_true')
    p.add_argument('--head-precision-stage',type=int,choices=(1,2),default=1)
    p.add_argument("--freeze-normalizers", action="store_true")
    p.add_argument("--transition-learning-rate",type=float)
    args = p.parse_args()
    if args.adapt_from is not None:args.resume=args.adapt_from
    if args.num_envs not in (8192,16384,32768) and not args.validation_only:
        p.error('Formal environment counts require a measured 8192/16384/32768 configuration')
    lineage=None
    parent_run=None
    if args.resume is not None:
        parent_run=json.loads((args.resume.resolve().parent/'run.json').read_text())
        if parent_run['task']!=args.task and args.adapt_from is None:
            p.error('Cross-task transfer requires --adapt-from, not an unlabeled resume')
        lineage=dict(kind='task-adaptation' if args.adapt_from else 'resume',
            checkpoint=str(args.resume.resolve()),task=parent_run['task'],
            checkpoint_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest(),
            parent_num_envs=parent_run['num_envs'])
    if args.transition_learning_rate is not None:
        assert args.adapt_from and parent_run['task']!=args.task
        assert 'WarmLR1e5' in args.task and args.transition_learning_rate==1e-5
    if parent_run and parent_run['task']!=args.task and 'WarmLR1e5' in args.task:
        assert args.transition_learning_rate==1e-5, 'Explicit transition required for this new recipe'
    smoke = json.loads(args.preflight.read_text())
    if args.transition_learning_rate is not None:
        from mjlab_microduck.custom_v2 import optimizer_transition as transition_module
        assert smoke['optimizer_transition']['after_initial_lr']==args.transition_learning_rate
        assert smoke['optimizer_transition']['source_sha256']==hashlib.sha256(Path(transition_module.__file__).read_bytes()).hexdigest()
    assert smoke['status'] == 'PASS' and smoke['task'] == args.task
    assert bool(smoke.get('normalization_recipe')) == args.freeze_normalizers, 'Normalizer preflight recipe differs'
    assert not args.freeze_normalizers or args.resume is not None, 'Freeze requires a mature parent'
    if parent_run and parent_run.get('normalization_recipe'):
        assert args.freeze_normalizers, 'Cannot silently resume statistic updates for a frozen parent'
    normalization_recipe = None
    frozen_snapshot = None
    if args.freeze_normalizers:
        from mjlab_microduck.custom_v2.frozen_normalization import RECIPE,freeze_normalizers,assert_frozen_normalizers
        normalization_recipe = RECIPE.copy()
        assert all(smoke['normalization_recipe'][k] == v for k,v in RECIPE.items())
        from mjlab_microduck.custom_v2 import frozen_normalization
        assert smoke['normalization_source_sha256'] == hashlib.sha256(Path(frozen_normalization.__file__).read_bytes()).hexdigest(), 'Normalizer source differs from preflight'
        assert smoke['normalization_unchanged_all_updates']
    assert bool(smoke.get('head_precision_recipe')) == args.head_precision, 'Preflight head recipe differs'
    assert not args.head_precision or args.resume is not None, 'Head precision requires a mastered recovery parent'
    assert smoke['physics_cuda_graph'] and min(smoke['motor_graph_replays']) > 0
    if not args.validation_only:
        assert smoke['envs'] == args.num_envs and smoke['throughput']['measured_updates'] >= 10
        assert smoke['throughput']['median_end_to_end_steps_per_s'] > 100_000, 'Throughput admission failed'
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    sources = sorted([*root.glob('src/**/*.py'), *root.glob('src/**/*.cpp'),
                      root/'scripts/train_v2_standup.py', root/'scripts/v2_python.sh'])
    hashes = {str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
    with tarfile.open(output/'source.tar.gz', 'w:gz') as archive:
        for f in sources:
            archive.add(f, arcname=str(f.relative_to(root)))
    cfg = load_env_cfg(args.task)
    head_recipe = None
    if args.head_precision:
        from mjlab_microduck.custom_v2.training import configure_head_precision_learning
        head_recipe = configure_head_precision_learning(cfg,args.head_precision_stage)
    if 'S0-O0-R1' in args.task or 'Firmware-LS3' in args.task:
        backend = smoke.get('measurement_backend')
        if not args.validation_only:
            assert backend is not None, 'P04 preflight must identify its actual measurement backend'
        if backend is not None:
            assert backend['frontend'] in ('CausalStream','CompiledCausalStream','FirmwareNativeStream')
            assert backend['pipeline'] in ('RegisterPipeline','CompiledRegisterPipeline')
            cfg.actions['joint_pos'].compile_measurement = backend['frontend'] == 'CompiledCausalStream'
            cfg.actions['joint_pos'].observer_frontend = ('firmware-ls3'
                if backend['frontend']=='FirmwareNativeStream' else 'bench')
            cfg.actions['joint_pos'].fuse_registers = backend['pipeline'] == 'CompiledRegisterPipeline'
            if cfg.actions['joint_pos'].fuse_registers:cfg.actions['joint_pos'].compile_measurement=True
            execution = smoke['measurement_execution']
            assert execution.get('spike_rejection',True)==cfg.actions['joint_pos'].spike_rejection, 'Preflight spike mode differs'
            assert execution.get('o0_local_history',False)==cfg.actions['joint_pos'].o0_local_history, 'Preflight O0 history differs'
            cfg.actions['joint_pos'].reset_graph_chunk = execution.get('reset_graph_chunk',1)
            measurement_root = root/'src/mjlab_microduck/sc0090_v2'
            for name,digest in execution['source_sha256'].items():
                assert hashlib.sha256((measurement_root/name).read_bytes()).hexdigest() == digest, f'Preflight source changed: {name}'

    if head_recipe is not None:
        for field in ('id','ranges','reward_weight','std','fine_std','fine_weight'):
            assert json.dumps(head_recipe[field])==json.dumps(smoke['head_precision_recipe'][field]), 'Head preflight recipe mismatch: '+field
    if head_recipe is not None:
        assert head_recipe.get('accuracy_bonus')==smoke['head_precision_recipe'].get('accuracy_bonus'), 'Accuracy reward preflight differs'
        if head_recipe.get('accuracy_bonus'):assert smoke['accuracy_oracle']['status']=='PASS'
    from mjlab_microduck.custom_v2.commands import sampling_contract
    assert smoke['head_sampling_contract']==sampling_contract(cfg.commands['head_pose']), 'Head command sampling differs from preflight'
    phase_enabled='first_timeout_deadline_reset' in cfg.events
    assert phase_enabled==bool(smoke.get('timeout_phase')), 'Timeout phase preflight differs'
    if phase_enabled:
        from mjlab_microduck.custom_v2 import timeout_phase as phase_module
        assert args.resume is not None and args.freeze_normalizers
        assert smoke['timeout_phase']['recipe']==phase_module.RECIPE
        assert smoke['timeout_phase']['source_sha256']==hashlib.sha256(Path(phase_module.__file__).read_bytes()).hexdigest()
        assert smoke['iterations']>=52 and smoke['timeout_phase_remaining_first_episodes']==0
        assert smoke['timeout_phase_default_after_reset']
    cfg.seed, cfg.scene.num_envs = 20261015, args.num_envs
    agent = load_rl_cfg(args.task)
    assert smoke['steps_per_env']==agent.num_steps_per_env
    assert smoke['learning_epochs']==agent.algorithm.num_learning_epochs
    assert smoke['minibatches']==agent.algorithm.num_mini_batches
    assert smoke['entropy_coef']==agent.algorithm.entropy_coef, 'Entropy preflight differs'
    assert smoke['distribution_cfg']==agent.actor.distribution_cfg, 'Distribution preflight differs'
    is_head_cap=(agent.actor.distribution_cfg.get('class_name')=='mjlab_microduck.custom_v2.head_exploration:HeadCappedGaussianDistribution')
    if is_head_cap:
        from mjlab_microduck.custom_v2 import head_exploration
        assert smoke['exploration_before']['source_sha256']==hashlib.sha256(Path(head_exploration.__file__).read_bytes()).hexdigest()
        assert smoke['first_minibatch_distribution_consistent_all_updates']
    if 'LateEntropy005' in args.task or is_head_cap:
        assert args.resume is not None and args.head_precision and args.freeze_normalizers, 'Late entropy requires explicit mature frozen-stat head parent'
    agent.logger, agent.upload_model, agent.save_interval = 'tensorboard', False, 250
    agent.seed = cfg.seed
    observation_transition=None
    curriculum_window_transition=None
    if parent_run and parent_run.get('position_robustness_recipe'):
        assert args.adapt_from is not None, 'Position robustness parent requires explicit adaptation'
        assert 'S0-O0-R1' in args.task, 'Position noise replacement requires actual S0/O0 measurement'
        assert 'encoder_bias' not in cfg.events
        assert all(cfg.observations['actor'].terms[n].noise is None for n in ('joint_pos','joint_vel'))
        observation_transition=dict(parent_recipe=parent_run['position_robustness_recipe'],
            handling='replaced, not added: the complete S0/O0 measurement owns actor joint q/v',
            retained='actor/critic/normalizers/Adam/recovery stage/head precision',
            prior_controls_are_baseline=True)
    write_json(output/'run.json', dict(task=args.task, device=args.device, num_envs=args.num_envs,
        allocation_updates=args.iterations, evaluation_interval=args.eval_interval,
        evaluation_samples_per_bucket_per_seed=args.eval_samples,
        seed=cfg.seed, source_sha256=hashes,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        dependencies={n:importlib.metadata.version(n) for n in
                      ['torch','mjlab','mujoco','mujoco-warp','warp-lang','better-actuator-models','rsl-rl-lib']},
        preflight=str(args.preflight.resolve()), agent=asdict(agent), environment=asdict(cfg),
        resume=str(args.resume) if args.resume else None,head_precision_recipe=head_recipe,
        lineage=lineage,observation_transition=observation_transition,normalization_recipe=normalization_recipe,
        requested_initial_learning_rate_transition=args.transition_learning_rate,
        planned_additional_controls=args.iterations*args.num_envs*agent.num_steps_per_env,
        curriculum_clock='actual per-environment control steps; not scaled by environment count'))
    class StopRequested(Exception):
        pass
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    env = ManagerBasedRlEnv(cfg, device=args.device)
    assert env.sim.use_cuda_graph
    from mjlab_microduck.custom_v2.exposure import CommandExposure
    exposure = CommandExposure(env)
    native_control_step = env.step
    def control_step(actions):
        exposure.record()
        return native_control_step(actions)
    env.step = control_step
    counts = wp.to_torch(env.sim.wp_data.nacon)
    native_step = env.sim.step
    def checked_step():
        native_step()
        torch._assert_async((counts <= env.sim.wp_data.naconmax).all(), 'V2 Warp contact overflow')
    env.sim.step = checked_step
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    # Identical network initialization despite distinct feedback setup RNG use.
    torch.manual_seed(cfg.seed + 100)
    runner = load_runner_cls(args.task)(wrapped, asdict(agent), str(output), args.device)
    if args.resume is None:
        assert env.common_step_counter==0 and runner.current_learning_iteration==0
        assert len(runner.alg.optimizer.state)==0, 'Fresh training must have empty Adam state'
        state=recovery._sc0090_recovery_state(env)
        assert state.stage==0 and not state.polish and head_recipe is None
        initial_state=runner.alg.save()
        import io
        initial_bytes=io.BytesIO();torch.save(initial_state,initial_bytes)
        write_json(output/'fresh-initialization.json',dict(status='PASS',
            origin='random initialization; no checkpoint or external weights loaded',
            network_seed=cfg.seed+100,checkpoint=None,optimizer_state_entries=0,
            common_step_counter=0,learning_iteration=0,recovery_stage=0,polish=False,
            head_precision_recipe=None,discovery=state.discovery,
            initial_state_sha256=hashlib.sha256(initial_bytes.getvalue()).hexdigest(),
            actor_normalizer_count=float(runner.alg.get_policy().obs_normalizer.count.item())))
        del initial_state,initial_bytes
    from mjlab_microduck.ppo_diagnostics import attach_ppo_diagnostics
    diagnostics = attach_ppo_diagnostics(runner, output/'ppo-diagnostics.jsonl')
    completed = 0
    inherited_controls=0
    rng_restore=None
    if args.resume:
        infos = runner.load(str(args.resume))
        completed = infos['v2']['completed_updates']
        inherited_controls=infos['v2'].get('controls',
            completed*parent_run['num_envs']*parent_run['agent']['num_steps_per_env'])
        checkpoint_state = torch.load(args.resume, map_location='cpu', weights_only=False)
        def same_state(actual, expected):
            if isinstance(expected, torch.Tensor):
                assert torch.equal(actual.detach().cpu(), expected), 'Resume tensor mismatch'
            elif isinstance(expected, dict):
                assert actual.keys() == expected.keys()
                for key in expected:
                    same_state(actual[key], expected[key])
            elif isinstance(expected, (list, tuple)):
                assert len(actual) == len(expected)
                for left,right in zip(actual,expected):
                    same_state(left,right)
            else:
                assert actual == expected
        restored = runner.alg.save()
        for key in ('actor_state_dict','critic_state_dict','optimizer_state_dict'):
            same_state(restored[key],checkpoint_state[key])
        assert runner.alg.learning_rate == runner.alg.optimizer.param_groups[0]['lr']
        assert env.common_step_counter == infos['env_state']['common_step_counter']
        write_json(output/'resume-verification.json',dict(
            status='PASS',checkpoint_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest(),
            completed_updates=completed,common_step_counter=env.common_step_counter,
            learning_rate=runner.alg.learning_rate,
            actor_critic_normalizers_optimizer_exact=True,
            previous_bootstrap_contract=checkpoint_state.get('terminal_critic_bootstrap'),
            current_bootstrap_contract=restored.get('terminal_critic_bootstrap'),
            source_changed=infos['v2'].get('source_sha256') != hashes,
            physics_resume='native-new-episodes', exposure_scope='current allocation segment'))
        del checkpoint_state, restored

        state = recovery._sc0090_recovery_state(env)
        state.load_state_dict(infos['v2']['recovery'])
        assert not args.head_precision or state.polish, 'Recovery has not reached polish'
        assert not infos['v2'].get('head_precision_recipe') or args.head_precision, 'Cannot silently drop the parent head recipe'
        env._sc0090_eval_history = infos['v2'].get('eval_history', {})
        if args.adapt_from is not None:
            previous_params=parent_run['environment']['events']['set_ground_state']['params']
            curriculum_window_transition=recovery.rebase_changed_frontier_clock(env,
                previous_params.get('roll_frontiers',recovery.SC0090_ROLL_FRONTIERS))
        runner.current_learning_iteration = completed
        from mjlab_microduck.custom_v2.rng_resume import restore_training_rng
        rng_restore=restore_training_rng(infos['v2'],parent_run,args.device)
        rng_preflight_reuse=validate_rng_preflight_reuse(smoke,rng_restore,
            checkpoint_sha256=lineage['checkpoint_sha256'],parent_task=parent_run['task'],
            task=args.task,is_adaptation=args.adapt_from is not None)
        write_json(output/'resume-rng-qualification.json',rng_preflight_reuse)
        write_json(output/'resume-rng.json',rng_restore)
        if state.polish:
            for name, weight in (('action_rate_l2',-.5),('arrival_damping',-.025),
                ('joint_torque_rate_l2',-.0005),('head_pose_bias',.5),('body_pose_tracking',.5)):
                env.reward_manager.get_term_cfg(name).weight = weight
        # Initial construction happened before restoring curriculum clocks.
        # Recreate native-new episodes from the restored stage/ranges, rather
        # than silently collecting a first rollout of stage-zero resets.
        env.reset()
        assert env.common_step_counter == infos['env_state']['common_step_counter']
        assert bool((state.spawn_stage == state.stage).all())
        write_json(output/'resume-environment.json',dict(status='PASS',stage=state.stage,
            polish=state.polish,all_spawn_stages_match=True,
            common_step_counter=env.common_step_counter,
            curriculum_window_transition=curriculum_window_transition,
            head_ranges=env.command_manager.get_term('head_pose').cfg.ranges,
            reward_weights={name:env.reward_manager.get_term_cfg(name).weight for name in
                ('action_rate_l2','arrival_damping','joint_torque_rate_l2','head_pose_bias','body_pose_tracking')},
            physics_resume='native-new-episodes after restoring clocks and RNG'))

    optimizer_transition = infos['v2'].get('optimizer_transition') if args.resume else None
    if args.transition_learning_rate is not None:
        from mjlab_microduck.custom_v2.optimizer_transition import apply_initial_learning_rate
        optimizer_transition=apply_initial_learning_rate(runner.alg,args.transition_learning_rate)
        write_json(output/'optimizer-transition.json',dict(status='PASS',**optimizer_transition,
            exact_parent_restore_already_verified=True,parent_checkpoint_sha256=lineage['checkpoint_sha256']))
    if args.freeze_normalizers:
        normalization_recipe,frozen_snapshot = freeze_normalizers(runner.alg)
        assert completed > 0
        write_json(output/'normalizer-freeze.json',dict(status='PASS',**normalization_recipe,
            parent_checkpoint_sha256=lineage['checkpoint_sha256'],
            applies_after_exact_parent_restore=True))
    exploration_recipe = None
    if hasattr(runner.alg.actor.distribution,'exploration_metadata'):
        exploration_recipe=runner.alg.actor.distribution.exploration_metadata()
        write_json(output/'exploration-restored.json',exploration_recipe)
    timeout_phase = None
    if phase_enabled:
        timeout_phase=phase_module.activate_first_timeout_phase(env,completed)
        write_json(output/'timeout-phase-initialization.json',timeout_phase)
    start = completed
    original_update = runner.alg.update
    def checked_update():
        nonlocal completed,stopping
        result = original_update()
        if frozen_snapshot is not None:
            assert_frozen_normalizers(runner.alg,frozen_snapshot)
        if exploration_recipe is not None:
            assert diagnostics.last['PPO/std_first_minibatch_max_abs_recompute_delta']==0., diagnostics.last
            assert diagnostics.last['PPO/kl_first_minibatch_max']<=1e-6, diagnostics.last
            assert diagnostics.last['PPO/ratio_first_minibatch_max_abs_error_from_one']<=.002, diagnostics.last
        completed += 1
        if phase_enabled and (diagnostics.last['PPO/kl_mean']>.5 or diagnostics.last['PPO/clip_fraction']>.8):
            write_json(output/'optimizer-guard-stop.json',dict(completed_updates=completed,
                metrics=diagnostics.last,limits=dict(pooled_kl=.5,clip_fraction=.8),
                reason='Keep the actual-N admission optimizer limits during the bounded phase experiment'))
            stopping=True
        if any(not np.isfinite(float(v)) for v in result.values()):
            raise FloatingPointError(f'Nonfinite PPO losses: {result}')
        if completed % 10 == 0 or completed <= start+5:
            write_json(output/'status.json', dict(status='RUNNING',pid=os.getpid(),
                completed_updates=completed, allocation_end=start+args.iterations,
                controls=inherited_controls+(completed-start)*args.num_envs*agent.num_steps_per_env,
                allocation_controls=(completed-start)*args.num_envs*agent.num_steps_per_env,
                inherited_controls=inherited_controls,
                common_step_counter=env.common_step_counter,
                updated_unix=time.time(), losses=result, command_exposure=exposure.report(),
                exploration=runner.alg.actor.distribution.exploration_metadata() if exploration_recipe else None,
                recovery_stage=recovery._sc0090_recovery_state(env).stage))
        if stopping:
            raise StopRequested()
        return result
    runner.alg.update = checked_update
    original_save = runner.save
    def save(path, infos=None):
        state = recovery._sc0090_recovery_state(env)
        infos = {**(infos or {}),'v2':dict(completed_updates=completed,
            task=args.task,num_envs=args.num_envs,steps_per_env=agent.num_steps_per_env,
            controls=inherited_controls+(completed-start)*args.num_envs*agent.num_steps_per_env,
            inherited_controls=inherited_controls,lineage=lineage,
            recovery=state.state_dict(),eval_history=getattr(env,'_sc0090_eval_history',{}),
            source_sha256=hashes, physics_resume='native-new-episodes',
            head_precision_recipe=head_recipe,normalization_recipe=normalization_recipe,
            exploration_recipe=runner.alg.actor.distribution.exploration_metadata() if exploration_recipe else None,
            optimizer_transition=optimizer_transition,timeout_phase=timeout_phase,
            observation_transition=observation_transition,
            curriculum_window_transition=curriculum_window_transition,
            command_exposure=exposure.report(), exposure_segment_start_update=start,
            rng_device=str(env.device),rng_restore=rng_restore,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())}
        original_save(path, infos)
    runner.save = save
    last_resume_path = None
    try:
        while completed < start+args.iterations and not stopping:
            amount = min(args.eval_interval-completed%args.eval_interval, start+args.iterations-completed)
            runner.current_learning_iteration = completed
            try:
                assert runner.alg.entropy_coef==agent.algorithm.entropy_coef, 'Restored algorithm entropy differs'
                runner.learn(amount, init_at_random_ep_len=False)
            except StopRequested:
                runner.current_learning_iteration = completed
                last_resume_path = output/f'checkpoint_{completed:06d}.pt'
                save(str(last_resume_path))
                break
            if runner.logger.writer is not None:
                runner.logger.writer.flush()
                runner.logger.writer.close()
            runner.current_learning_iteration = completed
            checkpoint = output/f'checkpoint_{completed:06d}.pt'
            save(str(checkpoint))
            last_resume_path = checkpoint
            if stopping:
                break
            state = recovery._sc0090_recovery_state(env)
            request = dict(schema=1,checkpoint_file=checkpoint.name,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                config_sha256=hashlib.sha256((output/'run.json').read_bytes()).hexdigest(),
                iteration=completed,stage=state.stage,seeds=[2026101501,2026101502],
                samples_per_group=args.eval_samples,duration_s=8.,deterministic=True)
            with torch.inference_mode():
                evidence=output/f'evaluation_{completed:06d}_evidence'
                evidence.mkdir(exist_ok=False)
                report = {**request,'cases':evaluate(args.task,runner.alg.get_policy(),args.device,state.stage,args.eval_samples,
                    compiled_measurement=cfg.actions['joint_pos'].compile_measurement,evidence_dir=evidence)}
            decision = recovery.sc0090_apply_recovery_evaluation(env,report,request)
            write_json(output/f'evaluation_{completed:06d}.json', {**report,'decision':decision})
            # Only proved recovery competence enables smoothing and head polish.
            if state.polish:
                for name, weight in (('action_rate_l2',-.5),('arrival_damping',-.025),
                    ('joint_torque_rate_l2',-.0005),('head_pose_bias',.5),('body_pose_tracking',.5)):
                    env.reward_manager.get_term_cfg(name).weight = weight
            # Preserve the exact evaluated file/hash. Curriculum decisions
            # change checkpoint metadata and belong in a separate resume file.
            last_resume_path = output/f'resume_{completed:06d}.pt'
            save(str(last_resume_path))
        write_json(output/'status.json', dict(status='STOPPED' if stopping else 'ALLOCATION_COMPLETE',
            pid=os.getpid(),completed_updates=completed,updated_unix=time.time(),
            controls=inherited_controls+(completed-start)*args.num_envs*agent.num_steps_per_env,
            inherited_controls=inherited_controls,
            allocation_controls=(completed-start)*args.num_envs*agent.num_steps_per_env,
            goal_complete=False, resume_checkpoint=str(last_resume_path),
            command_exposure=exposure.report(),
            exposure_segment_start_update=start,
            recovery_stage=recovery._sc0090_recovery_state(env).stage))
    finally:
        env.close()


if __name__ == '__main__':
    main()
