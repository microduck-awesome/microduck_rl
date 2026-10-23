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


def evaluate(task, policy, device, stage, samples, *, position_recipe=None,evidence_dir=None):
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    was_training = policy.training
    cases = []
    policy.eval()
    try:
        for seed in (2026101501, 2026101502):
            cfg = load_env_cfg(task, play=True)
            if position_recipe:
                from mjlab_microduck.custom_v2.training import configure_position_robustness_learning
                applied=configure_position_robustness_learning(cfg,position_recipe['stage'],require_corruption=False)
                assert applied==position_recipe, 'Unknown checkpoint position observation recipe'
            cfg.seed, cfg.scene.num_envs = seed, 7 * samples
            cfg.episode_length_s = 8.
            from mjlab.managers import RewardTermCfg
            cfg.curriculum, cfg.terminations = {}, {}
            cfg.rewards = {'stable_standing': RewardTermCfg(func=recovery.sc0090_stable_standing, weight=1.)}
            cfg.events = {n:t for n,t in cfg.events.items() if n in {
                'expand_bam_friction_fields','v2_runtime','mouth_reset','mouth_hold',
                'reset_base','set_ground_state','v2_reset_joint_guards','paired_error_reset'}}
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
                success = state.success & stable & ~invalid
                groups = {}
                for i, name in enumerate(recovery.SC0090_RECOVERY_BUCKETS):
                    mask = state.bucket == i
                    passing = mask & success
                    groups[name] = dict(trials=int(mask.sum()), wins=int(passing.sum()),
                        any_stable=int((mask & state.success).sum()), faults=int((mask & invalid).sum()),
                        first_stable_mean_s=float(first[passing].mean()) if passing.any() else None)
                cases.append(dict(seed=seed, finite=bool(torch.isfinite(env.sim.data.qpos).all()), groups=groups))
            finally:
                env.close()
    finally:
        policy.train(was_training)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)
    return cases


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
    p.add_argument('--resume', type=Path)
    p.add_argument('--resume-root', type=Path)
    p.add_argument('--resume-update', type=int, default=500)
    p.add_argument('--validation-only', action='store_true')
    p.add_argument('--head-precision', action='store_true')
    p.add_argument('--head-precision-stage',type=int,choices=(1,2,3,4),default=1)
    p.add_argument("--position-robustness-stage",type=int,choices=(0,1,2),default=0)
    args = p.parse_args()
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    assert world == 2, 'This launcher requires two synchronized GPU ranks'
    if args.resume_root is not None:
        assert args.resume is None
        args.resume = args.resume_root / f'rank{rank}' / f'resume_{args.resume_update:06d}.pt'
    args.device = f'cuda:{rank}'
    torch.cuda.set_device(rank)
    args.output = args.output / f'rank{rank}'
    if args.num_envs * world != 8192 and not args.validation_only:
        p.error('This baseline allocation uses the preflighted 8192 environments')
    smoke = json.loads(args.preflight.read_text())
    assert smoke['status'] == 'PASS' and smoke['task'] == args.task
    assert bool(smoke.get('head_precision_recipe')) == args.head_precision, 'Preflight head recipe differs'
    assert not args.head_precision or args.resume is not None, 'Head precision requires a mastered recovery parent'
    assert smoke['physics_cuda_graph'] and min(smoke['motor_graph_replays']) > 0
    if not args.validation_only:
        assert smoke['envs'] == 4096 and smoke['throughput']['measured_updates'] >= 9
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    assert hashlib.sha256((root/'src/mjlab_microduck/custom_v2/assets/robot.xml').read_bytes()).hexdigest() == smoke['robot_xml_sha256']
    sources = sorted([*root.glob('src/**/*.py'), *root.glob('src/**/*.cpp'),
                      root/'scripts/train3762_ddp.py', root/'scripts/distributed3762.py',
                      *root.glob('src/mjlab_microduck/custom_v2/assets/**/*.xml'),
                      *root.glob('src/mjlab_microduck/custom_v2/assets/assets/sim2real_shell_*.stl'),
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
    position_recipe = None
    if args.position_robustness_stage:
        assert args.resume is not None, 'Position robustness requires a trained recovery parent'
        from mjlab_microduck.custom_v2.training import configure_position_robustness_learning
        position_recipe=configure_position_robustness_learning(cfg,args.position_robustness_stage)
    assert position_recipe==smoke.get('position_robustness_recipe'), 'Position preflight recipe mismatch'
    if 'S0-O0-R1' in args.task:
        backend = smoke.get('measurement_backend')
        if not args.validation_only:
            assert backend is not None, 'P04 preflight must identify its actual measurement backend'
        if backend is not None:
            assert backend['frontend'] in ('CausalStream','CompiledCausalStream')
            assert backend['pipeline'] in ('RegisterPipeline','CompiledRegisterPipeline')
            cfg.actions['joint_pos'].compile_measurement = backend['frontend'] == 'CompiledCausalStream'
            cfg.actions['joint_pos'].fuse_registers = backend['pipeline'] == 'CompiledRegisterPipeline'
            execution = smoke['measurement_execution']
            measurement_root = root/'src/mjlab_microduck/sc0090_v2'
            for name,digest in execution['source_sha256'].items():
                assert hashlib.sha256((measurement_root/name).read_bytes()).hexdigest() == digest, f'Preflight source changed: {name}'

    if head_recipe is not None:
        for field in ('id','ranges','reward_weight','std','fine_std','fine_weight',
                      'precision_scale','command_sampling'):
            assert json.dumps(head_recipe.get(field))==json.dumps(smoke['head_precision_recipe'].get(field)), 'Head preflight recipe mismatch: '+field
    cfg.seed, cfg.scene.num_envs = 20261015 + rank, args.num_envs
    agent = load_rl_cfg(args.task)
    agent.logger, agent.upload_model, agent.save_interval = 'tensorboard', False, 250
    agent.seed = cfg.seed
    write_json(output/'run.json', dict(task=args.task, device=args.device, num_envs=args.num_envs,
        allocation_updates=args.iterations, evaluation_interval=args.eval_interval,
        evaluation_samples_per_bucket_per_seed=args.eval_samples,
        seed=cfg.seed, source_sha256=hashes,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        dependencies={n:importlib.metadata.version(n) for n in
                      ['torch','mjlab','mujoco','mujoco-warp','warp-lang','better-actuator-models','rsl-rl-lib']},
        preflight=str(args.preflight.resolve()), agent=asdict(agent), environment=asdict(cfg),
        resume=str(args.resume) if args.resume else None,head_precision_recipe=head_recipe,
        position_robustness_recipe=position_recipe))
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
    torch.manual_seed(20261015 + 100)
    runner = load_runner_cls(args.task)(wrapped, asdict(agent), str(output), args.device)
    from distributed3762 import install_global_statistics
    runner.logger.logger_type = "tensorboard"
    install_global_statistics(runner.alg)
    torch.manual_seed(20261015 + 1000 + rank)
    # Historical diagnostics replaces PPO.update and only supports one GPU.
    # Keep the native distributed PPO update; losses are recorded in status.json.
    completed = 0
    if args.resume:
        infos = runner.load(str(args.resume))
        completed = infos['v2']['completed_updates']
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
        assert not infos['v2'].get('position_robustness_recipe') or args.position_robustness_stage, 'Cannot silently drop the parent position recipe'
        env._sc0090_eval_history = infos['v2'].get('eval_history', {})
        runner.current_learning_iteration = completed
        torch.set_rng_state(infos["v2"]["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([r.cpu() for r in infos["v2"]["cuda_rng"]])
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
            head_ranges=env.command_manager.get_term('head_pose').cfg.ranges,
            reward_weights={name:env.reward_manager.get_term_cfg(name).weight for name in
                ('action_rate_l2','arrival_damping','joint_torque_rate_l2','head_pose_bias','body_pose_tracking')},
            physics_resume='native-new-episodes after restoring clocks and RNG'))

    start = completed
    original_update = runner.alg.update
    def checked_update():
        nonlocal completed, stopping
        result = original_update()
        completed += 1
        if any(not np.isfinite(float(v)) for v in result.values()):
            raise FloatingPointError(f'Nonfinite PPO losses: {result}')
        if completed % 10 == 0 or completed <= start+5:
            write_json(output/'status.json', dict(status='RUNNING',pid=os.getpid(),
                completed_updates=completed, allocation_end=start+args.iterations,
                controls=completed*args.num_envs*agent.num_steps_per_env,
                common_step_counter=env.common_step_counter,
                updated_unix=time.time(), losses=result, command_exposure=exposure.report(),
                recovery_stage=recovery._sc0090_recovery_state(env).stage))
        stop_flag = torch.tensor(int(stopping), device=args.device)
        torch.distributed.all_reduce(stop_flag, op=torch.distributed.ReduceOp.MAX)
        if bool(stop_flag):
            stopping = True
            raise StopRequested()
        return result
    runner.alg.update = checked_update
    original_save = runner.save
    def save(path, infos=None):
        state = recovery._sc0090_recovery_state(env)
        infos = {**(infos or {}),'v2':dict(completed_updates=completed,
            recovery=state.state_dict(),eval_history=getattr(env,'_sc0090_eval_history',{}),
            source_sha256=hashes, physics_resume='native-new-episodes',
            head_precision_recipe=head_recipe,position_robustness_recipe=position_recipe,
            command_exposure=exposure.report(), exposure_segment_start_update=start,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())}
        original_save(path, infos)
    runner.save = save
    last_resume_path = None
    try:
        while completed < start+args.iterations and not stopping:
            amount = min(args.eval_interval-completed%args.eval_interval, start+args.iterations-completed)
            runner.current_learning_iteration = completed
            try:
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
                samples_per_group=args.eval_samples,duration_s=8.,deterministic=True,
                position_observation_recipe=position_recipe)
            payload = [None]
            if rank == 0:
                with torch.inference_mode():
                    report = {**request,'cases':evaluate(args.task,runner.alg.get_policy(),args.device,state.stage,args.eval_samples,
                        position_recipe=position_recipe)}
                payload[0] = (request, report)
            torch.distributed.broadcast_object_list(payload, src=0)
            request, report = payload[0]
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
            world_size=world, global_envs=args.num_envs*world, goal_complete=False, resume_checkpoint=str(last_resume_path),
            command_exposure=exposure.report(),
            exposure_segment_start_update=start,
            recovery_stage=recovery._sc0090_recovery_state(env).stage))
    except BaseException as exc:
        write_json(output/'status.json', dict(status='FAILED', pid=os.getpid(),
            completed_updates=completed, updated_unix=time.time(),
            error=repr(exc), resume_checkpoint=str(last_resume_path)))
        raise
    finally:
        env.close()
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
