"""V4 fresh/legacy/resume state transitions and live MDP invariants."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace as NS
import json

import pytest
import torch

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_sc0090_v4_env_cfg import (
    make_training_program, make_sc0090_v4_env_cfg, make_program_rl_cfg,
)
from mjlab_microduck.tasks.program_runner import identify_checkpoint_task
from test_sc0090_finetune import fake_env


def program_env(task="recovery", n=14):
    env = fake_env(n)
    program = make_training_program(task)
    env.cfg = NS(curriculum={"training_program": NS(params={"program": program})},
                 events={k: None for k in ("randomize_com", "randomize_head_com", "push_robot")})
    commands = {k: NS(cfg=NS()) for k in ("head_pose", "body_pose")}
    env.command_manager = NS(get_term=commands.__getitem__)
    rewards = {k: NS(weight=0.) for k in ("action_rate_l2", "joint_torque_rate_l2", "body_pose_tracking",
                 "arrival_damping", "height_stand_sharp", "upright_sharp", "standing_composite")}
    events = {k: NS(params={}) for k in env.cfg.events}
    env.event_manager = NS(get_term_cfg=events.__getitem__)
    env.reward_manager = NS(get_term_cfg=rewards.__getitem__)
    return env


def report_for(env, step):
    env.common_step_counter = step
    state = mdp.sc0090_program_state(env)
    request = dict(schema=3, task=state["task"], program_hash=state["program_hash"],
        phase_index=state["phase_index"], global_step=step, iteration=step//24,
        checkpoint_sha256="checkpoint", config_sha256="config", seeds=[101, 102],
        samples_per_group=128, duration_s=8.)
    names = mdp.SC0090_RECOVERY_BUCKETS if state["task"] == "recovery" else tuple(k for k, _ in mdp.SC0090_PROGRAM_WALK_COMMANDS)
    cases = []
    for seed in request["seeds"]:
        cases.append(dict(seed=seed, finite=True,
            **{profile: {k: dict(trials=128, wins=128) for k in names} for profile in ("retention", "challenge")},
            metrics=dict(action_delta_rms=.1, torque_delta_rms=.01, settled_ang_rms=.1, rise_time_p95=2.)))
        if state['task'] == 'walk':
            cases[-1]['nominal'] = {name: dict(trials=1, wins=1) for name in names}
    return request, {**deepcopy(request), "cases": cases}


@pytest.mark.parametrize("task", ["walk", "recovery"])
def test_fresh_starts_easy_and_plan_is_stable(task):
    env = program_env(task)
    mdp.sc0090_program_curriculum(env, None, mdp.sc0090_program_spec(env))
    state = mdp.sc0090_program_state(env)
    assert state["phase_index"] == state["phase_enter_step"] == state["start_step"] == 0
    assert mdp.sc0090_program_phase(env)["name"] == "foundation_0"
    assert env.event_manager.get_term_cfg("randomize_com").params["ranges"] == (-.003, .003)
    assert env.command_manager.get_term("head_pose").cfg.ranges[0] == pytest.approx((-.055, .055))
    assert env.reward_manager.get_term_cfg("body_pose_tracking").weight == 0
    assert mdp.sc0090_program_hash(make_training_program(task)) == state["program_hash"]


@pytest.mark.parametrize("task", ["walk", "recovery"])
def test_global_iteration_6000_does_not_jump_to_hardest_phase(task):
    env = program_env(task)
    env.common_step_counter = 6000*24
    state = mdp.sc0090_program_restore(env, None, {"stage": 5, "polish": True})
    assert mdp.sc0090_program_phase(env)["name"] == "consolidate"
    assert state["phase_enter_step"] == state["start_step"] == 144000
    request, report = report_for(env, 144024)
    state = mdp.sc0090_program_apply_evaluation(env, report, request)
    assert state["passes"] == 1 and not state["last_result"]["advanced"]
    request, report = report_for(env, 146424)
    state = mdp.sc0090_program_apply_evaluation(env, report, request)
    assert state["last_result"]["advanced"]
    assert mdp.sc0090_program_phase(env)["name"] == "push_10"
    assert state["phase_enter_step"] == 146424


@pytest.mark.parametrize("stage", range(6))
def test_legacy_frontier_is_preserved_without_restarting_motor_dr(stage):
    env = program_env()
    env.common_step_counter = 144000
    mdp.sc0090_program_restore(env, None, {"stage": stage, "polish": False})
    phase = mdp.sc0090_program_phase(env)
    assert phase["frontier"] == stage and phase["head_scale"] == 1.
    mdp.sc0090_program_curriculum(env, None, mdp.sc0090_program_spec(env))
    assert mdp._sc0090_recovery_state(env).stage == stage


def test_resume_restores_phase_clock_passes_and_assessment_cadence():
    env = program_env()
    env.common_step_counter = 144000
    mdp.sc0090_program_restore(env, None, {"stage": 5, "polish": True})
    request, report = report_for(env, 145200)
    state = mdp.sc0090_program_apply_evaluation(env, report, request)
    state["last_attempt_step"] = 145200
    saved = deepcopy(state)
    other = program_env()
    other.common_step_counter = 145800
    mdp.sc0090_program_restore(other, saved)
    assert mdp.sc0090_program_state(other) == saved
    assert mdp.sc0090_program_state(other) is not saved
    assert other.common_step_counter-mdp.sc0090_program_state(other)["phase_enter_step"] == 1800
    with pytest.raises(ValueError):
        mdp.sc0090_program_apply_evaluation(other, report, request)


@pytest.mark.parametrize("key,value", [("schema", 2), ("phase_index", 999), ("phase_enter_step", 144001),
    ("program_hash", "wrong"), ("task", "walk"), ("last_attempt_step", 150000), ("passes", -1)])
def test_bad_checkpoint_state_is_rejected_before_mutation(key, value):
    env = program_env()
    env.common_step_counter = 144000
    state = mdp.sc0090_program_state(env)
    saved = deepcopy(state)
    saved[key] = value
    with pytest.raises(ValueError):
        mdp.sc0090_program_restore(env, saved)
    assert mdp.sc0090_program_state(env) == state


def test_consecutive_passes_and_retention_are_required():
    env = program_env()
    mdp.sc0090_program_restore(env, None, {"stage": 5, "polish": True})
    for step, bad in ((2400, False), (4800, True), (7200, False)):
        request, report = report_for(env, step)
        if bad:
            report["cases"][0]["retention"]["supine"]["wins"] = 120  # 93.75%
        state = mdp.sc0090_program_apply_evaluation(env, report, request)
        assert not state["last_result"]["advanced"]
    assert state["passes"] == 1
    request, report = report_for(env, 9600)
    assert mdp.sc0090_program_apply_evaluation(env, report, request)["last_result"]["advanced"]


@pytest.mark.parametrize("bad", ["missing_group", "missing_seed", "nonfinite", "bad_metric", "bad_counts", "sha", "stale_phase"])
def test_bad_report_cannot_advance_or_change_counters(bad):
    env = program_env()
    state = mdp.sc0090_program_state(env)
    before = deepcopy(state)
    request, report = report_for(env, 2400)
    if bad == "missing_group": report["cases"][0]["challenge"].pop("supine")
    elif bad == "missing_seed": report["cases"].pop()
    elif bad == "nonfinite": report["cases"][0]["finite"] = False
    elif bad == "bad_metric": report["cases"][0]["metrics"]["action_delta_rms"] = float("nan")
    elif bad == "bad_counts": report["cases"][0]["retention"]["supine"]["wins"] = 129
    elif bad == "sha": report["checkpoint_sha256"] = "changed"
    else: request["phase_index"] = report["phase_index"] = 1
    with pytest.raises(ValueError): mdp.sc0090_program_apply_evaluation(env, report, request)
    assert mdp.sc0090_program_state(env) == before


def test_smoothing_requires_measured_improvement_and_keeps_latency_bound():
    env = program_env()
    state = mdp.sc0090_program_state(env)
    phases = mdp.sc0090_program_spec(env)["phases"]
    state["phase_index"] = next(i for i,p in enumerate(phases) if p["name"] == "smooth_action_07")
    state["reference_metrics"] = dict(action_delta_rms=.1, rise_time_p95=2.)
    request, report = report_for(env, 2400)
    assert not mdp.sc0090_program_apply_evaluation(env, report, request)["last_result"]["passed"]
    request, report = report_for(env, 4800)
    for case in report["cases"]:
        case["metrics"]["action_delta_rms"] = .09
        case["metrics"]["rise_time_p95"] = 3.
    assert not mdp.sc0090_program_apply_evaluation(env, report, request)["last_result"]["passed"]


def test_episode_buffers_partial_reset_and_evaluation_balance():
    env = program_env(n=28)
    env._sc0090_program_eval_samples = 2
    mdp.sc0090_program_reset(env, None)
    buffers = mdp.sc0090_program_buffers(env)
    assert buffers["rehearsal"].tolist() == [i%4 < 2 for i in range(28)]
    buffers["pose_hold"][:] = 30
    env._prev_actuator_forces = torch.ones(28,14)
    ids = torch.tensor([2, 6, 19])
    mdp.sc0090_program_reset(env, ids)
    assert (buffers["pose_hold"][ids] == 0).all() and buffers["pose_hold"][3] == 30
    assert not env._prev_actuator_forces[ids].any() and env._prev_actuator_forces[3].sum() == 14


def test_body_commands_wait_for_recovery_and_rehearsal_stays_neutral():
    env = program_env(n=4)
    command = object.__new__(mdp.SC0090ProgramBodyCommand)
    command._env = env
    command.cfg = NS(tracking_enabled=True, recovery=True)
    command._command = torch.ones(4,6)*.1
    mdp._sc0090_recovery_state(env).success[:] = torch.tensor([False, True, True, True])
    mdp._sc0090_recovery_state(env).hold[:] = torch.tensor([0, 25, 25, 25])
    mdp.sc0090_program_buffers(env)["rehearsal"][2] = True
    env.scene["robot"].data.root_link_quat_w[3] = torch.tensor([0., 1., 0., 0.])
    expected = torch.zeros(4,6)
    expected[1] = .1
    assert torch.equal(command.command, expected)
    assert torch.all(command._command == .1)  # gating never destroys sampled targets


@pytest.mark.parametrize("task", ["walk", "recovery"])
def test_v4_keeps_motor_action_contract_and_has_all_live_terms(task):
    from mjlab_microduck.tasks.microduck_sc0090_finetune_env_cfg import make_sc0090_walk_v2_env_cfg
    from mjlab_microduck.tasks.microduck_sc0090_recovery_v3_env_cfg import make_sc0090_recovery_v3_env_cfg
    base = (make_sc0090_walk_v2_env_cfg if task=="walk" else make_sc0090_recovery_v3_env_cfg)()
    cfg = make_sc0090_v4_env_cfg(task)
    assert cfg.actions == base.actions and cfg.observations == base.observations and cfg.sim == base.sim
    assert cfg.scene.entities["robot"].articulation == base.scene.entities["robot"].articulation
    assert set(cfg.curriculum) == {"training_program"}
    assert cfg.events["push_robot"].func is mdp.sc0090_program_push
    assert cfg.rewards["body_pose_tracking"].func is mdp.body_pose_tracking_locomotion
    assert make_program_rl_cfg(task).training_program.checkpoint is None


def test_checkpoint_task_provenance_rejects_cross_family_and_unknown(tmp_path):
    path = tmp_path/'model_5999.pt'
    with pytest.raises(ValueError): identify_checkpoint_task(path, {})
    assert identify_checkpoint_task(path, {}, "walk") == "walk"
    assert identify_checkpoint_task(path, {"sc0090_recovery": {}}) == "recovery"
    with pytest.raises(ValueError): identify_checkpoint_task(path, {"sc0090_recovery": {}}, "walk")
    (tmp_path/'params').mkdir()
    (tmp_path/'params/agent.yaml').write_text('experiment_name: sc0090_walk_v2\nargs: !!python/tuple [1, 2]\n')
    assert identify_checkpoint_task(path, {}) == "walk"


def test_commanded_crouch_gets_target_reward_without_changing_nominal_recovery_gate():
    env = program_env(n=2)
    command = torch.zeros(2, 6)
    command[:, 2] = -.02
    env.command_manager.get_command = lambda _: command
    env.command_manager.get_term("body_pose").cfg.tracking_enabled = True
    env.scene["robot"].data.root_link_pos_w[:, 2] = .095
    state = mdp._sc0090_recovery_state(env)
    state.success[0] = True  # This world previously completed a nominal recovery.
    mdp.sc0090_program_buffers(env)["body_ready"][0] = True
    for _ in range(25):
        reward = mdp.sc0090_program_stable_standing(env)
    assert reward.tolist() == [1., 0.]
    assert state.success.tolist() == [True, False]
    assert state.hold.tolist() == [0, 0]  # Neither currently meets nominal z >= .105.


@pytest.mark.parametrize("load_cfg", [None, {"actor": True},
                                     {"actor": True, "critic": True, "optimizer": True, "iteration": True}])
def test_full_resume_restores_adaptive_lr_and_uses_next_iteration(monkeypatch, tmp_path, load_cfg):
    from mjlab_microduck.tasks.program_runner import SC0090ProgramRunner
    from mjlab_microduck.tasks import SC0090FineTuneRunner
    env = program_env()
    reset_phases = []
    env.reset = lambda: reset_phases.append(mdp.sc0090_program_phase(env)["name"])
    runner = object.__new__(SC0090ProgramRunner)
    runner.env, runner.device = NS(unwrapped=env), "cpu"
    runner.current_learning_iteration = 0
    runner.cfg = asdict(make_program_rl_cfg("recovery"))
    runner._abi = {"actor_obs": 61, "actions": 14, "motor_sha256": "motor"}
    runner.alg = NS(learning_rate=.0002, optimizer=NS(param_groups=[{"lr": .00003}]))
    infos = {"sc0090_recovery": {"stage": 5, "polish": True},
             "env_state": {"common_step_counter": 144000}}
    path = tmp_path / 'model_5999.pt'
    torch.save({"iter": 5999, "infos": infos, "optimizer_state_dict": {"param_groups": [{"lr": .00003}]}}, path)
    def load(self, *args, **kwargs):
        env.common_step_counter = 144000
        if kwargs.get("load_cfg") is None or kwargs["load_cfg"].get("iteration"):
            self.current_learning_iteration = 5999
        return infos
    monkeypatch.setattr(SC0090FineTuneRunner, "load", load)
    runner.load(path, load_cfg=load_cfg)
    actor_only = load_cfg == {"actor": True}
    assert runner.current_learning_iteration == (0 if actor_only else 6000)
    assert runner.alg.learning_rate == (.0002 if actor_only else .00003)
    assert env.common_step_counter == 144000
    assert reset_phases == ["consolidate"]
    assert mdp.sc0090_program_state(env)["phase_enter_step"] == 144000


def test_v4_budget_is_explicit_and_distributed_rejection_precedes_nccl(monkeypatch):
    from mjlab_microduck.tasks.program_runner import SC0090ProgramRunner, validate_runner_config
    from mjlab_microduck.tasks import SC0090FineTuneRunner
    cfg = asdict(make_program_rl_cfg("walk"))
    assert cfg["max_iterations"] is None
    with pytest.raises(ValueError, match="max-iterations"):
        validate_runner_config(cfg, "cuda:0", training=True)
    cfg["max_iterations"] = 17
    validate_runner_config(cfg, "cuda:0", training=True)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(SC0090FineTuneRunner, "__init__", lambda *a, **k: pytest.fail("entered NCCL constructor"))
    with pytest.raises(ValueError, match="single CUDA"):
        SC0090ProgramRunner(None, cfg, "log", "cuda:0")


@pytest.mark.parametrize("field,value", [("interval_iterations", True), ("interval_iterations", .5),
    ("interval_iterations", 0), ("seeds", (1, 1)), ("seeds", (-1, 2)), ("timeout_seconds", float("inf"))])
def test_bad_scheduling_configuration_rejected(field, value):
    from mjlab_microduck.tasks.program_runner import validate_runner_config
    cfg = asdict(make_program_rl_cfg("walk"))
    cfg["training_program"][field] = value
    with pytest.raises(ValueError):
        validate_runner_config(cfg, "cuda:0", training=False)


def test_completion_does_not_move_smoothing_baseline_again():
    env = program_env()
    state = mdp.sc0090_program_state(env)
    state["phase_index"] = len(mdp.sc0090_program_spec(env)["phases"])-1
    reference = dict(action_delta_rms=.2, torque_delta_rms=.02, settled_ang_rms=.2, rise_time_p95=2.)
    state["reference_metrics"] = deepcopy(reference)
    for step in (2400, 4800, 7200):
        request, report = report_for(env, step)
        state = mdp.sc0090_program_apply_evaluation(env, report, request)
        assert state["last_result"]["passed"]
        assert state["reference_metrics"] == reference
    assert state["complete"]


def test_rms_is_pooled_in_squared_units_and_latency_uses_worst_seed():
    env = program_env()
    request, report = report_for(env, 2400)
    report["cases"][0]["metrics"].update(action_delta_rms=0., rise_time_p95=1.)
    report["cases"][1]["metrics"].update(action_delta_rms=2., rise_time_p95=3.)
    before = deepcopy(mdp.sc0090_program_state(env))
    updated = mdp.sc0090_program_apply_evaluation(env, report, request, commit=False)
    assert mdp.sc0090_program_state(env) == before
    assert updated["last_result"]["metrics"]["action_delta_rms"] == pytest.approx(2**.5)
    assert updated["last_result"]["metrics"]["rise_time_p95"] == 3.


def test_torque_cost_does_not_compare_unrelated_episodes_or_charge_first_sample():
    env = program_env(n=2)
    data = env.scene["robot"].data
    data.actuator_force = torch.full((2, 14), 2.)
    assert mdp.sc0090_program_torque_rate(env).tolist() == [0., 0.]
    data.actuator_force += 1
    assert mdp.sc0090_program_torque_rate(env).tolist() == [14., 14.]
    mdp.sc0090_program_reset(env, torch.tensor([0]))
    data.actuator_force += 2
    assert mdp.sc0090_program_torque_rate(env).tolist() == [0., 56.]


def test_body_command_requires_new_nominal_hold_after_falling():
    env = program_env(n=1)
    buffers = mdp.sc0090_program_buffers(env)
    state = mdp._sc0090_recovery_state(env)
    state.hold[:] = 25
    state.success[:] = True
    buffers["body_ready"].copy_(mdp.sc0090_program_body_ready(env))
    state.hold[:] = 0
    env.scene["robot"].data.root_link_quat_w[:] = torch.tensor([0., 1., 0., 0.])
    buffers["body_ready"].copy_(mdp.sc0090_program_body_ready(env))
    env.scene["robot"].data.root_link_quat_w[:] = torch.tensor([1., 0., 0., 0.])
    assert not mdp.sc0090_program_body_ready(env).item()
    state.hold[:] = 24
    assert not mdp.sc0090_program_body_ready(env).item()
    state.hold[:] = 25
    assert mdp.sc0090_program_body_ready(env).item()


def test_pose_target_change_restarts_continuous_hold():
    env = program_env(n=1)
    env.command_manager.get_term("body_pose").cfg.tracking_enabled = True
    command = torch.zeros(1, 6)
    env.command_manager.get_command = lambda _: command
    mdp.sc0090_program_buffers(env)["body_ready"][:] = True
    for _ in range(25):
        mdp.sc0090_program_stable_standing(env)
    assert mdp.sc0090_program_buffers(env)["pose_hold"].item() == 25
    command[:, 2] = .001  # Remains in tolerance, but it is a new requested target.
    assert mdp.sc0090_program_stable_standing(env).item() == 0.
    assert mdp.sc0090_program_buffers(env)["pose_hold"].item() == 1


def test_fixed_eval_commands_are_written_during_reset_only_to_selected_worlds():
    n = len(mdp.SC0090_PROGRAM_WALK_COMMANDS)*4
    env = program_env("walk", n=n)
    term = object.__new__(mdp.SC0090ProgramVelocityCommand)
    term._env = env
    term.vel_command_b = torch.full((n,3), -9.)
    term.vel_command_w = term.vel_command_b.clone()
    for name in ("is_heading_env", "is_world_env", "is_forward_env", "is_standing_env"):
        setattr(term, name, torch.ones(n, dtype=torch.bool))
    env._sc0090_program_eval_samples = 2
    ids = torch.tensor([0, 4, 35, 43])
    term._resample_command(ids)
    expected = torch.tensor([v for _,v in mdp.SC0090_PROGRAM_WALK_COMMANDS])[ids//4]
    assert torch.equal(term.vel_command_b[ids], expected)
    assert torch.equal(term.vel_command_w[ids], expected)
    assert torch.all(term.vel_command_b[1] == -9.)


def test_checkpoint_atomic_publish_keeps_old_file_on_interruption(tmp_path):
    from mjlab_microduck.tasks.program_runner import atomic_write
    path = tmp_path/'model.pt'
    atomic_write(path, lambda f: f.write(b"complete old checkpoint"))
    def interrupted(stream):
        stream.write(b"partial new checkpoint")
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        atomic_write(path, interrupted)
    assert path.read_bytes() == b"complete old checkpoint"
    assert list(tmp_path.iterdir()) == [path]
    atomic_write(path, lambda f: f.write(b"complete new checkpoint"))
    assert path.read_bytes() == b"complete new checkpoint"


def test_run_directory_has_one_writer_without_waiting(tmp_path):
    from mjlab_microduck.tasks.program_runner import acquire_run_lock
    first = acquire_run_lock(tmp_path)
    try:
        with pytest.raises(BlockingIOError):
            acquire_run_lock(tmp_path)
    finally:
        first.close()
    acquire_run_lock(tmp_path).close()


def test_nonfinite_checkpoint_rejected_before_parent_load(monkeypatch, tmp_path):
    from mjlab_microduck.tasks.program_runner import SC0090ProgramRunner
    from mjlab_microduck.tasks import SC0090FineTuneRunner
    path = tmp_path/'bad.pt'
    torch.save({"actor_state_dict": {"weight": torch.tensor([float("nan")])}}, path)
    runner = object.__new__(SC0090ProgramRunner)
    monkeypatch.setattr(SC0090FineTuneRunner, "load", lambda *a, **k: pytest.fail("mutated live model"))
    with pytest.raises(ValueError, match="nonfinite"):
        runner.load(path)


def test_old_assessment_baseline_blocks_training_but_allows_actor_only_play(monkeypatch, tmp_path):
    from mjlab_microduck.tasks.program_runner import SC0090ProgramRunner
    from mjlab_microduck.tasks import SC0090FineTuneRunner
    env = program_env()
    env.common_step_counter = 144000
    env.reset = lambda: None
    state = mdp.sc0090_program_state(env)
    state.update(phase_index=20, reference_metrics={'settled_ang_rms': .2})
    runner = object.__new__(SC0090ProgramRunner)
    runner.env, runner.device = NS(unwrapped=env), 'cpu'
    runner.cfg = asdict(make_program_rl_cfg('recovery'))
    runner._abi = {}
    runner.current_learning_iteration = 0
    runner.alg = NS(learning_rate=.0002)
    infos = {'env_state': {'common_step_counter':144000}, 'sc0090_program': state,
             'sc0090_program_runner': {'abi': {}, 'evaluation_protocol': 2}}
    path = tmp_path/'old.pt'
    torch.save({'iter':5999,'infos':infos,'optimizer_state_dict':{'param_groups':[{'lr':.00003}]}}, path)
    loaded = []
    def parent_load(*args, **kwargs):
        loaded.append(kwargs['load_cfg'])
        return infos
    monkeypatch.setattr(SC0090FineTuneRunner, 'load', parent_load)
    with pytest.raises(ValueError, match='protocol 2'):
        runner.load(path)
    assert not loaded
    runner.load(path, load_cfg={'actor':True})
    assert loaded == [{'actor':True}]
    assert mdp.sc0090_program_state(env)['phase_index'] == 20
    assert mdp.sc0090_program_state(env)['reference_metrics'] == {}


def test_eval_timeout_reaps_real_child(tmp_path):
    import os, subprocess, sys
    from mjlab_microduck.tasks.program_runner import run_evaluation_process
    path = tmp_path/'child.log'
    with path.open('w') as stream, pytest.raises(subprocess.TimeoutExpired):
        run_evaluation_process([sys.executable, '-c',
            'import os,time; print(os.getpid(), flush=True); time.sleep(30)'], dict(os.environ), stream, .5)
    pid = int(path.read_text().strip())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.parametrize("stuck", [False, True])
def test_eval_interrupt_cleans_process_group_with_bounded_wait(monkeypatch, stuck):
    import subprocess
    from mjlab_microduck.tasks import program_runner as module
    waits, killed = [], []
    def wait(timeout):
        waits.append(timeout)
        if len(waits) == 1:
            raise KeyboardInterrupt
        if stuck:
            raise subprocess.TimeoutExpired('child', timeout)
        return -9
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **k: NS(pid=123, wait=wait, poll=lambda: None))
    monkeypatch.setattr(module.os, 'killpg', lambda *args: killed.append(args))
    with pytest.raises(module.EvaluationCleanupError if stuck else KeyboardInterrupt):
        module.run_evaluation_process(['child'], {}, None, 300.)
    assert waits == [300., 10.] and killed == [(123, module.signal.SIGKILL)]


@pytest.mark.parametrize("failure", [True, False])
def test_assessment_transaction_preserves_live_observations_until_reset(monkeypatch, tmp_path, failure):
    from mjlab_microduck.tasks import program_runner as module
    env = program_env()
    env.common_step_counter = 2400
    state = mdp.sc0090_program_state(env)
    state.update(passes=1, phase_enter_step=0)
    runner = object.__new__(module.SC0090ProgramRunner)
    runner.env, runner.device = NS(unwrapped=env), 'cpu'
    runner.cfg = asdict(make_program_rl_cfg('recovery'))
    runner.cfg['upload_model'] = False
    runner._evaluation_train_cfg = deepcopy(runner.cfg)
    runner._abi = {}
    runner.alg = NS(learning_rate=.0002, save=lambda: {"weight": torch.tensor([3.])})
    runner.current_learning_iteration = 99
    runner.logger = NS(writer=None)
    runner._export_checkpoint = lambda path: None
    def evaluate(command, *_):
        if failure:
            raise ValueError('bad report')
        from pathlib import Path
        request_path = Path(command[-1])
        request = json.loads(request_path.read_text())
        _, report = report_for(env, env.common_step_counter)
        report.update(request)
        for seed, case in zip(request['seeds'], report['cases']):
            case['seed'] = seed
        (request_path.parent/'report.json').write_text(json.dumps(report))
    monkeypatch.setattr(module, 'run_evaluation_process', evaluate)
    runner.save(tmp_path/'model.pt')
    saved = torch.load(tmp_path/'model.pt', weights_only=False)
    assert saved['infos']['sc0090_program']['passes'] == 0
    assert saved['infos']['sc0090_program']['last_result']['passed'] is (not failure)
    assert saved['infos']['sc0090_program']['phase_index'] == (0 if failure else 1)
    assert saved['infos']['sc0090_program']['last_attempt_step'] == 2400
    assert env.reward_manager.get_term_cfg('action_rate_l2').weight == 0.
    assert (tmp_path/'last_qualified.pt').exists() is (not failure)
    if not failure:
        qualified = torch.load(tmp_path/'last_qualified.pt', weights_only=False)
        assert qualified['infos'] == saved['infos']
        assert torch.equal(qualified['weight'], saved['weight'])


def test_automatic_eval_resets_preserve_observation_delay_clock():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers import TerminationTermCfg
    from mjlab.rl import RslRlVecEnvWrapper
    cfg = make_sc0090_v4_env_cfg('walk')
    cfg.scene.num_envs = len(mdp.SC0090_PROGRAM_WALK_COMMANDS)*4
    # Permanently fail half the worlds every step, while the others continue.
    cfg.terminations = {'forced': TerminationTermCfg(
        func=lambda env: torch.arange(env.num_envs, device=env.device)%2 == 0)}
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    try:
        env._sc0090_program_eval_samples = 2
        with torch.inference_mode():
            vec = RslRlVecEnvWrapper(env)
            expected = torch.tensor([v for _,v in mdp.SC0090_PROGRAM_WALK_COMMANDS], device=env.device).repeat_interleave(4,0)
            for i, (name, _) in enumerate(mdp.SC0090_PROGRAM_WALK_COMMANDS):
                if name == 'start_forward': expected[i*4:(i+1)*4] = 0.
                if name == 'stop_forward': expected[i*4:(i+1)*4, 0] = .1
            initial_obs = vec.get_observations()
            torch.testing.assert_close(initial_obs['actor'][:, 48:51], expected, rtol=0, atol=0)
            original_compute = env.observation_manager.compute
            ticks = []
            def compute(update_history=False):
                if update_history:
                    ticks.append(env.common_step_counter)
                return original_compute(update_history=update_history)
            env.observation_manager.compute = compute
            for _ in range(5):
                obs, _, done, _ = vec.step(torch.zeros((env.num_envs,14),device=env.device))
                assert done[::2].all()
                torch.testing.assert_close(obs['actor'][:, 48:51], expected, rtol=0, atol=0)
            assert ticks == [1,2,3,4,5]
    finally:
        env.close()


def test_walk_tracking_rejects_standing_reversed_and_half_speed_turns():
    contract = mdp.sc0090_walk_contract()
    commands = torch.tensor([[.08,0,0],[-.08,0,0],[0,0,.6],[0,0,-.6],[0,0,0]])
    assert mdp.sc0090_walk_tracking_pass(commands, commands, contract).all()
    assert not mdp.sc0090_walk_tracking_pass(torch.zeros_like(commands), commands, contract)[:4].any()
    assert not mdp.sc0090_walk_tracking_pass(-commands, commands, contract)[:4].any()
    assert not mdp.sc0090_walk_tracking_pass(commands*.5, commands, contract)[:4].any()
    assert not mdp.sc0090_walk_tracking_pass(torch.tensor([[.08,0.,0.]]), commands[4:], contract).any()


def test_randomized_pass_cannot_hide_a_nominal_turn_failure():
    env = program_env('walk')
    mdp.sc0090_program_restore(env, None)
    request, report = report_for(env, 2400)
    report['cases'][0]['nominal']['turn_right']['wins'] = 0
    assert not mdp.sc0090_program_apply_evaluation(env, report, request)['last_result']['passed']


def test_program_sampling_keeps_low_speed_range_and_separate_turn_signs():
    torch.manual_seed(42)
    command, bucket = mdp.sc0090_sample_program_walk_commands(
        20000, 'cpu', mdp.SC0090_PROGRAM_WALK_PROBABILITIES)
    assert set(bucket.tolist()) == set(range(9))
    for index, sign in ((1, 1), (2, -1)):
        x = sign*command[bucket == index, 0]
        assert (x >= .08).all() and (x <= .15).all()
    for index, sign in ((6, 1), (7, -1)):
        assert (command[bucket == index, :2] == 0).all()
        yaw = sign*command[bucket == index, 2]
        assert (yaw >= .2).all() and (yaw <= 1.).all()
        assert abs(float((bucket == index).float().mean())-.12) < .02
    cfg = make_sc0090_v4_env_cfg('walk')
    assert cfg.commands['twist'].bucket_probabilities == mdp.SC0090_PROGRAM_WALK_PROBABILITIES
    # At the accepted minimum, stationary motion earns <10% of this term.
    p = cfg.rewards['track_linear_velocity'].params
    import math
    assert math.exp(-(.08/(p['absolute_std']+p['relative_std']*.08))**2) < .1
    assert make_training_program('walk')['walk_contract'] == mdp.sc0090_walk_contract()


def test_transition_commands_change_at_control_boundary_with_standing_flags():
    n = len(mdp.SC0090_PROGRAM_WALK_COMMANDS)*4
    env = program_env('walk', n)
    env._sc0090_program_eval_samples = 2
    env._sc0090_program_evaluation_start_step = 1000
    env.common_step_counter = 1000
    term = NS(vel_command_b=torch.zeros(n,3), vel_command_w=torch.zeros(n,3),
              **{k:torch.ones(n,dtype=torch.bool) for k in
                 ('is_standing_env','is_heading_env','is_world_env','is_forward_env')})
    env.command_manager.get_term = lambda _: term
    indexes = {name: i*4 for i,(name,_) in enumerate(mdp.SC0090_PROGRAM_WALK_COMMANDS)}
    start, stop = indexes['start_forward'], indexes['stop_forward']
    mdp.sc0090_program_evaluation_commands(env, 2)
    assert term.is_standing_env[start] and not term.is_standing_env[stop]
    assert term.vel_command_b[start, 0] == 0 and term.vel_command_b[stop, 0] == .1
    env.common_step_counter += round(2/env.step_dt)-1
    mdp.sc0090_program_evaluation_command_step(env, None)
    assert term.vel_command_b[start, 0] == 0
    env.common_step_counter += 1
    mdp.sc0090_program_evaluation_command_step(env, None)
    assert term.vel_command_b[start, 0] == .08 and term.vel_command_b[stop, 0] == 0
    assert not term.is_standing_env[start] and term.is_standing_env[stop]
    assert not term.is_heading_env.any() and not term.is_world_env.any()
