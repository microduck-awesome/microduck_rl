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
    request = dict(schema=2, task=state["task"], program_hash=state["program_hash"],
        phase_index=state["phase_index"], global_step=step, iteration=step//24,
        checkpoint_sha256="checkpoint", config_sha256="config", seeds=[101, 102],
        samples_per_group=128, duration_s=8.)
    names = mdp.SC0090_RECOVERY_BUCKETS if state["task"] == "recovery" else tuple(k for k, _ in mdp.SC0090_PROGRAM_WALK_COMMANDS)
    cases = []
    for seed in request["seeds"]:
        cases.append(dict(seed=seed, finite=True,
            **{profile: {k: dict(trials=128, wins=128) for k in names} for profile in ("retention", "challenge")},
            metrics=dict(action_delta_rms=.1, torque_delta_rms=.01, settled_ang_rms=.1, rise_time_p95=2.)))
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
    for _ in range(25):
        reward = mdp.sc0090_program_stable_standing(env)
    assert reward.tolist() == [1., 0.]
    assert state.success.tolist() == [True, False]
    assert state.hold.tolist() == [0, 0]  # Neither currently meets nominal z >= .105.


@pytest.mark.parametrize("actor_only", [False, True])
def test_full_resume_restores_adaptive_lr_and_uses_next_iteration(monkeypatch, actor_only):
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
    infos = {"sc0090_recovery": {"stage": 5, "polish": True}}
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"iter": 5999, "infos": infos})
    def load(self, *args, **kwargs):
        env.common_step_counter = 144000
        if kwargs.get("load_cfg") is None:
            self.current_learning_iteration = 5999
        return infos
    monkeypatch.setattr(SC0090FineTuneRunner, "load", load)
    runner.load("model_5999.pt", load_cfg={"actor": True} if actor_only else None)
    assert runner.current_learning_iteration == (0 if actor_only else 6000)
    assert runner.alg.learning_rate == (.0002 if actor_only else .00003)
    assert env.common_step_counter == 144000
    assert reset_phases == ["consolidate"]
    assert mdp.sc0090_program_state(env)["phase_enter_step"] == 144000
