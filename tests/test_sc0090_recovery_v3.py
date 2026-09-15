"""Recovery promotion must represent evaluated ability, without changing PPO."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace as NS
import json
import subprocess

import cloudpickle
import pytest
import torch

from mjlab_microduck.tasks import SC0090FineTuneRunner, mdp
from mjlab_microduck.tasks.microduck_sc0090_finetune_env_cfg import (
    make_sc0090_recovery_v2_env_cfg, SC0090RecoveryV2RlCfg,
)
from mjlab_microduck.tasks.microduck_sc0090_recovery_v3_env_cfg import (
    make_sc0090_recovery_v3_env_cfg, SC0090RecoveryV3RlCfg,
)
from mjlab_microduck.tasks.recovery_eval_runner import SC0090RecoveryEvalRunner
from mjlab.rl import MjlabOnPolicyRunner
from test_sc0090_finetune import fake_env


def assessment(env, iteration=100, samples=128):
    request = dict(schema=1, checkpoint_sha256="checkpoint", config_sha256="config",
                   iteration=iteration, stage=mdp._sc0090_recovery_state(env).stage,
                   seeds=[71, 72], samples_per_group=samples, duration_s=8.0,
                   deterministic=True)
    report = {**deepcopy(request), "cases": [
        {"seed": seed, "finite": True, "groups": {
            name: {"trials": samples, "wins": samples if i < 4 else 0}
            for i, name in enumerate(mdp.SC0090_RECOVERY_BUCKETS)}}
        for seed in request["seeds"]]}
    return request, report


def test_v3_retains_physics_commands_observations_and_exploration():
    old, new = make_sc0090_recovery_v2_env_cfg(), make_sc0090_recovery_v3_env_cfg()
    for name in ("sim", "actions", "observations", "commands", "events", "terminations"):
        assert getattr(old, name) == getattr(new, name), name
    assert old.scene.entities["robot"].articulation == new.scene.entities["robot"].articulation
    for name, reward in old.rewards.items():
        assert reward == new.rewards[name], name
    for name in ("algorithm", "actor", "critic", "num_steps_per_env"):
        assert getattr(SC0090RecoveryV3RlCfg, name) == getattr(SC0090RecoveryV2RlCfg, name)
    assert new.curriculum["recovery_success"].params["advance_on_training"] is False
    assert "advance_on_training" not in old.curriculum["recovery_success"].params
    restored = cloudpickle.loads(cloudpickle.dumps({"env": new, "agent": asdict(SC0090RecoveryV3RlCfg)}))
    assert restored["env"].commands == new.commands
    assert restored["agent"]["recovery_evaluation"]["samples_per_group"] == 128


def test_neck_penalty_charges_actual_stop_and_is_zero_inside():
    env = fake_env(4)
    reward = make_sc0090_recovery_v3_env_cfg().rewards["neck_limit_proximity"]
    selector = deepcopy(reward.params["asset_cfg"])
    assert selector.joint_names == ("neck_pitch",)
    selector.joint_ids = [1]
    robot = env.scene["robot"].data
    robot.joint_pos = torch.tensor([[100., 0.], [100., 1.0472], [100., -1.0472], [100., 1.1472]])
    robot.joint_pos_limits = torch.tensor([[[-1., 1.], [-1.0472, 1.0472]]]).repeat(4, 1, 1)
    value = reward.weight * reward.func(env, asset_cfg=selector, margin=reward.params["margin"])
    assert torch.allclose(value, torch.tensor([0., -.4, -.4, -.6]), atol=1e-6)


@pytest.mark.parametrize("stage", [0, 5])
def test_training_success_cannot_advance_or_polish_v3(stage):
    env = fake_env()
    state = mdp._sc0090_recovery_state(env)
    state.stage = stage
    state.trials = state.wins = [1000] * 7
    env.common_step_counter = 2400
    mdp.sc0090_recovery_curriculum(env, [], advance_on_training=False)
    assert state.stage == stage and not state.polish
    assert state.rates == [1.] * 7


def test_assessment_advances_one_stage_and_preserves_existing_counters():
    env = fake_env()
    env.common_step_counter = 51648
    state = mdp._sc0090_recovery_state(env)
    state.total_trials = [1234] * 7
    request, report = assessment(env)
    result = mdp.sc0090_apply_recovery_evaluation(env, report, request)
    assert state.stage == 1 and not state.polish
    assert result["last_promotion_step"] == 51648
    assert state.total_trials == [1234] * 7
    assert result["counts"]["frontier"] == [256, 256]
    assert state.last_assessment_step == 51648
    assert state.trials == state.wins == [0] * 7
    with pytest.raises(ValueError):
        mdp.sc0090_apply_recovery_evaluation(env, report, request)


@pytest.mark.parametrize("failed_group", ["standing", "sitting", "prone", "frontier"])
def test_each_seed_must_pass_and_easy_skills_must_be_retained(failed_group):
    env = fake_env()
    env.common_step_counter = 2400
    request, report = assessment(env)
    report["cases"][0]["groups"][failed_group]["wins"] = 89  # 69.53%, second seed 100%
    result = mdp.sc0090_apply_recovery_evaluation(env, report, request)
    assert not result["promoted"]


@pytest.mark.parametrize("steps,samples", [(2399, 128), (2400, 127)])
def test_dwell_time_and_256_frontier_trials_required(steps, samples):
    env = fake_env()
    env.common_step_counter = steps
    request, report = assessment(env, samples=samples)
    assert not mdp.sc0090_apply_recovery_evaluation(env, report, request)["promoted"]


def test_dwell_time_is_measured_since_last_promotion_and_stale_reports_rejected():
    env = fake_env()
    env.common_step_counter = 2400
    request, report = assessment(env)
    mdp.sc0090_apply_recovery_evaluation(env, report, request)
    request, report = assessment(env, iteration=200)
    env.common_step_counter = 4799
    assert not mdp.sc0090_apply_recovery_evaluation(env, report, request)["promoted"]
    with pytest.raises(ValueError, match="already consumed"):
        mdp.sc0090_apply_recovery_evaluation(env, report, request)
    request, report = assessment(env, iteration=300)
    env.common_step_counter = 4800
    assert mdp.sc0090_apply_recovery_evaluation(env, report, request)["stage_after"] == 2


@pytest.mark.parametrize("bad", ["checkpoint", "config", "missing_seed", "missing_pose", "nonfinite",
                                  "short_trials", "invalid_wins", "wrong_seed", "wrong_stage"])
def test_invalid_assessments_do_not_mutate_curriculum(bad):
    env = fake_env()
    env.common_step_counter = 2400
    state = mdp._sc0090_recovery_state(env)
    request, report = assessment(env)
    if bad in ("checkpoint", "config"):
        report[f"{bad}_sha256"] = "mismatch"
    elif bad == "missing_seed":
        report["cases"].pop()
    elif bad == "missing_pose":
        report["cases"][1]["groups"].pop("supine")
    elif bad == "nonfinite":
        report["cases"][1]["finite"] = False
    elif bad == "short_trials":
        report["cases"][1]["groups"]["frontier"]["trials"] = 127
    elif bad == "invalid_wins":
        report["cases"][1]["groups"]["frontier"]["wins"] = 129
    elif bad == "wrong_seed":
        report["cases"][1]["seed"] = 100
    else:
        request["stage"] = report["stage"] = 2
    before = deepcopy(state.state_dict())
    with pytest.raises(ValueError):
        mdp.sc0090_apply_recovery_evaluation(env, report, request)
    assert state.state_dict() == before
    assert not hasattr(env, "_sc0090_eval_history")


@pytest.mark.parametrize("failed_group", ["left_side", "right_side", "supine", None])
def test_final_polish_requires_all_three_real_failure_poses(failed_group):
    env = fake_env()
    env.common_step_counter = 2400
    state = mdp._sc0090_recovery_state(env)
    state.stage = 5
    request, report = assessment(env)
    for case in report["cases"]:
        for counts in case["groups"].values():
            counts["wins"] = 128
    if failed_group:
        report["cases"][1]["groups"][failed_group]["wins"] = 89
    result = mdp.sc0090_apply_recovery_evaluation(env, report, request)
    assert result["polish"] == (failed_group is None)
    assert state.stage == 5 and not result["promoted"]


def test_fixed_pose_groups_support_partial_resets_without_corrupting_other_envs(monkeypatch):
    env = fake_env(14)
    def reset(env, ids, **kwargs):
        env.sim.data.qpos[ids, 3:7] = torch.tensor([2**-.5, 0., -2**-.5, 0.])
    monkeypatch.setattr(mdp, "set_random_ground_state", reset)
    mdp.sc0090_reset_recovery(env, None, evaluation_buckets=True)
    state = mdp._sc0090_recovery_state(env)
    assert state.bucket.tolist() == [i // 2 for i in range(14)]
    state.hold[:] = 25
    state.success[:] = True
    positions = env.sim.data.qpos.clone()
    ids = torch.tensor([11, 2, 7])
    mdp.sc0090_reset_recovery(env, ids, evaluation_buckets=True)
    mask = torch.ones(14, dtype=torch.bool)
    mask[ids] = False
    assert torch.equal(positions[mask], env.sim.data.qpos[mask])
    assert state.success[mask].all() and (state.hold[mask] == 25).all()
    assert not state.success[ids].any() and not state.hold[ids].any()
    assert state.bucket.tolist() == [i // 2 for i in range(14)]


@pytest.mark.parametrize("infos", [{}, {"sc0090_evaluation": {"last_attempt_iteration": 2150, "last_promotion_step": 51672}}])
def test_runner_resume_restores_evaluation_history_and_accepts_v2_checkpoints(monkeypatch, infos):
    monkeypatch.delenv("MICRODUCK_WARM_START", raising=False)
    monkeypatch.setattr(SC0090FineTuneRunner, "load", lambda *args: infos)
    runner = object.__new__(SC0090RecoveryEvalRunner)
    runner.env = NS(unwrapped=fake_env())
    runner.load("checkpoint")
    assert runner.env.unwrapped._sc0090_eval_history == infos.get("sc0090_evaluation", {})
    assert runner.env.unwrapped._sc0090_eval_history is not infos.get("sc0090_evaluation")


def test_runner_preserves_constructor_keys_consumed_by_rsl(monkeypatch):
    def initialize(self, env, cfg, *args, **kwargs):
        self.env, self.cfg, self.is_distributed = env, cfg, False
        cfg["algorithm"].pop("class_name")
        cfg["actor"].pop("class_name")
    monkeypatch.setattr(SC0090FineTuneRunner, "__init__", initialize)
    cfg = asdict(SC0090RecoveryV3RlCfg)
    before = deepcopy(cfg)
    runner = SC0090RecoveryEvalRunner(NS(unwrapped=fake_env()), cfg)
    assert runner._evaluation_train_cfg == before
    assert "class_name" not in runner.cfg["algorithm"]


def test_failed_evaluation_keeps_stage_and_persists_failure_without_repeating(monkeypatch, tmp_path):
    env = fake_env()
    env.cfg = NS()
    env.common_step_counter = 2400
    env._sc0090_eval_history = {}
    runner = object.__new__(SC0090RecoveryEvalRunner)
    runner.env, runner.cfg = NS(unwrapped=env), asdict(SC0090RecoveryV3RlCfg)
    runner._evaluation_train_cfg = deepcopy(runner.cfg)
    runner.current_learning_iteration, runner.device = 100, "cuda:0"
    writes, calls = [], []
    def save(self, path, infos):
        writes.append(deepcopy(infos))
        path.write_bytes(b"frozen policy")
    def timeout(*args, **kwargs):
        calls.append(kwargs)
        raise subprocess.TimeoutExpired(args[0], 240)
    monkeypatch.setattr(SC0090FineTuneRunner, "save", save)
    monkeypatch.setattr(MjlabOnPolicyRunner, "save", save)
    monkeypatch.setattr(subprocess, "run", timeout)
    rng = torch.get_rng_state().clone()
    runner.save(tmp_path / "model.pt")
    assert torch.equal(rng, torch.get_rng_state())
    assert mdp._sc0090_recovery_state(env).stage == 0
    assert calls[0]["env"]["CUDA_MPS_PIPE_DIRECTORY"] == ""
    assert "MICRODUCK_WARM_START" not in calls[0]["env"]
    assert writes[-1]["sc0090_evaluation"]["last_attempt_iteration"] == 100
    assert "TimeoutExpired" in writes[-1]["sc0090_evaluation"]["last_error"]["error"]
    runner.save(tmp_path / "model.pt")
    assert len(calls) == 1
