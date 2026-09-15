"""Failure regression gates for SC0090 V2 (CPU)."""
from copy import deepcopy
from types import SimpleNamespace as NS
import math

import pytest
import torch

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_sc0090_finetune_env_cfg import (
    make_sc0090_walk_v2_env_cfg, make_sc0090_recovery_v2_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg
from mjlab_microduck.tasks.microduck_standup_env_cfg import make_microduck_standup_env_cfg


class Scene(dict):
    def __init__(self, n):
        super().__init__()
        self.env_origins = torch.zeros(n, 3)


def fake_env(n=8):
    scene = Scene(n)
    scene["robot"] = NS(data=NS(
        root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1),
        root_link_pos_w=torch.tensor([[0., 0., .115]]).repeat(n, 1),
        root_link_lin_vel_b=torch.zeros(n, 3), root_link_ang_vel_b=torch.zeros(n, 3),
    ))
    scene["feet_ground_contact"] = NS(data=NS(found=torch.ones(n, 2, 1)))
    rewards = {k: NS(weight=0.) for k in (
        "action_rate_l2", "arrival_damping", "joint_torque_rate_l2")}
    return NS(num_envs=n, device="cpu", scene=scene, step_dt=.02,
              episode_length_buf=torch.zeros(n, dtype=torch.long), common_step_counter=0,
              reward_manager=NS(get_term_cfg=lambda name: rewards[name]),
              sim=NS(data=NS(qpos=torch.zeros(n, 21))))


def test_command_buckets_exercise_failed_regions_without_overrides():
    torch.manual_seed(19)
    p = mdp.SC0090VelocityCommandCfg.bucket_probabilities
    command, bucket = mdp.sc0090_sample_walk_commands(100000, "cpu", p)
    frequency = torch.bincount(bucket, minlength=8) / len(bucket)
    assert torch.allclose(frequency, torch.tensor(p), atol=.005)
    assert torch.all(command[bucket == 0] == 0)
    for group, sign, lo, hi in ((1, 1, .03, .15), (2, -1, .03, .15),
                                 (3, 1, .15, .4), (4, -1, .15, .35)):
        c = command[bucket == group]
        assert torch.all(c[:, 1:] == 0)
        assert torch.all((sign*c[:, 0] >= lo) & (sign*c[:, 0] <= hi))
    assert torch.all(command[bucket == 5][:, [0, 2]] == 0)
    assert torch.all(command[bucket == 6][:, :2] == 0)
    for group, axis in ((5, 1), (6, 2)):
        positive = (command[bucket == group, axis] > 0).float().mean()
        assert .48 < positive < .52


def test_slow_standing_no_longer_receives_near_perfect_tracking():
    env = fake_env(3)
    command = torch.tensor([[.1, 0., 0.], [0., 0., 0.], [.4, 0., 0.]])
    env.command_manager = NS(get_command=lambda _: command)
    reward = mdp.sc0090_track_linear_velocity(env)
    assert reward[0] < .25  # baseline .9048
    assert reward[1] == 1
    env.scene["robot"].data.root_link_lin_vel_b[:] = command
    assert torch.all(mdp.sc0090_track_linear_velocity(env) == 1)
    env.scene["robot"].data.root_link_ang_vel_b[:, :2] = 3.0
    assert torch.all(mdp.sc0090_track_yaw_velocity(env) == 1)
    env.scene["robot"].data.root_link_ang_vel_b[:, 2] = .6
    assert torch.all(mdp.sc0090_track_yaw_velocity(env) < .02)
    assert torch.all(mdp.sc0090_yaw_error(env) > 0)


def test_recovery_potential_orders_back_side_prone_and_stand():
    s = 2**-.5
    q = torch.tensor([[s, 0., -s, 0.], [s, s, 0., 0.],
                      [s, 0., s, 0.], [1., 0., 0., 0.]])
    pot = mdp.sc0090_recovery_potential(q, torch.tensor([.06, .06, .06, .115]))
    assert torch.allclose(pot, torch.tensor([0., .5, 1., 2.]), atol=1e-6)


def test_progress_stationary_and_closed_path_do_not_pay_or_leak_across_reset():
    env = fake_env(1)
    data = env.scene["robot"].data
    s = 2**-.5
    data.root_link_quat_w[:] = torch.tensor([s, 0., -s, 0.])
    data.root_link_pos_w[:, 2] = .06
    assert mdp.sc0090_recovery_progress(env).item() == 0
    env.episode_length_buf[:] = 2
    assert mdp.sc0090_recovery_progress(env).item() == 0
    data.root_link_quat_w[:] = torch.tensor([s, 0., s, 0.])
    up = mdp.sc0090_recovery_progress(env)
    assert up.item() > 0
    data.root_link_quat_w[:] = torch.tensor([s, 0., -s, 0.])
    down = mdp.sc0090_recovery_progress(env)
    assert down.item() < 0
    assert (up + down).abs().item() < 1e-6
    env.episode_length_buf[:] = 0
    data.root_link_quat_w[:] = torch.tensor([1., 0., 0., 0.])
    assert mdp.sc0090_recovery_progress(env).item() == 0


def test_success_requires_continuous_stable_contact_and_counts_once():
    env = fake_env(5)
    env.scene["robot"].data.root_link_pos_w[1, 2] = .06
    env.scene["robot"].data.root_link_ang_vel_b[2, 0] = 2.
    env.scene["feet_ground_contact"].data.found[3, 0] = 0
    state = mdp._sc0090_recovery_state(env)
    for _ in range(24):
        assert not mdp.sc0090_stable_standing(env).any()
    # Break the final env's hold before 0.5 s.
    env.scene["feet_ground_contact"].data.found[4, 0] = 0
    assert mdp.sc0090_stable_standing(env).tolist() == [1., 0., 0., 0., 0.]
    env.scene["feet_ground_contact"].data.found[4, 0] = 1
    for _ in range(24):
        mdp.sc0090_stable_standing(env)
    assert not state.success[4]
    mdp.sc0090_stable_standing(env)
    assert state.success[4]
    state.bucket[:] = torch.tensor([3, 6, 5, 4, 0])
    state.valid[:] = True
    mdp.sc0090_recovery_curriculum(env, torch.arange(5))
    assert state.wins[3] == 1 and state.wins[0] == 1
    mdp.sc0090_recovery_curriculum(env, torch.arange(5))
    assert state.total_trials == [1, 0, 0, 1, 1, 1, 1]


def test_curriculum_needs_frontier_success_sample_count_and_dwell_time():
    env = fake_env()
    state = mdp._sc0090_recovery_state(env)
    state.trials = [1000, 1000, 1000, 256, 100, 100, 100]
    state.wins = [1000, 1000, 1000, 0, 0, 0, 0]
    env.common_step_counter = 2400
    mdp.sc0090_recovery_curriculum(env, [])
    assert state.stage == 0 and not state.polish
    state.trials[3] = 255
    state.wins[3] = 255
    env.common_step_counter = 4800
    mdp.sc0090_recovery_curriculum(env, [])
    assert state.stage == 0
    state.trials[3] = state.wins[3] = 256
    env.common_step_counter = 4799
    mdp.sc0090_recovery_curriculum(env, [])
    assert state.stage == 0
    env.common_step_counter = 4800
    mdp.sc0090_recovery_curriculum(env, [])
    assert state.stage == 1
    assert not state.polish
    # Full-back and BOTH sides must pass before additional smoothing.
    state.stage = 5
    state.trials = [100]*7
    state.trials[3] = 256
    state.wins = state.trials.copy()
    state.wins[6] = 0
    env.common_step_counter = 7200
    mdp.sc0090_recovery_curriculum(env, [])
    assert not state.polish
    state.trials = [100]*7
    state.trials[3] = 256
    state.wins = state.trials.copy()
    env.common_step_counter = 9600
    mdp.sc0090_recovery_curriculum(env, [])
    assert state.polish


def test_checkpoint_state_restores_curriculum_but_discards_old_episodes():
    env = fake_env()
    state = mdp._sc0090_recovery_state(env)
    state.stage, state.last_assessment_step = 3, 7200
    state.trials[3] = 115
    saved = deepcopy(state.state_dict())
    restored = mdp.SC0090RecoveryState(env)
    restored.valid[:] = True
    restored.load_state_dict(saved)
    assert restored.state_dict() == saved
    assert not restored.valid.any()


def test_reverse_spawns_include_real_back_and_correct_left_right(monkeypatch):
    env = fake_env(30000)
    def fake_ground(env, ids, **kwargs):
        s = 2**-.5
        env.sim.data.qpos[ids, 3:7] = torch.tensor([s, 0., -s, 0.])
    monkeypatch.setattr(mdp, "set_random_ground_state", fake_ground)
    mdp.sc0090_reset_recovery(env, torch.arange(env.num_envs))
    state = mdp._sc0090_recovery_state(env)
    from mjlab.utils.lab_api.math import matrix_from_quat
    matrix = matrix_from_quat(env.sim.data.qpos[:, 3:7])
    assert torch.all(matrix[state.bucket == 3, 2, 0] < -.85)  # near prone
    assert torch.all(matrix[state.bucket == 4, 2, 1] < -.99)  # left down
    assert torch.all(matrix[state.bucket == 5, 2, 1] > .99)   # right down
    assert torch.all(matrix[state.bucket == 6, 2, 0] > .99)   # full back
    assert torch.bincount(state.bucket, minlength=7).min() > 1000


@pytest.mark.parametrize("factory,baseline", [
    (make_sc0090_walk_v2_env_cfg, make_microduck_velocity_env_cfg),
    (make_sc0090_recovery_v2_env_cfg, make_microduck_standup_env_cfg),
])
def test_finetune_preserves_runtime_layout_and_pins_dr(factory, baseline):
    before = baseline()
    cfg = factory()
    for group in ("actor", "critic"):
        assert list(cfg.observations[group].terms) == list(before.observations[group].terms)
    assert cfg.actions == before.actions
    assert cfg.events["randomize_com"].params["ranges"] == (-.015, .015)
    assert cfg.events["randomize_head_com"].params["ranges"] == (-.01, .01)
    assert baseline().events["randomize_com"].params["ranges"] == (-.003, .003)
    assert "expand_bam_friction_fields" in cfg.events
    assert cfg.commands["twist"].rel_forward_envs == 0
    assert factory(play=True).observations["actor"].terms.keys() == cfg.observations["actor"].terms.keys()


def test_recovery_does_not_polish_or_dilute_target_before_discovery():
    cfg = make_sc0090_recovery_v2_env_cfg()
    for key in ("arrival_damping", "joint_torque_rate_l2", "body_pose_tracking"):
        assert cfg.rewards[key].weight == 0
    assert cfg.rewards["standing_composite"].weight == 3.75
    assert cfg.rewards["remaining_fallen"].weight < 0
    assert "ground_state_mix" not in cfg.curriculum
    assert "com_upward_velocity" not in cfg.rewards
    assert "recovery_success" in cfg.curriculum
