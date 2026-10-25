"""Compare the baseline against the pinned official recipe on the actual V2 CAD."""

from dataclasses import asdict, fields
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import pytest
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.registry import list_tasks, load_rl_cfg
from mjlab_microduck.actuator.sc0090_v5 import SC0090V5ActuatorCfg, MODEL_SHA256
from mjlab_microduck.custom_v2.microduck_v2 import (
    ASSETS,
    HOME_HEIGHT,
    POLICY_JOINTS,
    TASK_ID,
    get_spec,
    make_v2_baseline_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)


def parameters_without_joint_selection(term):
    """Allow only the deliberate 14-joint selector adaptation in copied parameters."""
    params = dict(term.params)
    default = inspect.signature(term.func).parameters.get("asset_cfg")
    if "asset_cfg" not in params and default is not None:
        params["asset_cfg"] = default.default
    if "asset_cfg" in params:
        selector = asdict(params.pop("asset_cfg"))
        for key in ("joint_names", "joint_ids", "preserve_order"):
            selector.pop(key)
        params["asset_cfg"] = selector
    return params


@pytest.mark.parametrize("play", [False, True])
def test_observation_noise_delay_bias_and_privileged_critic_match_upstream(play):
    stock = make_microduck_velocity_env_cfg(play=play)
    cfg = make_v2_baseline_env_cfg(play=play)
    assert list(cfg.observations) == list(stock.observations)
    for name, group in stock.observations.items():
        actual = cfg.observations[name]
        for field in fields(group):
            if field.name != "terms":
                assert getattr(actual, field.name) == getattr(group, field.name)
        assert list(actual.terms) == list(group.terms)
        for key, expected in group.terms.items():
            term = actual.terms[key]
            for field in fields(term):
                if field.name != "params":
                    assert getattr(term, field.name) == getattr(expected, field.name), (
                        name,
                        key,
                        field.name,
                    )
            assert parameters_without_joint_selection(
                term
            ) == parameters_without_joint_selection(expected)
    actor = cfg.observations["actor"].terms
    assert (actor["joint_pos"].noise.n_min, actor["joint_pos"].noise.n_max) == (
        -0.001,
        0.001,
    )
    assert (actor["joint_vel"].noise.n_min, actor["joint_vel"].noise.n_max) == (
        -0.25,
        0.25,
    )
    assert (actor["joint_vel"].delay_min_lag, actor["joint_vel"].delay_max_lag) == (
        1,
        1,
    )
    assert actor["joint_pos"].params["biased"] is True
    assert cfg.observations["critic"].terms["joint_pos"].params["biased"] is False
    assert parameters_without_joint_selection(
        cfg.events["encoder_bias"]
    ) == parameters_without_joint_selection(stock.events["encoder_bias"])


def test_rewards_commands_general_randomization_and_ppo_are_inherited():
    stock = make_microduck_velocity_env_cfg()
    cfg = make_v2_baseline_env_cfg()
    motor_events = {
        "randomize_joint_friction",
        "randomize_joint_damping",
        "randomize_armature",
        "randomize_motor_gains",
    }
    assert set(cfg.events) == (set(stock.events) - motor_events) | {
        "mouth_hold",
        "mouth_reset",
    }
    for name, expected in stock.events.items():
        if name in motor_events or name == "reset_base":
            continue
        term = cfg.events[name]
        assert term.func is expected.func
        assert term.mode == expected.mode
        assert parameters_without_joint_selection(
            term
        ) == parameters_without_joint_selection(expected), name
    assert cfg.commands == stock.commands
    for name, expected in stock.rewards.items():
        term = cfg.rewards[name]
        assert term.func is expected.func
        assert term.weight == expected.weight
        assert parameters_without_joint_selection(
            term
        ) == parameters_without_joint_selection(expected), name
    for name, expected in stock.curriculum.items():
        assert cfg.curriculum[name].func is expected.func
        assert cfg.curriculum[name].params == expected.params
    agent, official = asdict(load_rl_cfg(TASK_ID)), asdict(MicroduckRlCfg)
    for key in ("experiment_name", "run_name"):
        agent.pop(key)
        official.pop(key)
    assert agent == official
    assert cfg.terminations.keys() == stock.terminations.keys()
    assert cfg.actions["joint_pos"].scale == stock.actions["joint_pos"].scale == 1.0
    assert type(cfg.actions["joint_pos"]) is JointPositionActionCfg
    assert cfg.actions["joint_pos"].clip is None
    assert tuple(cfg.actions["joint_pos"].actuator_names) == POLICY_JOINTS
    assert cfg.sim.mujoco.timestep == stock.sim.mujoco.timestep == 0.005
    assert cfg.decimation == stock.decimation == 4
    # Building V2 must not mutate the stock factory's future configurations.
    fresh = make_microduck_velocity_env_cfg()
    assert (
        fresh.scene.entities["robot"].spec_fn is stock.scene.entities["robot"].spec_fn
    )
    assert fresh.observations == stock.observations
    assert fresh.events == stock.events


def test_cad_motor_and_archive_independence():
    cfg = make_v2_baseline_env_cfg()
    model = get_spec().compile()
    assert float(model.body_mass.sum()) == pytest.approx(0.6246, abs=1e-10)
    assert model.nu == 15
    robot = cfg.scene.entities["robot"]
    assert robot.init_state.pos == (0.0, 0.0, HOME_HEIGHT)
    assert robot.init_state.joint_pos == {".*": 0.0}
    for actuator in robot.articulation.actuators:
        assert type(actuator) is SC0090V5ActuatorCfg
        assert (actuator.kp_fw, actuator.kd_fw, actuator.vin) == (15, 30, 11.1)
        assert (
            hashlib.sha256(Path(actuator.json_path).read_bytes()).hexdigest()
            == MODEL_SHA256
        )
        assert actuator.vin_range is None
    assert importlib.util.find_spec("mjlab_microduck.sc0090") is None
    assert importlib.util.find_spec("mjlab_microduck.sc0090_v2") is None
    assert not (ASSETS.parent / "training-assets").exists()
    assert not (ASSETS / "training.json").exists()
    manifest = json.loads((ASSETS.parent / "assembly-install.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((ASSETS.parent / name).read_bytes()).hexdigest() == digest
    assert TASK_ID in list_tasks()
    assert "Mjlab-Velocity-Flat-MicroDuck" in list_tasks()
    assert all("S0-O0-R1" not in task for task in list_tasks())
