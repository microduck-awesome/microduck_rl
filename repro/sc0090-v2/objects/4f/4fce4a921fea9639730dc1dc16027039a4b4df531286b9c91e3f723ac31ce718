"""V2 integration invariants on actual CAD and copied official configurations."""

import json
import pytest
from mjlab_microduck.custom_v2.microduck_v2 import (
    ASSETS,
    POLICY_JOINTS,
    get_spec,
    make_v2_sensor_env_cfg,
)
from mjlab_microduck.sc0090_v2.contract import load_contract
from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg


def test_v2_mass_home_joints_and_upstream_randomization():
    cfg = make_v2_sensor_env_cfg()
    stock = make_microduck_velocity_env_cfg()
    model = get_spec().compile()
    assert model.body_mass.sum() == pytest.approx(0.600941825, abs=1e-10)
    assert model.nu == 15
    assert tuple(cfg.actions["joint_pos"].actuator_names) == POLICY_JOINTS
    assert cfg.actions["joint_pos"].scale == stock.actions["joint_pos"].scale
    assert cfg.scene.entities["robot"].init_state.joint_pos == {".*": 0.0}
    removed = {
        "randomize_joint_friction",
        "randomize_joint_damping",
        "randomize_armature",
        "randomize_motor_gains",
        "encoder_bias",
    }
    for name in removed:
        assert name not in cfg.events
    for name, term in stock.events.items():
        if name not in removed | {"reset_base"}:
            assert name in cfg.events, name
            assert cfg.events[name].func is term.func, name
    assert cfg.observations["actor"].terms["joint_pos"].noise is None
    assert cfg.observations["actor"].terms["joint_vel"].delay_max_lag == 0
    assert "sc0090_feedback_fault" in cfg.terminations
    assert "sc0090_feedback_fault" not in stock.terminations
    assert stock.scene.entities["robot"].spec_fn is not get_spec
    assert cfg.sim.mujoco.timestep * cfg.decimation == pytest.approx(0.02)


def test_contract_rejects_timing_drift(tmp_path):
    contract = load_contract(ASSETS / "training.json")
    contract["timing"]["feedback_dt_s"] = 0.005
    path = tmp_path / "training.json"
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="timing"):
        load_contract(path)
