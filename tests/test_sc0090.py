"""Motor migration checks: provenance, control coordinates and friction DR."""

import hashlib
import json

import numpy as np
import pytest
import torch
from bam.actuator import TorchBackend

from mjlab_microduck.actuator.sc0090 import (
    SC0090_MODEL_PATH, SC0090_MODEL_SHA256, load_sc0090_model,
    SC0090_MAX_SPEED_RAD_S,
)
from mjlab_microduck.actuator.friction_dr_bam import FrictionDRBamActuator


def test_selected_fit_and_operating_point():
    assert hashlib.sha256(SC0090_MODEL_PATH.read_bytes()).hexdigest() == SC0090_MODEL_SHA256
    artifact = json.loads(SC0090_MODEL_PATH.read_text())
    model = load_sc0090_model()
    assert artifact["comparison_source"]["seed"] == 1
    assert artifact["physical_qualification"]["qualified"] is False
    for name, param in model.get_parameters().items():
        assert param.value == artifact[name]
    assert (model.actuator.vin, model.actuator.kp, model.actuator.kd) == (12., 20., 30.)
    assert model.actuator.deadband_count == 2
    assert model.actuator.radians_per_count == artifact["operating_point"]["servo_radians_per_count"]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_joint_coordinates_and_numpy_torch_control(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = load_sc0090_model()
    q = np.linspace(-1.5, 1.5, 14)[None].repeat(64, axis=0)
    # Bench bias/scale must not change a stationary robot joint's zero.
    np.testing.assert_allclose(model.actuator.compute_control(q, q, np.zeros_like(q), .005), 0, atol=1e-12)
    target = q + np.linspace(-.2, .2, 14)
    velocity = np.full_like(q, .3)
    act = model.actuator
    count_error = model.feedback_scale.value * (target-q) / act.radians_per_count
    deadzone = count_error - np.clip(count_error, -2, 2)
    expected = 12 * np.clip(model.pwm_per_p_count.value * (
        20 * deadzone - 30 * model.derivative_seconds.value * velocity / act.radians_per_count), -1, 1)
    np.testing.assert_allclose(act.compute_control(target, q, velocity, .005), expected, atol=1e-12)
    act.backend = TorchBackend()
    actual = act.compute_control(*(torch.tensor(x, device=device) for x in (target, q, velocity)), .005)
    np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=1e-12)


def test_friction_scaled_once_and_partial_reset():
    act = object.__new__(FrictionDRBamActuator)
    act._bam_model = load_sc0090_model()
    act.friction_scale = torch.ones(3, 1)
    act.default_friction_scale = act.friction_scale.clone()
    torque, external, stribeck = (torch.full((3, 14), v) for v in (.2, -.1, .4))
    baseline = act._compute_friction_budget(torque, external, stribeck)
    act.set_friction_scale(slice(None), torch.tensor([[.5], [1.], [1.5]]))
    torch.testing.assert_close(act._compute_friction_budget(torque, external, stribeck),
                               baseline * torch.tensor([[.5], [1.], [1.5]]))
    act.reset_friction_scale(torch.tensor([0, 2]))
    torch.testing.assert_close(act.friction_scale, torch.ones(3, 1))


@pytest.mark.parametrize("device", ["numpy", "cpu", "cuda:0"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_80rpm_drive_ceiling_retains_braking(device, dtype):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = load_sc0090_model()
    v = SC0090_MAX_SPEED_RAD_S
    q = np.zeros(6, dtype=dtype)
    dq = np.array([v, -v, v*1.2, -v*1.2, v*.9, -v*.9], dtype=dtype)
    target = np.array([10., -10., 10., -10., 10., -10.], dtype=dtype)
    if device != "numpy":
        model.actuator.backend = TorchBackend()
        q, dq, target = (torch.tensor(x, device=device) for x in (q, dq, target))
    act = model.actuator
    torque = act.compute_torque(act.compute_control(target, q, dq, .005), True, q, dq)
    power = torque * dq
    assert bool((power[:4] <= 8 * np.finfo(dtype).eps).all())
    assert bool((power[4:] > 0).all())
    braking = act.compute_torque(act.compute_control(-target, q, dq, .005), True, q, dq)
    assert bool((braking * dq < 0).all())


def test_both_training_tasks_use_sc0090_and_recovery_curriculum():
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg
    from mjlab_microduck.tasks.microduck_standup_env_cfg import make_microduck_standup_env_cfg
    for factory in (make_microduck_velocity_env_cfg, make_microduck_standup_env_cfg):
        cfg = factory()
        actuator = cfg.scene.entities["robot"].articulation.actuators[0]
        assert actuator.json_path == str(SC0090_MODEL_PATH)
        assert actuator.vin == 12
        assert actuator.delay_max_lag == 0 and actuator.use_identified_delay
        assert "expand_bam_friction_fields" in cfg.events
    stages = make_microduck_standup_env_cfg().curriculum["ground_state_mix"].params["param_stages"]
    assert stages[-1]["params"]["face_up_prob"] > 0
    assert stages[-1]["params"]["face_down_prob"] > 0
