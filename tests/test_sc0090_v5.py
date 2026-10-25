"""Freeze the identification equations independently of the upstream BAM adapter."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from bam.actuator import TorchBackend

from mjlab_microduck.actuator.sc0090_v5 import (
    MODEL_PATH,
    MODEL_SHA256,
    SC0090V5Actuator,
    SC0090V5ActuatorCfg,
    load_v5_model,
)


def test_frozen_identity_and_no_pendulum():
    assert hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() == MODEL_SHA256
    record = json.loads(MODEL_PATH.read_bytes())
    model = load_v5_model()
    for key, parameter in model.get_parameters().items():
        assert parameter.value == record[key]
    assert model.actuator.testbench is None
    assert (model.actuator.kp, model.actuator.kd, model.actuator.vin) == (15, 30, 11.1)


def test_modified_parameters_are_rejected(tmp_path):
    path = tmp_path / "model.json"
    record = json.loads(MODEL_PATH.read_bytes())
    record["kt"] *= 1.01
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="frozen"):
        load_v5_model(path)


@pytest.mark.parametrize("voltage", [10.2, 11.1, 12.4])
def test_vectorized_control_matches_raw_bench_coordinates(voltage):
    raw, robot = load_v5_model(vin=voltage, robot_coordinates=False), load_v5_model(vin=voltage)
    random = np.random.default_rng(42)
    q, dq, target = random.normal(size=(3, 64, 15))
    # Include exact HOME and deadband, plus saturated demands in both directions.
    q[0, :5] = 0
    target[0, :5] = [0, 0.001, -0.001, 10, -10]
    nominal_target = raw.feedback_bias.value + raw.feedback_scale.value * target
    expected = raw.actuator.compute_control(nominal_target, q, dq, 0.005)
    robot.actuator.backend = TorchBackend()
    actual = robot.actuator.compute_control(*map(torch.from_numpy, (target, q, dq)), 0.005)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(actual[0, :3], 0)
    np.testing.assert_allclose(actual[0, 3:5], [voltage, -voltage])
    expected_torque = (expected - raw.kt.value * dq) * raw.kt.value / raw.R.value
    torque = robot.actuator.compute_torque(actual, True, torch.from_numpy(q), torch.from_numpy(dq))
    np.testing.assert_allclose(torque.numpy(), expected_torque, rtol=1e-13, atol=1e-13)


def test_m6_friction_matches_identification_equation_all_signs_and_ties():
    model = load_v5_model()
    # Explicit opposite/same signs, zero and equal magnitudes catch the upstream
    # adapter's quadratic-gate discrepancy, which random motion often conceals.
    motor = np.repeat([-2.0, -1.0, 0.0, 1.0, 2.0], 5)
    external = np.tile([-2.0, -1.0, 0.0, 1.0, 2.0], 5)
    for velocity in [0.0, 0.01, -0.5, 8.73]:
        expected, damping = model.compute_frictions(motor, external, velocity)
        dummy = SimpleNamespace(_bam_model=model, friction_scale=torch.tensor([[1.0], [0.8]]))
        m, e = [torch.from_numpy(a).expand(2, -1) for a in (motor, external)]
        stribeck = torch.full_like(
            m, np.exp(-((abs(velocity) / model.dtheta_stribeck.value) ** model.alpha.value))
        )
        actual = SC0090V5Actuator._compute_friction_budget(dummy, m, e, stribeck)
        np.testing.assert_allclose(actual[0], expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            actual[1], expected * float(dummy.friction_scale[1]), rtol=1e-12, atol=1e-12
        )
        assert damping == model.friction_viscous.value


@pytest.mark.parametrize(
    "overrides",
    [
        dict(vin=0.0),
        dict(kp_fw=-1.0),
    ],
)
def test_invalid_operating_conditions_are_rejected(overrides):
    with pytest.raises(ValueError):
        SC0090V5ActuatorCfg(target_names_expr=(".*",), **overrides)
