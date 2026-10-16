"""Frozen v5 SC0090 M6 dynamics in calibrated robot coordinates.

The identified coefficients are unchanged. Firmware gains and supply are
operating conditions; bench gravity, payload and zero do not enter the robot.
This module leaves the upstream XL330 actuators and their tasks untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from bam.actuator import VoltageControlledActuator
from bam.actuators import actuators
from bam.mjlab import BamActuator, BamActuatorCfg
from bam.model import load_model_from_dict
from bam.parameter import Parameter
from bam.testbench import Pendulum


MODEL_PATH = Path(__file__).parent / "params" / "sc0090_v5_m6.json"
MODEL_SHA256 = "7c35a732575bfad135d5bf7c7eeb44f268ffd96862064f04e9e7a5105bbf6fb4"
IDENTIFICATION_BAM_REVISION = "3505fba49a8aa41f47c9a729c495f0c083757b0b"


class SC0090Feedback(VoltageControlledActuator):
    """The v5 firmware PWM/deadband law, with an explicit coordinate adapter."""

    def __init__(self):
        super().__init__(Pendulum, vin=11.1, kp=15.0, error_gain=1.0, max_pwm=1.0, max_current=None)
        self.kd = 30.0
        self.radians_per_count = np.deg2rad(330.0) / 1024
        self.deadband_count = 2.0
        self.robot_coordinates = False

    def initialize(self):
        # These initial values are replaced by the hash-checked frozen artifact.
        for name, value, low, high in (
            ("kt", 0.89240515, 0.1, 3.0),
            ("R", 27.0, 5.0, 60.0),
            ("armature", 0.0001, 0.000001, 0.01),
            ("pwm_per_p_count", 0.001, 0.0001, 0.01),
            ("derivative_seconds", 0.0, 0.0, 0.05),
            ("feedback_bias", 0.0, -0.3, 0.3),
            ("feedback_scale", 1.0, 0.5, 1.5),
        ):
            setattr(self.model, name, Parameter(value, low, high))

    def compute_control(self, q_target, q, dq, dt):
        model = self.model
        feedback = model.feedback_bias.value + model.feedback_scale.value * q
        if self.robot_coordinates:
            # Apply the SAME map to target and feedback: bench bias cannot move
            # robot HOME. Direction/zero calibration is an external joint map.
            q_target = model.feedback_bias.value + model.feedback_scale.value * q_target
        error = (q_target - feedback) / self.radians_per_count
        error -= self.backend.clamp(error, -self.deadband_count, self.deadband_count)
        derivative = self.kd * model.derivative_seconds.value * dq / self.radians_per_count
        duty = model.pwm_per_p_count.value * (self.kp * error - derivative)
        self.duty_cycle = self.backend.clamp(duty, -self.max_pwm, self.max_pwm)
        return self.vin * self.duty_cycle

    def get_extra_inertia(self):
        return self.model.armature.value

    def reset(self, env_ids=...):
        duty = getattr(self, "duty_cycle", None)
        if env_ids is ... or env_ids is None or duty is None:
            self.duty_cycle = None
        elif isinstance(duty, torch.Tensor):
            self.duty_cycle[env_ids] = 0


# Required by BAM's JSON loader. Only this actuator family is registered.
actuators["sc0090_feedback_v1"] = SC0090Feedback


def load_v5_model(path=MODEL_PATH, *, kp=15.0, kd=30.0, vin=11.1, robot_coordinates=True):
    """Load the exact P04 dynamics artifact without loading its pendulum."""
    blob = Path(path).read_bytes()
    if hashlib.sha256(blob).hexdigest() != MODEL_SHA256:
        raise ValueError("Expected the frozen sc0090-v5-m6-seed1 artifact used by P04")
    record = json.loads(blob)
    if record["bam_revision"] != IDENTIFICATION_BAM_REVISION:
        raise ValueError("Unexpected v5 identification provenance")
    for name, value, positive in (("P", kp, False), ("D", kd, False), ("voltage", vin, True)):
        if not math.isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError(f"Invalid SC0090 operating condition: {name}")
    settings = record["operating_point"]
    if settings["ki"] != 0 or not settings["goal_profile_zeroed"]:
        raise ValueError("The v5 fit requires I=0 and zeroed firmware goal profiles")
    model = load_model_from_dict(record)
    act = model.actuator
    act.kp, act.kd, act.vin = float(kp), float(kd), float(vin)
    act.radians_per_count = settings["servo_radians_per_count"]
    act.deadband_count = settings["deadband_count"]
    act.max_pwm = settings["max_pwm"]
    act.robot_coordinates = robot_coordinates
    return model


@dataclass(kw_only=True)
class SC0090V5ActuatorCfg(BamActuatorCfg):
    json_path: str | None = str(MODEL_PATH)
    kp_fw: float | None = 15.0
    kd_fw: float = 30.0
    vin: float | None = 11.1
    graph_compute: bool = True
    fast_friction_force: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.delay_min_lag or self.delay_max_lag:
            raise ValueError(
                "v5 identified command delay is zero; schedule bus commands separately"
            )
        if self.vin_range is not None or self.vin_drop_gain_range is not None:
            raise ValueError("Use the shared, measured V2 supply sampler, not XL330 voltage DR")
        if self.kp_fw is None or self.vin is None:
            raise ValueError("SC0090 firmware P and voltage must be explicit")
        load_v5_model(self.json_path, kp=self.kp_fw, kd=self.kd_fw, vin=self.vin)

    def build(self, entity, target_ids, target_names):
        return SC0090V5Actuator(self, entity, target_ids, target_names)


class SC0090V5Actuator(BamActuator):
    """BAM solver integration with the identification model's exact M6 gate."""

    def __init__(self, cfg, entity, target_ids, target_names):
        super().__init__(cfg, entity, target_ids, target_names)
        self._bam_model = load_v5_model(cfg.json_path, kp=cfg.kp_fw, kd=cfg.kd_fw, vin=cfg.vin)
        self.source_sha256 = MODEL_SHA256
        self._compute_graph = None
        self._graph_compute_supported = False
        self._capture_friction = False
        self._damping_scalar = None

    def edit_spec(self, spec, target_names):
        super().edit_spec(spec, target_names)
        for actuator in self._mjs_actuators:
            # PWM saturation and back-EMF already bound active motor effort.
            # A fixed stall-torque clip would incorrectly clip braking torque.
            # The XML's preview position range must not become a torque clamp.
            actuator.ctrllimited = False
            actuator.forcelimited = False

    def initialize(self, mj_model, model, data, device):
        super().initialize(mj_model, model, data, device)
        self.friction_scale = torch.ones_like(self.kp_scale)
        self.default_friction_scale = self.friction_scale.clone()
        self.firmware_kd_scale = torch.ones_like(self.kp_scale)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._bam_model.actuator.reset(env_ids)

    def reset_friction_scale(self, env_ids):
        self.friction_scale[env_ids] = self.default_friction_scale[env_ids]

    def set_friction_scale(self, env_ids, scale):
        self.friction_scale[env_ids] = scale

    def _compute_friction_budget(self, motor_torque, external_torque, stribeck_coeff):
        """Match bam.model.Model.compute_frictions, including opposite signs/ties."""
        m = self._bam_model
        friction = torch.full_like(motor_torque, m.friction_base.value)
        friction += torch.abs(
            external_torque * m.load_friction_external.value
            - motor_torque * m.load_friction_motor.value
        )
        friction += stribeck_coeff * (
            m.friction_stribeck.value
            + torch.abs(
                external_torque * m.load_friction_external_stribeck.value
                - motor_torque * m.load_friction_motor_stribeck.value
            )
        )
        ext, motor = external_torque.abs(), motor_torque.abs()
        quadratic = torch.where(
            ext < motor, m.load_friction_external_quad.value * ext.square(), 0.0
        ) + torch.where(ext > motor, m.load_friction_motor_quad.value * motor.square(), 0.0)
        friction += stribeck_coeff * quadratic * (external_torque.sign() != motor_torque.sign())
        return friction * self.friction_scale

    def _write_frictions(self, frictionloss, damping):
        if self._capture_friction:
            self._captured_friction = frictionloss
            return
        if self._damping_scalar is None or self._damping_scalar.device != frictionloss.device:
            self._damping_scalar = torch.tensor(damping, device=frictionloss.device,
                                                dtype=frictionloss.dtype)
        super()._write_frictions(frictionloss, self._damping_scalar)

    def _dof_friction_force(self, nv):
        if self.cfg.fast_friction_force:
            efc = self._data.efc
            types, ids, forces, counts = [self._as_tensor(x) for x in
                (efc.type, efc.id, efc.force, self._data.nefc)]
            if (forces.is_cuda and forces.dtype == torch.float32
                    and types.dtype == ids.dtype == counts.dtype == torch.int32):
                from .friction_force import friction_force
                return friction_force(types, ids, forces, counts, nv)
        return super()._dof_friction_force(nv)

    def compute(self, cmd):
        if self.cfg.graph_compute and self._graph_compute_supported:
            from .v5_graph import V5ComputeGraph
            if self._compute_graph is None:
                self._compute_graph = V5ComputeGraph(self)
            return self._compute_graph.compute(cmd, self._compute_eager)
        return self._compute_eager(cmd)

    def _compute_eager(self, cmd):
        # Keep actual velocity in the firmware D term; upstream kd_scale is
        # electrical damping, a different quantity from the firmware D register.
        model, act = self._bam_model, self._bam_model.actuator
        act.vin = self.vin_tensor
        act.kp = self.cfg.kp_fw * self.kp_scale
        act.kd = self.cfg.kd_fw * self.firmware_kd_scale
        control = act.compute_control(cmd.position_target, cmd.pos, cmd.vel, self._dt)
        torque = act.compute_torque(control, True, cmd.pos, cmd.vel * self.kd_scale)
        bias = self._as_tensor(self._data.qfrc_bias)
        constraint = self._as_tensor(self._data.qfrc_constraint)
        previous = self._as_tensor(self._data.qfrc_actuator)[:, self._dof_ids]
        external = (
            -bias[:, self._dof_ids]
            + constraint[:, self._dof_ids]
            - self._dof_friction_force(bias.shape[-1])[:, self._dof_ids]
        )
        stribeck = torch.exp(
            -torch.pow(cmd.vel.abs() / model.dtheta_stribeck.value, model.alpha.value)
        )
        budget = self._compute_friction_budget(previous, external, stribeck)
        self._write_frictions(budget, model.friction_viscous.value)
        return torque
