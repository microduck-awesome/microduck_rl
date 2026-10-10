"""Frozen v5 M6 torque law, with body integration owned by MuJoCo.

Equations follow servo_bam/native_rollout.c (pinned BAM reference). Measurement
play belongs to S and is deliberately absent here. NumPy and Torch call the
same torque function; neither function integrates or resets physical state.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

M6_SHA256 = "7c35a732575bfad135d5bf7c7eeb44f268ffd96862064f04e9e7a5105bbf6fb4"


def load_frozen(path: str | Path) -> dict:
    blob = Path(path).read_bytes()
    if hashlib.sha256(blob).hexdigest() != M6_SHA256:
        raise ValueError("Expected yesterday's frozen v5 sc0090-v5-m6-seed1 parameters")
    p = json.loads(blob)
    if p["model"] != "m6" or p["actuator"] != "sc0090_feedback_v1":
        raise ValueError("Unsupported model identity")
    if p["command_delay"] != 0:
        raise ValueError("This frozen integration requires the identified zero command delay")
    return p


def torque(p, q, v, goal, enabled, settings, dt, xp=np):
    """Return applied motor-minus-friction torque, control volts, motor torque.

    settings = [dt, P, D, voltage, body J, gravity amplitude, rad/count,
                deadband counts, PWM limit]. Last dimension indexes settings.
    Gravity is included only when computing friction/stopping impulse; MuJoCo
    supplies gravity to the body exactly once. No lagged motor-load shortcut.
    """
    s = settings
    clamp = xp.clip if xp is np else xp.clamp
    count_error = (goal - (p["feedback_bias"] + p["feedback_scale"] * q)) / s[..., 6]
    count_error = count_error - clamp(count_error, -s[..., 7], s[..., 7])
    duty = p["pwm_per_p_count"] * (
        s[..., 1] * count_error - s[..., 2] * p["derivative_seconds"] * v / s[..., 6]
    )
    duty = clamp(duty, -s[..., 8], s[..., 8])
    control = s[..., 3] * duty
    motor = (p["kt"] * control / p["R"] - p["kt"] ** 2 * v / p["R"]) * enabled
    external = -s[..., 5] * xp.sin(q + p["q_offset"])
    stribeck = xp.exp(-(xp.abs(v / p["dtheta_stribeck"]) ** p["alpha"]))
    friction = p["friction_base"]
    friction = friction + xp.abs(external * p["load_friction_external"] - motor * p["load_friction_motor"])
    friction = friction + stribeck * p["friction_stribeck"]
    friction = friction + xp.abs(
        external * p["load_friction_external_stribeck"] - motor * p["load_friction_motor_stribeck"]
    ) * stribeck
    opposite = xp.sign(external) != xp.sign(motor)
    friction = friction + xp.where(
        opposite & (xp.abs(external) < xp.abs(motor)),
        stribeck * p["load_friction_external_quad"] * external**2, 0.0,
    )
    friction = friction + xp.where(
        opposite & (xp.abs(external) > xp.abs(motor)),
        stribeck * p["load_friction_motor_quad"] * motor**2, 0.0,
    )
    stop = ((s[..., 4] + p["armature"]) / dt) * v + (motor + external)
    budget = friction + p["friction_viscous"] * xp.abs(v)
    applied = motor - xp.sign(stop) * xp.minimum(xp.abs(stop), budget)
    return applied, control, motor


def make_spec(p: dict, settings, mass: float, *, joint_limits=None) -> mujoco.MjSpec:
    """Build a fixed-base pendulum with the measured J and first moment.

    q=0 uses the v5 external-reference coordinates. The frozen q_offset rotates
    the gravity frame, without applying an extra coordinate map to observations.
    """
    s = np.asarray(settings, dtype=float)
    moment = s[5] / 9.80665
    com = moment / mass
    inertia_com = s[4] - mass * com**2
    if not (mass > 0 and inertia_com > 0 and np.isfinite(s).all()):
        raise ValueError("Measured pendulum mass/J/first moment are inconsistent")
    qw, qy = math.cos(p["q_offset"] / 2), math.sin(p["q_offset"] / 2)
    limits = 'limited="false"' if joint_limits is None else f'limited="true" range="{joint_limits[0]} {joint_limits[1]}"'
    xml = f'''<mujoco model="sc0090-v5-testbench">
      <compiler angle="radian"/>
      <option timestep="{s[0]}" gravity="0 0 -9.80665" integrator="Euler"/>
      <worldbody><body name="gravity_frame" pos="0 0 0.3" quat="{qw} 0 {qy} 0">
        <body name="arm">
          <joint name="sc0090" type="hinge" axis="0 1 0" {limits}
                 armature="{p['armature']}" damping="0" frictionloss="0"/>
          <inertial pos="0 0 {-com}" mass="{mass}"
                    diaginertia="{inertia_com} {inertia_com} {min(inertia_com, 1e-6)}"/>
          <geom type="capsule" fromto="0 0 0 0 0 {-com * 1.5}" size="0.004"
                contype="0" conaffinity="0" rgba="0.2 0.6 0.8 1"/>
        </body>
      </body></worldbody>
      <actuator><motor name="sc0090" joint="sc0090" gear="1"/></actuator>
    </mujoco>'''
    return mujoco.MjSpec.from_string(xml)


class CpuM6:
    """Continuous MuJoCo body plus frozen torque producer for runner/dry-run."""

    def __init__(self, p, settings, mass):
        self.p = p
        self.settings = np.asarray(settings, dtype=float).copy()
        self.model = make_spec(p, self.settings, mass).compile()
        self.data = mujoco.MjData(self.model)

    def reset(self, q=0.0, v=0.0):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0], self.data.qvel[0] = q, v
        mujoco.mj_forward(self.model, self.data)

    def step(self, goal, enabled=True):
        q, v = self.data.qpos[0], self.data.qvel[0]
        applied, control, _ = torque(self.p, q, v, goal, enabled, self.settings, self.settings[0])
        self.data.ctrl[0] = applied
        mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("Nonfinite SC0090 simulation")
        return q, v, float(control)
