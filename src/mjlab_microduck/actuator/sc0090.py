"""Shared SC0090 M6 model and calibrated joint coordinates for GPU/CPU sim."""

import hashlib
import json
import math
from pathlib import Path

from bam.model import load_model_from_dict
from servo_bam.two_load.feedback import FeedbackSC0090

SC0090_MODEL_PATH = Path(__file__).with_name("sc0090_m6.json")
SC0090_MODEL_SHA256 = "6efbe8f2c1a1f7dbb6518f8e43693d1c6e3694eee4b3f2dce009e3bba6b97466"
SC0090_VIN = 12.0
SC0090_KP = 20.0
SC0090_KD = 30.0
SC0090_MAX_SPEED_RPM = 80.0
SC0090_MAX_SPEED_RAD_S = SC0090_MAX_SPEED_RPM * 2 * math.pi / 60
# Revision 2 corrects GPU M6 quadratic gating and CPU DOF-friction indexing.
# This changes implementation, not the archived identified model parameters.
SC0090_DYNAMICS_REVISION = 2


class RobotSC0090Actuator(FeedbackSC0090):
    """Targets are calibrated robot radians, rather than raw bench commands.

    Apply the same calibration to target and feedback. This preserves the
    fitted gain/scale while keeping every robot joint's own zero: the bench
    bias must not move HOME. The fitted q_offset is used only by BAM's bench
    dynamics, never by the robot MJCF.
    """

    max_speed_rad_s = SC0090_MAX_SPEED_RAD_S

    def compute_control(self, q_target, q, dq, dt):
        target = self.model.feedback_bias.value + self.model.feedback_scale.value * q_target
        super().compute_control(target, q, dq, dt)
        # At the user-specified output speed ceiling, cap voltage at back-EMF
        # to stop accelerating in that direction. Preserve braking, passive
        # dynamics and the fitted torque curve below the ceiling.
        bemf_duty = self.backend.clamp(
            self.model.kt.value * dq / self.vin, -self.max_pwm, self.max_pwm
        )
        high = self.max_pwm + (dq >= self.max_speed_rad_s) * (bemf_duty - self.max_pwm)
        low = -self.max_pwm + (dq <= -self.max_speed_rad_s) * (bemf_duty + self.max_pwm)
        self.duty_cycle = self.backend.clamp(self.duty_cycle, low, high)
        return self.vin * self.duty_cycle


def use_robot_coordinates(model, max_speed_rpm=SC0090_MAX_SPEED_RPM):
    previous = model.actuator
    actuator = RobotSC0090Actuator()
    actuator.__dict__.update(previous.__dict__)
    actuator.max_speed_rad_s = max_speed_rpm * 2 * math.pi / 60
    model.actuator = actuator
    return model


def load_sc0090_model(kp_fw=SC0090_KP, vin=SC0090_VIN, max_current=None):
    """Load the exact archived fit, restoring its firmware operating point."""
    if max_current is not None:
        raise ValueError("The SC0090 fit models PWM saturation, not a firmware current limiter")
    blob = SC0090_MODEL_PATH.read_bytes()
    if hashlib.sha256(blob).hexdigest() != SC0090_MODEL_SHA256:
        raise ValueError("SC0090 M6 source differs from the selected v3 artifact")
    artifact = json.loads(blob)
    model = load_model_from_dict(artifact)
    model.actuator.load_control_settings(artifact["operating_point"])
    model.actuator.kp = kp_fw
    model.actuator.vin = vin
    model.actuator.max_current = max_current
    return use_robot_coordinates(model)
