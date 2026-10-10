"""mjlab actuator for the frozen SC0090 v5 M6 torque law.

The voltage-controller target uses nominal servo radians. The physical joint
state remains in external-reference radians; the identified feedback map is
applied once inside ``torque``. MuJoCo owns integration and gravity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch

from mjlab.actuator.actuator import Actuator, ActuatorCfg, ActuatorCmd, TransmissionType

from .physics import load_frozen, torque


@dataclass(kw_only=True)
class SC0090M6ActuatorCfg(ActuatorCfg):
    parameter_path: str | Path
    settings: tuple[float, ...]

    def __post_init__(self):
        super().__post_init__()
        p = load_frozen(self.parameter_path)
        if self.transmission_type != TransmissionType.JOINT:
            raise ValueError('SC0090 is a single-joint transmission')
        if self.delay_max_lag != 0:
            raise ValueError('Frozen v5 M6 has zero identified command delay')
        s = np.asarray(self.settings, dtype=float)
        if s.shape != (9,) or not np.isfinite(s).all() or np.any(s[[0, 3, 4, 6, 8]] <= 0):
            raise ValueError('Expected nine finite SC0090 operating settings')
        if self.armature is not None and self.armature != p['armature']:
            raise ValueError('Do not override the frozen rotor armature')
        if self.frictionloss not in (None, 0.) or self.viscous_damping not in (None, 0.):
            raise ValueError('Frozen torque already includes friction and damping')

    def build(self, entity, target_ids, target_names):
        return SC0090M6Actuator(self, entity, target_ids, target_names)


class SC0090M6Actuator(Actuator[SC0090M6ActuatorCfg]):
    def __init__(self, cfg, entity, target_ids, target_names):
        super().__init__(cfg, entity, target_ids, target_names)
        if len(target_ids) != 1:
            raise ValueError('The SC0090 bench actuator requires exactly one joint')
        self.parameters = load_frozen(cfg.parameter_path)

    @property
    def command_field(self):
        return 'position'

    def edit_spec(self, spec, target_names):
        for name in target_names:
            matches = [a for a in spec.actuators if a.target == name]
            if len(matches) != 1:
                raise ValueError(f'Expected one torque motor for {name}, got {len(matches)}')
            act = matches[0]
            act.set_to_motor()
            act.gear = [1., 0., 0., 0., 0., 0.]
            act.ctrllimited = False
            act.forcelimited = False
            self._mjs_actuators.append(act)
            joint = spec.joint(name)
            joint.armature = self.parameters['armature']
            joint.frictionloss = 0.
            joint.damping = np.zeros((3, 1))

    def initialize(self, mj_model, model, data, device):
        super().initialize(mj_model, model, data, device)
        if not np.isclose(mj_model.opt.timestep, self.cfg.settings[0], rtol=0., atol=1e-12):
            raise ValueError('Torque stopping impulse and physics timestep must agree')
        if mj_model.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER:
            raise ValueError('This frozen M6 integration was qualified with Euler')
        shape = (data.nworld, 1)
        base = torch.tensor(self.cfg.settings, dtype=torch.float32, device=device)
        self.settings = base.expand(*shape, 9).clone()
        self.default_settings = self.settings.clone()
        self.enabled = torch.ones(shape, dtype=torch.bool, device=device)
        self.fault = torch.zeros(shape, dtype=torch.bool, device=device)
        self.control_volts = torch.zeros(shape, device=device)
        self.motor_torque = torch.zeros(shape, device=device)
        self.applied_torque = torch.zeros(shape, device=device)

    def compute(self, cmd: ActuatorCmd):
        valid = torch.isfinite(cmd.position_target) & torch.isfinite(cmd.pos) & torch.isfinite(cmd.vel)
        self.fault |= ~valid
        q = torch.where(valid, cmd.pos, 0.)
        v = torch.where(valid, cmd.vel, 0.)
        goal = torch.where(valid, cmd.position_target,
                           self.parameters['feedback_bias'] + self.parameters['feedback_scale']*q)
        applied, volts, motor = torque(self.parameters, q, v, goal,
            self.enabled & ~self.fault, self.settings, self.cfg.settings[0], xp=torch)
        self.fault |= ~(torch.isfinite(applied) & torch.isfinite(volts) & torch.isfinite(motor))
        self.control_volts.copy_(torch.where(self.fault, 0., volts))
        self.motor_torque.copy_(torch.where(self.fault, 0., motor))
        self.applied_torque.copy_(torch.where(self.fault, 0., applied))
        return self.applied_torque

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.settings[ids] = self.default_settings[ids]
        self.enabled[ids] = True
        self.fault[ids] = False
        self.control_volts[ids] = 0.
        self.motor_torque[ids] = 0.
        self.applied_torque[ids] = 0.
