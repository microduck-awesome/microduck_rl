"""SC0090 entity factory and continuous batched mjlab/MuJoCo Warp replay."""

from __future__ import annotations

import numpy as np
import torch

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg

from .actuator import SC0090M6ActuatorCfg
from .physics import load_frozen, make_spec


def make_entity_cfg(parameter_path, settings, mass, *, joint_limits=None):
    parameters = load_frozen(parameter_path)
    settings = tuple(float(v) for v in settings)
    return EntityCfg(
        spec_fn=lambda: make_spec(parameters, settings, mass, joint_limits=joint_limits),
        articulation=EntityArticulationInfoCfg(actuators=(SC0090M6ActuatorCfg(
            target_names_expr=('sc0090',), parameter_path=parameter_path, settings=settings),)),
        init_state=EntityCfg.InitialStateCfg(joint_pos={'sc0090': 0.}, joint_vel={'.*': 0.}),
        collisions=(),
    )


class GpuM6Batch:
    """One continuous physical world per record, with no teacher resets."""

    def __init__(self, parameter_path, settings, masses, *, device='cuda:0'):
        self.device = device
        settings = np.asarray(settings, dtype=float)
        masses = np.asarray(masses, dtype=float)
        n = len(settings)
        if settings.shape != (n, 9) or masses.shape != (n,) or n == 0:
            raise ValueError('One operating point and mass are required per world')
        if not np.isfinite(settings).all() or not np.isfinite(masses).all() or np.any(masses <= 0):
            raise ValueError('Nonfinite or nonpositive operating point')
        if not np.all(settings[:, 0] == settings[0, 0]):
            raise ValueError('All worlds must share the physical timestep')
        self.dt = float(settings[0, 0])
        entity_cfg = make_entity_cfg(parameter_path, settings[0], masses[0])
        self.scene = Scene(SceneCfg(num_envs=n, terrain=None, entities={'bench': entity_cfg}), device)
        model = self.scene.compile()
        self.sim = Simulation(num_envs=n, model=model, device=device,
            cfg=SimulationCfg(nconmax=1, njmax=1, mujoco=MujocoCfg(
                timestep=self.dt, integrator='euler', gravity=(0., 0., -9.80665))))
        self.scene.initialize(self.sim.mj_model, self.sim.model, self.sim.data)
        self.entity = self.scene['bench']
        self.actuator = self.entity.actuators[0]
        self.sim.expand_model_fields(('body_mass', 'body_ipos', 'body_inertia'))
        body_id = model.body('bench/arm').id
        com = settings[:, 5]/9.80665/masses
        inertia = settings[:, 4]-masses*com**2
        if np.any(inertia <= 0):
            raise ValueError('Inconsistent pendulum mass, inertia and first moment')
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
        self.sim.model.body_mass[:, body_id] = tensor(masses)
        self.sim.model.body_ipos[:, body_id, :] = tensor(np.column_stack([com*0, com*0, -com]))
        self.sim.model.body_inertia[:, body_id, :] = tensor(
            np.column_stack([inertia, inertia, np.minimum(inertia, 1e-6)]))
        self.sim.recompute_constants(RecomputeLevel.set_const)
        self.actuator.settings.copy_(tensor(settings).unsqueeze(1))
        self.actuator.default_settings.copy_(self.actuator.settings)

    def reset(self, initial_state):
        state = torch.as_tensor(initial_state, dtype=torch.float32, device=self.device)
        if state.shape != (self.actuator.settings.shape[0], 2) or not torch.isfinite(state).all():
            raise ValueError('Expected finite physical q/v initial states')
        self.sim.reset()
        self.scene.reset()
        self.entity.write_joint_state_to_sim(state[:, :1], state[:, 1:])
        self.sim.forward()

    def rollout(self, offsets, entries):
        lengths = np.diff(offsets)
        max_length = int(lengths.max())
        n = len(lengths)
        if n != self.actuator.settings.shape[0] or entries.shape != (int(offsets[-1]), 4):
            raise ValueError('Replay world count or entry dimensions do not match')
        commands = np.empty((max_length, n, 2), dtype=np.float32)
        for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            commands[:end-start, i] = entries[start:end, [0, 3]]
            commands[end-start:, i] = entries[end-1, [0, 3]]
        commands = torch.as_tensor(commands, device=self.device)
        trace = torch.empty((max_length, n, 3), device=self.device)
        self.reset(entries[offsets[:-1], 1:3])
        for k in range(max_length):
            trace[k, :, 0] = self.entity.data.joint_pos[:, 0]
            trace[k, :, 1] = self.entity.data.joint_vel[:, 0]
            self.actuator.enabled.copy_(commands[k, :, 1:2].bool())
            self.entity.set_joint_position_target(commands[k, :, 0:1])
            self.scene.write_data_to_sim()
            trace[k, :, 2] = self.actuator.control_volts[:, 0]
            self.sim.step()
            self.scene.update(self.dt)
        if self.actuator.fault.any() or not torch.isfinite(trace).all():
            raise FloatingPointError('Latched invalid input or nonfinite GPU M6 trajectory')
        trace = trace.cpu().numpy()
        return np.concatenate([trace[:length, i] for i, length in enumerate(lengths)])
