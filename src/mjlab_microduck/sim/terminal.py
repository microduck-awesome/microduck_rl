"""Isolated forward/sensing buffers for terminal-observation collection.

No integration, reset, actuator computation or model mutation runs here. The
live model is read-only during capture; dynamic inputs are copied and valid
sensor caches may be borrowed read-only while live writers are fenced. This
helper does not install itself into a training task.
"""

from copy import copy
from contextlib import contextmanager

import mujoco_warp as mjwarp
import torch
import warp as wp

from mjlab.sensor.builtin_sensor import BuiltinSensor
from mjlab.sensor.contact_sensor import ContactSensor
from mjlab.sensor.sensor_context import SensorContext
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.sim.sim import Simulation
from mjlab.sim.sim_data import WarpBridge


class TerminalSimulation:
    # Inputs to forward dynamics. Solver/contact workspaces are recomputed, not
    # copied. Warmstart is an input even though qacc itself is an output.
    INPUTS = ('time', 'qpos', 'qvel', 'act', 'qacc_warmstart', 'ctrl',
              'qfrc_applied', 'xfrc_applied', 'eq_active', 'mocap_pos', 'mocap_quat')

    def __init__(self, env):
        live = env.sim
        self._wp_stream = wp.Stream(device=live.wp_device)
        self._torch_stream = torch.cuda.ExternalStream(self._wp_stream.cuda_stream, device=live.device)
        with self.execution(live):
            self._initialize(env)

    def _initialize(self, env):
        live = env.sim
        self._live_forward_graph = live.forward_graph
        self._live_model = live.wp_model
        if any(getattr(live.wp_model.callback, key) for key in ('control', 'sensor')):
            raise ValueError('Terminal forward does not support custom model callbacks')
        sim = object.__new__(Simulation)
        sim.cfg, sim.device, sim.wp_device = live.cfg, live.device, live.wp_device
        sim.num_envs, sim._mj_model, sim._wp_model = live.num_envs, live.mj_model, live.wp_model
        with wp.ScopedDevice(live.wp_device):
            sim._wp_data = mjwarp.make_data(
                live.mj_model, nworld=env.num_envs,
                naconmax=live.wp_data.naconmax, naccdmax=live.wp_data.naccdmax,
                njmax=live.wp_data.njmax, njmax_nnz=live.wp_data.njmax_nnz,
            )
        sim._data_bridge = WarpBridge(sim._wp_data)
        sim._model_bridge = live._model_bridge
        sim.use_cuda_graph = live.use_cuda_graph
        sim.forward_graph = sim.sense_graph = None
        sim._sensor_context = None
        scene = copy(env.scene)
        scene._entities = {}
        scene._sensors = {}
        for name, entity in env.scene.entities.items():
            clone = copy(entity)
            clone._data = copy(entity.data)
            clone._data.data = sim.data
            scene._entities[name] = clone
        for name, sensor in env.scene.sensors.items():
            if type(sensor) not in (BuiltinSensor, ContactSensor, TerrainHeightSensor):
                raise ValueError(f'Unsupported terminal sensor: {type(sensor).__name__}')
            clone = copy(sensor)
            if isinstance(sensor, ContactSensor):
                clone._slots = [copy(slot) for slot in sensor._slots]
            clone.initialize(live.mj_model, live.model, sim.data, live.device)
            clone._cache_valid = False
            clone._cached_data = None
            scene._sensors[name] = clone
        rays = [s for s in scene.sensors.values() if isinstance(s, TerrainHeightSensor)]
        scene._sensor_context = None
        if rays:
            scene._sensor_context = sim._sensor_context = SensorContext(
                mj_model=live.mj_model, model=live.wp_model, data=sim.wp_data,
                camera_sensors=[], raycast_sensors=rays, device=live.device)
        self.sim, self.scene = sim, scene
        self.kinematics_graph = None
        if not sim.use_cuda_graph:
            # Keep the same isolated buffers/stream fencing on older drivers.
            # Simulation.forward/sense already provide the eager fallback.
            return
        with wp.ScopedDevice(live.wp_device):
            with wp.ScopedCapture() as capture:
                mjwarp.forward(sim.wp_model, sim.wp_data)
            sim.forward_graph = capture.graph
            with wp.ScopedCapture() as capture:
                sim.wp_data.sensordata.zero_()
                mjwarp.kinematics(sim.wp_model, sim.wp_data)
                mjwarp.com_pos(sim.wp_model, sim.wp_data)
                mjwarp.com_vel(sim.wp_model, sim.wp_data)
                mjwarp.sensor_vel(sim.wp_model, sim.wp_data)
            self.kinematics_graph = capture.graph
            if rays:
                with wp.ScopedCapture() as capture:
                    sim._sense_kernel()
                sim.sense_graph = capture.graph

    def sync(self, env):
        if env.sim.wp_model is not self._live_model or env.sim.forward_graph is not self._live_forward_graph:
            raise RuntimeError('Live model/graph changed; rebuild terminal forward before capture')
        with wp.ScopedDevice(self.sim.wp_device):
            for name in self.INPUTS:
                source, target = getattr(env.sim.wp_data, name), getattr(self.sim.wp_data, name)
                if source.size:
                    wp.copy(target, source)
        for name, sensor in env.scene.sensors.items():
            clone = self.scene.sensors[name]
            clone._cache_valid = sensor._cache_valid
            if isinstance(sensor, BuiltinSensor):
                # Builtin caches are views of sensordata, and therefore see
                # forward's new values even when the cache is already valid.
                clone._cached_data = clone._data_view if sensor._cache_valid else None
            else:
                # Supported critic terms only read cached heights/contact data.
                # Both live writer streams wait for capture before resetting
                # these arrays, so copying read-only cache tensors is needless.
                clone._cached_data = sensor._cached_data if sensor._cache_valid else None
            if isinstance(sensor, ContactSensor) and not sensor._cache_valid:
                if sensor._air_time_state is not None:
                    for key, value in vars(sensor._air_time_state).items():
                        getattr(clone._air_time_state, key).copy_(value)
                if sensor._history_state is not None:
                    for key, value in sensor._history_state.items():
                        clone._history_state[key].copy_(value)

    def kinematic_forward(self):
        if self.kinematics_graph is not None:
            wp.capture_launch(self.kinematics_graph)
        else:
            with wp.ScopedDevice(self.sim.wp_device):
                self.sim.wp_data.sensordata.zero_()
                mjwarp.kinematics(self.sim.wp_model, self.sim.wp_data)
                mjwarp.com_pos(self.sim.wp_model, self.sim.wp_data)
                mjwarp.com_vel(self.sim.wp_model, self.sim.wp_data)
                mjwarp.sensor_vel(self.sim.wp_model, self.sim.wp_data)

    @contextmanager
    def execution(self, live):
        # All shadow bridges/graphs use one fixed stream. Fence both writers
        # of the live state on entry and both possible reset/DR writers on exit.
        # GPU events provide ordering without a device-wide/host synchronization.
        caller = torch.cuda.current_stream(live.device)
        physics = torch.cuda.ExternalStream(wp.get_stream(live.wp_device).cuda_stream, device=live.device)
        self._torch_stream.wait_stream(caller)
        self._torch_stream.wait_stream(physics)
        try:
            with torch.cuda.stream(self._torch_stream), wp.ScopedStream(self._wp_stream):
                yield caller
        finally:
            caller.wait_stream(self._torch_stream)
            physics.wait_stream(self._torch_stream)

    def memo(self, env, shadow_env):
        """Keep small manager copies from recursively cloning the live scene."""
        result = {id(env): shadow_env, id(env.scene): self.scene, id(env.sim): self.sim}
        result.update({id(entity): self.scene.entities[name] for name, entity in env.scene.entities.items()})
        result.update({id(sensor): self.scene.sensors[name] for name, sensor in env.scene.sensors.items()})
        result[id(env.command_manager.cfg)] = env.command_manager.cfg
        for term in env.command_manager._terms.values():
            result[id(term.cfg)] = term.cfg
            limits = getattr(term, '_joint_delta_limits', None)
            if limits is not None:
                result[id(limits)] = limits
        return result
