"""V2 reset correctness and instance-local GPU optimizations.

Ported from backup/local-work-20261015 SC0090EpisodeResetV1 and recompute.
This changes no calibrated motor coefficients, time steps, or solver options.
"""

import re
import torch
from dataclasses import dataclass
from mjlab.managers.event_manager import requires_model_fields
from mjlab.terrains.terrain_entity import TerrainEntity, TerrainEntityCfg


@dataclass
class V2TerrainCfg(TerrainEntityCfg):
    """Omit the viewer sites that V2 already deletes before compilation."""
    skip_environment_origin_markers: bool = True


def install_origin_marker_filter():
    # mjlab 1.3.0 constructs TerrainEntity directly, bypassing cfg.build().
    # Scope the hook to an explicit V2 config. Other tasks retain native sites.
    if getattr(TerrainEntity._add_env_origin_sites, '_v2_scoped_filter', False):
        return
    native = TerrainEntity._add_env_origin_sites
    def add(self):
        if (isinstance(self.cfg, V2TerrainCfg)
                and self.cfg.skip_environment_origin_markers):
            return
        native(self)
    add._v2_scoped_filter = True
    TerrainEntity._add_env_origin_sites = add


def configure_origin_markers(cfg):
    if cfg.scene.terrain is not None:
        install_origin_marker_filter()
        if not isinstance(cfg.scene.terrain, V2TerrainCfg):
            cfg.scene.terrain = V2TerrainCfg(**vars(cfg.scene.terrain))
    return cfg


def remove_origin_markers(spec):
    # N viewer-only origin sites otherwise cause N*N transforms every substep.
    for site in list(spec.sites):
        if re.fullmatch(r"env_origin_\d+", site.name) and site.group == 4:
            spec.delete(site)


class V2EpisodeReset:
    def __init__(self, env):
        from mjlab_microduck.actuator.sc0090_v5 import SC0090V5Actuator
        actuators = env.scene['robot'].actuators
        if not actuators or any(not isinstance(a, SC0090V5Actuator) for a in actuators):
            raise ValueError('V2 reset requires frozen V5 BAM actuators')
        self.env = env
        self.ids = torch.unique(torch.cat([a._dof_ids for a in actuators]))
        self.defaults = {}
        for name in ('dof_frictionloss', 'dof_damping'):
            field = getattr(env.sim.model, name)
            if field.ndim != 2 or field.shape[0] != env.num_envs:
                raise ValueError('V2 BAM model fields must be per-world')
            self.defaults[name] = env.sim.get_default_field(name)[self.ids].clone()
        self.native_reset = env._reset_idx
        self.native_commands = env.command_manager.reset
        self.native_write = env.scene.write_data_to_sim
        self.inside_reset = False
        self.pending_write = False

    def install(self):
        self.env._reset_idx = self.reset
        self.env.command_manager.reset = self.reset_commands
        self.env.event_manager.reset = self.reset_events
        self.env.scene.write_data_to_sim = self.write

    def reset(self, env_ids=None):
        if self.inside_reset or self.pending_write:
            raise RuntimeError('Unsupported V2 reset/write/forward order')
        env = self.env
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
        self.inside_reset = True
        try:
            self.native_reset(ids)
            for name, default in self.defaults.items():
                getattr(env.sim.model, name)[ids[:, None], self.ids] = default
            for name in ('qfrc_actuator', 'qfrc_bias', 'qfrc_constraint'):
                getattr(env.sim.data, name)[ids] = 0.
            self.pending_write = True
        finally:
            self.inside_reset = False

    def reset_commands(self, env_ids=None):
        if self.inside_reset:
            self.env.episode_length_buf[slice(None) if env_ids is None else env_ids] = 0
        return self.native_commands(env_ids)

    def reset_events(self, env_ids=None):
        manager = self.env.event_manager
        for configs in manager._mode_class_term_cfgs.values():
            for cfg in configs:
                cfg.func.reset(env_ids=env_ids)
        ids = slice(None) if env_ids is None else env_ids
        count = self.env.num_envs if env_ids is None else len(env_ids)
        for cfg, timer in zip(manager._mode_term_cfgs.get('interval', []),
                              manager._interval_term_time_left, strict=True):
            if not cfg.is_global_time:
                lo, hi = cfg.interval_range_s
                timer[ids] = torch.rand(count, device=self.env.device) * (hi - lo) + lo
        return {}

    def write(self):
        if self.pending_write:
            self.pending_write = False
            return
        return self.native_write()


@requires_model_fields(
    'dof_frictionloss', 'dof_damping', 'actuator_acc0', 'actuator_biasprm',
    'cam_pos0', 'cam_poscom0', 'cam_mat0', 'light_pos0', 'light_poscom0', 'light_dir0',
)
def install_runtime(env, env_ids=None):
    from mjlab_microduck.sim.recompute import prepare_recompute_storage, install_recompute_graph_cache
    prepare_recompute_storage(env.sim)
    install_recompute_graph_cache(env.sim)
    for actuator in env.scene['robot'].actuators:
        actuator._graph_compute_supported = env.sim.use_cuda_graph
    if hasattr(env, '_v2_episode_reset'):
        raise RuntimeError('V2 reset adapter installed twice')
    env._v2_episode_reset = V2EpisodeReset(env)
    env._v2_episode_reset.install()


def configure_runtime(cfg):
    from mjlab.managers import EventTermCfg
    configure_origin_markers(cfg)
    # Preserve an existing terrain modifier when installing this optimization.
    original = cfg.scene.spec_fn
    if original is None:
        cfg.scene.spec_fn = remove_origin_markers
    else:
        def modify(spec):
            original(spec)
            remove_origin_markers(spec)
        cfg.scene.spec_fn = modify
    cfg.events['v2_runtime'] = EventTermCfg(func=install_runtime, mode='startup')
    return cfg
