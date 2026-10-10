"""Quantized SC0090 position action and independent causal sensor clock."""
from dataclasses import dataclass
import math

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from .actions import ActionConfig
from .registers import RegisterPipeline


@dataclass(kw_only=True)
class SC0090ActionCfg(ActionTermCfg):
    contract: ActionConfig
    fit_path: str
    weight_path: str
    S: str
    O: str
    random_phase: bool = False
    randomization_path: str | None = None
    warmup_us: int = 280000

    def build(self, env):
        return SC0090Action(self, env)


class SC0090Action(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.pipeline = RegisterPipeline(cfg.fit_path, cfg.weight_path, self.num_envs,
            cfg.S, cfg.O, device=self.device, seed=env.cfg.seed or 42,
            random_phase=cfg.random_phase,randomization_path=cfg.randomization_path)
        self._raw = torch.zeros((self.num_envs, 1), device=self.device)
        self.clock = torch.full((self.num_envs,), 1000000+cfg.warmup_us,
                               dtype=torch.int64, device=self.device)
        c = cfg.contract
        self.zero_goal = c.count_from_physical(0.)
        self.goal_count = torch.full((self.num_envs,), self.zero_goal, device=self.device)
        self.goal = (self.goal_count-c.zero_count)*c.radians_per_count
        self.fault = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_attempts = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.reset()

    @property
    def action_dim(self):
        return 1

    @property
    def raw_action(self):
        return self._raw

    def reset(self, env_ids=None):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[ids] = True
        self._raw[ids] = 0.
        self.goal_count[ids] = self.zero_goal
        self.goal[ids] = (self.zero_goal-self.cfg.contract.zero_count)*self.cfg.contract.radians_per_count
        self.fault[ids] = False
        # Controlled hold is an equilibrium of the frozen D (audited for 3s).
        # Warm only selected worlds. No ideal state enters actor observations.
        stationary = torch.zeros(self.num_envs, device=self.device)
        remaining = mask.clone()
        for attempt in range(8):
            selected = remaining.nonzero().flatten()
            if selected.numel() == 0:
                break
            self.pipeline.reset(selected)
            self.reset_attempts[selected] += 1
            for elapsed in range(0, self.cfg.warmup_us+1, 5000):
                time = 1000000+elapsed
                if 0 < elapsed < self.cfg.warmup_us and elapsed % 20000 == 0:
                    self.pipeline.stream.command(time, self.goal_count, mask=remaining)
                self.pipeline.advance(time, stationary, stationary, mask=remaining)
            _, ready = self.pipeline.query(1000000+self.cfg.warmup_us)
            remaining &= ~ready
        if remaining.any():
            raise RuntimeError('SC0090 completed-output warmup failed; no fallback observation')
        self.clock[ids] = 1000000+self.cfg.warmup_us
        self._entity.set_joint_position_target(self.goal[:, None])

    def process_actions(self, actions):
        if not torch.isfinite(actions).all():
            self.fault |= ~torch.isfinite(actions).all(-1)
            raise FloatingPointError('Nonfinite SC0090 policy action; command not published')
        c = self.cfg.contract
        self._raw.copy_(actions)
        physical = (actions[:, 0]*c.action_scale_rad).clamp(c.minimum_rad, c.maximum_rad)
        wanted = torch.round((physical-c.calibration_bias)/c.calibration_scale/c.radians_per_count+c.zero_count)
        lower = max(c.minimum_count, math.ceil((c.minimum_rad-c.calibration_bias)/c.calibration_scale/c.radians_per_count+c.zero_count))
        upper = min(c.maximum_count, math.floor((c.maximum_rad-c.calibration_bias)/c.calibration_scale/c.radians_per_count+c.zero_count))
        wanted = wanted.clamp(lower, upper)
        delta = math.floor(c.maximum_rate_rad_s*c.policy_dt_s/(c.calibration_scale*c.radians_per_count))
        self.goal_count.copy_(torch.minimum(self.goal_count+delta, torch.maximum(self.goal_count-delta, wanted)))
        self.goal = (self.goal_count-c.zero_count)*c.radians_per_count
        # Ideal simulated bus: a zero-duration completed/confirmed transaction.
        # Sensor reads retain sampled duration and 5ms production availability.
        self.pipeline.stream.command(self.clock, self.goal_count)

    def apply_actions(self):
        self.pipeline.advance(self.clock, self._entity.data.joint_pos[:, 0],
                              self._entity.data.joint_vel[:, 0])
        self._entity.set_joint_position_target(self.goal[:, None])
        self.clock += 5000

    def estimate(self):
        return self.pipeline.query(self.clock)
