"""Whole-robot joint targets and the independent 5 ms sensor producer clock."""

from dataclasses import dataclass
import torch
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg

from mjlab_microduck.actuator.sc0090_v5 import load_v5_model
from .feedback import WholeRobotFeedback


@dataclass(kw_only=True)
class V2JointPositionActionCfg(JointPositionActionCfg):
    archive_path: str
    randomization_stage: str = "nominal"
    command_rate_rad_s: float | None = None
    compile_measurement: bool = False
    fuse_registers: bool = False
    reset_graph_chunk: int = 1
    observer_frontend: str = 'bench'
    spike_rejection: bool = True
    o0_local_history: bool = False

    def build(self, env):
        return V2JointPositionAction(self, env)


class V2JointPositionAction(JointPositionAction):
    def __init__(self, cfg, env):
        if cfg.command_rate_rad_s is not None and (
            type(cfg.command_rate_rad_s) not in (float, int)
            or not 0 < cfg.command_rate_rad_s <= 20
        ):
            raise ValueError("Command slew must be null or finite within (0, 20] rad/s")
        super().__init__(cfg, env)
        if self.action_dim != 14:
            raise ValueError("V2 policy must have exactly 14 joints")
        mouth, _ = self._entity.find_joints("^mouth$")
        self._all_ids = torch.cat(
            (self._target_ids, torch.tensor(mouth, device=self.device))
        )
        self.feedback = WholeRobotFeedback(
            cfg.archive_path,
            self.num_envs,
            device=self.device,
            seed=env.cfg.seed or 42,
            stage=cfg.randomization_stage,
            compile_stream=cfg.compile_measurement,
            fuse_registers=cfg.fuse_registers,
            reset_graph_chunk=cfg.reset_graph_chunk,
            observer_frontend=cfg.observer_frontend,
            spike_rejection=cfg.spike_rejection,
            o0_local_history=cfg.o0_local_history,
        )
        if cfg.observer_frontend=='firmware-ls3':
            limits=self._entity.data.joint_pos_limits[:,self._all_ids]
            self.feedback.pipeline.stream.position_min.copy_(limits[...,0].flatten())
            self.feedback.pipeline.stream.position_max.copy_(limits[...,1].flatten())
        env.unwrapped._v2_joint_feedback = self.feedback
        model = load_v5_model()
        self._feedback_bias, self._feedback_scale = (
            model.feedback_bias.value,
            model.feedback_scale.value,
        )
        self._goal = torch.zeros((self.num_envs, 15), device=self.device)
        self._count = self.feedback.counts(self._goal)
        self.reset()

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device)
        if env_ids is not None:
            ids = ids[env_ids] if isinstance(env_ids, slice) else env_ids
        super().reset(ids)
        pose = self._entity.data.joint_pos[:, self._all_ids].clone()
        self.feedback.reset(ids, pose)
        self._count[ids] = self.feedback.counts(pose)[ids]
        self._goal[ids] = self.feedback.physical_target(self._count)[ids]
        for actuator in self._entity.actuators:
            actuator.vin_tensor[ids, 0] = self.feedback.voltage[ids]
        self._apply_targets()

    def process_actions(self, actions):
        if actions.is_cuda:
            torch._assert_async(torch.isfinite(actions).all(), "Nonfinite V2 policy action")
        elif not torch.isfinite(actions).all():
            raise FloatingPointError("Nonfinite V2 policy action")
        super().process_actions(actions)
        limits = self._entity.data.joint_pos_limits[:, self._all_ids]
        wanted = self._goal.clone()
        wanted[:, :14] = self._processed_actions
        wanted[:, 14:] = getattr(self._env.unwrapped, "_v2_mouth_target", 0.0)
        wanted = wanted.clamp(limits[..., 0], limits[..., 1])
        if self.cfg.command_rate_rad_s is not None:
            rate = self.cfg.command_rate_rad_s
            if rate <= 0:
                raise ValueError("Command slew must be positive or explicitly disabled")
            wanted = self._goal + (wanted - self._goal).clamp(
                -rate * self._env.step_dt, rate * self._env.step_dt
            )
        # A reset pose can start beyond a joint stop; mechanical guards still win over slew.
        wanted = wanted.clamp(limits[..., 0], limits[..., 1])
        self._count.copy_(self.feedback.counts(wanted))
        # Subsequent slew starts at the quantized, count-clamped wire target.
        self._goal.copy_(self.feedback.physical_target(self._count))
        self.feedback.command(self._count)

    def _apply_targets(self):
        target = self.feedback.physical_goal_for_motor(
            self._count, self._feedback_bias, self._feedback_scale
        )
        self._entity.set_joint_position_target(target, joint_ids=self._all_ids)

    def apply_actions(self):
        self.feedback.advance(
            self._entity.data.joint_pos[:, self._all_ids],
            self._entity.data.joint_vel[:, self._all_ids],
        )
        self._apply_targets()
