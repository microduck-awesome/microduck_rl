"""CUDA replay of the frozen V5 tensor program (no changed arithmetic)."""

from dataclasses import replace
import torch


def _key(value):
    return (value.data_ptr(), value.shape, value.stride(), value.dtype, value.device)


class V5ComputeGraph:
    """Stable inputs; invalidate on model storage/parameter changes.

    Based on backup/local-work-20261015 actuator/compute_graph.py. V5 has no
    lagged battery-sag dependency: measured supply is read directly each tick.
    Friction bridge writes stay outside capture, on their native stream.
    """

    def __init__(self, owner):
        self.owner = owner
        self.graph = None
        self.signature = None
        self.eager_calls = 0
        self.capture_count = 0
        self.replays = 0

    def compute(self, cmd, eager):
        owner = self.owner
        if (not cmd.pos.is_cuda or not torch.is_inference_mode_enabled()
                or torch.is_grad_enabled() or self.eager_calls < 2):
            self.eager_calls += 1
            return eager(cmd)
        data, act = owner._data, owner._bam_model.actuator
        external = [owner.vin_tensor, owner.kp_scale, owner.kd_scale,
                    owner.firmware_kd_scale, owner.friction_scale, owner._dof_ids]
        external += [owner._as_tensor(x) for x in (
            data.qfrc_actuator, data.qfrc_bias, data.qfrc_constraint,
            data.efc.type, data.efc.id, data.efc.force, data.nefc)]
        constants = tuple((n, p.value) for n, p in owner._bam_model.get_parameters().items())
        signature = (tuple(_key(v) for v in external), constants,
                     owner.cfg.kp_fw, owner.cfg.kd_fw, owner._dt,
                     act.radians_per_count, act.deadband_count, act.max_pwm,
                     act.robot_coordinates, owner.cfg.fast_friction_force,
                     tuple((v.shape, v.dtype, v.device) for v in
                           (cmd.position_target, cmd.pos, cmd.vel)))
        if self.graph is None or signature != self.signature:
            self.graph = None
            self.cmd = replace(cmd, position_target=cmd.position_target.clone(),
                               pos=cmd.pos.clone(), vel=cmd.vel.clone())
            graph = torch.cuda.CUDAGraph()
            owner._capture_friction = True
            try:
                with torch.inference_mode(False), torch.no_grad(), torch.cuda.graph(graph):
                    self.torque = eager(self.cmd)
                self.friction = owner._captured_friction
                self.outputs = {n: getattr(act, n) for n in ('vin', 'kp', 'kd', 'duty_cycle')}
            finally:
                owner._capture_friction = False
            self.graph, self.signature = graph, signature
            self.capture_count += 1
        else:
            self.cmd.position_target.copy_(cmd.position_target)
            self.cmd.pos.copy_(cmd.pos)
            self.cmd.vel.copy_(cmd.vel)
        self.graph.replay()
        self.replays += 1
        for name, value in self.outputs.items():
            setattr(act, name, value)
        owner._write_frictions(self.friction, owner._bam_model.friction_viscous.value)
        return self.torque.clone()
