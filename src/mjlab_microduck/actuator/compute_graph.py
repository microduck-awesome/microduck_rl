"""Replay the original SC0090 tensor program, retaining its FP32 operations."""

from dataclasses import replace

import torch


def _tensor_key(value):
    if value is None:
        return None
    return (value.data_ptr(), value.shape, value.stride(), value.dtype, value.device)


class ActuatorComputeGraph:
    """Own stable command/duty inputs; invalidate when external storage changes.

    The delegated SC0090 method contains no random sampling. Capturing records
    its existing operators and operation order, without compiler fusion or
    changes to precision. Friction writes remain on the original bridge stream
    outside the graph. Partial reset can replace duty_cycle; copy its current
    value on every call, preserving the lagged shared-battery calculation.
    """

    def __init__(self, actuator):
        self.actuator = actuator
        self.graph = None
        self.key = None
        self.eager_calls = 0
        self.capture_count = 0

    def _key(self, cmd):
        owner, act = self.actuator, self.actuator._bam_model.actuator
        data, efc = owner._data, owner._data.efc
        external = [owner.vin_tensor, owner.vin_drop_resistance, owner.kp_scale,
                    owner.kd_scale, owner.firmware_kd_scale, owner.friction_scale,
                    owner._dof_ids]
        external += [owner._as_tensor(x) for x in (
            data.qfrc_actuator, data.qfrc_bias, data.qfrc_constraint,
            efc.type, efc.id, efc.force, data.nefc)]
        # DR mutates tensor contents, which replay reads directly. Scalar
        # parameters/flags are baked into kernel arguments and need recapture.
        constants = tuple((name, p.value) for name, p in owner._bam_model.get_parameters().items())
        constants += tuple((name, getattr(owner._bam_model, name)) for name in
                           ("stribeck", "load_dependent", "directional", "quadratic"))
        constants += tuple((name, getattr(act, name)) for name in
                           ("radians_per_count", "deadband_count", "max_pwm", "max_speed_rad_s"))
        return (tuple(_tensor_key(t) for t in external), constants,
                owner._base_kp, owner._base_kd, owner._dt, owner.cfg.vin_min,
                owner.cfg.fast_friction_force,
                (act.duty_cycle.shape, act.duty_cycle.dtype, act.duty_cycle.device),
                tuple((t.shape, t.dtype, t.device) for t in (cmd.position_target, cmd.pos, cmd.vel)))

    def compute(self, cmd, eager):
        owner, act = self.actuator, self.actuator._bam_model.actuator
        duty = getattr(act, "duty_cycle", None)
        # Execute initialization as real steps; no invisible warmup updates.
        # Full reset restores None and must follow the original no-sag branch.
        # Static inputs belong to training's inference context. Other callers
        # retain eager instead of mutating those buffers outside that context.
        if (not cmd.pos.is_cuda or not torch.is_inference_mode_enabled() or torch.is_grad_enabled()
                or duty is None or self.eager_calls < 2):
            result = eager(cmd)
            self.eager_calls += 1
            return result
        key = self._key(cmd)
        if self.graph is None or self.key != key:
            self.graph = None
            self.command = replace(cmd, position_target=cmd.position_target.clone(),
                                   pos=cmd.pos.clone(), vel=cmd.vel.clone())
            self.duty = duty.clone()
            graph = torch.cuda.CUDAGraph()
            owner._capture_friction = True
            try:
                # Allocate PyTorch's graph/RNG bookkeeping as ordinary tensors,
                # so this capture cannot poison later captures by other users
                # of the default CUDA generator (e.g. an optimizer).
                with torch.inference_mode(False), torch.no_grad(), torch.cuda.graph(graph):
                    act.duty_cycle = self.duty
                    self.motor = eager(self.command)
                self.friction = owner._captured_friction
                self.outputs = {name: getattr(act, name) for name in ("vin", "kp", "kd", "duty_cycle")}
            finally:
                owner._capture_friction = False
            self.graph, self.key = graph, key
            self.capture_count += 1
        else:
            self.command.position_target.copy_(cmd.position_target)
            self.command.pos.copy_(cmd.pos)
            self.command.vel.copy_(cmd.vel)
            self.duty.copy_(duty)
        self.graph.replay()
        # Reset and eager fallback may have replaced these Python references.
        for name, value in self.outputs.items():
            setattr(act, name, value)
        owner._write_frictions(self.friction, owner._bam_model.friction_viscous.value)
        # Preserve compute's fresh-result ownership for callers retaining it.
        return self.motor.clone()
