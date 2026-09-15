"""SC0090 M6 with resettable friction DR and optional backlash feedback."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import torch
from servo_bam.rl_actuator import SC0090BamActuator, SC0090BamActuatorCfg
from .sc0090 import use_robot_coordinates, SC0090_MAX_SPEED_RPM
from mjlab.actuator.actuator import ActuatorCmd


class FrictionDRBamActuator(SC0090BamActuator):
    """SC0090 actuator; upstream BAM applies friction_scale exactly once."""

    def __init__(self, cfg, entity, target_ids, target_names):
        super().__init__(cfg, entity, target_ids, target_names)
        use_robot_coordinates(self._bam_model, cfg.max_speed_rpm)

    def initialize(self, mj_model, model, data, device) -> None:
        super().initialize(mj_model, model, data, device)
        # kp_scale is (num_envs, 1); mirror it for a per-env friction multiplier.
        self.friction_scale = torch.ones_like(self.kp_scale)
        self.default_friction_scale = self.friction_scale.clone()
        # The selected fit has zero delay: avoid copying a (2,N,14) history
        # buffer every physics substep when its output is exactly the input.
        if self._bam_model.command_delay.value == 0.0:
            self._identified_delay = None
        self._damping_scalar = None
        self._damping_value = None
        self._compute_graph = None
        self._graph_compute_supported = False  # Set by the V2 startup event.
        self._capture_friction = False

    def compute(self, cmd):
        if self.cfg.graph_compute and self._graph_compute_supported:
            from .compute_graph import ActuatorComputeGraph
            if self._compute_graph is None:
                self._compute_graph = ActuatorComputeGraph(self)
            return self._compute_graph.compute(cmd, super().compute)
        return super().compute(cmd)

    def _write_frictions(self, frictionloss: torch.Tensor, damping: float) -> None:
        if getattr(self, "_capture_friction", False):
            self._captured_friction = frictionloss
            return
        if self.cfg.fast_friction_writes:
            # Assigning a Python scalar through CUDA advanced indexing makes
            # PyTorch copy a host scalar and synchronize the stream each time.
            # Keep the value on the device. Still write every substep, so DR,
            # partial resets and replacement model arrays retain BAM semantics.
            if (self._damping_scalar is None or self._damping_value != damping
                    or self._damping_scalar.device != frictionloss.device
                    or self._damping_scalar.dtype != frictionloss.dtype):
                self._damping_scalar = torch.tensor(
                    damping, device=frictionloss.device, dtype=frictionloss.dtype)
                self._damping_value = damping
            damping = self._damping_scalar
        super()._write_frictions(frictionloss, damping)

    def _dof_friction_force(self, nv: int) -> torch.Tensor:
        if self.cfg.fast_friction_force:
            efc = self._data.efc
            types, ids, forces, counts = (
                self._as_tensor(x) for x in (efc.type, efc.id, efc.force, self._data.nefc))
            if (forces.is_cuda and forces.dtype == torch.float32
                    and types.dtype == ids.dtype == counts.dtype == torch.int32):
                from .friction_force import friction_force
                return friction_force(types, ids, forces, counts, nv)
        return super()._dof_friction_force(nv)

    def set_friction_scale(self, env_ids, friction_scale: torch.Tensor) -> None:
        self.friction_scale[env_ids] = friction_scale

    def reset_friction_scale(self, env_ids) -> None:
        self.friction_scale[env_ids] = self.default_friction_scale[env_ids]


@dataclass(kw_only=True)
class FrictionDRBamActuatorCfg(SC0090BamActuatorCfg):
    """Drop-in for BamActuatorCfg that builds a friction-DR-capable actuator."""

    max_speed_rpm: float = SC0090_MAX_SPEED_RPM
    fast_friction_writes: bool = False
    fast_friction_force: bool = False
    graph_compute: bool = False

    def __post_init__(self):
        super().__post_init__()
        if not math.isfinite(self.max_speed_rpm) or self.max_speed_rpm <= 0:
            raise ValueError("SC0090 maximum output speed must be positive and finite")

    def build(self, entity, target_ids, target_names) -> FrictionDRBamActuator:
        return FrictionDRBamActuator(self, entity, target_ids, target_names)


class BacklashEncoderBamActuator(FrictionDRBamActuator):
    """FrictionDRBamActuator whose firmware PD reads the encoder THROUGH backlash.

    Backlash models (robot_groundcontact_backlash.xml) put an unactuated
    ``passive_<joint>_backlash`` hinge in series with each servo joint: the
    servo joint is the motor output, the backlash joint is the play between it
    and the link, and the link angle is their sum.

    On the real servo the magnetic encoder sits on the OUTPUT side of that
    play, so the firmware position loop closes on main+backlash — while the
    servo winds through the dead zone the measured position (and hence the PD
    error) doesn't change. This subclass reproduces that: ``cmd.pos`` fed to
    BAM's voltage control law becomes qpos[main] + qpos[backlash].

    ``cmd.vel`` is left motor-side on purpose: in BAM it drives back-EMF and
    friction, which are rotor physics, not an encoder-derived firmware signal.

    Degrades to a plain FrictionDRBamActuator on models without backlash
    joints (per-joint mask), so it is safe to use on any microduck model.
    """

    def initialize(self, mj_model, model, data, device) -> None:
        super().initialize(mj_model, model, data, device)
        name_to_local = {n: i for i, n in enumerate(self.entity.joint_names)}
        ids, mask = [], []
        for name in self._target_names:
            bl_id = name_to_local.get(f"passive_{name}_backlash")
            ids.append(0 if bl_id is None else bl_id)
            mask.append(0.0 if bl_id is None else 1.0)
        self._backlash_joint_ids = torch.tensor(ids, dtype=torch.long, device=device)
        self._backlash_mask = torch.tensor(mask, dtype=torch.float32, device=device)
        n_backlash = int(self._backlash_mask.sum().item())
        print(
            f"[BacklashEncoderBamActuator] encoder-through-backlash feedback on "
            f"{n_backlash}/{len(mask)} joints"
        )

    def get_command(self, data) -> ActuatorCmd:
        cmd = super().get_command(data)
        pos = cmd.pos + data.joint_pos[:, self._backlash_joint_ids] * self._backlash_mask
        return dataclasses.replace(cmd, pos=pos)


@dataclass(kw_only=True)
class BacklashEncoderBamActuatorCfg(FrictionDRBamActuatorCfg):
    """FrictionDRBamActuatorCfg whose PD feedback reads through backlash joints."""

    def build(self, entity, target_ids, target_names) -> BacklashEncoderBamActuator:
        return BacklashEncoderBamActuator(self, entity, target_ids, target_names)
