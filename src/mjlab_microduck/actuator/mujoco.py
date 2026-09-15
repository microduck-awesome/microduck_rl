"""CPU BAM correction for MuJoCo's DOF-friction constraint indices."""

import mujoco
import numpy as np
from bam.mujoco import MujocoController


def dof_friction_forces(model, data):
    """Generalized friction force, indexed by DOF (not by joint)."""
    selected = data.efc_type == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
    ids = data.efc_id[selected]
    if np.any((ids < 0) | (ids >= model.nv)):
        raise ValueError("Invalid active DOF-friction constraint index")
    result = np.zeros(model.nv, dtype=data.qfrc_constraint.dtype)
    np.add.at(result, ids, data.efc_force[selected])
    return result


class DofFrictionMujocoController(MujocoController):
    """Keep pinned BAM control/supply arithmetic; correct its friction lookup.

    BAM 3505fba's CPU update matches efc_id to joint_indexes. A free or ball
    joint makes those differ from DOF indices. The GPU implementation already
    uses DOF indices. Recompute only the pure friction budget after the upstream
    update, before any physics step; do not advance the control law twice.
    """

    def update(self):
        super().update()
        model, data = self.mujoco_model, self.mujoco_data
        ids = self.dof_indexes
        external = (-data.qfrc_bias + data.qfrc_constraint
                    - dof_friction_forces(model, data))[ids]
        loss, damping = self.model.compute_frictions(
            data.qfrc_actuator[ids], external, data.qvel[ids])
        model.dof_frictionloss[ids] = loss
        model.dof_damping[ids] = damping
