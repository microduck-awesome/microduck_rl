"""Check CPU friction against MuJoCo's constraint Jacobian, including free DOFs."""
import mujoco
import numpy as np
import pytest

from mjlab_microduck.actuator.mujoco import dof_friction_forces, DofFrictionMujocoController
from mjlab_microduck.actuator.sc0090 import load_sc0090_model


@pytest.mark.parametrize("floating", [False, True])
def test_friction_uses_generalized_dofs_with_interleaved_passive_joint(floating):
    free = '<freejoint/>' if floating else ''
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <worldbody><body pos="0 0 1">{free}<geom type="sphere" size=".03"/>
      <body><joint name="a" frictionloss=".07"/><geom type="sphere" size=".02"/>
      <body pos="0 0 .1"><joint name="passive_b" frictionloss=".11"/>
      <geom type="sphere" size=".02"/>
      <body pos="0 0 .1"><joint name="c" frictionloss=".19"/>
      <geom type="sphere" size=".02"/></body></body></body></body></worldbody>
      <actuator><motor name="a" joint="a"/><motor name="c" joint="c"/></actuator>
      </mujoco>''')
    data = mujoco.MjData(model)
    data.qvel[:] = np.linspace(.1, 1., model.nv)
    data.ctrl[:] = [.13, -.17]
    mujoco.mj_forward(model, data)
    # Independent oracle: project friction rows through the real constraint
    # Jacobian. A joint-index lookup fails this test with a floating base.
    force = data.efc_force.copy()
    force[data.efc_type != mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF] = 0
    expected = np.zeros(model.nv)
    mujoco.mj_mulJacTVec(model, data, expected, force)
    np.testing.assert_allclose(dof_friction_forces(model, data), expected, atol=1e-12)
    assert np.linalg.norm(expected) > 0

    bam = load_sc0090_model()
    controller = DofFrictionMujocoController(bam, ['a', 'c'], model, data)
    mujoco.mj_forward(model, data)
    ids = controller.dof_indexes
    external = (-data.qfrc_bias + data.qfrc_constraint - dof_friction_forces(model, data))[ids]
    expected_loss, expected_damping = bam.compute_frictions(
        data.qfrc_actuator[ids], external, data.qvel[ids])
    passive = int(model.joint('passive_b').dofadr[0])
    before_passive = model.dof_frictionloss[passive]
    controller.update()
    np.testing.assert_allclose(model.dof_frictionloss[ids], expected_loss, atol=1e-12)
    np.testing.assert_allclose(model.dof_damping[ids], expected_damping, atol=1e-12)
    assert model.dof_frictionloss[passive] == before_passive
