"""Execution optimizations must preserve BAM writes and reset-time physics."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from bam.mjlab import BamActuator
from mjlab_microduck.actuator.friction_dr_bam import FrictionDRBamActuator
from mjlab_microduck.sim.recompute import install_recompute_graph_cache


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_friction_writes_match_bam_after_external_write_and_reallocation(device, dtype):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    def make(fast):
        act = object.__new__(FrictionDRBamActuator)
        act.cfg = NS(fast_friction_writes=fast)
        act._num_envs = 4
        # Include interleaved passive DOFs; unselected columns must survive.
        act._dof_ids = torch.tensor([1, 3, 6], device=device)
        act._friction_fields_checked = False
        act._damping_scalar = act._damping_value = None
        act._mjwarp_model = NS(
            dof_frictionloss=torch.full((4, 8), .7, device=device, dtype=dtype),
            dof_damping=torch.full((4, 8), .9, device=device, dtype=dtype))
        return act

    reference, optimized = make(False), make(True)
    budget = torch.arange(12, device=device, dtype=dtype).reshape(4, 3) / 10
    for damping in (0., .12, .12, .03):
        for act in (reference, optimized):
            # Replacement storage must be used even after the cache is warm.
            for name in ("dof_frictionloss", "dof_damping"):
                value = getattr(act._mjwarp_model, name).clone()
                value[1, :] = .55
                setattr(act._mjwarp_model, name, value)
        BamActuator._write_frictions(reference, budget, damping)
        optimized._write_frictions(budget, damping)
        for name in ("dof_frictionloss", "dof_damping"):
            torch.testing.assert_close(getattr(optimized._mjwarp_model, name),
                                       getattr(reference._mjwarp_model, name), rtol=0, atol=0)


def test_non_graph_simulation_keeps_original_methods():
    sim = NS(use_cuda_graph=False, recompute_constants=lambda level: level,
             create_graph=lambda: None)
    original = sim.recompute_constants
    assert install_recompute_graph_cache(sim) is None
    assert sim.recompute_constants is original


@pytest.mark.parametrize("level_name", ["set_const_fixed", "set_const_0", "set_const"])
def test_recompute_graph_matches_eager_after_parameter_changes_and_expansion(level_name):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    import mujoco
    import warp as wp
    from mjlab.managers.event_manager import RecomputeLevel
    from mjlab.sim import Simulation, SimulationCfg

    model = mujoco.MjModel.from_xml_string('''
    <mujoco><worldbody><body pos="0 0 1"><joint name="a" type="hinge"/>
    <geom type="capsule" size=".05" fromto="0 0 0 0 0 .3" mass="1"/>
    <body pos="0 0 .3"><joint name="b" axis="0 1 0"/>
    <geom type="sphere" size=".08" mass=".5"/></body></body></worldbody>
    <actuator><motor joint="a"/><motor joint="b"/></actuator></mujoco>''')
    sim = Simulation(8, SimulationCfg(nconmax=16, njmax=16), model, "cuda:0")
    if not sim.use_cuda_graph:
        pytest.skip("CUDA graph driver/mempool support unavailable")
    sim.expand_model_fields(("body_mass", "body_inertia", "body_ipos", "qpos0",
                             "dof_armature", "body_subtreemass", "body_invweight0",
                             "dof_invweight0"))
    eager = sim.recompute_constants
    cache = install_recompute_graph_cache(sim)
    assert install_recompute_graph_cache(sim) is cache
    level = getattr(RecomputeLevel, level_name)

    def snapshot():
        # These include reset-derived fields AND physics scratch/state arrays.
        fields = {"model": ("body_subtreemass", "body_invweight0", "dof_invweight0",
                             "body_mass", "body_inertia", "qpos0"),
                  "data": ("qpos", "qvel", "qM", "qLD", "xpos", "xquat", "xipos")}
        return {group + "." + name: getattr(getattr(sim, group), name).clone()
                for group, names in fields.items() for name in names}

    for i in range(3):
        # Repeated calls must read changed per-world values, without replaying
        # stale capture-time values or leaking DR into other worlds.
        sim.model.body_mass[1:4, 1:] = 1. + i * .25
        sim.model.body_inertia[1:4, 1:] *= 1.1
        sim.model.qpos0[2:5] = .1 * (i + 1)
        sim.data.qpos[:] = torch.linspace(-.3, .4, 16, device="cuda:0").reshape(8, 2)
        qpos = sim.data.qpos.clone()
        eager(level)
        expected = snapshot()
        # Destroy derived outputs so a missing/no-op replay cannot pass.
        if level_name != "set_const_0":
            sim.model.body_subtreemass[:] = 0.
        if level_name != "set_const_fixed":
            sim.model.body_invweight0[:] = 0.
            sim.model.dof_invweight0[:] = 0.
        sim.recompute_constants(level)
        actual = snapshot()
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0, msg=name)
        torch.testing.assert_close(sim.data.qpos.clone(), qpos, rtol=0, atol=0)
        assert (level in cache.graphs) == (level_name != "set_const_fixed")
        if i == 1:
            sim.expand_model_fields(("body_gravcomp",))
            assert not cache.graphs
    wp.synchronize()


def test_only_v2_enables_optimizations_and_physics_configuration_is_preserved():
    from mjlab_microduck.tasks.microduck_sc0090_finetune_env_cfg import (
        make_sc0090_walk_v2_env_cfg, make_sc0090_recovery_v2_env_cfg)
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg
    from mjlab_microduck.tasks.microduck_standup_env_cfg import make_microduck_standup_env_cfg
    for base, updated in ((make_microduck_velocity_env_cfg, make_sc0090_walk_v2_env_cfg),
                          (make_microduck_standup_env_cfg, make_sc0090_recovery_v2_env_cfg)):
        a, b = base(), updated()
        assert "cache_reset_constants" not in a.events
        assert b.events["cache_reset_constants"].mode == "startup"
        old = a.scene.entities["robot"].articulation.actuators[0]
        new = b.scene.entities["robot"].articulation.actuators[0]
        assert not old.fast_friction_writes and new.fast_friction_writes
        assert not old.fast_friction_force and new.fast_friction_force
        assert not old.graph_compute and new.graph_compute
        assert old.max_speed_rpm == new.max_speed_rpm == 80
        assert a.sim == b.sim and a.decimation == b.decimation


@pytest.mark.parametrize("task", ["walk", "recovery"])
def test_removing_origin_markers_preserves_physics_sensors_and_origins(task):
    import mujoco
    from mjlab.scene import Scene
    from mjlab_microduck.tasks import mdp
    from mjlab_microduck.tasks.microduck_sc0090_finetune_env_cfg import (
        make_sc0090_walk_v2_env_cfg, make_sc0090_recovery_v2_env_cfg)

    cfg = (make_sc0090_walk_v2_env_cfg() if task == "walk"
           else make_sc0090_recovery_v2_env_cfg())
    cfg.scene.num_envs = 64
    cfg.scene.spec_fn = None
    scene = Scene(cfg.scene, device="cpu")
    reference = scene.compile()
    origins = scene.env_origins.clone()
    mdp.sc0090_remove_origin_markers(scene.spec)
    optimized = scene.compile()
    assert reference.nsite == optimized.nsite + 64
    assert optimized.nsite == 7
    torch.testing.assert_close(scene.env_origins, origins, rtol=0, atol=0)
    for name in ("nq", "nv", "nu", "nbody", "njnt", "ngeom", "nsensor", "nsensordata"):
        assert getattr(reference, name) == getattr(optimized, name)
    # Site IDs can move; physical parameters, collisions and all actual sensor
    # definitions must survive unchanged, with object references mapped by name.
    for name in dir(reference):
        if name.startswith(("body_", "jnt_", "dof_", "geom_", "actuator_")):
            value = getattr(reference, name)
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, getattr(optimized, name), err_msg=name)
    for i in range(reference.nsensor):
        objtype = reference.sensor_objtype[i]
        assert optimized.sensor_objtype[i] == objtype
        assert mujoco.mj_id2name(reference, objtype, reference.sensor_objid[i]) == mujoco.mj_id2name(
            optimized, objtype, optimized.sensor_objid[i])
    sites = {optimized.site(i).name for i in range(optimized.nsite)}
    assert sites == {reference.site(i).name for i in range(reference.nsite)
                     if not reference.site(i).name.startswith("env_origin_")}
    a, b = mujoco.MjData(reference), mujoco.MjData(optimized)
    if reference.nkey:
        mujoco.mj_resetDataKeyframe(reference, a, 0)
        mujoco.mj_resetDataKeyframe(optimized, b, 0)
    rng = np.random.default_rng(42)
    for _ in range(100):
        control = rng.uniform(-.1, .1, reference.nu)
        a.ctrl[:] = b.ctrl[:] = control
        mujoco.mj_step(reference, a)
        mujoco.mj_step(optimized, b)
        for name in ("qpos", "qvel", "qacc", "qfrc_actuator", "qfrc_constraint", "sensordata"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
        for name in sites:
            np.testing.assert_array_equal(a.site(name).xpos, b.site(name).xpos)
            np.testing.assert_array_equal(a.site(name).xmat, b.site(name).xmat)


@pytest.mark.parametrize("task", ["walk", "recovery"])
def test_robot_recompute_preserves_fp32_accuracy_and_motor_torque_is_exact(task):
    """Bound existing FP32 atomic-reduction noise; require exact motor writes."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.event_manager import RecomputeLevel
    from mjlab_microduck.tasks.microduck_sc0090_finetune_env_cfg import (
        make_sc0090_walk_v2_env_cfg, make_sc0090_recovery_v2_env_cfg)

    cfg = (make_sc0090_walk_v2_env_cfg() if task == "walk"
           else make_sc0090_recovery_v2_env_cfg())
    cfg.scene.num_envs = 64
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        cache = getattr(env.sim, "_microduck_recompute_cache", None)
        if cache is None:
            pytest.skip("CUDA graph driver/mempool support unavailable")
        import warp as wp
        assert env.sim.wp_model.actuator_acc0.shape == (64, 14)
        assert env.sim.wp_model.actuator_acc0.strides[0] > 0
        assert env.sim.wp_model.stat.meaninertia.shape == (64,)
        assert env.sim.wp_model.stat.meaninertia.strides[0] > 0
        fields = {"model": ("body_subtreemass", "body_invweight0", "dof_invweight0",
                             "actuator_acc0", "body_mass", "body_inertia", "body_ipos"),
                  "data": ("qpos", "qvel", "qM", "qLD", "xpos", "xquat", "xipos")}

        def snapshot():
            values = {group + "." + name: getattr(getattr(env.sim, group), name).clone()
                      for group, names in fields.items() for name in names}
            values["model.meaninertia"] = wp.to_torch(env.sim.wp_model.stat.meaninertia).clone()
            return values

        with torch.inference_mode():
            for i in range(3):
                ids = (torch.arange(64, device="cuda:0") if i == 0
                       else torch.arange(i - 1, 64, 2, device="cuda:0"))
                env.reset(seed=42 + i, env_ids=ids)
                level = RecomputeLevel.set_const
                cache._original_recompute(level)
                expected = snapshot()
                cache._original_recompute(level)
                repeat = snapshot()
                # Replaying must recompute the values, rather than just leave
                # matching results from the reference execution in memory.
                env.sim.model.body_invweight0[:] = 0.
                env.sim.model.dof_invweight0[:] = 0.
                env.sim.recompute_constants(level)
                actual = snapshot()
                derived = {"model.body_subtreemass", "model.body_invweight0",
                           "model.dof_invweight0", "model.actuator_acc0", "model.meaninertia",
                           "data.qM", "data.qLD"}
                for name in expected:
                    # The reference itself uses nondeterministic FP32 atomic
                    # sums on branching trees. Apply the SAME strict bound to
                    # reference/reference and reference/graph; do not compare
                    # whole trajectories as if MuJoCo Warp were bitwise stable.
                    rtol, atol = (1e-6, 1e-7) if name in derived else (0, 0)
                    torch.testing.assert_close(repeat[name], expected[name], rtol=rtol, atol=atol)
                    torch.testing.assert_close(actual[name], repeat[name], rtol=rtol, atol=atol)

                act = env.scene["robot"].actuators[0]
                cmd = act.get_command(env.scene["robot"].data)
                # Battery sag uses previous duty. Restore it before comparing
                # the same command through reference and optimized writes.
                previous_duty = getattr(act._bam_model.actuator, "duty_cycle", None)
                previous_duty = None if previous_duty is None else previous_duty.clone()
                act.cfg.fast_friction_writes = False
                act.cfg.fast_friction_force = False
                act.cfg.graph_compute = False
                reference_torque = act.compute(cmd).clone()
                expected_friction = env.sim.model.dof_frictionloss.clone()
                expected_damping = env.sim.model.dof_damping.clone()
                act._bam_model.actuator.duty_cycle = previous_duty
                act.cfg.fast_friction_writes = True
                act.cfg.fast_friction_force = True
                act.cfg.graph_compute = True
                actual_torque = act.compute(cmd)
                torch.testing.assert_close(actual_torque, reference_torque, rtol=0, atol=0)
                torch.testing.assert_close(env.sim.model.dof_frictionloss.clone(), expected_friction,
                                           rtol=0, atol=0)
                torch.testing.assert_close(env.sim.model.dof_damping.clone(), expected_damping,
                                           rtol=0, atol=0)
                env.step(torch.zeros((64, 14), device="cuda:0"))
    finally:
        env.close()


@pytest.mark.parametrize("device,dtype", [("cpu", torch.float32), ("cuda:0", torch.float32),
                                         ("cuda:0", torch.float64)])
def test_friction_force_keeps_mask_indices_and_partial_updates(device, dtype):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    import mujoco
    friction_type = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)
    act = object.__new__(FrictionDRBamActuator)
    act.cfg = NS(fast_friction_force=True)
    act._device = device
    worlds, rows, nv = 17, 83, 29
    # Strided arrays exercise the Warp/Torch bridge's layout preservation.
    types = torch.full((worlds, rows * 2), friction_type + 1, dtype=torch.int32, device=device)[:, ::2]
    ids = torch.full_like(types, -999)
    forces = torch.full((worlds, rows * 2), float("nan"), device=device, dtype=dtype)[:, ::2]
    counts = torch.full((worlds * 2,), rows, dtype=torch.int32, device=device)[::2]
    act._data = NS(efc=NS(type=types, id=ids, force=forces), nefc=counts)
    generator = torch.Generator(device=device).manual_seed(74)
    for i in range(3):
        # One valid friction row per DOF, mixed with non-friction and stale
        # rows. Invalid IDs/NaN forces in ignored rows must not be dereferenced.
        active = slice(i % 2, None, 2)
        types[active, :nv] = friction_type
        ids[active, :nv] = torch.arange(nv, device=device, dtype=torch.int32)
        forces[active, :nv] = torch.randn((len(range(i % 2, worlds, 2)), nv),
                                          generator=generator, device=device, dtype=dtype)
        counts[active] = nv - i * 5
        expected = BamActuator._dof_friction_force(act, nv)
        actual = act._dof_friction_force(nv)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Repeated indices retain scatter-add semantics; exact integer-valued
    # forces avoid ambiguity from parallel floating-point summation order.
    counts[:] = rows
    types[:] = friction_type
    ids[:] = 3
    forces[:] = 0.125
    torch.testing.assert_close(act._dof_friction_force(nv),
                               BamActuator._dof_friction_force(act, nv), rtol=0, atol=0)


@pytest.mark.parametrize("task", ["walk", "recovery"])
def test_actuator_graph_matches_original_exactly_through_resets_and_dr(task):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from dataclasses import replace
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab_microduck.tasks.microduck_sc0090_finetune_env_cfg import (
        make_sc0090_walk_v2_env_cfg, make_sc0090_recovery_v2_env_cfg)
    cfg = (make_sc0090_walk_v2_env_cfg() if task == "walk"
           else make_sc0090_recovery_v2_env_cfg())
    cfg.scene.num_envs = 64
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        if not env.sim.use_cuda_graph:
            pytest.skip("CUDA graph support unavailable")
        robot = env.scene["robot"]
        owner = robot.actuators[0]
        act = owner._bam_model.actuator
        with torch.inference_mode():
            env.reset(seed=47)
            seen_graph = None
            for step in range(17):
                if step in (5, 11):
                    owner.reset(torch.arange(step % 2, 64, 2, device="cuda:0"))
                if step == 8:
                    owner.reset()
                    assert act.duty_cycle is None
                if step == 10:
                    # New storage must invalidate captures, while in-place DR
                    # must be observed without rebuilding the graph.
                    owner.kp_scale = owner.kp_scale.clone()
                if step == 12:
                    owner._bam_model.friction_viscous.value *= 1.05
                if step == 16:
                    owner._graph_compute_supported = False
                owner.kp_scale[::3] = 0.75 + step * .01
                owner.kd_scale[1::3] = 1.05 - step * .01
                owner.firmware_kd_scale[2::3] = 0.95 + step * .01
                owner.friction_scale[::2] = 0.8 + step * .015
                owner.vin_tensor[1::2] = 10.8 + step * .1
                cmd = owner.get_command(robot.data)
                speeds = torch.tensor([-10., -act.max_speed_rad_s, -1e-6, 0., 1e-6,
                                       act.max_speed_rad_s, 10.], device="cuda:0")
                cmd = replace(cmd, position_target=cmd.pos + torch.randn_like(cmd.pos) * .1,
                              vel=speeds.repeat(2).expand_as(cmd.vel))
                duty = act.duty_cycle
                duty = None if duty is None else duty.clone()
                owner.cfg.graph_compute = False
                owner.cfg.fast_friction_force = False
                expected_motor = owner.compute(cmd).clone()
                expected = {name: getattr(act, name).clone() for name in ("vin", "kp", "kd", "duty_cycle")}
                expected["friction"] = env.sim.model.dof_frictionloss.clone()
                expected["damping"] = env.sim.model.dof_damping.clone()
                act.duty_cycle = duty
                owner.cfg.fast_friction_force = True
                owner.cfg.graph_compute = True
                actual_motor = owner.compute(cmd)
                torch.testing.assert_close(actual_motor, expected_motor, rtol=0, atol=0)
                for name in ("vin", "kp", "kd", "duty_cycle"):
                    torch.testing.assert_close(getattr(act, name), expected[name], rtol=0, atol=0)
                torch.testing.assert_close(env.sim.model.dof_frictionloss.clone(), expected["friction"], rtol=0, atol=0)
                torch.testing.assert_close(env.sim.model.dof_damping.clone(), expected["damping"], rtol=0, atol=0)
                graph = owner._compute_graph.graph
                if step in (6, 7, 9) and seen_graph is not None:
                    assert graph is seen_graph
                if step in (10, 12):
                    assert graph is not seen_graph
                seen_graph = graph
                env.step(torch.zeros((64, 14), device="cuda:0"))
                torch.testing.assert_close(actual_motor, expected_motor, rtol=0, atol=0)
            assert owner._compute_graph.graph is not None
        # Training captures under inference_mode; a subsequent evaluator may
        # use no_grad instead. Its input copies must not mutate inference-only
        # buffers outside their creation context.
        # mjlab lazily creates its tensor wrappers inside the training context
        # too. Rewrap the SAME physical storage outside inference_mode so that
        # the original eager writer itself supports this comparison.
        env.sim._model_bridge.clear_cache()
        for name in ("dof_frictionloss", "dof_damping"):
            assert not getattr(env.sim.model, name).is_inference()
        owner._graph_compute_supported = True
        for context in (torch.no_grad, torch.inference_mode):
            with context():
                cmd = owner.get_command(robot.data)
                duty = act.duty_cycle.clone()
                owner.cfg.graph_compute = False
                expected_motor = owner.compute(cmd).clone()
                act.duty_cycle = duty
                owner.cfg.graph_compute = True
                actual_motor = owner.compute(cmd)
                torch.testing.assert_close(actual_motor, expected_motor, rtol=0, atol=0)
        # The actuator must not leave the default generator's graph state in
        # inference-only tensors and break a later independent CUDA capture.
        value = torch.ones(8, device="cuda:0")
        independent = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(independent):
            doubled = value * 2
        independent.replay()
        torch.testing.assert_close(doubled, value * 2, rtol=0, atol=0)
    finally:
        env.close()


def test_friction_kernel_flags_invalid_active_ids_without_writing_out_of_bounds():
    import mujoco
    import warp as wp
    from mjlab_microduck.actuator.friction_force import _extract_friction
    friction_type = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)
    with wp.ScopedDevice("cpu"):
        types = wp.full((1, 4), friction_type, dtype=wp.int32)
        ids = wp.array(np.array([[-1, 2, 1, 100]], dtype=np.int32))
        forces = wp.ones((1, 4), dtype=wp.float32)
        counts = wp.array(np.array([3], dtype=np.int32))
        output = wp.zeros((1, 2), dtype=wp.float32)
        valid = wp.ones(1, dtype=wp.int32)
        wp.launch(_extract_friction, dim=(1, 4), inputs=[types, ids, forces, counts, output, valid])
        np.testing.assert_array_equal(output.numpy(), [[0., 1.]])
        np.testing.assert_array_equal(valid.numpy(), [0])
