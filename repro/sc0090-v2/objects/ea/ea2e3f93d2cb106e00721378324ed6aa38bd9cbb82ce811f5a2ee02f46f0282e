"""Record a three-second, non-resetting BAM HOME hold from small joint disturbances.

This diagnoses the CAD pose and actuator integration, not learned balance or
hardware stability. The report retains tilt and velocity, including failures.
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.custom_v2.microduck_v2 import make_v2_bam_env_cfg, get_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = make_v2_bam_env_cfg(play=True)
    cfg.scene.num_envs = 3
    cfg.seed = 42
    cfg.events = {
        name: term
        for name, term in cfg.events.items()
        if name in {"expand_bam_friction_fields", "mouth_hold"}
    }
    cfg.curriculum = {}
    cfg.terminations = {}
    cfg.rewards = {}
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        env.reset()
        robot = env.scene["robot"]
        pose = robot.data.joint_pos.clone()
        pose[1:] += 0.01 * torch.randn_like(pose[1:])
        robot.write_joint_state_to_sim(pose, torch.zeros_like(pose))
        env.sim.forward()
        maximum = torch.zeros(3)
        for _ in range(150):
            obs, *_ = env.step(torch.zeros(3, 14))
            assert all(torch.isfinite(value).all() for value in obs.values())
            tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))
            maximum = torch.maximum(maximum, tilt)
        model = env.sim.mj_model
        cad = get_spec().compile()
        for i in range(1, cad.nbody):
            actual = model.body("robot/" + cad.body(i).name)
            np.testing.assert_allclose(actual.mass, cad.body(i).mass, rtol=0, atol=1e-12)
            np.testing.assert_allclose(actual.inertia, cad.body(i).inertia, rtol=0, atol=1e-12)
        dofs = [model.joint("robot/" + name).dofadr[0] for name in robot.joint_names]
        report = dict(
            task="V2 frozen M6, nominal 11.1 V, P15/D30/I0, ideal diagnostic feedback",
            seconds=3,
            environments=3,
            reset_during_hold=False,
            perturbation_rad_std=0.01,
            maximum_tilt_deg=torch.rad2deg(maximum).tolist(),
            final_tilt_deg=torch.rad2deg(tilt).tolist(),
            final_root_height_m=robot.data.root_link_pos_w[:, 2].tolist(),
            final_max_joint_speed_rad_s=robot.data.joint_vel.abs().amax(-1).tolist(),
            nominal_mass_kg=float(model.body_mass.sum()),
            armature_kg_m2=model.dof_armature[dofs].tolist(),
            no_fall_under_15_degrees=bool((torch.rad2deg(maximum) < 15).all()),
            hardware_validated=False,
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
