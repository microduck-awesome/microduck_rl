#!/usr/bin/env python3
"""Measure explicit policy/motor profiles with the same CPU gait metrics.

This is an offline nominal comparison, not a hardware test or training entry.
Profiles name their ONNX, XML, motor, voltage and output processing explicitly.
No recovery switching or automatic resets are allowed inside a rollout.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from diagnose_gait import summarize
from sc0090_server import CONTROL_DT, PHYSICS_DT, rehearsal


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize_motion(trace, dt):
    result = summarize(trace, dt)
    # Both vectors are in WORLD axes (mj_objectVelocity(..., flg_local=0)).
    relative = trace['head_angular_velocity'] - trace['torso_angular_velocity']
    result['head_relative_to_torso_angular_speed_rms_rad_s'] = float(
        np.sqrt(np.mean(np.sum(relative ** 2, axis=1))))
    return result


class Probe:
    def __init__(self, profile):
        self.profile = profile
        if profile['motor'] == 'xl330':
            from bam.model import load_model
            # Bundled XL330 fits changed between BAM commits. The official
            # policy must use an explicitly pinned fit, not the installed one.
            self.motor_json = Path(profile['motor_json'])
            if sha256(self.motor_json) != profile['motor_json_sha256']:
                raise ValueError('Official motor parameter hash mismatch')
            motor = load_model(str(self.motor_json))
            motor.actuator.kp = profile['kp']
            motor.actuator.vin = profile['vin']
            motor.actuator.max_current = None
        elif profile['motor'] == 'sc0090':
            motor = rehearsal.load_bam_model(profile['kp'], profile['vin'], None)
            self.motor_json = rehearsal.SC0090_MODEL_PATH
        else:
            raise ValueError('Unknown motor profile')
        self.model, self.data, self.motor, _ = rehearsal.load_mujoco_with_bam(
            profile['scene'], motor, PHYSICS_DT,
            profile['supply_resistance_ohm'], profile['vin_min'])
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.opt.iterations = 10
        self.model.opt.ls_iterations = 20
        self.feet = [self.model.geom(f'{s}_foot_collision').id for s in ('left', 'right')]
        self.floor = self.model.geom('floor').id
        for geom in range(self.model.ngeom):
            if (self.model.geom(geom).name or '').endswith('_collision'):
                self.model.geom_condim[geom] = 3 if geom in self.feet else 1
            if geom in self.feet:
                self.model.geom_priority[geom] = 1
                self.model.geom_friction[geom, 0] = 1.
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.policy = rehearsal.PolicyInference(
            self.model, self.data, profile['policy'], bam_ctrl=self.motor,
            action_scale=profile['action_scale'], use_projected_gravity=True,
            new_cmd_obs=True, joint_velocity_obs_lag=profile['velocity_obs_lag'],
            session_options=options, providers=['CPUExecutionProvider'])
        self.names = [self.model.joint(int(j)).name for j in self.model.actuator_trnid[:, 0]]
        session = self.policy.walking_session
        meta = session.get_modelmeta().custom_metadata_map
        assert session.get_inputs()[0].shape == [1, 61]
        assert session.get_outputs()[0].shape == [1, 14]
        assert meta['joint_names'].split(',') == self.names
        # Export metadata is rounded to 3 decimal places; use the precise HOME
        # from the shared inference path and verify that it rounds identically.
        assert np.allclose(np.asarray(meta['default_joint_pos'].split(','), float),
                           self.policy.default_pose, atol=0.00051, rtol=0)
        self.neck = [i for i, n in enumerate(self.names) if 'head' in n or 'neck' in n]
        self.neck_dofs = np.asarray(self.policy.joint_qvel_indices)[self.neck]
        self.head = self.model.body('jaw_soft').id
        self.torso = self.model.body('trunk_base').id
        self.alpha = np.full(14, profile['legs_alpha'], dtype=float)
        self.alpha[self.neck] = profile['head_alpha']
        assert np.isfinite(self.alpha).all() and ((self.alpha > 0) & (self.alpha <= 1)).all()
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.policy.joint_qpos_indices] = self.policy.default_pose
        qa = self.policy._trunk_qpos_adr
        self.data.qpos[qa:qa + 7] = [0., 0., .12, 1., 0., 0., 0.]
        mujoco.mj_forward(self.model, self.data)
        rehearsal.reset_bam_controller(self.motor)
        self.policy.last_action.fill(0)
        self.policy.reset_observation_history()
        self.policy.command.fill(0)
        self.policy.vel_cmd.fill(0)
        self.filtered_target = self.policy.default_pose.copy()

    def step(self, speed):
        self.policy.vel_cmd[:] = [speed, 0., 0.]
        self.policy._update_command()
        if not np.isfinite(self.policy.get_observations()).all():
            raise FloatingPointError('Nonfinite observation')
        action = self.policy.infer()
        if action.shape != (14,) or not np.isfinite(action).all():
            raise FloatingPointError('Invalid action')
        target = self.policy.default_pose + self.policy.action_scale * action
        # Preserve bit-identical unfiltered execution for the Pages cross-check.
        if np.all(self.alpha == 1):
            self.filtered_target = target.copy()
        else:
            self.filtered_target += self.alpha * (target - self.filtered_target)
        self.policy.set_position_targets(self.filtered_target)
        for _ in range(round(CONTROL_DT / PHYSICS_DT)):
            warnings = self.data.warning.number.copy()
            self.motor.update()
            mujoco.mj_step(self.model, self.data)
            if (self.data.warning.number > warnings).any():
                raise FloatingPointError('MuJoCo physics warning')
            if not all(np.isfinite(x).all() for x in
                       (self.data.qpos, self.data.qvel, self.data.qacc, self.data.ctrl)):
                raise FloatingPointError('Nonfinite physics')
        mujoco.mj_forward(self.model, self.data)

    def frame(self):
        normal = np.zeros(2)
        for cid, c in enumerate(self.data.contact):
            pair = {int(c.geom1), int(c.geom2)}
            if self.floor not in pair:
                continue
            for i, geom in enumerate(self.feet):
                if geom in pair:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, cid, force)
                    normal[i] += max(0., force[0])
        omega = []
        for body in (self.head, self.torso):
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body, vel, 0)
            omega.append(vel[:3].copy())
        rotation = self.data.xmat[self.torso].reshape(3, 3)
        qa = self.policy._trunk_qpos_adr
        w, x, y, z = self.data.qpos[qa + 3:qa + 7]
        yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
        va = int(self.model.joint('trunk_base_freejoint').dofadr[0])
        vx, vy = self.data.qvel[va:va + 2]
        tilt = np.arccos(np.clip(-self.policy.get_projected_gravity()[2], -1., 1.))
        return dict(
            velocity=[np.cos(yaw)*vx + np.sin(yaw)*vy, -np.sin(yaw)*vx + np.cos(yaw)*vy],
            foot_normal_force=normal, foot_position=self.data.geom_xpos[self.feet].copy(),
            forward=rotation[:, 0].copy(), head_angular_velocity=omega[0],
            torso_angular_velocity=omega[1], neck_velocity=self.data.qvel[self.neck_dofs].copy()*30/np.pi,
            # Radians sent to the motors, including any explicit output preset.
            neck_action=self.filtered_target[self.neck].copy(),
            head_relative_position=rotation.T @ (self.data.xpos[self.head] - self.data.xpos[self.torso]),
            height=float(self.data.qpos[qa + 2]), tilt_rad=float(tilt),
            joint_velocity_rad_s=self.data.qvel[self.policy.joint_qvel_indices].copy(),
            target_rad=self.filtered_target.copy(),
            fell=bool(self.data.qpos[qa + 2] < .08 or tilt > np.pi/4))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--speeds', nargs='+', type=float, default=[.08, .1, .2, .3])
    parser.add_argument('--duration', type=float, default=32.)
    parser.add_argument('--settle', type=float, default=2.)
    parser.add_argument('--window', type=float, default=10.)
    args = parser.parse_args()
    if (not np.isfinite([args.duration, args.settle, args.window, *args.speeds]).all()
            or args.settle < 0 or args.window < 2 * CONTROL_DT
            or args.duration < args.settle + args.window):
        parser.error('Require finite durations and at least one full measurement window')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profiles = json.loads(args.profiles.read_text())
    rows, provenance = [], []
    for profile in profiles:
        probe = Probe(profile)
        paths = [Path(profile['policy']), Path(profile['scene']), probe.motor_json]
        hashes = {str(p.resolve()): sha256(p) for p in paths}
        provenance.append(dict(profile=profile, sha256=hashes,
                               model_mass_kg=float(probe.model.body_mass.sum()),
                               neck_joint_names=[probe.names[i] for i in probe.neck]))
        for speed in args.speeds:
            probe.reset()
            frames = []
            for _ in range(round(args.duration / CONTROL_DT)):
                probe.step(speed)
                frames.append(probe.frame())
            trace = {k: np.asarray([f[k] for f in frames]) for k in frames[0]}
            if not all(np.isfinite(v).all() for v in trace.values()):
                raise FloatingPointError('Nonfinite trace')
            np.savez_compressed(args.output_dir / f"{profile['label']}_{speed:.3f}.npz", **trace)
            start, width = round(args.settle/CONTROL_DT), round(args.window/CONTROL_DT)
            windows = []
            for a in range(start, len(frames)-width+1, width):
                sample = {k: v[a:a+width] for k, v in trace.items()}
                windows.append(dict(measurement_s=[a*CONTROL_DT, (a+width)*CONTROL_DT],
                                    fell=bool(sample['fell'].any()), **summarize_motion(sample, CONTROL_DT)))
            row = dict(profile=profile['label'], command_m_s=speed,
                       fell=bool(trace['fell'].any()),
                       first_fall_s=(float((np.flatnonzero(trace['fell'])[0]+1)*CONTROL_DT)
                                     if trace['fell'].any() else None), windows=windows)
            measured = {k: v[start:] for k, v in trace.items()}
            row['measured'] = dict(measurement_s=[start*CONTROL_DT, len(frames)*CONTROL_DT],
                                   **summarize_motion(measured, CONTROL_DT))
            rows.append(row)
            print(json.dumps(dict(profile=row['profile'], speed=speed, fell=row['fell'],
                  windows=[{k:v for k,v in x.items() if k!='flight_durations_s'} for x in windows])), flush=True)
        assert all(sha256(p) == digest for p, digest in hashes.items())
    report = dict(provenance=provenance, conditions=dict(policy_hz=1/CONTROL_DT,
                  physics_hz=1/PHYSICS_DT, duration_s=args.duration, settle_s=args.settle,
                  window_s=args.window, observation='61D, projected gravity, zero head/body commands',
                  environment='flat, foot friction 1, no DR, no pushes, deterministic HOME init',
                  controls='no recovery switches, resets or extra action delay during rollout',
                  interpretation='windows are one continuous trajectory, not independent seeds'), rows=rows)
    (args.output_dir/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
