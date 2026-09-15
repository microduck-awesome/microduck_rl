#!/usr/bin/env python3
"""Measure walking style using the same unfiltered CPU execution as Pages.

All samples are at policy boundaries (50 Hz). These measurements describe a
nominal rollout; they are not hardware measurements or style acceptance gates.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from sc0090_server import CONTROL_DT, Demo, default_models


def segments(mask):
    """Return complete true runs, excluding runs cut by either window edge."""
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(edges == 1),
                                           np.flatnonzero(edges == -1))
            if a > 0 and b < len(mask)]


def summarize(trace, dt):
    contacts = trace['foot_normal_force'] > 0
    durations, touchdown_events, strides = [], [], []
    for foot in range(2):
        flights = segments(~contacts[:, foot])
        # Keep raw flight durations as well as debounced touchdown counts.
        durations.extend((end - start) * dt for start, end in flights)
        touchdowns = [end for start, end in flights
                      if end - start >= 2 and end + 2 <= len(contacts)
                      and contacts[end:end + 2, foot].all()]
        touchdown_events.extend((i, foot) for i in touchdowns)
        for a, b in zip(touchdowns, touchdowns[1:]):
            displacement = trace['foot_position'][b, foot] - trace['foot_position'][a, foot]
            # Project onto the horizontal heading at the first touchdown.
            direction = trace['forward'][a, :2]
            norm = np.linalg.norm(direction)
            if norm < np.finfo(float).eps:
                continue  # A fallen torso may have no horizontal heading.
            direction = direction / norm
            strides.append(float(displacement[:2] @ direction))
    head_omega = trace['head_angular_velocity']
    torso_omega = trace['torso_angular_velocity']
    result = {
        'mean_forward_speed_m_s': float(trace['velocity'][:, 0].mean()),
        'touchdowns_per_s': len(touchdown_events) / (len(contacts) * dt),
        'median_same_foot_stride_m': float(np.median(strides)) if strides else None,
        'median_flight_s': float(np.median(durations)) if durations else None,
        'flight_durations_s': durations,
        'double_support_fraction': float(contacts.all(axis=1).mean()),
        'head_angular_speed_rms_rad_s': float(np.sqrt(np.mean(np.sum(head_omega ** 2, axis=1)))),
        'torso_angular_speed_rms_rad_s': float(np.sqrt(np.mean(np.sum(torso_omega ** 2, axis=1)))),
        'head_angular_acceleration_rms_rad_s2': float(np.sqrt(np.mean(np.sum((np.diff(head_omega, axis=0) / dt) ** 2, axis=1)))),
        'neck_joint_rpm_rms': np.sqrt(np.mean(trace['neck_velocity'] ** 2, axis=0)).tolist(),
        'neck_action_delta_rms_rad': float(np.sqrt(np.mean(np.diff(trace['neck_action'], axis=0) ** 2))),
        'head_position_range_m': np.ptp(trace['head_relative_position'], axis=0).tolist(),
    }
    # Describe frequency content, without treating a band boundary as a target.
    frequencies = np.fft.rfftfreq(len(head_omega), dt)
    spectrum = np.abs(np.fft.rfft(head_omega - head_omega.mean(axis=0), axis=0)) ** 2
    power = spectrum.sum(axis=1)
    result['head_angular_velocity_peak_hz'] = (float(frequencies[1 + np.argmax(power[1:])])
                                               if power.sum() > 0 else None)
    result['head_angular_velocity_power_above_6hz_fraction'] = float(power[frequencies > 6].sum() / max(power.sum(), 1e-30))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--walking', type=Path)
    parser.add_argument('--recovery', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--speeds', nargs='+', type=float, default=[.08, .1, .2, .3])
    parser.add_argument('--duration', type=float, default=12.)
    parser.add_argument('--settle', type=float, default=2.)
    args = parser.parse_args()
    if (not np.isfinite([args.duration, args.settle, *args.speeds]).all()
            or args.settle < 0 or args.duration - args.settle < 2 * CONTROL_DT):
        parser.error('Require finite settings and at least two measured control steps')
    walking, recovery = default_models()
    walking = (args.walking or walking).resolve(strict=True)
    recovery = (args.recovery or recovery).resolve(strict=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    demo = Demo(walking, recovery)
    head = demo.model.body('jaw_soft').id
    torso = demo.model.body('trunk_base').id
    feet = [demo.model.geom(f'{side}_foot_collision').id for side in ('left', 'right')]
    neck = [i for i, name in enumerate(demo.joint_names) if 'neck' in name or 'head' in name]
    dofs = np.asarray(demo.policy.joint_qvel_indices)[neck]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (walking, recovery)}
    rows = []
    for speed in args.speeds:
        demo.reset('standing')
        frames, fell = [], False
        for step in range(round(args.duration / CONTROL_DT)):
            demo.step([speed, 0, 0])
            state = demo.status()
            fell |= state['recovering']
            if step < round(args.settle / CONTROL_DT):
                continue
            normal = np.zeros(2)
            for contact_id, contact in enumerate(demo.data.contact):
                pair = {int(contact.geom1), int(contact.geom2)}
                if demo.floor_geom not in pair:
                    continue
                for i, geom in enumerate(feet):
                    if geom in pair:
                        force = np.zeros(6)
                        mujoco.mj_contactForce(demo.model, demo.data, contact_id, force)
                        normal[i] += max(0., force[0])
            velocities = []
            for body in (head, torso):
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(demo.model, demo.data, mujoco.mjtObj.mjOBJ_BODY,
                                        body, velocity, 0)
                velocities.append(velocity[:3].copy())
            rotation = demo.data.xmat[torso].reshape(3, 3)
            frames.append(dict(
                velocity=state['velocity'], foot_normal_force=normal,
                foot_position=demo.data.geom_xpos[feet].copy(), forward=rotation[:, 0].copy(),
                head_angular_velocity=velocities[0], torso_angular_velocity=velocities[1],
                neck_velocity=demo.data.qvel[dofs].copy() * 30 / np.pi,
                neck_action=demo.policy.last_action[neck].copy(),
                head_relative_position=rotation.T @ (demo.data.xpos[head] - demo.data.xpos[torso])))
        trace = {key: np.asarray([frame[key] for frame in frames]) for key in frames[0]}
        if not all(np.isfinite(value).all() for value in trace.values()):
            raise FloatingPointError('Nonfinite gait trace')
        row = dict(command_m_s=speed, fell=fell, **summarize(trace, CONTROL_DT))
        rows.append(row)
        np.savez_compressed(args.output_dir / f'speed_{speed:.3f}.npz', **trace)
        print(json.dumps({k: v for k, v in row.items() if k != 'flight_durations_s'}), flush=True)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha for p, sha in hashes.items())
    report = dict(models_sha256=hashes, dynamics_revision=demo.status()['dynamics_revision'],
                  policy_hz=1 / CONTROL_DT, measurement_s=[args.settle, args.duration],
                  conditions='Pages CPU execution, nominal 12 V, no DR, no external pushes, zero head/body commands',
                  touchdown_definition='positive foot-floor normal force; preceding flight and following contact each at least 2 samples',
                  neck_joint_names=[demo.joint_names[i] for i in neck], rows=rows)
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
