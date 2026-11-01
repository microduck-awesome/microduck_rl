#!/usr/bin/env python3
"""CPU static geometry screen for historical standup reset pose ranges.

This is not a replacement for reset-manager/GPU dynamic preflight. It screens
floor and self contacts at representative starting states, before integration.
"""
import json
import math
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'src/mjlab_microduck/custom_v2/assets/scene.xml'
JOINTS = ('left_hip_yaw','left_hip_roll','left_hip_pitch','left_knee','left_ankle',
          'neck_pitch','head_pitch','head_yaw','head_roll','right_hip_yaw',
          'right_hip_roll','right_hip_pitch','right_knee','right_ankle')
SIT = {1:.0873,2:-.10,3:-1.10,4:-.85,10:-.0873,11:-.10,12:-1.10,13:-.85}


def quat_mul(a, b):
    result = np.zeros(4)
    mujoco.mju_mulQuat(result, np.asarray(a), np.asarray(b))
    return result


def main():
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)
    floor = m.geom('floor').id
    rng = np.random.default_rng(20261101)
    results = {}
    # One fixed seed for 7 buckets and six frontier settings; 128 each.
    for bucket in ('standing','sitting','prone','frontier150','frontier120','frontier90',
                   'frontier60','frontier30','frontier0','left_side','right_side','supine'):
        floor_depth, self_depth, examples = [], [], []
        for i in range(128):
            mujoco.mj_resetDataKeyframe(m, d, m.key('home').id)
            yaw = rng.uniform(-math.pi, math.pi)
            cy, sy = math.cos(yaw/2), math.sin(yaw/2)
            if bucket in ('standing','sitting'):
                roll, pitch = rng.uniform(-math.radians(10), math.radians(10), size=2)
                qx = [math.cos(roll/2), math.sin(roll/2), 0., 0.]
                qy = [math.cos(pitch/2), 0., math.sin(pitch/2), 0.]
                quat = quat_mul(quat_mul([cy,0.,0.,sy], qy), qx)
                d.qpos[2] = rng.uniform(.12857121733465776,.13857121733465776) if bucket == 'standing' else rng.uniform(.0634,.0834)
                if bucket == 'sitting':
                    for k, val in SIT.items():
                        d.qpos[m.joint(JOINTS[k]).qposadr[0]] = val
                    for name in JOINTS:
                        joint = m.joint(name)
                        addr = joint.qposadr[0]
                        d.qpos[addr] += rng.normal(0., .12)
                        d.qpos[addr] = np.clip(d.qpos[addr],joint.range[0]+.002,joint.range[1]-.002)
            else:
                s = 2 ** -.5
                quat = np.array([s*cy, -s*sy, s*cy, s*sy]) if bucket == 'prone' else np.array([s*cy,s*sy,-s*cy,s*sy])
                d.qpos[2] = rng.uniform(.05,.09)
                if bucket.startswith('frontier'):
                    angle = math.radians(int(bucket.removeprefix('frontier')) + rng.uniform(0,30)) * rng.choice([-1,1])
                    quat = quat_mul(quat, [math.cos(angle/2),0.,0.,math.sin(angle/2)])
                elif bucket in ('left_side','right_side'):
                    angle = math.pi/2 * (1 if bucket == 'left_side' else -1)
                    quat = quat_mul(quat, [math.cos(angle/2),0.,0.,math.sin(angle/2)])
            d.qpos[3:7] = quat
            mujoco.mj_forward(m,d)
            f, s = 0., 0.
            for j in range(d.ncon):
                c = d.contact[j]
                depth = max(0., -float(c.dist))
                if floor in (c.geom1,c.geom2):
                    f = max(f,depth)
                else:
                    s = max(s,depth)
            floor_depth.append(f)
            self_depth.append(s)
            if max(f,s) > .005 and len(examples) < 3:
                examples.append({'sample':i,'floor_depth_m':f,'self_depth_m':s,'qpos':d.qpos.tolist()})
        results[bucket] = {'samples':128,'floor_max_m':max(floor_depth),'floor_p95_m':float(np.quantile(floor_depth,.95)),
                           'floor_over_5mm':sum(x>.005 for x in floor_depth),
                           'self_max_m':max(self_depth),'self_p95_m':float(np.quantile(self_depth,.95)),
                           'self_over_5mm':sum(x>.005 for x in self_depth),'examples':examples}
    output = {'protocol':'static approximation: HOME joints except noisy sit; random root yaw/tilt/heights; no PPO or step',
              'seed':20261101,'model':str(MODEL.relative_to(ROOT)),'results':results}
    out = ROOT / 'docs/v3-reset-geometry-audit.json'
    out.write_text(json.dumps(output,indent=2)+'\n')
    for name,r in results.items():
        print(f'{name:12} floor>5mm={r["floor_over_5mm"]}/128 max={1000*r["floor_max_m"]:.1f}mm self>5mm={r["self_over_5mm"]}/128 max={1000*r["self_max_m"]:.1f}mm')


if __name__ == '__main__':
    main()
