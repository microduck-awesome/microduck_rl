#!/usr/bin/env python3
"""Audit the copied V3 assembly and compare its joint contract with A6804."""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/l40-vm01/Documents/ai-research/microduck/microduck-assembly-v3-home-second')
V3 = ROOT / 'src/mjlab_microduck/custom_v2/assets'
V2 = Path('/home/l40-vm01/Documents/ai-research/microduck/microduck-v2-repro/repro/sc0090-v2/checkpoints/20261024-rise3762-collision-adaptation/candidate-6804/model')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model(path):
    return mujoco.MjModel.from_xml_path(str(path))


def joints(m):
    return [m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]


def main():
    files = [p for p in (SOURCE / 'model').rglob('*') if p.is_file()]
    mismatches = [str(p.relative_to(SOURCE / 'model')) for p in files
                  if not (V3 / p.relative_to(SOURCE / 'model')).is_file()
                  or sha(p) != sha(V3 / p.relative_to(SOURCE / 'model'))]
    m2, m3 = model(V2 / 'robot.xml'), model(V3 / 'robot.xml')
    j2, j3 = joints(m2), joints(m3)
    metadata = json.loads((V3 / 'model_metadata.json').read_text())
    home = m3.key('home').qpos
    details = {}
    for name in j3:
        a, b = m2.joint(name), m3.joint(name)
        details[name] = {'v2_axis_local': a.axis.tolist(), 'v3_axis_local': b.axis.tolist(),
                         'v2_range': a.range.tolist(), 'v3_range': b.range.tolist(),
                         'v3_home': float(home[b.qposadr[0]]),
                         'v3_home_clearance': float(min(home[b.qposadr[0]] - b.range[0], b.range[1] - home[b.qposadr[0]]))}
    out = {'assembly_commit': '46a1b2c1d371def5775d33e67424755f253e9881',
           'source_files_count': len(files), 'mismatches': mismatches,
           'metadata_sha_match': sha(SOURCE / 'reports/model_metadata.json') == sha(V3 / 'model_metadata.json'),
           'v2_mass_kg': float(m2.body_mass.sum()), 'v3_mass_kg': float(m3.body_mass.sum()),
           'v2_hinge_order': j2, 'v3_hinge_order': j3,
           'v3_home_qpos_max_difference_metadata': float(np.max(np.abs(home - metadata['home_qpos']))),
           'v3_collision_geoms': [{'name': m3.geom(i).name, 'contype': int(m3.geom_contype[i]),
                                   'conaffinity': int(m3.geom_conaffinity[i])}
                                  for i in range(m3.ngeom) if m3.geom_contype[i] or m3.geom_conaffinity[i]],
           'joints': details}
    assert not mismatches and out['metadata_sha_match'] and j2 == j3
    assert out['v3_home_qpos_max_difference_metadata'] < 1e-9
    assert all(d['v3_home_clearance'] > 0 for name, d in details.items() if name != 'mouth')
    assert details['mouth']['v3_home_clearance'] >= 0
    output = ROOT / 'docs/v3-assembly-source-audit.json'
    output.write_text(json.dumps(out, indent=2) + '\n')
    print(f'PASS: {len(files)} identical files; {len(j3)} hinges; mass={out["v3_mass_kg"]:.12f} kg; report={output}')


if __name__ == '__main__':
    main()
