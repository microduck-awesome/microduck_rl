from pathlib import Path
import json, numpy as np, mujoco
from scipy.spatial.transform import Rotation
A=Path(__file__).resolve().parents[1]/'src/mjlab_microduck/custom_v2/assets'
spec=mujoco.MjSpec.from_file(str(A/'robot.xml'))
spec.worldbody.add_geom(name='audit_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[2,2,.1])
m=spec.compile();d=mujoco.MjData(m)
rows=[]
for name,angles in [('supine',[0,-90,0]),('prone',[0,90,0]),('left',[90,0,0]),('right',[-90,0,0]),('rear_seated',[0,-60,0])]:
 mujoco.mj_resetData(m,d);q=Rotation.from_euler('xyz',angles,degrees=True).as_quat();d.qpos[3:7]=q[[3,0,1,2]];d.qpos[2]=.25
 touched=set();peak=0;penetration=0
 final_forces=[]
 for i in range(600):
  mujoco.mj_step(m,d)
  assert np.isfinite(d.qpos).all()
  for ci,c in enumerate(d.contact):
   names=[m.geom(int(g)).name for g in c.geom]
   if 'audit_floor' in names:
    touched.update(names);penetration=min(penetration,float(c.dist))
    force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force)
    if i==599:final_forces.append(dict(geoms=names,distance_m=float(c.dist),normal_force_N=float(force[0])))
  peak=max(peak,d.ncon)
 rows.append(dict(pose=name,ground_geoms=sorted(touched-{'audit_floor'}),peak_contacts=peak,min_distance_m=penetration,final_qpos=d.qpos.tolist(),final_contacts=final_forces))
assert any(any(g.startswith('sim2real_shell_') for g in row['ground_geoms']) for row in rows)
report=dict(status='PASS',scope='passive gravity drop, not policy behavior qualification',mass_kg=float(m.body_mass.sum()),cases=rows)
Path('../collision-check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
