"""Five fixed points on one authorized HOME/ClearSeat path, not a pose search."""
import os,json,hashlib,math
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial import ConvexHull,QhullError
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'sitstand-short-lesson-worktree';xml=W/'src/mjlab_microduck/custom_v2/assets/robot.xml'
spec=mujoco.MjSpec.from_file(str(xml));spec.worldbody.add_geom(name='audit_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[0,0,.1],contype=1,conaffinity=2);m=spec.compile();d=mujoco.MjData(m);floor=m.geom('audit_floor').id
pose={'left_hip_roll':.0873,'right_hip_roll':-.0873,'left_hip_pitch':-.7,'right_hip_pitch':-.7,'left_knee':-.8,'right_knee':-.8,'left_ankle':-.15,'right_ankle':-.15};feet=[m.geom(n).id for n in ('left_foot_collision','right_foot_collision')];mesh={}
for g in feet:
 mid=m.geom_dataid[g];a=m.mesh_vertadr[mid];mesh[g]=m.mesh_vert[a:a+m.mesh_vertnum[mid]].copy()
def verts(g):return mesh[g]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
def margin(points,com):
 xy=np.unique(np.round(points[:,:2],9),axis=0)
 if len(xy)<3:return None
 try:h=ConvexHull(xy)
 except QhullError:return None
 return float(-(h.equations[:,:2]@com[:2]+h.equations[:,2]).max())
rows=[]
for alpha in (0,.25,.5,.75,1):
 d.qpos[:]=m.qpos0;d.qvel[:]=0;d.qpos[:3]=[0,0,.12];d.qpos[3:7]=[1,0,0,0]
 for n,v in pose.items():d.qpos[m.joint(n).qposadr[0]]=alpha*v
 mujoco.mj_forward(m,d);d.qpos[2]-=min(verts(g)[:,2].min() for g in feet);mujoco.mj_forward(m,d)
 com=(m.body_mass[:,None]*d.xipos).sum(0)/m.body_mass.sum();vs=[verts(g) for g in feet];patch=np.concatenate([v[v[:,2]<=.0005] for v in vs]);whole=np.concatenate(vs)
 contacts=[{'geoms':[m.geom(int(i)).name for i in c.geom],'distance_m':float(c.dist)} for c in d.contact if floor not in c.geom and c.dist<0]
 rows.append(dict(alpha=alpha,root_height_upright_touch_floor_m=float(d.qpos[2]),COM=com.tolist(),whole_foot_projection_inside_margin_m=margin(whole,com),near_floor05mm_projection_inside_margin_m=margin(patch,com),foot_z_span_m=[float(v[:,2].max()-v[:,2].min()) for v in vs],self_contacts=contacts,static_orientation_assumption='rootupright, rootz adjusted so lowestfoot vertex touchesfloor; not a solveddynamic pose'))
report=dict(status='FIXED_PATH_CPU_SCREEN_COMPLETE',candidate_paths=1,points=5,rows=rows,model_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),formal_controls_added=0,gpu_controls_added=0,scope='Kinematic crosssections only; COM inside fullfoot hull is not dynamic/stiffness proof, and narrow contact patch can expand after actualroot orientation adjustment. No interpolation reward/teacher labels written.')
(R/'sitstand-slow-path-cpu-001.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
