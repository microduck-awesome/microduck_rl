from pathlib import Path
import json,mujoco,numpy as np,torch
r=Path(__file__).resolve().parent;m=mujoco.MjModel.from_xml_path(str(r/'roulade-repaired-prep-worktree/src/mjlab_microduck/custom_v2/assets/robot.xml'));d=mujoco.MjData(m);saved=torch.load(r/'roulade-physical-003/initial-state.pt',weights_only=False,map_location='cpu');q=saved['initial_qpos'][100].numpy().copy();q[:3]-=saved['env_origins'][100].numpy()
rows=[]
for label,joints in [('original',[]),('right_hip_yaw_bound',['right_hip_yaw']),('right_hip_roll_bound',['right_hip_roll']),('both_right_lateral_bounds',['right_hip_yaw','right_hip_roll'])]:
 z=q.copy()
 for name in joints:
  adr=int(m.joint(name).qposadr[0]);z[adr]=np.clip(z[adr],-.04,.04)
 d.qpos[:]=z;mujoco.mj_forward(m,d);cs=[dict(geoms=[m.geom(int(c.geom1)).name,m.geom(int(c.geom2)).name],depth=float(-c.dist)) for c in d.contact[:d.ncon] if c.dist<0 and m.geom_bodyid[c.geom1] and m.geom_bodyid[c.geom2]];rows.append(dict(variant=label,max_depth=max([x['depth'] for x in cs],default=0.),contacts=cs))
(r/'roulade-noise-bound-cpu-001.json').write_text(json.dumps(dict(status='SAME_FAILED_Q_CAUSAL_GEOMETRY',world=100,rows=rows,scope='All other q/root held identical; no dynamics or skill claim'),indent=2)+'\n');print(rows)
