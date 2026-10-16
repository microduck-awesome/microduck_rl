import os,json,hashlib
from pathlib import Path
import mujoco,numpy as np,torch
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
r=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');w=r/'groundpick-repaired-worktree'
spec=mujoco.MjSpec.from_file(str(w/'src/mjlab_microduck/custom_v2/assets/robot.xml'));spec.worldbody.add_geom(name='audit_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[0,0,.1],contype=1,conaffinity=2);m=spec.compile();d=mujoco.MjData(m)
s=torch.load(r/'groundpick-lowstart-0128-probe-001/traces.pt',map_location='cpu',weights_only=False);qs=s['policy_switch']['qpos'].numpy();vs=s['policy_switch']['qvel'].numpy();i=m.joint('neck_pitch').qposadr[0];head=m.geom('ground_head_shell_collision').id;floor=m.geom('audit_floor').id;mouth=m.site('mouth_tip').id
rows=[]
for q,v in zip(qs,vs):
 d.qpos[:]=q;d.qvel[:]=v;d.qpos[i]-=.06;mujoco.mj_forward(m,d)
 rows.append(dict(mouth_height_m=float(d.site_xpos[mouth,2]),mouth_down_cos=float(-d.site_xmat[mouth].reshape(3,3)[2,0]),head_floor_clearance_m=float(mujoco.mj_geomDistance(m,d,head,floor,1.,None))))
report=dict(status='LOCAL_GEOMETRY_PASS_NOT_DYNAMIC',delta_neck_rad=-.06,worlds=64,mouth_height_range_m=[min(x['mouth_height_m'] for x in rows),max(x['mouth_height_m'] for x in rows)],down_cos_min=min(x['mouth_down_cos'] for x in rows),head_shell_floor_min_clearance_m=min(x['head_floor_clearance_m'] for x in rows),mouth_gate_original_max_m=.025,all_pass=all(0<=x['mouth_height_m']<=.025 and x['mouth_down_cos']>=.95 and x['head_floor_clearance_m']>0 for x in rows),scope='same64 captured physical prehold qpos; one chosen neck offset; root/other joints unchanged; quiet/body original conditions passed001 but need new dynamics to confirm',rows=rows)
(r/'groundpick-lowstart-clearance64.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
