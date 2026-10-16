import os,json,hashlib
from pathlib import Path
import mujoco,numpy as np,torch
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'groundpick-repaired-worktree'
spec=mujoco.MjSpec.from_file(str(W/'src/mjlab_microduck/custom_v2/assets/robot.xml'))
if not any(g.type==mujoco.mjtGeom.mjGEOM_PLANE for g in spec.geoms):spec.worldbody.add_geom(name='audit_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[0,0,.1],contype=1,conaffinity=2)
m=spec.compile();d=mujoco.MjData(m)
s=torch.load(R/'groundpick-lowstart-0128-probe-001/traces.pt',map_location='cpu',weights_only=False);q=s['policy_switch']['qpos'][0].numpy();assert q.shape==d.qpos.shape
neck=m.body('neck').id;headb=[]
for b in range(m.nbody):
 p=b
 while p>0 and p!=neck:p=m.body_parentid[p]
 if p==neck:headb.append(b)
headg=[i for i in range(m.ngeom) if int(m.geom_bodyid[i]) in headb and (m.geom_contype[i] or m.geom_conaffinity[i])]
plane=m.geom('audit_floor').id
rows=[]
for name,delta in [('base',0),('neck_pitch',-.02),('neck_pitch',.02),('head_pitch',-.02),('head_pitch',.02)]:
 d.qpos[:]=q
 if name!='base':d.qpos[m.joint(name).qposadr[0]]+=delta
 mujoco.mj_forward(m,d)
 distances=[(m.geom(i).name,float(mujoco.mj_geomDistance(m,d,plane,i,1.,None))) for i in headg]
 rows.append(dict(joint=name,delta=delta,mouth_height=float(d.site_xpos[m.site('mouth_tip').id,2]),mouth_down_cos=float(-d.site_xmat[m.site('mouth_tip').id].reshape(3,3)[2,0]),head_floor_distances=sorted(distances,key=lambda x:x[1])[:5],head_floor_contacts=[dict(geoms=[m.geom(int(g)).name for g in c.geom],distance=float(c.dist)) for c in d.contact if plane in c.geom and any(int(g) in headg for g in c.geom)]))
x=s['measurements'];gates={}
for name,v in {'mouth_range':(x['mouth_height']>=0)&(x['mouth_height']<=.025),'downcos':x['mouth_down_cos']>=.95,'tilt':x['tilt']<np.pi/3,'bothfeet':x['feet'].all(-1),'no_self':~x['self_contact'],'no_head':~x['head_contact'],'speed':x['linear'].norm(dim=-1)<.05,'omega':x['omega']<.5,'ready':x['feedback_ready'],'finite':x['finite']}.items():gates[name]=dict(last200ms_all_trials=int(v[40:50].all(0).sum()),switch=int(v[49].sum()))
r=dict(status='CONTACT_REASON_IDENTIFIED',original_probe='groundpick-lowstart-0128-probe-001',gates=gates,rows=rows,head_force_last200ms_range=[float(x['head_force'][40:50].min()),float(x['head_force'][40:50].max())],qualification_correction='Old physical reference recorded excess force above1N, not contact found; physical quiet pose is not no-contact low-goal qualified',scope='CPU frozen physical state with original CAD and added terrain plane; no integration/model mutation; small local kinematic differences are not dynamic qualification')
(R/'groundpick-lowstart-contact-audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
