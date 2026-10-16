"""Six bounded kinematic starts; candidates only, not dynamic/skill qualification."""
from pathlib import Path
import json,hashlib
import numpy as np
import mujoco
from scipy.optimize import least_squares
w=Path(__file__).resolve().parents[1];xml=w/'src/mjlab_microduck/custom_v2/assets/robot.xml'
m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m)
names=['left_hip_pitch','left_knee','left_ankle','neck_pitch','head_pitch']
ids=[m.joint(n).qposadr[0] for n in names];right=[m.joint(n.replace('left_','right_')).qposadr[0] for n in names[:3]]
sites=[m.site(n).id for n in ['mouth_tip','left_foot','right_foot']];root_id=m.body('trunk_base').id
lo=np.array([*m.jnt_range[[m.joint(n).id for n in names],0]+.02,-.65,.045]);hi=np.array([*m.jnt_range[[m.joint(n).id for n in names],1]-.02,.65,.14])
def apply(x):
 d.qpos[:]=m.qpos0;d.qvel[:]=0;d.qpos[ids]=x[:5];d.qpos[right]=x[:3]
 d.qpos[3:7]=[np.cos(x[5]/2),0,np.sin(x[5]/2),0];d.qpos[2]=x[6]
 mujoco.mj_forward(m,d)
def cost(x,height):
 apply(x);mouth,left,rightp=d.site_xpos[sites];mx=d.site_xmat[sites[0]].reshape(3,3)
 fl=d.site_xmat[sites[1]].reshape(3,3);fr=d.site_xmat[sites[2]].reshape(3,3)
 center=(left+rightp)*.5;com=d.subtree_com[root_id]
 return np.array([(mouth[2]-height)/.02,(mx[2,0]+1)/.15,left[2]/.005,rightp[2]/.005,
   fl[0,2]/.2,fl[1,2]/.2,fr[0,2]/.2,fr[1,2]/.2,
   (com[0]-center[0])/.025,(com[1]-center[1])/.025, max(0,-fl[2,2])/.1,max(0,-fr[2,2])/.1])
rows=[]
for height in (.015,.030,.045):
 for start in ([-.4,.9,-.5,1.,.5,0.,.105],[-.8,1.5,-.7,1.6,-.1,0.,.085]):
  fit=least_squares(cost,np.clip(start,lo+1e-5,hi-1e-5),args=(height,),bounds=(lo,hi),max_nfev=150,ftol=1e-8,xtol=1e-8,gtol=1e-8)
  apply(fit.x);mouth,left,rightp=d.site_xpos[sites];mx=d.site_xmat[sites[0]].reshape(3,3)
  contacts=[{'geoms':[m.geom(int(g)).name for g in c.geom],'dist_m':float(c.dist)} for c in d.contact if c.dist<-.0001]
  pose={n:float(v) for n,v in zip(names,fit.x[:5])};pose.update({n.replace('left_','right_'):pose[n] for n in names[:3]})
  feet=np.stack([left,rightp]);com=d.subtree_com[root_id];stable_support=abs(com[0]-feet[:,0].mean())<.025 and feet[:,1].min()<com[1]<feet[:,1].max()
  rows.append(dict(target_mouth_height_m=height,optimizer_success=bool(fit.success),nfev=fit.nfev,cost=float(fit.cost),pose=pose,root_pitch_rad=float(fit.x[5]),root_height_m=float(fit.x[6]),mouth_height_m=float(mouth[2]),mouth_down_cos=float(-mx[2,0]),foot_site_z=feet[:,2].tolist(),support_com=bool(stable_support),self_penetrations=contacts,geometric_candidate=bool(not contacts and stable_support and abs(mouth[2]-height)<.005 and -mx[2,0]>.95 and np.abs(feet[:,2]).max()<.003)))
report=dict(status='BOUNDED_CPU_GEOMETRY_SCREEN_ONLY',model_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),source_model_unchanged=True,candidate_budget=6,results=rows,scope='No dynamics, no claim of equilibrium/transition/skill, no collision masks changed; foot-site constraints are a sufficient candidate search, not proof against other valid paths')
(w/'runs-v2/groundpick-reach-screen-001.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
