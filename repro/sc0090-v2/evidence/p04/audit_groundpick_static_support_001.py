"""Bounded CPU static support audit plus one constrained solve; no dynamics."""
import os,json,math,hashlib,time
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import numpy as np,torch,mujoco
from scipy.spatial import ConvexHull,QhullError
from scipy.optimize import minimize
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'groundpick-repaired-worktree';xml=W/'src/mjlab_microduck/custom_v2/assets/robot.xml'
spec=mujoco.MjSpec.from_file(str(xml));spec.worldbody.add_geom(name='audit_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[0,0,.1],contype=1,conaffinity=2)
m=spec.compile();d=mujoco.MjData(m);floor=m.geom('audit_floor').id;feet=[m.geom(n).id for n in ('left_foot_collision','right_foot_collision')];mouth=m.site('mouth_tip').id;neck=m.body('neck').id
mass=float(m.body_mass.sum());headb=[]
for b in range(m.nbody):
 parent=b
 while parent>0 and parent!=neck:parent=int(m.body_parentid[parent])
 if parent==neck:headb.append(b)
colliders=[i for i in range(m.ngeom) if i!=floor and (m.geom_contype[i] or m.geom_conaffinity[i])]
headg=[i for i in colliders if int(m.geom_bodyid[i]) in headb]
mesh={}
for i in feet:
 mid=m.geom_dataid[i];st=m.mesh_vertadr[mid];mesh[i]=np.array(m.mesh_vert[st:st+m.mesh_vertnum[mid]],dtype=float)
def vertices(i):return mesh[i]@d.geom_xmat[i].reshape(3,3).T+d.geom_xpos[i]
def hull_info(points,com,head=None):
 pts=np.unique(np.round(np.asarray(points)[:,:2],10),axis=0)
 if len(pts)<3:return dict(valid=False,points_xy=pts.tolist(),reason='fewer_than3_distinct_points')
 try:h=ConvexHull(pts)
 except QhullError:return dict(valid=False,points_xy=pts.tolist(),reason='collinear_contact_points')
 n=h.equations[:,:2];b=h.equations[:,2];violation=n@com[:2]+b;margin=float(-violation.max());res=dict(valid=True,points_xy=pts[h.vertices].tolist(),area_m2=float(h.volume),signed_inside_margin_m=margin,com_inside=margin>=0)
 if head is not None:
  hd=n@head[:2]+b;v=violation>0
  if not v.any():alpha=0.
  elif np.any(hd[v]<=0):alpha=None
  else:alpha=float(np.max(violation[v]/hd[v]))
  res.update(head_min_vertical_load_fraction=alpha,head_min_vertical_force_N=None if alpha is None else alpha*mass*9.81,head_signed_inside_margin_m=float(-hd.max()))
 return res

def snapshot(q):
 d.qpos[:]=q;d.qvel[:]=0;mujoco.mj_forward(m,d)
 com=(m.body_mass[:,None]*d.xipos).sum(0)/mass
 foot_contacts=[];head_contacts=[];other_contacts=[];self_contacts=[]
 for c in d.contact:
  gs=list(map(int,c.geom))
  if floor in gs:
   i=gs[0] if gs[1]==floor else gs[1]
   row=dict(geom=m.geom(i).name,position=c.pos.tolist(),dist_m=float(c.dist))
   if i in feet:foot_contacts.append(row)
   elif i in headg:head_contacts.append(row)
   else:other_contacts.append(row)
  elif c.dist<0:self_contacts.append(dict(geoms=[m.geom(i).name for i in gs],dist_m=float(c.dist)))
 allfoot=np.concatenate([vertices(i) for i in feet]);lowest=np.concatenate([v[v[:,2]<=v[:,2].min()+.0005] for v in [vertices(i) for i in feet]])
 # Use the actual contacting head location when present, not mouth tip.
 head=np.mean([c['position'] for c in head_contacts],axis=0) if head_contacts else None
 support_all=hull_info(allfoot,com,head);support_sole=hull_info(lowest,com,head)
 support_contact=hull_info(np.array([c['position'] for c in foot_contacts]).reshape(-1,3),com,head) if foot_contacts else dict(valid=False,reason='no_foot_contacts')
 return dict(com_world=com.tolist(),total_nominal_CAD_mass_kg=mass,mouth_height_m=float(d.site_xpos[mouth,2]),mouth_down_cos=float(-d.site_xmat[mouth].reshape(3,3)[2,0]),foot_contacts=foot_contacts,head_contacts=head_contacts,head_contact_centroid=None if head is None else head.tolist(),other_ground_contacts=other_contacts,self_contacts=self_contacts,full_foot_projection=support_all,lowest_05mm_foot_patch=support_sole,actual_CPU_foot_contacts=support_contact)
rows=[]
for name in ('001','002'):
 p=R/f'groundpick-lowstart-0128-probe-{name}';s=torch.load(p/'traces.pt',map_location='cpu',weights_only=False);states=s['policy_switch']['qpos'].numpy();items=[snapshot(q) for q in states]
 summaries={}
 for key in ('full_foot_projection','lowest_05mm_foot_patch','actual_CPU_foot_contacts'):
  valid=[x[key] for x in items if x[key]['valid']];forces=[x['head_min_vertical_force_N'] for x in valid if x.get('head_min_vertical_force_N') is not None]
  summaries[key]=dict(valid_worlds=len(valid),com_outside_worlds=sum(not x['com_inside'] for x in valid),inside_margin_range_m=[min(x['signed_inside_margin_m'] for x in valid),max(x['signed_inside_margin_m'] for x in valid)] if valid else None,min_head_vertical_force_range_N=[min(forces),max(forces)] if forces else None)
 rows.append(dict(probe=name,worlds=64,nominal_CAD_support=summaries,actual_GPU_recorded_head_force_norm_N=[float(s['measurements']['head_force'][40:50].min()),float(s['measurements']['head_force'][40:50].max())],states=items))
# Exactly one constrained correction solve from002 nominal target; no multistart/scan.
base=json.loads((R/'groundpick-lowstart-0128-probe-002/traces.pt').with_name('report.json').read_text())
pose=json.loads((R/'groundpick-reach-screen-001.json').read_text())['results'][0];pose['pose']['neck_pitch']-=.06
qbase=m.qpos0.copy()
for name,v in pose['pose'].items():qbase[m.joint(name).qposadr[0]]=v
names=['left_hip_pitch','left_knee','left_ankle'];right=[n.replace('left_','right_') for n in names]
# Fixed material sole vertices selected at HOME, independent of candidate selection.
d.qpos[:]=m.qpos0;d.qpos[2]=.12;mujoco.mj_forward(m,d);sole={i:np.flatnonzero(vertices(i)[:,2]<=vertices(i)[:,2].min()+.0005) for i in feet}
x0=np.array([pose['pose'][n] for n in names]+[pose['root_pitch_rad'],pose['root_height_m']])
bounds=[tuple(m.jnt_range[m.joint(n).id]+np.array([.01,-.01])) for n in names]+[(-.65,.65),(.06,.14)]
scales=np.array([.3,.3,.3,.2,.02]);calls=[0]
def apply(x):
 d.qpos[:]=qbase;d.qvel[:]=0
 for i,n in enumerate(names):d.qpos[m.joint(n).qposadr[0]]=d.qpos[m.joint(right[i]).qposadr[0]]=x[i]
 d.qpos[:3]=[0,0,x[4]];d.qpos[3:7]=[math.cos(x[3]/2),0,math.sin(x[3]/2),0];mujoco.mj_forward(m,d)
def geometry(x):
 calls[0]+=1;apply(x);foot=np.concatenate([vertices(i)[sole[i]] for i in feet]);com=(m.body_mass[:,None]*d.xipos).sum(0)/mass
 h=ConvexHull(foot[:,:2]);margin=-float((h.equations[:,:2]@com[:2]+h.equations[:,2]).max())
 clearance=min(float(mujoco.mj_geomDistance(m,d,floor,i,1.,None)) for i in headg)
 selfdist=min([float(c.dist) for c in d.contact if floor not in c.geom]+[.01])
 return dict(mouth_z=float(d.site_xpos[mouth,2]),down=float(-d.site_xmat[mouth].reshape(3,3)[2,0]),com_margin=margin,head_clearance=clearance,self_distance=selfdist,foot_z_min=float(foot[:,2].min()),foot_z_max=float(foot[:,2].max()),foot_z_mean=float(foot[:,2].mean()),foot_flat_delta=float(foot[:,2].max()-foot[:,2].min()))
def inequality(x):
 g=geometry(x)
 return np.array([(g['mouth_z']-.005)/.02,(.025-g['mouth_z'])/.02,(g['down']-.95)/.05,(g['head_clearance']-.003)/.01,(g['com_margin']-.005)/.01,(g['self_distance']+.000001)/.01,(.0005-g['foot_flat_delta'])/.005,(g['foot_z_min']+.00025)/.005,(.00025-g['foot_z_max'])/.005])
def objective(x):return float(np.sum(((x-x0)/scales)**2))
fit=minimize(objective,x0,method='SLSQP',bounds=bounds,constraints=[dict(type='ineq',fun=inequality)],options={'maxiter':120,'ftol':1e-9,'disp':False})
g=geometry(fit.x);ineq=inequality(fit.x);apply(fit.x);candidate=snapshot(d.qpos.copy());candidate_pose={n:float(d.qpos[m.joint(n).qposadr[0]]) for n in pose['pose']}
solve=dict(status='STATIC_CANDIDATE_ONLY' if np.min(ineq)>=-1e-5 else 'NO_QUALIFIED_STATIC_CANDIDATE',success=bool(fit.success),message=fit.message,iterations=int(fit.nit),geometry_calls=calls[0],single_start=x0.tolist(),solution=fit.x.tolist(),pose=candidate_pose,root_pitch_rad=float(fit.x[3]),root_height_m=float(fit.x[4]),objective=float(fit.fun),normalized_inequalities=ineq.tolist(),geometry=g,full_snapshot=candidate,constraints='mouth5–25mm/down>=.95; head-ground>=3mm; nominal CAD COM>=5mm inside fixed HOME material sole projection; bilateral symmetric hip/knee/ankle; footplane maxspread.5mm and entire sole z±.25mm; no self penetration>1um; head target unchanged from002',dynamic_qualified=False)
result=dict(status='CPU_STATIC_SUPPORT_AUDIT_COMPLETE',source_model_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),formal_controls_added=0,gpu_controls_added=0,simulation_integration_steps=0,nominal_CAD_mass_kg=mass,mass_provenance='Full original inertialfromgeom=false CAD body masses; per-world GPU randomized trunk pseudo-inertia and COM fields were not saved, so these are nominal-CAD reconstructions, not exact randomized world forces',support_definitions={'actual_CPU_foot_contacts':'CPU collision contacts from original full CAD at captured real qpos, not GPU contact-force reconstruction','lowest_05mm_foot_patch':'vertices within.5mm of minimum z perfoot at saved pose','full_foot_projection':'all vertices of both foot collision meshes projected to floor; generous maximum support bound','head_min_force':'minimum vertical load needed with only foot convex support and contacting head point; no tangential forces, accelerations or active joint torque inference'},saved_pose_results=rows,single_constrained_solve=solve,limitations=['Static geometric candidate does not prove v5/feedback dynamic hold or transition feasibility.','Foot friction, dynamic angular momentum, support pressure distribution and exact randomized inertia are not reconstructed.','No GPU probe or training authorized by this CPU result.'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(R/'groundpick-static-support-audit-001.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'mass':mass,'saved':[{'probe':x['probe'],'nominal_CAD_support':x['nominal_CAD_support'],'GPU_head_force':x['actual_GPU_recorded_head_force_norm_N']} for x in rows],'solve':{k:v for k,v in solve.items() if k!='full_snapshot'}},indent=2))
