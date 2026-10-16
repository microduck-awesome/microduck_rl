"""Static support and actuator feasibility on saved geometry; no integration."""
from pathlib import Path
import hashlib,importlib.util,json,math,os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import numpy as np,mujoco
from scipy.optimize import linprog,minimize
from scipy.linalg import qr
from scipy.spatial import ConvexHull
R=Path(__file__).resolve().parent;W=R/'groundpick-repaired-worktree'
source=W/'src/mjlab_microduck/physics/cpu_warp_contacts.py'
sp=importlib.util.spec_from_file_location('fixed_cpu_warp_contacts',source);mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
xml=W/'src/mjlab_microduck/custom_v2/assets/robot.xml';spec=mujoco.MjSpec.from_file(str(xml));spec.worldbody.add_geom(name='audit_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[0,0,.1],contype=1,conaffinity=2)
m=spec.compile();d=mujoco.MjData(m);patch=mod.CpuWarpFootContacts(m)
v5=json.loads((W/'src/mjlab_microduck/actuator/params/sc0090_v5_m6.json').read_text())
mapping=np.load(W/'src/mjlab_microduck/custom_v2/training-assets/observer/qualification/O1-native100-seed42.npz')['mapping'].reshape(2)
candidate=json.loads((R/'groundpick-executable-endpoint-005.json').read_text());names=candidate['joint_order'];qids=np.array([m.joint(n).qposadr[0] for n in names]);dofs=np.array([m.joint(n).dofadr[0] for n in names]);limits=np.array([m.jnt_range[m.joint(n).id] for n in names]);floor=m.geom('audit_floor').id;feet=[m.geom(n).id for n in ['left_foot_collision','right_foot_collision']];mouth=m.site('mouth_tip').id
scale=v5['operating_point']['servo_radians_per_count'];gain=v5['kt']*11.1/v5['R'];dead=2.;K=v5['pwm_per_p_count']*15.
def count_for(command):return np.rint(np.clip((np.asarray(command)-mapping[0])/mapping[1]/scale+512,20,1001))
def torque(count,q):
    err=((count-512)*scale-v5['feedback_bias']-v5['feedback_scale']*q)/scale
    return gain*np.clip(K*(err-np.clip(err,-dead,dead)),-1,1)
def invert(required,q):
    duty=required/gain;err=duty/K+np.sign(duty)*dead
    count=np.rint(np.clip((v5['feedback_bias']+v5['feedback_scale']*q)/scale+err+512,20,1001))
    command=mapping[0]+mapping[1]*(count-512)*scale
    return count,command,torque(count,q)
mesh={}
for i in feet:
    mi=m.geom_dataid[i];start=m.mesh_vertadr[mi];mesh[i]=np.array(m.mesh_vert[start:start+m.mesh_vertnum[mi]],dtype=float)
def vertices(i):return mesh[i]@d.geom_xmat[i].reshape(3,3).T+d.geom_xpos[i]
d.qpos[:]=m.qpos0;d.qpos[2]=.12;mujoco.mj_forward(m,d)
sole={};home=[]
for i in feet:
    v=vertices(i);z=v[:,2];levels=[]
    for eps in [1e-6,1e-5,1e-4,.0005]:
        ids=np.flatnonzero(z<=z.min()+eps);xy=v[ids,:2];area=float(ConvexHull(xy).volume) if len(ids)>=3 else 0.
        levels.append(dict(tolerance_m=eps,vertices=len(ids),z_spread_m=float(np.ptp(z[ids])),area_m2=area))
    ids=np.flatnonzero(z<=z.min()+1e-5);assert len(ids)>=3
    hull=ConvexHull(v[ids,:2]);sole[i]=ids[hull.vertices];home.append(dict(geom=m.geom(i).name,levels=levels,selected_indices=sole[i].tolist()))
q=np.array(candidate['geometry']['q']);d.qpos[:]=m.qpos0;d.qpos[qids]=q;d.qpos[:3]=[0,0,candidate['root_z']];p=candidate['root_pitch_rad'];d.qpos[3:7]=[math.cos(p/2),0,math.sin(p/2),0];d.qvel[:]=0;mujoco.mj_forward(m,d)
contacts=[]
for c in d.contact:
    if floor not in c.geom:continue
    g=int(c.geom[0] if c.geom[1]==floor else c.geom[1])
    if g in feet:contacts.append(dict(geom=g,point=np.array(c.pos),dist=float(c.dist)))
projected=[dict(geom=i,point=v.copy(),dist=float(v[2])) for i in feet for v in vertices(i)[sole[i]]]
def analyze(points,payload,label):
    if not points:return dict(kind=label,payload_kg=payload,status='NO_SUPPORT_POINTS')
    jt=[]
    for row in points:
        jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jac(m,d,jp,jr,row['point'],int(m.geom_bodyid[row['geom']]));jt.append(jp[2])
    A=np.stack(jt,1);jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jac(m,d,jp,jr,d.site_xpos[mouth],int(m.site_bodyid[mouth]));b=d.qfrc_bias+jp[2]*payload*9.81
    baseA=A[:6];baseb=b[:6];rank=np.linalg.matrix_rank(baseA,tol=1e-10);_,_,piv=qr(baseA.T,pivoting=True,mode='economic');selected=piv[:rank]
    lp=linprog(np.zeros(A.shape[1]),A_eq=baseA[selected],b_eq=baseb[selected],bounds=[(0,None)]*A.shape[1],method='highs')
    output=dict(kind=label,payload_kg=payload,points=len(points),root_balance_rank=int(rank),solver_success=bool(lp.success),message=lp.message)
    if not lp.success:return output
    output['full_root_residual_max']=float(np.max(np.abs(baseA@lp.x-baseb)))
    if output['full_root_residual_max']>1e-7:output['status']='NO_STATIC_ROOT_BALANCE';return output
    low=torque(count_for(limits[:,0]+.005),q);high=torque(count_for(limits[:,1]-.005),q)
    # tau_required = gravity/payload - contact reaction; nominal still pose.
    Bj=A[dofs];bj=b[dofs]
    constrained=linprog(np.zeros(A.shape[1]),A_ub=np.concatenate((-Bj,Bj)),b_ub=np.concatenate((high-bj,bj-low)),A_eq=baseA[selected],b_eq=baseb[selected],bounds=[(0,None)]*A.shape[1],method='highs')
    force0=constrained.x if constrained.success else lp.x
    def fun(f):return float(np.sum((bj-Bj@f)**2))
    fit=minimize(fun,force0,jac=lambda f:-2*Bj.T@(bj-Bj@f),method='SLSQP',bounds=[(0,None)]*len(force0),constraints=[dict(type='eq',fun=lambda f:baseA[selected]@f-baseb[selected],jac=lambda f:baseA[selected])],options={'maxiter':100,'ftol':1e-12})
    f=fit.x if fit.success else force0;t=bj-Bj@f;counts,command,actual=invert(t,q)
    oldmotor=torque(count_for(np.array(candidate['geometry']['command'])),q)
    output.update(status='ACTIVE_MOTOR_ONLY_FEASIBLE' if constrained.success else 'ACTIVE_MOTOR_ONLY_INFEASIBLE',friction_not_used=True,
        constrained_message=constrained.message,least_norm_force_N=f.tolist(),root_balance_residual_max=float(np.abs(baseA@f-baseb).max()),required_torque_Nm=t.tolist(),
        old_candidate_motor_torque_at_desired_pose=oldmotor.tolist(),candidate_torque_deficit=(oldmotor-t).tolist(),
        lower_available_motor_torque=low.tolist(),upper_available_motor_torque=high.tolist(),
        torque_balance_wire_counts=counts.tolist(),torque_balance_action_targets=command.tolist(),quantized_motor_torque=actual.tolist(),
        wire_action_inside_original_limits=((command>=limits[:,0])&(command<=limits[:,1])).tolist(),
        minimum_torque_active_force_N=constrained.x.tolist() if constrained.success else None,
        minimum_torque_optimization_success=bool(fit.success))
    return output
result=dict(status='STATIC_TORQUE_AND_WARP_SUPPORT_AUDIT_COMPLETE',formal_controls=0,diagnostic_controls=0,simulation_integration_steps=0,
    model_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),contact_adapter_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),home_sole=home,
    actual_CPU_Warp_contacts=[dict(geom=m.geom(x['geom']).name,point=x['point'].tolist(),dist_m=x['dist']) for x in contacts],
    desired_pose_sole_geometry=[dict(geom=m.geom(i).name,z_range_m=[float(vertices(i)[sole[i],2].min()),float(vertices(i)[sole[i],2].max())]) for i in feet],
    joint_order=names,nominal_motor_stall_Nm=gain,observer_mapping=mapping.tolist(),
    analyses=[analyze(points,load,label) for label,points in [('active_Warp_foot_contacts',contacts),('projected_true_HOME_sole_corners_relaxed_support',projected)] for load in [0,.03]],
    limitations=['Zero-velocity static force balance with vertical reactions only; no tangential forces or contact-pressure reconstruction.',
      'Active-motor feasibility without relying on static friction is stronger than mathematical static equilibrium with friction; failure does not prove task impossibility.',
      'Nominal CAD mass/ipos only; saved GPU startup mass DR not substituted.','Projected sole corners are a relaxed geometric support bound, not active contacts.',
      'These inverse commands are diagnostic calculations, not dynamically validated candidates.'])
out=R/'groundpick-static-torque-006.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');patch.close()
print(json.dumps(dict(status=result['status'],home_sole=home,contacts=result['actual_CPU_Warp_contacts'],analyses=[{k:v for k,v in x.items() if k in ['kind','payload_kg','status','points','full_root_residual_max','solver_success','message']} for x in result['analyses']]),indent=2))
