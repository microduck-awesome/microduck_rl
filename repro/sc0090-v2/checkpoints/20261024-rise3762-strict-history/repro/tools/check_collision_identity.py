"""CPU model identity and contact support check; no policy initialization or learning."""
from pathlib import Path
import json,hashlib,mujoco,numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261024-strict3762/sources');relative=Path('src/mjlab_microduck/custom_v2/assets/robot.xml')
old=mujoco.MjModel.from_xml_path(str(BASE/'.static-cache'/relative));new=mujoco.MjModel.from_xml_path(str(BASE/'standup-allocation-001'/relative))
fields=['body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat','body_parentid','jnt_pos','jnt_axis','jnt_range','jnt_qposadr','qpos0','actuator_trnid','actuator_ctrlrange','actuator_gear']
for field in fields:np.testing.assert_array_equal(getattr(old,field),getattr(new,field),err_msg=field)
geom_fields=['geom_bodyid','geom_type','geom_pos','geom_quat','geom_size','geom_contype','geom_conaffinity','geom_condim','geom_friction','geom_solref','geom_solimp']
for i in range(old.ngeom):
 name=old.geom(i).name;assert name;other=new.geom(name).id
 for field in geom_fields:np.testing.assert_array_equal(getattr(old,field)[i],getattr(new,field)[other],err_msg=name+':'+field)
assert new.ngeom-old.ngeom==32
spec=mujoco.MjSpec.from_file(str(BASE/'standup-allocation-001'/relative));spec.worldbody.add_geom(name='audit_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[2,2,.1]);m=spec.compile();d=mujoco.MjData(m);rows=[]
for name,axis,angle in [('supine',[0,1,0],-np.pi/2),('prone',[0,1,0],np.pi/2),('side_positive',[1,0,0],np.pi/2),('side_negative',[1,0,0],-np.pi/2),('rear_tilt',[0,1,0],-np.pi/3)]:
 mujoco.mj_resetData(m,d);d.qpos[2]=.25;quat=np.zeros(4);mujoco.mju_axisAngle2Quat(quat,np.array(axis,dtype=float),angle);d.qpos[3:7]=quat;seen=set();final=[]
 for step in range(600):
  mujoco.mj_step(m,d);assert np.isfinite(d.qpos).all()
  for ci,c in enumerate(d.contact):
   names=[m.geom(int(g)).name for g in c.geom]
   if 'audit_floor'in names:
    seen.update(names)
    if step==599:
     force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force);final.append(dict(geoms=names,distance_m=float(c.dist),force_N=float(force[0])))
 if name!='prone':assert any(any(g.startswith('sim2real_shell_')for g in x['geoms'])and x['force_N']>0 for x in final),name
 rows.append(dict(pose=name,contact_geoms=sorted(seen),final_contacts=final))
report=dict(status='PASS',model_fields_exact=fields,existing_geom_fields_exact=geom_fields,old_geom_count=old.ngeom,new_geom_count=new.ngeom,mass_kg=float(new.body_mass.sum()),cases=rows,scope='CPU compiled identity and passive contact support only; not policy performance',formal_updates=0)
(ROOT/'collision-identity-review.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items()if k!='cases'})
