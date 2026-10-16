from pathlib import Path
import json,mujoco,torch,numpy as np,hashlib
root=Path(__file__).resolve().parent;source=root/'roulade-repaired-prep-worktree';a=torch.load(root/'roulade-physical-003/initial-state.pt',weights_only=False,map_location='cpu');m=mujoco.MjModel.from_xml_path(str(source/'src/mjlab_microduck/custom_v2/assets/robot.xml'));d=mujoco.MjData(m)
rows=[]
for i,q in enumerate(a['initial_qpos'].numpy()):
 d.qpos[:]=q;d.qpos[:3]-=a['env_origins'][i].numpy();mujoco.mj_forward(m,d);cs=[]
 for c in d.contact[:d.ncon]:
  if c.dist<0 and m.geom_bodyid[c.geom1] and m.geom_bodyid[c.geom2]:cs.append(dict(geoms=[m.geom(int(c.geom1)).name,m.geom(int(c.geom2)).name],depth=float(-c.dist)))
 if cs:rows.append(dict(index=i,group=i%8,max_depth=max(c['depth'] for c in cs),q={m.joint(j).name:float(q[m.jnt_qposadr[j]]) for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE},contacts=cs))
r=dict(status='EXACT_NATIVE_RESET_CPU_REPLAY',source=str(root/'roulade-physical-003/initial-state.pt'),source_sha256=hashlib.sha256((root/'roulade-physical-003/initial-state.pt').read_bytes()).hexdigest(),failed_gate=[x for x in rows if x['max_depth']>=.001],all_self_contact=rows,policy_joint_order_expected=['left_hip_yaw','left_hip_roll','left_hip_pitch','left_knee','left_ankle','neck_pitch','head_pitch','head_yaw','head_roll','right_hip_yaw','right_hip_roll','right_hip_pitch','right_knee','right_ankle'])
(root/'roulade-actual-reset-failure-001.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['failed_gate'],indent=2));print('all self count',len(rows))
