from pathlib import Path
import json,mujoco,torch,collections
R=Path(__file__).resolve().parent;torch.set_num_threads(2)
m=mujoco.MjModel.from_binary_path(str(R/'ballkick-contact-witness-model-002.mjb'));d=mujoco.MjData(m)
def self_contacts(q):
 d.qpos[:]=q;d.qvel[:]=0;mujoco.mj_forward(m,d)
 return [(m.geom(int(c.geom[0])).name,m.geom(int(c.geom[1])).name,float(c.dist))for c in d.contact if m.geom(int(c.geom[0])).name.startswith('robot/')and m.geom(int(c.geom[1])).name.startswith('robot/')]
result={}
for stage in ('nominal','measured'):
 t=torch.load(R/f'ballkick-standing-handoff-{stage}-001/traces.pt',weights_only=False,map_location='cpu');initial=t['initial']['qpos'].numpy();x=t['measurements'];rows=[];pairs=collections.Counter()
 for i,q in enumerate(initial):
  contacts=self_contacts(q);pairs.update((a,b)for a,b,_ in contacts)
  rows.append(dict(world=i,contacts=contacts))
 j=[]
 for idx in range(m.njnt):
  adr=int(m.jnt_qposadr[idx]);name=m.joint(idx).name
  if name.startswith('robot/') and int(m.jnt_type[idx])==int(mujoco.mjtJoint.mjJNT_HINGE):
   j.append(dict(joint=name,initial=float(initial[0,adr]),after_first_control=float(x['raw_qpos'][0,0,adr]),after_second_control=float(x['raw_qpos'][1,0,adr])))
 result[stage]=dict(initial_self_trials=sum(bool(x['contacts'])for x in rows),initial_worlds=len(rows),pairs=[dict(pair=k,contact_points=v)for k,v in pairs.items()],first_world_joints=j,initial_rows=rows)
(R/'ballkick-initial-self-review-001.json').write_text(json.dumps(dict(status='INITIAL_SELF_CONTACT_AUDITED',results=result,scope='All saved initial qpos reconstructed in same immutable scene geometry with CPU MuJoCo; no dynamics integration and no initial-state repair.'),indent=2)+'\n')
print(json.dumps({k:{x:v[x]for x in ('initial_self_trials','pairs','first_world_joints')}for k,v in result.items()},indent=2))
