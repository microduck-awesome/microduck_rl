import json,hashlib,ast
from pathlib import Path
import numpy as np,mujoco
r=Path(__file__).resolve().parent;s=r/'roulade-repaired-prep-worktree';prior=json.loads((r/'roulade-cfg-cpu-001.json').read_text());m=mujoco.MjModel.from_xml_path(str(s/'src/mjlab_microduck/custom_v2/assets/robot.xml'));d=mujoco.MjData(m)
names=next(ast.literal_eval(x.value) for x in ast.parse((s/'src/mjlab_microduck/custom_v2/microduck_v2.py').read_text()).body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='POLICY_JOINTS' for t in x.targets));qids=[int(m.joint(n).qposadr[0]) for n in names];rng=np.random.default_rng(2610);target=np.array(prior['tuck']);depth=[]
for i in range(256):
 blend=rng.uniform(.3,1.);noise=rng.normal(0,.08,14);q=blend*target+noise.clip(-.04,.04);d.qpos[:]=m.qpos0;d.qpos[2]=1.;d.qpos[qids]=q;mujoco.mj_forward(m,d);depth.append(max([-float(c.dist) for c in d.contact[:d.ncon] if c.dist<0 and m.geom_bodyid[c.geom1] and m.geom_bodyid[c.geom2]],default=0.))
assert max(depth)==0.,max(depth)
prior.update(bounded_noise=.04,static_noisy_tuck_cases=256,max_self_penetration_m=max(depth),evidence='Same256 CPU RNG targets/draws as001; only noise.clip±.04 changes. Full config inheritance unchanged except new clip parameter and metadata; GPU verifies actual resolved clip.')
prior['updated_source_sha256']={str(p.relative_to(s)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [s/'src/mjlab_microduck/tasks/mdp.py',s/'src/mjlab_microduck/custom_v2/roulade.py']};(r/'roulade-cfg-cpu-002.json').write_text(json.dumps(prior,indent=2)+'\n');print('PASS same256 bounded resets max self depth',max(depth))
