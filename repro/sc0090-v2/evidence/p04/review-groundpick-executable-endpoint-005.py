"""Read saved GPU evidence, identify actual contacts and equilibrium-model error."""
from pathlib import Path
import hashlib,json,math,os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import torch,mujoco,numpy as np
R=Path(__file__).resolve().parent;W=R/'groundpick-repaired-worktree';P=R/'groundpick-executable-endpoint-hold-005'
torch.set_num_threads(2)
d=torch.load(P/'traces.pt',map_location='cpu',weights_only=False);report=json.loads((P/'report.json').read_text());fit=json.loads((R/'groundpick-executable-endpoint-005.json').read_text())
xml=W/'src/mjlab_microduck/custom_v2/assets/robot.xml';spec=mujoco.MjSpec.from_file(str(xml));spec.worldbody.add_geom(name='audit_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[0,0,.1],contype=1,conaffinity=2)
m=spec.compile();md=mujoco.MjData(m);floor=m.geom('audit_floor').id
final=d['chain']['qpos'][-1].double().numpy();assert final.shape==(64,m.nq)
grouped={};rows=[]
for i,q in enumerate(final):
    md.qpos[:]=q;md.qpos[:3]-=d['initial']['origins'][i].numpy();md.qvel[:]=0;mujoco.mj_forward(m,md)
    contacts=[]
    for c in md.contact:
        if floor in c.geom or c.dist>=0:continue
        names=tuple(sorted(m.geom(int(g)).name for g in c.geom));row=dict(geoms=names,distance_m=float(c.dist));contacts.append(row)
        x=grouped.setdefault(names,dict(worlds=set(),distances=[]));x['worlds'].add(i);x['distances'].append(float(c.dist))
    rows.append(dict(world=i,payload_kg=float(d['measurements']['payload_kg'][-1,i]),self_contacts=contacts))
desired=np.array(fit['geometry']['q']);command=np.array(fit['geometry']['command']);results=[]
for payload in [0,.03]:
    selection=torch.arange(64)%2==int(payload>0);q=d['measurements']['physical_q'][100:150,selection].double().mean((0,1)).numpy()
    actual_goal=d['chain']['o0_physical_target'][100:150,selection,:14].double().mean((0,1)).numpy()
    results.append(dict(payload_kg=payload,actual_mean_q=q.tolist(),desired_q=desired.tolist(),command=command.tolist(),
        measured_error_from_desired_deg=np.rad2deg(q-desired).tolist(),
        measured_command_equilibrium_offset_deg=np.rad2deg(q-command).tolist(),
        native_quantized_physical_goal_mean=actual_goal.tolist(),
        physical_goal_minus_requested_command_rad=(actual_goal-command).tolist()))
out=dict(status='EXECUTABLE_STATIC_CANDIDATE_FAILED_REAL_HOLD',formal_controls_added=0,diagnostic_controls=9600,
    original_gates_unchanged=True,low_goal_tail1s=[x['low_goal_tail1s'] for x in report['results']],
    outcome_summary=[{k:x[k] for k in ['payload_kg','trials','low_goal_tail1s','head_contact_settled_trials','self_contact_settled_trials','fault_trials','nonfinite_trials']} for x in report['results']],
    contact_witnesses=[dict(geoms=list(k),worlds=len(v['worlds']),distance_range_m=[min(v['distances']),max(v['distances'])]) for k,v in grouped.items()],
    rows=rows,equilibrium_model_review=results,
    trace_sha256=hashlib.sha256((P/'traces.pt').read_bytes()).hexdigest(),model_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    decision='No healthy endpoint established; do not use this pose as a successful reverse-curriculum reset or teacher. No automatic further candidate or PPO. Static feasibility and a constant measured offset are insufficient to certify contact/actuator equilibrium.',
    scope='CPU forward at saved finalGPUqpos to identify self geometry only, not new dynamics or exact randomized force reconstruction.')
(R/'groundpick-executable-endpoint-review-005.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ['status','outcome_summary','contact_witnesses','equilibrium_model_review']},indent=2))
