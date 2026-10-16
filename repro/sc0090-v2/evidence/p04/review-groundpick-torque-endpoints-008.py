"""Close endpoint diagnostics against unchanged saved physical acceptance gates."""
from pathlib import Path
import hashlib,json,math,os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import torch
R=Path(__file__).resolve().parent;torch.set_num_threads(2)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[];initials={}
for label,directory in [('torque007','groundpick-torque-balanced-hold-007'),('margin008_nominal','groundpick-torque-margin-hold-nominal-008'),('margin008_measured','groundpick-torque-margin-hold-measured-008')]:
    p=R/directory;d=torch.load(p/'traces.pt',map_location='cpu',weights_only=False);x=d['measurements'];initials[label]=d['initial']
    components=dict(ready=x['feedback_ready'],finite=x['finite'],no_head=~x['head_contact'],no_self=~x['self_contact'],both_feet=x['feet'].all(-1),
        linear=x['linear'].norm(dim=-1)<.05,omega=x['omega']<.5,mouth_low=x['mouth_height']>=0,mouth_high=x['mouth_height']<=.025,
        mouth_down=x['mouth_down_cos']>=.95,tilt=x['tilt']<math.radians(60))
    independent=torch.stack(list(components.values())).all(0);assert torch.equal(independent,x['low_goal'])
    failure_rows=[]
    for i in torch.nonzero(~independent[100:].all(0)).flatten().tolist():
        failure_rows.append(dict(world=i,payload_kg=float(x['payload_kg'][-1,i]),failed_tail_components=[k for k,v in components.items() if not bool(v[100:,i].all())],
            mouth_tail_range_m=[float(x['mouth_height'][100:,i].min()),float(x['mouth_height'][100:,i].max())]))
    report=json.loads((p/'report.json').read_text())
    rows.append(dict(label=label,directory=str(p),report_sha256=sha(p/'report.json'),trace_sha256=sha(p/'traces.pt'),
        formal_controls=0,diagnostic_controls=report['diagnostic_controls'],original_gate_recomputed_exact=True,
        full3s_control_sample_head_contact_trials=int(x['head_contact'].any(0).sum()),
        full3s_control_sample_self_contact_trials=int(x['self_contact'].any(0).sum()),
        original_results=report['results'],failure_rows=failure_rows,
        scope='Contacts and task gates sampled at original50Hz after-reward boundary; camera recorded200Hz. No claim substep contact history was captured.'))
a,b=initials['margin008_nominal'],initials['margin008_measured'];pair={k:dict(exact=torch.equal(a[k],b[k]),max_abs=float((a[k]-b[k]).abs().max()))for k in a}
result=dict(status='DYNAMIC_HEALTHY_LOW_ENDPOINT_DEMONSTRATED_WITH_MEASURED_HEIGHT_RESIDUAL',formal_controls_added=0,diagnostic_controls=28800,
    total_groundpick_formal_controls_unchanged=75497472,results=rows,margin008_between_condition_initials=pair,
    low_hold_summary={'nominal':64,'measured':61,'trials_per_condition':64},
    retained_static_pose=str(R/'groundpick-torque-balanced-margin-008.json'),
    limits='Only fixed-target low endpoint holds tested. Not a trained policy, complete down/up skill,all40g loads or hardware acceptance. Three measured empty trials exceed unchanged25mm mouthheight; keep failures.',
    next='Stop endpoint search; prepare bounded full-history GroundPick PPO with validated physical lowpose exposure and minimal contact-reward alignment. No truth or fixed controller injected into actor.',
    script_sha256=sha(Path(__file__)))
(R/'groundpick-torque-endpoints-closeout-008.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],summary=result['low_hold_summary'],failure_rows={x['label']:x['failure_rows'] for x in rows},pair=pair),indent=2))
