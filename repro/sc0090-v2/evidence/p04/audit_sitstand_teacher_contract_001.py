import json,re,hashlib,ast,os
from pathlib import Path
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'sitstand-short-lesson-worktree';T=R/'rise-head-micro-exposure-worktree'
a=json.loads((R/'head2-micro-exposure-allocation-001/run.json').read_text());b=json.loads((R/'sitstand-lesson4-allocation-001/run.json').read_text())
def norm(v):
 if isinstance(v,dict):return {k:norm(x) for k,x in v.items()}
 if isinstance(v,list):return [norm(x) for x in v]
 if isinstance(v,str):
  v=re.sub(r' at 0x[0-9a-f]+','',v)
  for r in (W,T):v=v.replace(str(r),'<SOURCE>')
 return v
at=a['environment']['observations']['actor']['terms'];bt=b['environment']['observations']['actor']['terms'];assert tuple(at)==tuple(bt)
assert all(norm(at[k])==norm(bt[k]) for k in at if k!='body_command')
assert bt['body_command']['params']=={'dim':6} and 'zero_command_padding' in bt['body_command']['func']
assert norm(a['environment']['actions'])==norm(b['environment']['actions'])
assert a['environment']['scene']['entities']['robot']['init_state']['joint_pos']==b['environment']['scene']['entities']['robot']['init_state']['joint_pos']=={'.*':0.}
core=[k for k in a['source_sha256'] if k.startswith('src/mjlab_microduck/sc0090_v2/') or k.startswith('src/mjlab_microduck/actuator/') or k in ('src/mjlab_microduck/custom_v2/fast_native.py','src/mjlab_microduck/custom_v2/head_exploration.py')]
changed=[k for k in core if a['source_sha256'][k]!=b['source_sha256'].get(k)];assert not changed,changed
xml='src/mjlab_microduck/custom_v2/assets/robot.xml';assert (W/xml).read_bytes()==(T/xml).read_bytes()
results={}
for stage in ('nominal','measured'):
 d=json.loads((R/f'sitstand-lesson4-0384-{stage}/report.json').read_text());results[stage]={k:[dict(tail_hold=x['tail_hold'],head_tail=x['head_step_tail_band_hold'],faults=x['faults'],tilt_over45=x['tilt_over45']) for x in v] for k,v in d['groups'].items()}
physical=json.loads((R/'sitstand-clear-pose-batch-002/report.json').read_text())['candidates'][0]
report=dict(status='PASS_EXPLICIT_OBSERVATION_ACTION_ADAPTER_NOT_PHYSICAL_TRANSFER',teacher_checkpoint=str(R/'head2-micro-exposure-allocation-001/checkpoint_001856.pt'),student_reference=str(R/'sitstand-lesson4-allocation-001/checkpoint_000384.pt'),task=b['task'],actor_width=61,actions=14,proprio_slots=[0,48],posture_slot=48,teacher_twist_slots=[48,51],head_slots=[51,55],body_slots=[55,61],teacher_adapter='Clone actual61 raw measured obs; set48:51=0 and55:61=0; preserve0:48 and51:55; use frozen1856 actor and its own normalizer. Never load its zero-twist normalizer into SitStand student.',action_contract_equal=True,home_zero=True,matched_core_files=len(core),matched_robot_xml_sha256=hashlib.sha256((W/xml).read_bytes()).hexdigest(),command_semantics={'actor_flag':'0stand/1sit; raw generated_commands, no scale/delay','alpha':'internal2s slew initialized from actual spawnheight, reward/critic only; not an extra actor phase input','dwell_s':[3.5,6.5],'stand_height':.112,'clear_sit_height':.075999},physics_differences={'teacher_solver':[10,20],'sitstand_solver':[30,50],'teacher_contact_capacity':35,'sitstand_contact_capacity':64,'both_dt':.005,'both_control_dt':.02},existing_lesson384=results,clear_seat_physical_evidence=physical,evidence_limits=['ClearSeat original3s physical audit saved self-contact/tilt/velocity/fault but not feet/headfound; not full new no-contact qualification.','Old128/128 failed linear interpolation was original conflicting SIT target, not a proven failure of every possibleClearSeat transition. No interpolation will be copied.','Static SIT action label only has physical evidence near sit endpoint; applying it from standing is an explicit reachability probe, not an already valid wholetransition teacher.','Posture flag separates goals, but alpha history remains internal; no claim all aliasing eliminated.'],one_probe={'worlds':64,'controls':38400,'seconds':12,'groups':'four initialstate×firstgoal combinations16each; first4s/reverse4s/return4s','stage':'nominal actualS0/O0','standing_action':'frozen1856 mean from adapted realobs','sitting_action':'qualified ClearSeat absolute14 vector withheadHOME; nojointinterpolation','head_commands':'zero thisminimalprobe','measurements':'original stable+separate bothfeet/headfound/self andheadHOME, 200Hzcamera/20msexposure perphase withhealthdenominators; actualobs/actions/targets saved','PPO_updates':0},source_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (W/'src/mjlab_microduck/tasks/mdp.py',W/'src/mjlab_microduck/custom_v2/sitstand_clear.py',W/'src/mjlab_microduck/custom_v2/sitstand_lesson.py',W/'scripts/evaluate_v2_sitstand.py')},formal_controls_added=0)
(R/'sitstand-teacher-contract-001.json').write_text(json.dumps(report,indent=2)+'\n');print(report['status'],report['matched_core_files'])
