"""Independent saved-result review of the fixed64 teacher/control continuation."""
from pathlib import Path
import hashlib,json,math
import torch
r=Path(__file__).resolve().parent;output=r/'roulade-teacher-control-0320-review.json';assert not output.exists()
torch.set_num_threads(2)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
result=dict(status='FIXED_PAIR_COMPLETE_SKILL_UNQUALIFIED',additional_formal_controls=50331648,whole_roulade_controls=150994944,conditions={},training={},evidence={})
for arm in ('teacher','control'):
 p=r/f'roulade-{arm}-allocation-001/report.json';d=read(p)
 assert d['status']=='ALLOCATION_COMPLETE' and d['completed_updates']==320 and d['allocation_controls']==25165824
 result['training'][arm]={k:d[k] for k in ('completed_updates','allocation_controls','throughput','terminal_bootstrap_samples','contact_peak','contact_capacity','training_metrics')}
 result['evidence'][str(p)]=sha(p)
for stage in ('nominal','measured'):
 traces={};reports={};condition={}
 for arm in ('parent256','teacher','control'):
  name=f'roulade-endpoint-0256-{stage}' if arm=='parent256' else f'roulade-{arm}-0320-{stage}'
  reports[arm]=read(r/name/'report.json');traces[arm]=torch.load(r/name/'traces.pt',map_location='cpu',weights_only=False)
  result['evidence'][str(r/name/'report.json')]=sha(r/name/'report.json')
  result['evidence'][str(r/name/'traces.pt')]=sha(r/name/'traces.pt')
  m=traces[arm]['measurements'];initial=traces[arm]['initial'];idx=torch.arange(len(initial['qpos']));rows=[]
  for group,reported in enumerate(reports[arm]['results']):
   select=idx%8<4 if group==0 else idx%8==group+3;healthy=m['body_stable'][-25:,select].all(0);head=m['stable'][-25:,select].all(0)
   assert int(healthy.sum())==reported['body_landing_tail500ms'] and int(head.sum())==reported['neutral_head_landing_tail500ms']
   physical=m['physical_q'][-25:,select,5:9]
   rows.append(dict(**reported,
       head_home_tail_mae_body_healthy_deg=torch.rad2deg(physical[:,healthy].abs().mean((0,1))).tolist() if bool(healthy.any()) else None,
       initial_root_height_m=float((initial['qpos'][select,2]-initial['origins'][select,2]).mean()),
       tail_root_height_m=float(m['height'][-25:,select].mean()),tail_tilt_deg=float(torch.rad2deg(m['tilt'][-25:,select]).mean()),
       body_healthy_and_frontier330=int((healthy&(m['frontier'][-1,select]>=math.radians(330))).sum())))
  condition[arm]=rows
 exact={}
 for key,left in traces['teacher']['initial'].items():
  right=traces['control']['initial'][key]
  exact[key]=dict(exact=torch.equal(left,right),maximum_absolute_difference=float((left.float()-right.float()).abs().max()))
 condition['teacher_control_initial_comparison']=exact
 result['conditions'][stage]=condition
result['scope']='One fixed64 continuation per arm from original256; no validation checkpoint used. Same standing/midroll5s evaluation and unchanged healthy/rotation/support/head definitions. Initial equality is checked, not presumed. Teacher body stability is not complete roll or neutral head. Midroll inherited latch/progress never proves standing-origin roll.'
output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({stage:{arm:[{k:row[k] for k in ('origin','spawn_pitch_deg','core_completion','body_landing_tail500ms','neutral_head_landing_tail500ms','supported_frontier_deg_mean','head_home_tail_mae_body_healthy_deg','body_healthy_and_frontier330')} for row in data[arm]] for arm in ('teacher','control')} for stage,data in result['conditions'].items()},indent=2))
