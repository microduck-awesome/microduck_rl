"""CPU-only review of already saved fixed-condition SitStand curve reports."""
from pathlib import Path
import json,hashlib,re
import torch
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');nodes=(320,448,576,640);results={};pairing={};config={}
assets=json.loads((R/'sitstand-history-hold320-worktree/assets-equality.json').read_text());assert assets['status']=='EXACT' and assets['count']==283
for row in assets['assets']:
 for root in ('sitstand-history-base-worktree','sitstand-history-hold320-worktree'):
  assert hashlib.sha256((R/root/row['path']).read_bytes()).hexdigest()==row['sha256']
for condition in ('nominal','measured'):
 reference=None;reference_cfg=None;pairing[condition]={}
 for node in nodes:
  out=R/(f'sitstand-history-base-{node:04d}-{condition}' if node==320 else f'sitstand-history-hold320-{node:04d}-{condition}');rep=json.loads((out/'report.json').read_text());assert rep['numerical_source_verified']and rep['actor_normalizer_unchanged']
  trace=torch.load(out/'traces.pt',map_location='cpu',weights_only=False);initial=trace['initial'];del trace
  if reference is None:reference=initial
  pairing[condition][str(node)]={k:dict(exact=torch.equal(v,reference[k]),max_abs=float((v-reference[k]).abs().max()))for k,v in initial.items()}
  text=(out/'eval-config.json').read_text();normalized=re.sub(r' at 0x[0-9a-fA-F]+','',text).replace('sitstand-history-hold320-worktree','sitstand-history-base-worktree')
  if reference_cfg is None:reference_cfg=normalized
  config[f'{condition}_{node}']=dict(equal_to_first=normalized==reference_cfg,sha256=hashlib.sha256(normalized.encode()).hexdigest())
  groups={}
  for name,phases in rep['groups'].items():
   groups[name]=[]
   for phase in phases:
    summary={k:phase[k]for k in('trials','tail_hold','faults','head_step_trials','head_step_reached','head_step_tail_band_hold','tilt_over45','head_tail_mae_deg')}
    summary['camera']=phase['camera'];groups[name].append(summary)
  results[f'{condition}_{node}']=dict(report=str(out/'report.json'),report_sha256=hashlib.sha256((out/'report.json').read_bytes()).hexdigest(),groups=groups)
report=dict(status='SAVED_CURVE_REVIEW_COMPLETE',formal_controls=125829120,all_sitstand_formal_controls=478150656,diagnostic_controls_new=6*256*600,initial_pairing=pairing,config_comparison=config,results=results,all_initial_exact=all(v['exact']for stage in pairing.values()for point in stage.values()for v in point.values()),all_eval_configs_equal_within_condition=all(v['equal_to_first']for v in config.values()),scope='Same fixed play conditions and command clock; actor+normalizer change by training. Parent320 compared to448/576/640 using saved reports only. Config normalization substitutes only isolated worktree root after283noncode assets exact and removes function addresses; raw config files preserved. No extra sim/PPO.')
(R/'sitstand-history-hold320-curve-review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in ('results','initial_pairing','config_comparison')},indent=2))
