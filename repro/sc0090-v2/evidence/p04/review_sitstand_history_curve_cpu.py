"""CPU-only review of already saved fixed-condition SitStand curve reports."""
from pathlib import Path
import json,hashlib,re
import torch
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');nodes=(64,128,192,256,320);results={};pairing={};config={}
for condition in ('nominal','measured'):
 reference=None;reference_cfg=None;pairing[condition]={}
 for node in nodes:
  out=R/f'sitstand-history-base-{node:04d}-{condition}';rep=json.loads((out/'report.json').read_text());assert rep['numerical_source_verified']and rep['actor_normalizer_unchanged']
  trace=torch.load(out/'traces.pt',map_location='cpu',weights_only=False);initial=trace['initial'];del trace
  if reference is None:reference=initial
  pairing[condition][str(node)]={k:dict(exact=torch.equal(v,reference[k]),max_abs=float((v-reference[k]).abs().max()))for k,v in initial.items()}
  text=(out/'eval-config.json').read_text();normalized=re.sub(r' at 0x[0-9a-fA-F]+','',text)
  if reference_cfg is None:reference_cfg=normalized
  config[f'{condition}_{node}']=dict(equal_to_first=normalized==reference_cfg,sha256=hashlib.sha256(normalized.encode()).hexdigest())
  groups={}
  for name,phases in rep['groups'].items():
   groups[name]=[]
   for phase in phases:
    summary={k:phase[k]for k in('trials','tail_hold','faults','head_step_trials','head_step_reached','head_step_tail_band_hold','tilt_over45','head_tail_mae_deg')}
    summary['camera']=phase['camera'];groups[name].append(summary)
  results[f'{condition}_{node}']=dict(report=str(out/'report.json'),report_sha256=hashlib.sha256((out/'report.json').read_bytes()).hexdigest(),groups=groups)
report=dict(status='SAVED_CURVE_REVIEW_COMPLETE',formal_controls=125829120,all_sitstand_formal_controls=352321536,diagnostic_controls=10*256*600,initial_pairing=pairing,config_comparison=config,results=results,all_initial_exact=all(v['exact']for stage in pairing.values()for point in stage.values()for v in point.values()),all_eval_configs_equal_within_condition=all(v['equal_to_first']for v in config.values()),scope='Same fixed play conditions and command clock; actor+normalizer change by training. Early64/128 were observations only, not recipe efficacy rejection; no extra sim/PPO.')
(R/'sitstand-history-base-curve-review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in ('results','initial_pairing','config_comparison')},indent=2))
