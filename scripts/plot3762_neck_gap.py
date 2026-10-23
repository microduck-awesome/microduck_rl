"""Compare recorded, same-seed physical head/stance traces without new rollouts."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT.parent/'neck-gap-diagnostic';OUT.mkdir(exist_ok=True)
paths={'old3762':ROOT.parent/'head-stage1/head-old','new5524':ROOT.parent/'head-stage4/head'}
fig,axes=plt.subplots(3,2,figsize=(12,8),sharex=True);summary={}
for col,name in enumerate(['neck_pitch_current_+1','neck_pitch_current_-1']):
 summary[name]={}
 for label,path in paths.items():
  report=json.loads((path/'report.json').read_text());cases=[r for r in report['records']if r['seed']==2026101503];index=next(i for i,r in enumerate(cases)if r['name']==name);repeats=cases[index]['trials'];ids=slice(index*repeats,(index+1)*repeats)
  data=np.load(path/'traces-2026101503.npz');q=np.rad2deg(data['physical_head_rad'][:,ids,0]);cmd=np.rad2deg(data['command_rad'][:,ids,0]);tilt=np.rad2deg(data['health'][:,ids,1]);rate=data['health'][:,ids,2];t=(np.arange(q.shape[0])+1)*.02
  if label=='old3762':axes[0,col].plot(t,cmd.mean(1),'k--',label='target')
  for row,values in enumerate((q,tilt,rate)):
   axes[row,col].plot(t,values.mean(1),label=label);axes[row,col].fill_between(t,values.min(1),values.max(1),alpha=.12)
  summary[name][label]=dict(target_deg=float(cmd[150].mean()),physical_neck_final_step_deg=float(q[175:200].mean()),final_step_body_tilt_deg=float(tilt[175:200].mean()),final_step_body_rate_rad_s=float(rate[175:200].mean()),all_trials_valid=not bool(data['invalid'][:,ids].any()))
 axes[0,col].set_title(name);axes[2,col].set_xlabel('time (s)')
for row,title in enumerate(['physical neck angle (deg)','body tilt (deg)','body angular rate (rad/s)']):
 axes[row,0].set_ylabel(title)
 for ax in axes[row]:ax.grid(alpha=.3);ax.legend()
fig.suptitle('Same corrected collision model, same seed, 8 trials; bands show min/max')
fig.tight_layout();fig.savefig(OUT/'neck-response.png',dpi=150);fig.savefig(OUT/'neck-response.svg')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
