from pathlib import Path
import json,hashlib,re,statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261024-strict3762');OUT=BASE/'final-evaluation';REP=BASE/'regression-replay'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canon(x):
 if isinstance(x,dict):return {k:canon(v) for k,v in x.items()}
 if isinstance(x,list):return [canon(v) for v in x]
 if isinstance(x,str):return re.sub(r' at 0x[0-9a-fA-F]+(?=>)','',x)
 return x
summary=dict(schema=1,status='EVALUATED_WITH_LIMITATIONS',model_checks=read(OUT/'artifact-verification.json'),comparison_config_identity={},recovery={},head={},joint_limits={},regression={},limitations=['Development-seed simulation evaluation, not held-out or hardware qualification','Exact configuration/history replay does not promise bitwise trajectories or equal learned performance','Large-angle head stability and transient physical limit penetration remain material limitations'])
for kind,seeds in [('recovery',[2026101501,2026101502]),('head',[2026101503,2026101504])]:
 checks=[]
 for seed in seeds:
  a=OUT/f'{kind}-new'/f'eval-config-{seed}.json';b=OUT/f'{kind}-old'/f'eval-config-{seed}.json';assert canon(read(a))==canon(read(b)),(kind,seed)
  checks.append(dict(seed=seed,effective_config_equal_ignoring_callable_addresses=True,new_config_sha256=sha(a),old_config_sha256=sha(b)))
 summary['comparison_config_identity'][kind]=checks
neck={}
for label in ['new','old']:
 recovery=read(OUT/f'recovery-{label}/report.json');head=read(OUT/f'head-{label}/report.json');limits=read(OUT/f'joint-limits-{label}.json')
 assert recovery['checkpoint_sha256']==head['checkpoint_sha256']
 groups={name:dict(wins=sum(c['groups'][name]['wins'] for c in recovery['cases']),trials=sum(c['groups'][name]['trials'] for c in recovery['cases'])) for name in recovery['cases'][0]['groups']}
 summary['recovery'][label]=dict(checkpoint_sha256=head['checkpoint_sha256'],groups=groups,wins=sum(g['wins'] for g in groups.values()),trials=sum(g['trials'] for g in groups.values()),report_sha256=sha(OUT/f'recovery-{label}/report.json'))
 ranges={}
 for group in ['zero','small','current','final']:
  rows=[r for r in head['records'] if r['range']==group]
  ranges[group]=dict(records=len(rows),steady_mae_deg=[statistics.mean(r['step']['steady_mae_deg_mean'][i] for r in rows) for i in range(4)])
 selected=[r for r in head['records'] if r['name']=='neck_pitch_final_+1'];assert len(selected)==2
 data={'q':[],'health':[],'command':[]}
 for seed in [2026101503,2026101504]:
  rows=[r for r in head['records'] if r['seed']==seed];idx=next(i for i,r in enumerate(rows) if r['name']=='neck_pitch_final_+1');sl=slice(idx*8,(idx+1)*8)
  with np.load(OUT/f'head-{label}'/f'traces-{seed}.npz') as trace:
   data['q'].append(trace['physical_head_rad'][:,sl,0].copy());data['health'].append(trace['health'][:,sl].copy());data['command'].append(trace['command_rad'][:,sl,0].copy())
 data={k:np.concatenate(v,axis=1) for k,v in data.items()};neck[label]=data;h=data['health'][100:200];tilt=np.degrees(h[...,1]);severe=((h[...,0]<.08)|(tilt>45)).any(0)
 summary['head'][label]=dict(checkpoint_sha256=head['checkpoint_sha256'],axis_order=head['joint_order'],current_ranges=head['current_ranges'],final_ranges=head['final_ranges'],range_averages=ranges,neck_plus63=dict(target_deg=float(1.1*180/np.pi),steady_mae_deg_mean=statistics.mean(r['step']['steady_mae_deg_mean'][0] for r in selected),per_seed=[dict(seed=r['seed'],mae_deg=r['step']['steady_mae_deg_mean'][0],stable_fraction=r['step']['stable_fraction'],height_min_m=r['step']['height_min_m'],tilt_p95_deg=r['step']['tilt_p95_deg']) for r in selected],pooled_stable_fraction=float(h[...,3].mean()),pooled_height_min_m=float(h[...,0].min()),pooled_tilt_p95_deg=float(np.quantile(tilt,.95)),trials=16,severe_low_posture_trials=int(severe.sum()),severe_low_posture_definition='Any step with trunk height<0.08m or tilt>45deg during the two-second command; descriptive diagnostic, not a new training gate'),report_sha256=sha(OUT/f'head-{label}/report.json'))
 joints=[]
 for row in limits['seeds']:
  records=[r for r in head['records'] if r['seed']==row['seed']]
  for j in row['joints']:joints.append(dict(j,seed=row['seed'],case=records[j['worst_env']//8]['name']))
 worst=min(joints,key=lambda j:j['min_clearance_deg']);summary['joint_limits'][label]=dict(minimum_clearance_deg=worst['min_clearance_deg'],worst=worst,max_near_half_degree_duration_s=max(j['longest_within_half_degree_s'] for j in joints),scope=limits['scope'],report_sha256=sha(OUT/f'joint-limits-{label}.json'))
assert summary['head']['new']['current_ranges']==summary['head']['old']['current_ranges'] and summary['head']['new']['final_ranges']==summary['head']['old']['final_ranges']
assert np.array_equal(neck['new']['command'],neck['old']['command'])
legacy=read(Path('/home/l40-vm01/data2/experiments/microduck-v2-20261023/rise3762-collision/release/rise3762-new-collision-20261024/evidence/head-new.json'))
summary['previous_adaptive_6804_reference']=dict(checkpoint_sha256=legacy['checkpoint_sha256'],neck_plus63_steady_mae_deg_mean=statistics.mean(r['step']['steady_mae_deg_mean'][0] for r in legacy['records'] if r['name']=='neck_pitch_final_+1'),scope='Archived prior evaluation, not rerun in this comparison; not a single-factor causal attribution')
# Raw trajectory view: means and full trial envelopes, not confidence intervals.
fig,axes=plt.subplots(4,1,figsize=(9,10),sharex=True);t=(np.arange(300)+1)*.02
for label,color,title in [('old','#2468b4','Old 3762'),('new','#d65a1c','Strict-history 3762')]:
 d=neck[label];arrays=[np.degrees(d['q']),d['health'][...,0]*100,np.degrees(d['health'][...,1]),d['health'][...,3]*100]
 for ax,a in zip(axes,arrays):
  ax.plot(t,a.mean(1),color=color,label=title,lw=1.8)
  if ax is not axes[-1]:ax.fill_between(t,a.min(1),a.max(1),color=color,alpha=.12)
axes[0].plot(t,np.degrees(neck['new']['command'][:,0]),'k--',label='Command',lw=1.2)
for ax,label in zip(axes,['Neck offset (deg)','Trunk height (cm)','Trunk tilt (deg)','Stable fraction (%)']):
 ax.set_ylabel(label);ax.grid(alpha=.2);ax.axvspan(2,4,color='gray',alpha=.07)
axes[0].legend(loc='upper left',fontsize=9);axes[-1].set_xlabel('Time (s)');axes[-1].set_xlim(0,6);axes[-1].set_ylim(-3,103)
fig.suptitle('+63 degree neck command: tracking and body stability\n16 trials, same two seeds and corrected collision; bands show min-max')
fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/'neck63-comparison.png',dpi=170);fig.savefig(OUT/'neck63-comparison.svg');plt.close(fig)
# Recovery replay localizes the failed motion stage without inferring a reward cause.
replay_series={};initial={}
for n in [2250,2500,2750]:
 report=read(REP/str(n)/'report.json');series={'height':[],'tilt':[],'stable':[]};rows=[]
 for seed in [2026101501,2026101502]:
  meta=read(REP/str(n)/f'metadata-{seed}.json');sl=slice(640,768)
  with np.load(REP/str(n)/f'traces-{seed}.npz') as a:
   q=a['qpos'][:,sl].copy();qv=a['qvel'][:,sl].copy();hold=a['stable_hold'][:,sl].copy();init=a['initial_qpos'].copy();origins=a['env_origins'][sl].copy()
  if n==2250:initial[seed]=init
  diff=float(np.max(np.abs(init-initial[seed])));assert diff==0
  z=q[...,2]-origins[None,:,2];w,x,y,zz=np.moveaxis(q[...,3:7],-1,0);tilt=np.degrees(np.arccos(np.clip(1-2*(x*x+y*y),-1,1)));dofs=[j['dof_address'] for j in meta['policy_joints']]
  series['height'].append(z);series['tilt'].append(tilt);series['stable'].append(hold>=25)
  rows.append(dict(seed=seed,initial_qpos_max_difference_from2250=diff,max_stable_hold_controls=int(hold.max()),final_height_median_m=float(np.median(z[-25:])),final_tilt_median_deg=float(np.median(tilt[-25:])),last2s_joint_speed_rms_rad_s=float(np.sqrt(np.mean(qv[-100:,:,dofs]**2))),trace_sha256=meta['trace_sha256']))
 replay_series[n]={k:np.concatenate(v,axis=1) for k,v in series.items()}
 native=read(Path(report['checkpoint']).parent/f'evaluation_{n:06d}.json');differences=[]
 for a,z in zip(report['cases'],native['cases']):
  for name,g in a['groups'].items():
   for key in ('wins','trials','any_stable','faults'):
    if g[key]!=z['groups'][name][key]:differences.append(dict(seed=a['seed'],bucket=name,field=key,native=z['groups'][name][key],replay=g[key]))
 summary['regression'][str(n)]=dict(right_wins=sum(c['groups']['right_side']['wins'] for c in report['cases']),right_trials=256,total_wins=sum(g['wins'] for c in report['cases'] for g in c['groups'].values()),all_counts_exactly_match_native=report['same_counts_as_native_reference'],other_count_differences=differences,seeds=rows,report_sha256=sha(REP/str(n)/'report.json'))
fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True);t=(np.arange(400)+1)*.02
for n,color in [(2250,'#2468b4'),(2500,'#ca3433'),(2750,'#2e8b57')]:
 for ax,key,scale in zip(axes,['height','tilt','stable'],[100,1,100]):
  a=replay_series[n][key].astype(float)*scale;ax.plot(t,np.median(a,axis=1) if key!='stable' else a.mean(1),color=color,label=str(n),lw=1.8)
  if key!='stable':ax.fill_between(t,np.quantile(a,.1,axis=1),np.quantile(a,.9,axis=1),color=color,alpha=.12)
for ax,label in zip(axes,['Trunk height (cm)','Trunk tilt (deg)','Stable fraction (%)']):ax.set_ylabel(label);ax.grid(alpha=.2)
axes[0].legend(title='Checkpoint');axes[-1].set_xlabel('Time (s)');axes[-1].set_xlim(0,8);axes[-1].set_ylim(-3,103)
fig.suptitle('Right-side recovery replay: identical initial poses\n256 trials per checkpoint; median and 10-90 percentile bands')
fig.tight_layout(rect=(0,0,1,.93));fig.savefig(OUT/'regression-right-side.png',dpi=170);fig.savefig(OUT/'regression-right-side.svg');plt.close(fig)
summary['normalizer_ablation']={}
for network,norm in [(2250,2500),(2500,2250)]:
 p=BASE/'normalizer-ablation'/f'network{network}_norm{norm}'/'report.json';d=read(p);assert d['hybrid_policy_not_saved_or_trained']
 summary['normalizer_ablation'][f'network{network}_norm{norm}']=dict(right_wins=sum(c['groups']['right_side']['wins'] for c in d['cases']),right_trials=256,report_sha256=sha(p),diagnostic_only=True)
summary['regression_interpretation']=dict(observation='2500 stays near5cm trunk height and90deg tilt throughout right-side recovery; 2250/2750 stand from identical initial poses',localization='Failure tracks the2500 network parameters under both tested normalizers. Restoring2250 normalization does not rescue it;2250 network with2500 normalization succeeds.',not_proven='The exact reward term or gradient sequence causing the parameter change is not isolated. This is a recovered, reproducible regression, not ordinary sample-count noise.',no_rollback_or_rescue_training=True)
summary['analysis_script_sha256']=sha(Path(__file__))
(OUT/'comparison-summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({'new_recovery':summary['recovery']['new']['wins'],'old_recovery':summary['recovery']['old']['wins'],'new_neck63':summary['head']['new']['neck_plus63'],'old_neck63':summary['head']['old']['neck_plus63'],'normalizer_ablation':summary['normalizer_ablation']},indent=2))
