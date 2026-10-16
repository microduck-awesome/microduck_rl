"""One offline audit of existing immutable physical head traces. No simulation."""
from pathlib import Path
from collections import defaultdict
import json,hashlib,numpy as np
B=Path(__file__).resolve().parent
sources={'phase_control':B/'head2-timeout-phase-1984-head-measured/report.json','phase_accuracy':B/'head2-phase-accuracy-1984-head-measured/report.json'}

def scores(error,cmd):
 active=np.abs(cmd)>1e-9;count=active.sum(-1);band=np.maximum(.1*np.abs(cmd),.002)
 norm=np.abs(error)/band;axis=1/(1+norm)
 mean=np.where(count>0,(axis*active).sum(-1)/np.maximum(count,1),axis.mean(-1))
 worst=np.where(count>0,np.where(active,norm,0).max(-1),norm.max(-1));minimum=np.where(count>0,1/(1+worst),axis.mean(-1))
 old=3*(.5*np.exp(-(error/.5)**2)+.5*np.exp(-(error/.04)**2)).mean(-1)
 return mean,minimum,old,worst
# Existing mean can rank persistent failure above simultaneous success.
cmd=np.array([[.195,.195,0,0]],dtype=np.float64);band=np.maximum(.1*abs(cmd),.002)
fail=np.zeros_like(cmd);fail[0,1]=1.2*band[0,1];ok=np.zeros_like(cmd);ok[0,:2]=.9*band[0,:2]
f=scores(fail,cmd);s=scores(ok,cmd)
assert f[3][0]>1 and s[3][0]<1 and f[0][0]>s[0][0] and f[1][0]<s[1][0]
synthetic=dict(command=cmd.tolist(),failing_errors_rad=fail.tolist(),passing_errors_rad=ok.tolist(),failure={k:float(v[0]) for k,v in zip(['mean_accuracy','worst_accuracy','old_head2','worst_normalized_error'],f)},success={k:float(v[0]) for k,v in zip(['mean_accuracy','worst_accuracy','old_head2','worst_normalized_error'],s)})
models={};case_models={}
for label,path in sources.items():
 report=json.loads(path.read_text());summary_path=path.parent.with_name(path.parent.name.replace('-head-measured','-head-summary'))/'report.json';expected=json.loads(summary_path.read_text())['reports'][0]['summary'];groups=defaultdict(list);examples=[];counts=defaultdict(int);case_rows=[]
 for seed in sorted({r['seed'] for r in report['records']}):
  rows=[r for r in report['records'] if r['seed']==seed];tp=path.parent/f'traces-{seed}.npz'
  with np.load(tp) as d:
   q=d['physical_head_rad'].astype(np.float64);command=d['command_rad'].astype(np.float64);bad=d['invalid'].astype(bool);stable=d['health'][...,3].astype(bool)
  targets=np.concatenate([np.repeat(np.asarray(row['target'],dtype=np.float32)[None],row['trials'],axis=0) for row in rows]).astype(np.float64)
  assert np.array_equal(command[100:200],np.repeat(targets[None],100,axis=0))
  mean,worst,old,norm=scores(q[100:200]-command[100:200],command[100:200]);tail=slice(75,100)
  eligible=stable[75:100].all(0)&stable[175:200].all(0)&~bad.any(0)
  passed=(norm[tail]<=1).all(0)&~bad[100:200].any(0);persistent_fail=(norm[tail]>1).all(0)&~bad[100:200].any(0)
  m=mean[tail].mean(0);w=worst[tail].mean(0);g=old[tail].mean(0);oldstack=g+.5*m;newstack=g+.5*w
  offset=0
  for row in rows:
   n=row['trials'];idx=np.arange(offset,offset+n);offset+=n;active=sum(abs(v)>1e-9 for v in row['target'])
   if not active:continue
   key=f'step/{row["range"]}/{active}-axis';values=dict(seed=seed,name=row['name'],group=key,trials=n,tail_hold=int(passed[idx].sum()),eligible=int(eligible[idx].sum()),tail_mean_accuracy=float(m[idx].mean()),tail_worst_accuracy=float(w[idx].mean()),tail_old_head2=float(g[idx].mean()),tail_common_mean_stack=float(oldstack[idx].mean()),tail_common_worst_stack=float(newstack[idx].mean()),tail_worst_normalized_error_mean=float(norm[tail,idx].mean()))
   groups[key].append(values);case_rows.append(values)
   if active<2:continue
   good=idx[eligible[idx]&passed[idx]];failidx=idx[eligible[idx]&persistent_fail[idx]]
   for fi in failidx:
    for gi in good:
     counts['persistent_fail_vs_success_pairs']+=1
     if m[fi]>m[gi]+1e-6:counts['mean_bonus_wrong_order']+=1
     if oldstack[fi]>oldstack[gi]+1e-6:
      counts['old_head2_plus_mean_bonus_wrong_order']+=1
      corrected=newstack[fi]<newstack[gi]-1e-6
      counts['wrong_head_stack_order_corrected_by_worst_bonus']+=int(corrected)
      examples.append(dict(seed=seed,name=row['name'],group=key,fail_trial=int(fi-idx[0]),success_trial=int(gi-idx[0]),fail_mean=float(m[fi]),success_mean=float(m[gi]),fail_worst=float(w[fi]),success_worst=float(w[gi]),fail_mean_stack=float(oldstack[fi]),success_mean_stack=float(oldstack[gi]),fail_worst_stack=float(newstack[fi]),success_worst_stack=float(newstack[gi]),corrected=bool(corrected)))
 for key,rs in groups.items():assert sum(x['tail_hold'] for x in rs)==expected[key]['final_half_s_hold_pass'],(label,key)
 totals={key:dict(trials=sum(x['trials'] for x in rs),tail_hold=sum(x['tail_hold'] for x in rs),eligible=sum(x['eligible'] for x in rs),**{field:sum(x[field]*x['trials'] for x in rs)/sum(x['trials'] for x in rs) for field in ['tail_mean_accuracy','tail_worst_accuracy','tail_old_head2','tail_common_mean_stack','tail_common_worst_stack','tail_worst_normalized_error_mean']}) for key,rs in groups.items()}
 models[label]=dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),groups=totals,ranking_counts=dict(counts),examples=examples[:50],case_rows=case_rows,original_hold_metric_parity=True)
 case_models[label]={(x['seed'],x['name']):x for x in case_rows}
reversals=[]
for key,a in case_models['phase_control'].items():
 z=case_models['phase_accuracy'][key]
 if z['tail_hold']<a['tail_hold'] and z['tail_mean_accuracy']>a['tail_mean_accuracy']:
  reversals.append(dict(seed=key[0],name=key[1],group=a['group'],control=a,accuracy=z,head_substack_also_increases=z['tail_common_mean_stack']>a['tail_common_mean_stack']))
out=dict(status='OFFLINE_COMPLETE',scope='Stateless old Head2 tracking + proposed/actual independent accuracy term from saved physical reward-boundary trajectories. Excludes body, action and stateful bias rewards; cannot establish full-return ranking or intervention efficacy.',limitation='No new physics, no source/model mutation. Within-case success/failure trials have different random initial perturbations. Cross-model cases match commands/seeds, not full captured initial state.',synthetic=synthetic,models=models,cross_model_cases_mean_bonus_rises_while_tail_hold_falls=reversals,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
path=B/'head2-active-axis-aggregation-audit.json';assert not path.exists();path.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(synthetic=synthetic,models={k:dict(ranking_counts=v['ranking_counts'],groups=v['groups']) for k,v in models.items()},cross_model_reversals=len(reversals)),indent=2))
