"""CPU-only action/measurement/strict-gate review of four fixed diagnostics."""
from pathlib import Path
import torch,math,json,hashlib,importlib.util
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'roulade-history-worktree';torch.set_num_threads(2);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((R/'roulade-history-late-teacher-plan-001.json').read_text());assert json.loads((R/'roulade-history-late-teacher-status-002.json').read_text())['status']=='FOUR_LATE_LANDING_DIAGNOSTICS_COMPLETE'
weights={k:torch.load(plan[k],map_location='cpu',weights_only=False)['actor_state_dict']for k in ['student','teacher']}
for k in weights:
 run=json.loads((Path(plan[k]).parent/'run.json').read_text());assert run['agent']['actor']['activation']=='elu' and run['agent']['actor']['hidden_dims']==[512,256,128]
spec=importlib.util.spec_from_file_location('camera_cpu',W/'src/mjlab_microduck/custom_v2/head_metrics.py');head=importlib.util.module_from_spec(spec);spec.loader.exec_module(head)
def inference(raw,sd):
 x=(raw-sd['obs_normalizer._mean'])/(sd['obs_normalizer._std']+.01)
 for i in [0,2,4,6]:
  x=torch.nn.functional.linear(x,sd[f'mlp.{i}.weight'],sd[f'mlp.{i}.bias'])
  if i!=6:x=torch.nn.functional.elu(x)
 return x

def summarize(values,mask):
 if not bool(mask.any()):return None
 return {k:dict(mean_over_trials=float(v[mask].mean()),p95_over_trials=float(torch.quantile(v[mask],.95)),max_over_trials=float(v[mask].max()))for k,v in values.items()}
loaded={};rows=[];legs=[0,1,2,3,4,9,10,11,12,13]
for arm in plan['arms']:
 for stage in plan['conditions']:
  p=R/f'roulade-history-late-teacher-{arm}-{stage}-001';report=json.loads((p/'report.json').read_text());x=torch.load(p/'traces.pt',map_location='cpu',weights_only=False);loaded[arm,stage]=x;m=x['measurements'];g=x['gate_snapshot'];mask=x['eligible_at2p5s']
  expected=g['head_latch']&(g['frontier']>=math.radians(330))&(g['tilt']<math.pi/4)&(g['height']>.04)&(g['omega']<2)&g['feedback_ready']&g['finite'];assert torch.equal(mask,expected)
  assert bool((x['actor_obs_preaction'][:,:,48:]==0).all())
  previous_error=float((x['actor_obs_preaction'][1:,:,34:48]-x['actions'][:-1]).abs().max());assert previous_error<1e-6
  route=x['student_actions'].clone()
  if arm=='late_teacher':route[125:]=torch.where(mask[None,:,None],x['teacher_actions'][125:],route[125:])
  assert torch.equal(route,x['actions'])
  sample_steps=[0,25,50,75,100,124,125,130,150,175,200,225,249];sample=x['actor_obs_preaction'][sample_steps].reshape(-1,61);parity={}
  for role,key in [('student','student_actions'),('teacher','teacher_actions')]:
   actual=inference(sample,weights[role]);want=x[key][sample_steps].reshape(-1,14);error=float((actual-want).abs().max());assert error<2e-5,(role,error);parity[role]=error
  finite=m['finite'].all(0);feedback=m['feedback_ready'].all(0)
  body=m['feet'].all(-1)&~m['head_contact']&~m['self_contact']&((m['height']-.112).abs()<.012)&(m['tilt']<math.pi/12)&(m['linear'].norm(dim=-1)<.05)&(m['omega'].norm(dim=-1)<.5)&(m['physical_q'][:,:,legs].abs().mean(-1)<.2)&m['feedback_ready']&m['finite'];neutral=body&(m['physical_q'][:,:,5:9].abs().amax(-1)<.1);assert torch.equal(neutral,m['stable']);land=body[-25:].all(0);home=neutral[-25:].all(0)
  contacts=x['contacts200'];assert contacts.shape==(1000,64,4);strictcontact=contacts[-100:,:,:2].all((0,2))&~contacts[-100:,:,2:].any((0,2));motion=m['omega'][:,:,1].abs()>.5;motion&=m['support'].int().cumsum(0)>0;support=(m['support']&motion).sum(0)/motion.sum(0).clamp_min(1);flat=torch.where(motion,m['lateral_axis_z'].abs(),0).amax(0)
  geometric=(x['geometric_unwrapped_rad'][-1]+x['initial']['spawn_angle'])>=math.radians(345);eligible=m['head_latch'][-1]&(m['frontier'][-1]>=math.radians(330));core=land&eligible&geometric&finite&feedback&(support>=.95)&(flat<math.sin(math.pi/4))
  camera=head.camera_metrics(x['camera_quat_w'][-101:]);groups=[]
  for i in range(5):
   select=torch.arange(64)%8<4 if i==0 else torch.arange(64)%8==i+3;original=report['results'][i]
   for key,want in [('core_completion',core),('strict_neutral_head_completion',core&home),('body_landing_tail500ms',land),('neutral_head_landing_tail500ms',home)]:assert int(want[select].sum())==original[key]
   count=dict(group='standing'if i==0 else 'mid'+str([90,180,270,330][i-1]),trials=int(select.sum()),eligible_at2p5s=int((mask&select).sum()),tail_body=int((land&select).sum()),tail_neutral_head=int((home&select).sum()),tail_body_neutral_200Hz_clean_feet=int((home&strictcontact&select).sum()),original_core=int((core&select).sum()),original_strict_core=int((core&home&select).sum()),post_handoff_tilt_over45=int((m['tilt'][125:,select]>math.pi/4).any(0).sum()),post_handoff_min_height_m=float(m['height'][125:,select].min()),tail_head_mae_deg=torch.rad2deg(m['physical_q'][-25:,select,5:9].abs().mean((0,1))).tolist(),tail_head_force200_mean_N=float(x['head_force200'][-100:,select].mean()),camera_all=summarize(camera,select),camera_healthy_trials=int((select&home&strictcontact).sum()),camera_healthy=summarize(camera,select&home&strictcontact))
   groups.append(count)
  rows.append(dict(arm=arm,stage=stage,groups=groups,independent_cpu_action_max_error=parity,previous_executed_action_input_max_error=previous_error,feedback_faults=int((~feedback).sum()),nonfinite=int((~finite).sum()),report_sha256=sha(p/'report.json'),original_report=str(p/'report.json')))
pairs=[]
for stage in plan['conditions']:
 a,b=loaded['policy',stage],loaded['late_teacher',stage];eq={k:torch.equal(v,b['initial'][k])for k,v in a['initial'].items()};assert all(eq.values());pairs.append(dict(stage=stage,initial_exact=eq,prefix_max_abs_qpos=float((a['preaction_qpos'][:125]-b['preaction_qpos'][:125]).abs().max()),prefix_max_abs_actor_obs=float((a['actor_obs_preaction'][:125]-b['actor_obs_preaction'][:125]).abs().max()),eligible_set_equal=torch.equal(a['eligible_at2p5s'],b['eligible_at2p5s'])))
out=dict(status='FOUR_LATE_LANDING_DIAGNOSTICS_CPU_REVIEWED',diagnostic_controls=64000,new_formal_controls=0,rows=rows,pairs=pairs,scope='Teacher reachability only; original standingorigin fullroll gates remain. Midroll starts retain explicit priorangle/latch. External teacher intervention is not trainedpolicy or deployment. No terminal accepts missingfeedback. CPU actionparity includes each ownnormalizer and actualpreviousactions.',limitations=['FiniteN64 diagnostic; not formal skillacceptance.','Initialexactness alone doesnotprove prefix trajectory equality.','200Hz tailcontactscreen supplements original50Hz gates; rollphaseheadcontact is intended.','A local teacher can be useful without satisfying fullroll; no substitution of success definition.'],script_sha256=sha(Path(__file__)))
(R/'roulade-history-late-teacher-closeout-001.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'rows':[{k:v for k,v in row.items()if k!='original_report'}for row in rows]},indent=2))
