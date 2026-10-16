"""Original three-cycle gates + supplementary200Hz contactscreen; CPUonly."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,hashlib,time,importlib.util,math
import torch
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'groundpick-target-precision-worktree';A=R/'groundpick-target-precision-allocation-001';D=Path('/home/l40-vm01/Documents/ai-research/microduck/microduck-v2-p04/docs');torch.set_num_threads(2);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
status=json.loads((R/'groundpick-target-precision-pipeline-status-002.json').read_text());assert status['status']=='GROUNDPICK_TARGET_PRECISION_COMPLETE_NO_AUTOMATIC_CONTINUATION'
formal=json.loads((A/'report.json').read_text());assert formal['completed_updates']==768 and formal['allocation_controls']==150994944
q=json.loads((R/'groundpick-target-precision-qualification-001.json').read_text());assert all(sha(W/k)==v for k,v in q['source_sha256'].items())
steps=[json.loads(x)for x in(A/'native-optimizer-step-callback.jsonl').read_text().splitlines()];assert len(steps)==256 and all(x['optimizer_step_calls']==120 and x['total_optimizer_step_calls']==120*(i+1)for i,x in enumerate(steps))
spec=importlib.util.spec_from_file_location('ground_camera_cpu',W/'src/mjlab_microduck/custom_v2/head_metrics.py');head=importlib.util.module_from_spec(spec);spec.loader.exec_module(head)
def cam(values,mask):
 if not bool(mask.any()):return None
 return {k:dict(mean_over_trials=float(v[mask].mean()),p95_over_trials=float(torch.quantile(v[mask],.95)),max_over_trials=float(v[mask].max()))for k,v in values.items()}
rows=[];refs={};configs={};pairs={};camera_diffs=[]
for node in [512,576,640,704,768]:
 for stage in ['nominal','measured']:
  p=R/(f'groundpick-precision-reference-0512-{stage}' if node==512 else f'groundpick-target-precision-{node:04d}-{stage}');report=json.loads((p/'report.json').read_text());trace=torch.load(p/'traces.pt',map_location='cpu',weights_only=False);m=trace['measurements'];contact=trace['head_self_found_200Hz'];assert contact.shape==(2400,256,2)
  import re
  configtext=(p/'config.json').read_text().replace(str(R/'groundpick-health-history-worktree'),str(W))
  current=json.loads(re.sub(r' at 0x[0-9a-fA-F]+',' at <address>',configtext));initial=trace['initial']
  actual_params=current['rewards']['mouth_ground_proximity']['params'];assert (actual_params['std'],actual_params['target_height'])==((.1,0.)if node==512 else(.025,.020))
  current['rewards']['mouth_ground_proximity']['params']['std']=.1;current['rewards']['mouth_ground_proximity']['params']['target_height']=0.
  if stage not in refs:refs[stage]={k:v.clone()for k,v in initial.items()};configs[stage]=current
  pair={k:torch.equal(v,refs[stage][k])for k,v in initial.items()};assert all(pair.values());assert current==configs[stage];pairs[f'{node}-{stage}']=pair
  cfree=~contact.reshape(600,4,256,2).any((1,3));healthy=m['finite']&m['feedback_ready']&~m['head_contact']&~m['self_contact']&m['feet'].all(-1);motion=healthy&(m['tilt']<math.pi/3);quiet=motion&(m['linear'].norm(dim=-1)<.05)&(m['omega']<.5);standbody=quiet&((m['height']-.112).abs()<.012)&(m['tilt']<math.pi/12)
  group_results=[];totals=[dict(cycle=i,trials=256,low=0,stand=0,clean=0,complete=0,head1deg=0,fault=0,strict200Hz_low=0,strict200Hz_stand=0,strict200Hz_complete=0)for i in range(3)]
  for group in range(8):
   sel=torch.arange(256)%8==group;rr=report['results'][group];outcycles=[]
   for cycle in range(3):
    lo=cycle*200;hi=lo+200;phase=m['phase'][lo:hi,sel];lowphase=(phase>=.375)&(phase<.425);low=m['low_goal'][lo:hi,sel]&lowphase;attained=low.unfold(0,5,1).all(-1).any(0);stand=m['standing_goal'][hi-25:hi,sel].all(0);clean=healthy[lo:hi,sel].all(0);fault=(~m['feedback_ready'][lo:hi,sel]).any(0);complete=attained&stand&clean&~fault
    saved=rr['cycles'][cycle]
    for key,want in [('low_goal_sustained_100ms',attained),('return_stand_tail_500ms',stand),('clean_cycle',clean),('complete_cycle',complete),('feedback_fault_trials',fault)]:assert int(want.sum())==saved[key]
    strictlow=(low&cfree[lo:hi,sel]).unfold(0,5,1).all(-1).any(0);strictstand=(m['standing_goal'][hi-25:hi,sel]&cfree[hi-25:hi,sel]).all(0);strictclean=clean&cfree[lo:hi,sel].all(0);strictcomplete=strictlow&strictstand&strictclean&~fault
    row=dict(saved_original=saved,head_found_200Hz_trials=int(contact[lo*4:hi*4,sel,0].any(0).sum()),self_found_200Hz_trials=int(contact[lo*4:hi*4,sel,1].any(0).sum()),supplementary_contactscreen=dict(low=int(strictlow.sum()),stand=int(strictstand.sum()),complete=int(strictcomplete.sum()),scope='Adds recorded200Hz contact screening; original50Hz bodyvelocity/feet gates retained, no newacceptance threshold'),camera_checked={})
    for name,p0,p1,hm in [('descent',0.,.375,motion),('low_hold',.375,.425,quiet),('rise',.425,.8,motion),('stand_hold',.8,1.,standbody)]:
     indices=((phase[:,0]>=p0)&(phase[:,0]<p1)).nonzero().flatten()
     if len(indices)<2:assert saved['camera'][name]is None;continue
     first,last=lo+int(indices[0]),lo+int(indices[-1])+1;values=head.camera_metrics(trace['camera_quat_w'][first*4:last*4+1,sel],dt=.005,exposure_s=.02);original=hm[first:last,sel].all(0);strict=original&cfree[first:last,sel].all(0);parity=cam(values,original);record=saved['camera'][name]
     assert int(original.sum())==record['healthy_entire_window_trials']
     if parity is not None:
      for k in parity:
       for v in parity[k]:
        expected=record['healthy_entire_window'][k][v];actual=parity[k][v];camera_diffs.append(dict(node=node,stage=stage,group=group,cycle=cycle,phase=name,metric=k,aggregate=v,cpu=actual,gpu=expected,abs_error=abs(actual-expected),relative_error=abs(actual-expected)/max(abs(expected),1e-12)))
     else:assert record['healthy_entire_window']is None
     row['camera_checked'][name]=dict(original_healthy_trials=int(original.sum()),contactscreen_healthy_trials=int(strict.sum()),contactscreen_camera=cam(values,strict),raw_all_original=record['all_trials'])
    outcycles.append(row);totals[cycle]['low']+=int(attained.sum());totals[cycle]['stand']+=int(stand.sum());totals[cycle]['clean']+=int(clean.sum());totals[cycle]['complete']+=int(complete.sum());totals[cycle]['head1deg']+=saved['head_return_all_axes_1deg_hold'];totals[cycle]['fault']+=int(fault.sum());totals[cycle]['strict200Hz_low']+=int(strictlow.sum());totals[cycle]['strict200Hz_stand']+=int(strictstand.sum());totals[cycle]['strict200Hz_complete']+=int(strictcomplete.sum())
   group_results.append(dict(payload_kg=rr['payload_kg'],push=rr['training_interval_pushes'],cycles=outcycles))
  rows.append(dict(node=node,stage=stage,initial_exact_with512=pair,config_exact_except_two_reward_params=True,cycle_totals=totals,groups=group_results,report_sha256=sha(p/'report.json'),runtime_qualified=False,hardware_tested=False));del trace,m,contact

worst=sorted(camera_diffs,key=lambda x:x['abs_error'],reverse=True);out=dict(status='CAMERA_NUMERICAL_PARITY_DIAGNOSIS_ONLY',comparisons=len(camera_diffs),over_old_absolute_tolerance=sum(x['abs_error']>2e-5 for x in camera_diffs),worst_absolute=worst[:30],worst_relative=sorted(camera_diffs,key=lambda x:x['relative_error'],reverse=True)[:30]);(R/'groundpick-precision-camera-parity-audit-001.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k not in ['worst_absolute','worst_relative']}));print(json.dumps(worst[:5],indent=2))
