"""Original three-cycle gates + supplementary200Hz contactscreen; CPUonly."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,hashlib,time,importlib.util,math
import torch
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'groundpick-health-history-worktree';A=R/'groundpick-health-history-allocation-001';D=Path('/home/l40-vm01/Documents/ai-research/microduck/microduck-v2-p04/docs');torch.set_num_threads(2);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
status=json.loads((R/'groundpick-health-history-pipeline-status.json').read_text());assert status['status']=='GROUNDPICK_HEALTH_HISTORY_COMPLETE_NO_AUTOMATIC_CONTINUATION'
formal=json.loads((A/'report.json').read_text());assert formal['completed_updates']==512 and formal['allocation_controls']==301989888
q=json.loads((R/'groundpick-health-history-qualification-001.json').read_text());assert all(sha(W/k)==v for k,v in q['source_sha256'].items())
steps=[json.loads(x)for x in(A/'native-optimizer-step-callback.jsonl').read_text().splitlines()];assert len(steps)==512 and all(x['optimizer_step_calls']==120 and x['total_optimizer_step_calls']==120*(i+1)for i,x in enumerate(steps))
spec=importlib.util.spec_from_file_location('ground_camera_cpu',W/'src/mjlab_microduck/custom_v2/head_metrics.py');head=importlib.util.module_from_spec(spec);spec.loader.exec_module(head)
def cam(values,mask):
 if not bool(mask.any()):return None
 return {k:dict(mean_over_trials=float(v[mask].mean()),p95_over_trials=float(torch.quantile(v[mask],.95)),max_over_trials=float(v[mask].max()))for k,v in values.items()}
rows=[];refs={};configs={};pairs={}
for node in [128,256,384,512]:
 for stage in ['nominal','measured']:
  p=R/f'groundpick-health-history-{node:04d}-{stage}';report=json.loads((p/'report.json').read_text());trace=torch.load(p/'traces.pt',map_location='cpu',weights_only=False);m=trace['measurements'];contact=trace['head_self_found_200Hz'];assert contact.shape==(2400,256,2)
  current=json.loads((p/'config.json').read_text());initial=trace['initial']
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
     if parity is not None:assert all(abs(parity[k][v]-record['healthy_entire_window'][k][v])<=2e-5 for k in parity for v in parity[k])
     else:assert record['healthy_entire_window']is None
     row['camera_checked'][name]=dict(original_healthy_trials=int(original.sum()),contactscreen_healthy_trials=int(strict.sum()),contactscreen_camera=cam(values,strict),raw_all_original=record['all_trials'])
    outcycles.append(row);totals[cycle]['low']+=int(attained.sum());totals[cycle]['stand']+=int(stand.sum());totals[cycle]['clean']+=int(clean.sum());totals[cycle]['complete']+=int(complete.sum());totals[cycle]['head1deg']+=saved['head_return_all_axes_1deg_hold'];totals[cycle]['fault']+=int(fault.sum());totals[cycle]['strict200Hz_low']+=int(strictlow.sum());totals[cycle]['strict200Hz_stand']+=int(strictstand.sum());totals[cycle]['strict200Hz_complete']+=int(strictcomplete.sum())
   group_results.append(dict(payload_kg=rr['payload_kg'],push=rr['training_interval_pushes'],cycles=outcycles))
  rows.append(dict(node=node,stage=stage,initial_exact_with128=pair,config_exact_with128=True,cycle_totals=totals,groups=group_results,report_sha256=sha(p/'report.json'),runtime_qualified=False,hardware_tested=False));del trace,m,contact
wall=json.loads((A/'shared-process-phase-wall.json').read_text());live=[json.loads(x)for x in(A/'curriculum-live.jsonl').read_text().splitlines()]
out=dict(status='GROUNDPICK_FULL_STAGE_CLOSED_NO_AUTOMATIC_CONTINUATION',new_formal_controls=301989888,prior_formal_controls=75497472,all_history_formal_controls=377487360,actual_optimizer_steps=61440,clock=formal['common_step_counter'],full_formal_shared_phase_wall=wall,steady_throughput=formal['throughput'],full_worker_including_admission_eval_wait_s=status['worker_wall_s'],actual_live_managers_and_reset_exposure=live,rows=rows,initial_pairing=pairs,scope='Original12s3cycle battery thresholds preserved. Cycles share worlds; counts not independenttrials. New200Hzcontact screens supplement50Hzoriginalhealth; headcameraactual200Hz/20msexposure, notcalibratedpixelblur.',limitations=['Traininglowgoal includes lowresetstarts, not independentfullcycle mastery.','Originalactor61stillphaseencoded realfeedback only; no fixedcommandteacher used.','Nominal/measured effects not isolated intervention unless allotherconditionspaired; primarycomparisons arewithincondition acrossnodes.','Fullformalphase timing includesenvbuild but imports shared; workerwall alsoincludesCPU/wait/qualification/8evaluations.'],source_unchanged=True)
(R/'groundpick-health-history-final-review.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
lines=['\n## 固定512与原协议四节点最终曲线\n','| 节点 | 条件 | 3周期低位/各256 | 3周期回站/各256 | 完整/各256 | fault/各256 |','|---:|---|---|---|---|---|']
for row in rows:lines.append('|'+ '|'.join([str(row['node']),row['stage'],str([x['low']for x in row['cycle_totals']]),str([x['stand']for x in row['cycle_totals']]),str([x['complete']for x in row['cycle_totals']]),str([x['fault']for x in row['cycle_totals']])])+'|')
lines+=['',f"正式新增301,989,888/实际61,440 Adam，Ground历史377,487,360，预算关闭不自动续训。完整formal阶段{wall['seconds_including_environment_and_validation']:.3f}s/{wall['controls_per_second']:.2f}controls/s；稳态median{formal['throughput']['median_control_per_second']:.2f}/aggregate{formal['throughput']['aggregate_control_per_second']:.2f}。worker含资格/等待/8末评另列{status['worker_wall_s']:.3f}s。",'','各组四载荷/推扰、原gate parity、初态配置逐位核验、200Hz额外head/self筛查及相机健康分母见`groundpick-health-history-final-review.json`。不以静态008或训练reset命中替代完整周期能力，所有候选保留、实机未验证。']
p=D/'sc0090-v2-groundpick-health-history.md';text=p.read_text();old=text.split('\n')[2];assert old.startswith('当前：');text=text.replace(old,'当前：固定fresh512和原12s双条件四节点末评已全部完成，正式301,989,888新增、Ground全历史377,487,360；预算关闭，无自动续训。最终原门槛曲线及头稳/接触分母见本文末及`groundpick-health-history-final-review.json`。',1);p.write_text(text+'\n'.join(lines)+'\n')
p=D/'sc0090-v2-groundpick-agent.md';text=p.read_text();old=text.split('\n')[0];assert old.startswith('当前：');text=text.replace(old,'当前：HealthHistory512完整学习与原12s双条件四节点评测结束，Ground正式全历史377,487,360，预算关闭；完整能力/头稳结论见[固定阶段最终曲线](sc0090-v2-groundpick-health-history.md)。',1);p.write_text(text)
print(json.dumps({'status':out['status'],'rows':[{k:v for k,v in row.items()if k in ['node','stage','cycle_totals']}for row in rows]},indent=2))
