"""CPU-only bounded final review; no inference, reward recomputation, or rollout."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,pathlib,hashlib,time,importlib.util,math
import torch
R=pathlib.Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'ballkick-target-precision-worktree'
A=R/'ballkick-target-precision-allocation-001';D=pathlib.Path('/home/l40-vm01/Documents/ai-research/microduck/microduck-v2-p04/docs')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
_spec=importlib.util.spec_from_file_location('recorded_head_metrics',W/'src/mjlab_microduck/custom_v2/head_metrics.py');_head=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_head)
def cam_aggregate(values,mask):
 if not bool(mask.any()):return None
 return {k:dict(mean_over_trials=float(v[mask].mean()),p95_over_trials=float(torch.quantile(v[mask],.95)))for k,v in values.items()}
status=json.loads((R/'ballkick-target-precision-pipeline-status.json').read_text())
assert status['status']=='BALLKICK_TARGET_PRECISION_COMPLETE_NO_AUTOMATIC_CONTINUATION',status
formal=json.loads((A/'report.json').read_text());run=json.loads((A/'run.json').read_text());wall=json.loads((A/'shared-process-phase-wall.json').read_text());qualification=json.loads((R/'ballkick-target-precision-qualification-001.json').read_text())
assert formal['allocation_controls']==150994944 and formal['completed_updates']==512
assert all(sha(W/k)==v for k,v in qualification['source_sha256'].items())
steps=[json.loads(x)for x in(A/'native-optimizer-step-callback.jsonl').read_text().splitlines()]
assert len(steps)==256 and all(x['optimizer_step_calls']==120 and x['total_optimizer_step_calls']==(i+1)*120 for i,x in enumerate(steps))
rows=[];refs={};pairing={};hashes={}
for node in [320,384,448,512]:
 for stage in ['nominal','measured']:
  p=R/f'ballkick-target-precision-{node:04d}-{stage}'
  report=json.loads((p/'report.json').read_text());trace=torch.load(p/'traces.pt',map_location='cpu',weights_only=False);m=trace['measurements'];raw=trace['physics_raw_contacts'];bits=trace['physics_kick_state']
  assert report['trials']==256 and raw.shape==(1000,256,4) and bits.shape==(1000,256,2)
  # No partial resets in this battery: bad monotonic, bad implies validfalse.
  assert not bool((bits[1:,:,1].long()<bits[:-1,:,1].long()).any())
  assert not bool((bits[:,:,0]&bits[:,:,1]).any())
  assert not bool((raw[:,:,3]&~bits[:,:,1]).any())
  assert not bool((raw[:,:,3]&bits[:,:,0]).any())
  assert int(raw[:,:,0].any(0).sum())==report['raw_correct_foot_touch_trials_200hz']
  assert int(bits[:,:,0].any(0).sum())==report['healthy_valid_ever_trials_200hz']
  init=trace['initial']
  if stage not in refs:refs[stage]={k:v.clone()for k,v in init.items()}
  paired={k:torch.equal(refs[stage][k],v)for k,v in init.items()}
  pairing[f'{node}-{stage}']=paired
  def first_summary(x):
   seen=x.any(0);idx=x.long().argmax(0)
   return dict(trials=int(seen.sum()),first_substep_index_median=float(idx[seen].double().median())if bool(seen.any())else None,scope='0-based native substep callback index, no interpolated physical onset')
  camera_checks={}
  motion=m['finite']&m['feedback_ready']&~m['head_contact']&~m['self_contact']&(m['height']>=.08)&(m['tilt']<=math.pi/4)
  for name,start,end,healthy in [('prepare_0_to_05s',0,25,motion),('motion_05_to_2s',25,100,motion),('return_4_to_5s',200,250,m['standing'])]:
   values=_head.camera_metrics(trace['camera_quat_w'][start*4:end*4+1],dt=.005,exposure_s=.02)
   original=healthy[start:end].all(0);strict=original&~raw[start*4:end*4,:,2:].any((0,2))
   parity=cam_aggregate(values,original);saved=report['camera'][name]['healthy_entire_window']
   assert int(original.sum())==report['camera'][name]['healthy_entire_window_trials']
   if parity is not None:
    assert saved is not None
    assert all(abs(parity[k][v]-saved[k][v])<=2e-5 for k in parity for v in parity[k]),name
   else:assert saved is None
   camera_checks[name]=dict(original50Hz_health_trials=int(original.sum()),extra200Hz_head_self_contact_rejected_trials=int((original&~strict).sum()),contact_screened_healthy_trials=int(strict.sum()),camera_contact_screened=cam_aggregate(values,strict),scope='Same original50Hz body/velocity/feet criteria, additionally no head/self found in saved200Hz window; original protocol results retained. This is not200Hz body-state certification.')
  row={k:report[k]for k in ['stage','trials','raw_correct_foot_touch_trials_200hz','healthy_valid_ever_trials_200hz','bad_latch_ever_trials_200hz','raw_self_touch_trials_200hz','valid_contact_trials','clean_motion_trials','rolling_target_last1s_trials','body_standing_last05s_trials','body_and_neutral_head_last05s_trials','complete_kick_trials','complete_kick_and_neutral_head_trials','feedback_fault_trials','nonfinite_trials','ball_forward_distance_mean_m','ball_tail_forward_speed_mean_mps','ball_tail_lateral_abs_mean_mps','head_tail_home_mae_deg','camera']}
  row.update(node=node,camera_cpu_parity_and_200Hz_contact_screen=camera_checks,first_raw_correct_contact=first_summary(raw[:,:,0]),first_self_contact=first_summary(raw[:,:,3]),first_bad_latch=first_summary(bits[:,:,1]),tail1s_height_m=float(m['height'][-50:].mean()),tail1s_tilt_deg=float(torch.rad2deg(m['tilt'][-50:]).mean()),tail1s_linear_speed_mps=float(m['linear'][-50:].norm(dim=-1).mean()),tail1s_angular_speed_radps=float(m['angular'][-50:].norm(dim=-1).mean()),tail1s_both_feet_fraction=float(m['feet'][-50:].all(-1).float().mean()),tail1s_standing_sample_fraction=float(m['standing'][-50:].float().mean()),initial_paired_with320=paired,qualification='UNQUALIFIED_UNLESS_ALL_ORIGINAL_GATES_PASS',runtime_qualified=False,hardware_tested=False)
  rows.append(row);hashes[str(p/'report.json')]=sha(p/'report.json');del trace,m,raw,bits
parent_rows=[];parent_pairing={}
for stage in ['nominal','measured']:
 p=R/f'ballkick-health-history-0256-{stage}'
 report=json.loads((p/'report.json').read_text());trace=torch.load(p/'traces.pt',map_location='cpu',weights_only=False)
 pair={k:torch.equal(trace['initial'][k],v)for k,v in refs[stage].items()};assert all(pair.values());parent_pairing[stage]=pair
 parent_rows.append({k:v for k,v in report.items()if k not in ['source_sha256']})
 del trace
out=dict(status='STAGE_CLOSED_NO_AUTOMATIC_CONTINUATION',recorded_unix=time.time(),formal_new_controls=150994944,previous_formal_controls=326762496,all_history_formal_controls=477757440,completed_updates=512,native_Adam_steps=len(steps)*120,native_clock=formal['common_step_counter'],full_shared_process_phase=wall,steady_throughput=formal['throughput'],worker_wall_including_admission_and_evaluation=status['worker_wall_s'],live_reward_weights_by_node=[json.loads(x)for x in(A/'curriculum-live.jsonl').read_text().splitlines()],rows=rows,initial_pairing=pairing,report_sha256=hashes,mature_original_parent=str(R/'ballkick-health-history-allocation-001/checkpoint_000256.pt'),original_parent256_results=parent_rows,parent256_initial_pairing=parent_pairing,source_unchanged=True,camera_metric_source_sha256=sha(W/'src/mjlab_microduck/custom_v2/head_metrics.py'),scope='Frozen deterministic native five-second battery. Rawcontact differs from supportedhealthy valid. Camera uses recorded200Hz head site and original healthy denominators; no hardware/complete-skill inference from a single component. Counts across nodes are repeated trials, not independent samples.',limitations=['Validation and formal budget separate; full-worker wall also includes qualification/evaluation.','Per-node healthy-camera denominators differ, so aggregate camera means are not causal matched-subset comparisons.','Historical official successful BallKick raw run/checkpoint not located; dose/curriculum based on actual template and explicit scaling.'])
f=R/'ballkick-target-precision-final-review.json';f.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
lines=['\n## 成熟精度256续段最终曲线（同原门槛）\n','| 节点 | 条件 | raw足触球 | 健康valid | clean | 滚速方向 | 身体保持 | 身体+头中性 | 完整 | fault |','|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:lines.append('| '+ ' | '.join(map(str,[r['node'],r['stage'],r['raw_correct_foot_touch_trials_200hz'],r['healthy_valid_ever_trials_200hz'],r['clean_motion_trials'],r['rolling_target_last1s_trials'],r['body_standing_last05s_trials'],r['body_and_neutral_head_last05s_trials'],r['complete_kick_trials'],r['feedback_fault_trials']]))+' |')
lines+=['',f"新增正式150,994,944，累计477,757,440；实际30,720次Adam、nativeclock{formal['common_step_counter']}。完整formal共享进程阶段{wall['seconds_including_environment_and_validation']:.3f}s/{wall['controls_per_second']:.2f}controls/s；稳态median{formal['throughput']['median_control_per_second']:.2f}、aggregate{formal['throughput']['aggregate_control_per_second']:.2f}。完整worker含资格和8次末评另列{status['worker_wall_s']:.3f}s，不混为学习吞吐。","",'各条件320/384/448/512的原始qpos/qvel/actor_obs/origins配对核验、200Hz首次触球/自碰时刻和分窗口相机RMS/20ms曝光及健康分母见 `p04/ballkick-target-precision-final-review.json`。完整资格不由某一分项代替；未完成原全门槛者仍未合格，runtime_qualified/hardware_tested=false。预算到此关闭，不自动续训。']
p=D/'sc0090-v2-ballkick-target-precision.md';text=p.read_text();old=text.split('\n')[2];assert old.startswith('当前：');text=text.replace(old,'当前：成熟精度256连续正式及原nominal/measured四节点评测全部完成，预算关闭，无自动续训。新增150,994,944、BallKick全历史477,757,440；最终曲线和头稳/健康分母见本文末及 `p04/ballkick-target-precision-final-review.json`。',1);text+='\n'.join(lines)+'\n';p.write_text(text)
p=D/'sc0090-v2-ballkick-agent.md';text=p.read_text();first=text.split('\n')[0];assert first.startswith('当前：');text=text.replace(first,'当前：成熟精度256续段及双条件评测已结束；Ball累计477,757,440正式controls，预算关闭，无自动续训。各原门槛结论见[完整曲线](sc0090-v2-ballkick-target-precision.md)及 `ballkick-target-precision-final-review.json`。',1);p.write_text(text)
print(json.dumps(dict(status=out['status'],rows=[{k:v for k,v in r.items()if k in ['node','stage','raw_correct_foot_touch_trials_200hz','healthy_valid_ever_trials_200hz','clean_motion_trials','rolling_target_last1s_trials','body_standing_last05s_trials','complete_kick_trials','feedback_fault_trials']}for r in rows],initial_pairing=pairing),indent=2))
