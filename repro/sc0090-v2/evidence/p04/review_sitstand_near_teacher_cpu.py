"""Saved data only; retain original gates and independently report contact screen."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,hashlib,time
import torch
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04')
PREFIX='sitstand-near-teacher-lr3e6';OUT=R/(PREFIX+'-final-review.json')
state=json.loads((R/(PREFIX+'-pipeline-status.json')).read_text());assert state['status']=='SITSTAND_NEAR_TEACHER_COMPLETE_NO_AUTOMATIC_CONTINUATION'
assert not Path('/proc/'+str(state['pid'])).exists()
rows=[];reference={};torch.set_num_threads(2)
for node in (640,768,896,960):
 for stage in ('nominal','measured'):
  root=R/(f'sitstand-history-hold320-{node:04d}-{stage}'if node==640 else f'{PREFIX}-{node:04d}-{stage}')
  report=json.loads((root/'report.json').read_text());tr=torch.load(root/'traces.pt',map_location='cpu',weights_only=False);initial=tr['initial'];x=tr['traces'];mode=tr['mode'];assert x.shape[:2]==(600,256)
  if node==640:reference[stage]=initial
  pairing={k:torch.equal(v,reference[stage][k])for k,v in initial.items()}
  agg={key:dict(trials=0,body_hold=0,body_head1deg=0,body_contactscreen=None if node==640 else 0,body_clean_head1deg=None if node==640 else 0)for key in('SIT','STAND')};heads=dict(trials=0,reached=0,tail_band_hold=0);faults=0;camera=[]
  for m,name in enumerate(('stand_to_stand','stand_to_sit','sit_to_stand','sit_to_sit')):
   subset=mode==m
   for phase,g in enumerate(report['groups'][name]):
    target=(m%2)if phase in(0,2)else 1-(m%2);kind='SIT'if target else'STAND';tail=x[(phase+1)*200-25:(phase+1)*200,subset]
    body=tail[:,:,4].bool().all(0);head=(tail[:,:,7]<torch.pi/180).all(0)
    assert int(body.sum())==g['tail_hold'];assert int((body&head).sum())==g['head_tail_all_axes_1deg']
    agg[kind]['trials']+=g['trials'];agg[kind]['body_hold']+=int(body.sum());agg[kind]['body_head1deg']+=int((body&head).sum())
    heads['trials']+=g['head_step_trials'];heads['reached']+=g['head_step_reached'];heads['tail_band_hold']+=g['head_step_tail_band_hold'];faults+=g['faults']
    if node!=640:
     c=tr['contacts_200Hz'][((phase+1)*200-25)*4:((phase+1)*200)*4,subset].bool();clean=c[:,:,:2].all(-1).all(0)&(~c[:,:,2:].any(-1)).all(0)
     assert int((body&clean).sum())==g['tail_body_plus200Hz_feet_nohead_noself'];assert int((body&clean&head).sum())==g['tail_body_clean_head_all_axes1deg']
     agg[kind]['body_contactscreen']+=int((body&clean).sum());agg[kind]['body_clean_head1deg']+=int((body&clean&head).sum())
    camera.append(dict(initial_group=name,phase=phase,posture_command=kind,windows=g['camera']))
  rows.append(dict(node=node,stage=stage,original_gate_counts=agg,head_response=heads,trial_phase_faults=faults,initial_pairing_with640=pairing,camera_with_original_and_extra_health_denominators=camera,actor_normalizer_unchanged=report['actor_normalizer_unchanged'],report_sha256=hashlib.sha256((root/'report.json').read_bytes()).hexdigest()))
formal=R/(PREFIX+'-allocation-001');rep=json.loads((formal/'report.json').read_text());counter=[json.loads(l)for l in(formal/'native-optimizer-step-callback.jsonl').read_text().splitlines()];assert len(counter)==320 and counter[-1]['allocation_optimizer_steps']==25600
bc=[json.loads(l)for l in(formal/'bc-diagnostics.jsonl').read_text().splitlines()];assert len(bc)==320
receipt=dict(review_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),status='FIXED_STAGE_CLOSED_NO_AUTOMATIC_CONTINUATION',recorded_unix=time.time(),formal_new_controls=rep['allocation_controls'],all_history_formal_controls=603979776,actual_PPO_Adam=counter[-1]['allocation_optimizer_steps'],actual_BC_Adam=counter[-1]['allocation_BC_steps'],PPO_parent_LR_preserved=json.loads((formal/'mature-load-observer.json').read_text()),full_shared_phase_wall=json.loads((formal/'shared-process-phase-wall.json').read_text()),full_worker_wall=state['worker_wall_s'],full_worker_wall_scope=state.get('worker_wall_scope','Original singleworker lifecycle'),throughput=rep['throughput'],BC_max_meanKL=max(b['losses']['expert_bc_kl_mean']for b in bc),BC_stand_frames=sum(b['context']['effective_stand_frames']for b in bc),BC_SIT_anchor_frames=sum(b['context']['sit_anchor_frames']for b in bc),BC_ignored_STAND_frames=sum(b['context']['ignored_stand_frames']for b in bc),rows=rows,source_unchanged=True,scope='Original12s command/reverse/return gates preserved; new200Hz feet/head/self screen separate. Camera includes intended motion; denominator reported per phase.',limitations=['Original640 lacks this full200Hz head contact trace; its body0 already excludes any clean stand pass.','Exact initial state checked independently; no assertion of identical closed-loop trajectories.','No hardware/runtime qualification.'])
OUT.write_text(json.dumps(receipt,indent=2)+'\n');print('CPU_REVIEW_PASS',OUT,flush=True)
D=Path('/home/l40-vm01/Documents/ai-research/microduck/microduck-v2-p04/docs');p=D/'sc0090-v2-sitstand-near-teacher.md';s=p.read_text();p.write_text('# 当前阶段已完成\n\nBC3e-6固定320正式和六项原评测已完成，禁止自动追加。CPU完整复算见 `sitstand-near-teacher-lr3e6-final-review.json`；是否技能合格按原指标，不能用预算结束代替。\n\n'+s)
# Static artifact uses the already independently checked original-gate counts.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for stage,color in(('nominal','#1764ab'),('measured','#cf6c18')):
 rs=[r for r in rows if r['stage']==stage];nodes=[r['node']for r in rs]
 for ax,posture in zip(axes[0],('SIT','STAND')):
  ax.plot(nodes,[100*r['original_gate_counts'][posture]['body_hold']/r['original_gate_counts'][posture]['trials']for r in rs],marker='o',color=color,label=stage);ax.set_title(posture+' body hold, original gate')
 axes[1,0].plot(nodes,[100*sum(v['body_head1deg']for v in r['original_gate_counts'].values())/sum(v['trials']for v in r['original_gate_counts'].values())for r in rs],marker='o',color=color,label=stage);axes[1,0].set_title('Body + simultaneous 1 degree head hold')
 axes[1,1].plot(nodes,[100*r['head_response']['reached']/r['head_response']['trials']for r in rs],marker='o',color=color,label=stage);axes[1,1].set_title('Commanded head response, original gate')
for ax in axes.flat:ax.set_xlabel('Checkpoint update');ax.set_ylabel('Percent of trials');ax.set_ylim(-2,102);ax.grid(alpha=.2);ax.legend()
fig.suptitle('SitStand near-STAND teacher: fixed recipe, full 12 s evaluation')
fig.savefig(R/(PREFIX+'-curve.png'),dpi=160);plt.close(fig)
