import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,hashlib
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');A=R/'ballkick-health-history-allocation-001';torch.set_num_threads(2)
acc=EventAccumulator(str(A),size_guidance={'scalars':0});acc.Reload();course={}
for tag in acc.Tags()['scalars']:
 if tag.startswith('Curriculum/'):
  rows=acc.Scalars(tag);course[tag]={str(n):[{'step':x.step,'value':x.value}for x in rows if x.step==n-1]for n in [256,320,384]}
weights=[json.loads(x)for x in(A/'curriculum-live.jsonl').read_text().splitlines()];by={x['update']:x['live_reward_weights']for x in weights};assert by[256]==by[320]==by[384]
results=[]
for stage in ['nominal','measured']:
 arms={}
 for n in [256,384]:
  p=R/f'ballkick-health-history-{n:04d}-{stage}';x=torch.load(p/'traces.pt',map_location='cpu',weights_only=False);m=x['measurements'];raw=x['physics_raw_contacts'];bits=x['physics_kick_state'];report=json.loads((p/'report.json').read_text());init=x['initial'];displacement=((m['ball_position'][-1,:,:2]+init['origins'][:,:2]-init['qpos'][:,-7:-5])*x['kick_direction']).sum(-1)
  motion=m['finite']&m['feedback_ready']&~m['head_contact']&~m['self_contact']&(m['height']>=.08)&(m['tilt']<=torch.pi/4)
  gate=dict(valid=m['valid_kick'].any(0),clean=motion.all(0)&~m['bad_contact'].any(0),target=m['rolling_target'][-50:].all(0),body=m['standing'][-25:].all(0),distance=displacement>=.25,head=(m['physical_q'][-25:,:,5:9].abs().amax(-1)<.1).all(0))
  complete=gate['valid']&gate['clean']&gate['target']&gate['body']&gate['distance'];assert int(complete.sum())==report['complete_kick_trials'];assert int((complete&gate['head']).sum())==report['complete_kick_and_neutral_head_trials']
  detail=dict(underspeed=(m['ball_forward'][-50:]<.8).any(0),overspeed=(m['ball_forward'][-50:]>1.2).any(0),lateral=(m['ball_lateral'][-50:].abs()>.2).any(0),body_linear=(m['linear'][-25:].norm(dim=-1)>=.15).any(0),body_omega=(m['angular'][-25:].norm(dim=-1)>=1.5).any(0),body_not_bothfeet=(~m['feet'][-25:].all(-1)).any(0),body_height=(m['height'][-25:]<.102).any(0),body_tilt=(m['tilt'][-25:]>torch.pi/9).any(0),raw_self=raw[:,:,3].any(0),raw_head=raw[:,:,2].any(0))
  arms[n]=dict(gates=gate,complete=complete,detail=detail,initial=init,head=qhead if False else m['physical_q'][-25:,:,5:9].abs().mean(0),positive_ball_rate=(12*m['ball_forward'].clamp(0,1)*m['valid_kick']).mean(0),targetspeedmean=m['ball_forward'][-50:].mean(0))
 pair={k:torch.equal(arms[256]['initial'][k],arms[384]['initial'][k])for k in arms[256]['initial']};assert all(pair.values())
 lost=arms[256]['complete']&~arms[384]['complete'];gain=~arms[256]['complete']&arms[384]['complete'];both=arms[256]['complete']&arms[384]['complete']
 results.append(dict(stage=stage,exact_initial=pair,success256=int(arms[256]['complete'].sum()),success384=int(arms[384]['complete'].sum()),retained=int(both.sum()),lost=int(lost.sum()),gained=int(gain.sum()),lost_failure_gates384={k:int((lost&~v).sum())for k,v in arms[384]['gates'].items()},lost_detail384={k:int((lost&v).sum())for k,v in arms[384]['detail'].items()},all_failures={str(n):{k:int(v.sum())for k,v in arms[n]['detail'].items()}for n in arms},head_tail_abs_deg={str(n):torch.rad2deg(arms[n]['head']).mean(0).tolist()for n in arms},tail_speed_mean={str(n):float(arms[n]['targetspeedmean'].mean())for n in arms},positive_ball_rate_mean={str(n):float(arms[n]['positive_ball_rate'].mean())for n in arms},scope='Gates overlap; differences are paired initialworld policy outcomes, not identified causal mechanism.'))
out=dict(status='CPU_PAIRED_256_TO384_CURRICULUM_AND_FAILURE_AUDIT',newGPUcontrols=0,newformalcontrols=0,actual_logged_curriculum=course,live_reward_weights256=by[256],all_live_rewards_256_320_384_equal=True,head_command_semantics='No activeheadcommand inBall; originalHOME neck pose term, no headtarget curriculum',results=results,conclusion='No evidence of post256 curriculum escalation if actualmanagerlogs match. Frozen256course would then be same numericalcontinuation, not correctivefactor; do not attribute regression to changed taxes. Known completeaction emerged by256 and persisted partially384; preserve256. Need specific nextmechanism or explicit same-recipe budget rather than fictitious freeze contrast.',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(R/'ballkick256-384-course-failure-cpu-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
