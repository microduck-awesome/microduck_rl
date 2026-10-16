from pathlib import Path
import torch,json,hashlib
r=Path(__file__).resolve().parent;rows=[]
for stage in ('nominal','measured'):
 p=r/f'ballkick-right-0085-{stage}/traces.pt'
 if not p.exists():continue
 x=torch.load(p,map_location='cpu',weights_only=False)['measurements'];tail=slice(-50,None)
 row=dict(stage=stage,trials=x['height'].shape[1],tail_height_mean=float(x['height'][tail].mean()),tail_height_p05=float(torch.quantile(x['height'][tail],.05)),tail_tilt_mean_deg=float(torch.rad2deg(x['tilt'][tail]).mean()),tail_linear_mean=float(x['linear'][tail].norm(dim=-1).mean()),tail_omega_mean=float(x['angular'][tail].norm(dim=-1).mean()),tail_both_feet_fraction=float(x['feet'][tail].all(-1).float().mean()),tail_head_contact_fraction=float(x['head_contact'][tail].float().mean()),tail_self_contact_fraction=float(x['self_contact'][tail].float().mean()),tail_bad_contact_trials=int(x['bad_contact'][tail].any(0).sum()),tail_valid_contact_trials=int(x['valid_kick'][tail].any(0).sum()),tail_q_mean=x['physical_q'][tail].mean((0,1)).tolist(),trace_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 forward=12*x['ball_forward'].clamp(0,1)*x['valid_kick'];overshoot=-4*(x['ball_forward']-1).clamp(0,5)
 fail=~x['standing'];row['reconstructed_positive_forward_reward_rate_tail_mean']=float(forward[tail].mean());row['positive_forward_reward_tail_paid_on_nonstanding_fraction']=float((forward[tail]*fail[tail]).sum()/forward[tail].sum().clamp_min(1e-9));row['positive_forward_reward_rate_on_nonstanding_tail_mean']=float((forward[tail]*fail[tail]).mean());rows.append(row)
(r/'ballkick-right-0085-saved-diagnostic.json').write_text(json.dumps(dict(status='SAVED_TRACE_PHYSICAL_AND_PARTIAL_REWARD_DIAGNOSIS',rows=rows,scope='Reconstructs only ball-forward/overspeed from saved physical boundary and actual latch; not full reward, causal PPO gradient or evidence of a chosen repair'),indent=2)+'\n');print(json.dumps(rows))
