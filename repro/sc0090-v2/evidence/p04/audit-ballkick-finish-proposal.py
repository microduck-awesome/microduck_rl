from pathlib import Path
import torch,json,hashlib
r=Path(__file__).resolve().parent;torch.set_num_threads(2);rows=[]
for stage in ('nominal','measured'):
 path=r/f'ballkick-right-0085-{stage}/traces.pt';d=torch.load(path,map_location='cpu',weights_only=False)['measurements'];v=d['valid_kick'];lin=d['linear'].square().sum(-1);ang=d['angular'].square().sum(-1)
 lateral=-4*d['ball_lateral'].abs().clamp(max=1.)*v
 settle=-2*(lin/(lin+.15**2)+ang/(ang+1.5**2)+(~d['feet'].all(-1)).float())*v
 old=12*d['ball_forward'].clamp(0,1)*v-4*(d['ball_forward']-1).clamp(0,5)
 for phase,sl in [('whole',slice(None)),('tail1s',slice(-50,None))]:
  rows.append(dict(stage=stage,phase=phase,old_ball_forward_and_overshoot=float(old[sl].mean()),new_direction_penalty=float(lateral[sl].mean()),new_postkick_settle_penalty=float(settle[sl].mean()),partial_sum_after_new_terms=float((old+lateral+settle)[sl].mean()),valid_fraction=float(v[sl].float().mean()),pre_valid_new_cost_max=float(torch.where(~v,lateral.abs()+settle.abs(),0.).max()),first_valid_contact_s_median=float(((v.long().argmax(0)+1)*.02).median()),trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
out=dict(status='CPU_SAVED_TRACE_PROPOSAL_AUDIT',rows=rows,terms=dict(direction='-4*valid*min(abs(ball_lateral),1)',postkick='-2*valid*(v2/(v2+.15^2)+omega2/(omega2+1.5^2)+not_both_feet)'),invariants='original12 forward and-4 overshoot/pose/head/PPO preserved; no new positive reward, timer, hidden state, reward-side state mutation or actor input',counterfactual='quiet bothfeet body with same ball state removes settle cost; lateral0 removes direction cost; before valid kick both are exactly0',scope='Only original ball speed+overshoot and two proposed terms; not full return or causal training improvement')
(r/'ballkick-finish-proposal-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
