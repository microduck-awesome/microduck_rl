from pathlib import Path
import json,hashlib,torch
r=Path(__file__).resolve().parent;p=r/'ballkick-physical-002';x=torch.load(p/'trace.pt',map_location='cpu',weights_only=False)
s=x['physics_kick_state'].bool();t=x['physics_touches'].bool();v=x['values'];bad=(t[:,:,1]|t[:,:,2]).long().cummax(0).values.bool()
assert torch.equal(s[:,:,1],bad)
assert not bool((s[:,:,0]&bad).any())
assert not bool((s[:,:,0]&~(t[:,:,0]&t[:,:,3]).long().cummax(0).values.bool()).any())
assert torch.equal(s[3::4,:,0],v['valid_kick'].bool())
assert torch.equal(s[3::4,:,1],v['bad_contact'].bool())
missed=t[:,:,0].reshape(-1,4,t.shape[1]).any(1)&~t[3::4,:,0]
reward=x['rewards'][:,:,x['reward_names'].index('ball_forward_velocity')]
assert bool((reward[~v['valid_kick'].bool()]==0).all())
report=dict(status='PASS_PHYSICAL_ATTRIBUTION_REVIEW_NOT_POLICY_ACCEPTANCE',scope_correction='002 includes new positive reward contact gate and pose-dependent floor clearance. Actual contact/latch recorded at200Hz, control traces50Hz; original report scope string was stale and preserved.',bad_contact_prefix_exact=True,no_unattributed_valid_latch=True,control_boundary_latches_exact=True,positive_reward_without_valid_contact=0,correct_touch_20ms_windows_missed_by_boundary_only=int(missed.sum()),physics_controls=32000,formal_controls=0,inputs_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (p/'report.json',p/'trace.pt',p/'initial-geometry.json')},limitation='Incoming-ball static-foot bounce is attribution positive control, not proof of learned active kick. Healthy contact predicate tested physically and in CPU transition tests; no policy admission yet.')
(r/'ballkick-physical-002-review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
