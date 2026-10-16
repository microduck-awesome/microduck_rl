"""Execute actual pinned reward functions on explicit analytic state fixtures.

No claim that synthetic deceleration is the V2/ball physical trajectory.
"""
from pathlib import Path
import ast,hashlib,json,types,torch
r=Path(__file__).resolve().parent;p=r/'ballkick-repaired-worktree/src/mjlab_microduck/tasks/mdp.py'
s=p.read_text();tree=ast.parse(s);names={'_ball_kick_dir','ball_forward_velocity','ball_speed_overshoot_penalty'}
code='from __future__ import annotations\n'+'\n\n'.join(ast.get_source_segment(s,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names)
namespace={'torch':torch};exec(compile(code,str(p),'exec'),namespace)
device='cuda:1';v=torch.tensor([0.,.5,1.,1.5,2.,4.],device=device)
def reward(speed):
 data=types.SimpleNamespace(root_link_lin_vel_w=torch.stack((speed,torch.zeros_like(speed),torch.zeros_like(speed)),-1))
 env=types.SimpleNamespace(scene={'ball':types.SimpleNamespace(data=data)},num_envs=len(speed),device=device)
 return 12*namespace['ball_forward_velocity'](env,max_speed=1.)-4*namespace['ball_speed_overshoot_penalty'](env,target_speed=1.)
actual=reward(v);assert torch.equal(actual,torch.tensor([0.,6.,12.,10.,8.,0.],device=device))
# Rate reward over5s, explicit constant deceleration0.2m/s^2.
t=torch.arange(250,device=device)*.02;curves=[]
for initial in (1.,1.5,2.):
 speed=(initial-.2*t).clamp_min(0);pay=reward(speed);curves.append(dict(initial_speed=initial,undiscounted_reward_integral=float(pay.sum()*.02),discounted_sum_gamma099=float((pay*torch.pow(.99,torch.arange(250,device=device))).sum()*.02)))
assert curves[1]['undiscounted_reward_integral']>curves[0]['undiscounted_reward_integral']
out=dict(status='ACTUAL_REWARD_SEMANTICS_AUDITED_NOT_PHYSICS_VALIDATION',source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),functions=sorted(names),instantaneous_speeds=v.cpu().tolist(),instantaneous_reward_rates=actual.cpu().tolist(),contact_gate='Neither actual reward reads contact identity, robot posture, or valid kick history',synthetic_rolling_counterexample=dict(deceleration_mps2=.2,duration_s=5.,dt=.02,curves=curves),scope='Native instantaneous objective peaks1m/s; explicit synthetic rolling integral can prefer overspeed. This refutes a universal optimum claim only, not a measured V2 failure. Need actual ball physics/contact audit before choosing reward repair; no policy or training controls.')
p=r/'ballkick-reward-semantics-001.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
