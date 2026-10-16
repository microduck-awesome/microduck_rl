"""Evaluate pure landing reward terms on saved states, without advancing rollout state."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import ast,hashlib,json,math
from pathlib import Path
from types import SimpleNamespace as NS
import torch
torch.set_num_threads(2)
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04')
W=R/'roulade-history-worktree'
source=W/'src/mjlab_microduck/tasks/mdp.py'
functions=['standing_composite_score','_roulade_completion_gate','roulade_landing_composite','roulade_upright_after_roll','roulade_height_after_roll','roulade_landing_sharp','roulade_stand_tax','roulade_rise_velocity']
tree=ast.parse(source.read_text())
selected=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in functions]
assert len(selected)==len(functions)
ns=dict(torch=torch,math=math,_DEFAULT_ASSET_CFG=NS(name='robot'),
        _servo_default_joint_pos=lambda env,robot:torch.zeros_like(robot.data.joint_pos),
        _servo_joint_pos=lambda env,robot:robot.data.joint_pos,
        _update_roulade_accum=lambda env,robot:None,
        _roulade_state=lambda env:(env.saved_accum,env.saved_frontier,torch.zeros_like(env.saved_frontier)))
exec('from __future__ import annotations\n'+ast.unparse(ast.Module(body=selected,type_ignores=[])),ns)
cfg=json.loads((R/'roulade-history-1536-nominal-001/config.json').read_text())
terms=['roulade_landing_composite','roulade_upright_after_roll','roulade_height_after_roll','roulade_landing_sharp','roulade_stand_tax','roulade_rise_velocity']
rows=[]
for stage in ['nominal','measured']:
    trace=torch.load(R/f'roulade-history-1536-{stage}-001/traces.pt',map_location='cpu',weights_only=False)
    m=trace['measurements'];n=250*256
    env=NS(saved_accum=m['accum'].flatten(),saved_frontier=m['frontier'].flatten(),_roulade_head_latch=m['head_latch'].flatten())
    root=torch.zeros(n,3);root[:,2]=m['height'].flatten()
    data=NS(root_link_pos_w=root,root_link_quat_w=m['root_quat'].flatten(0,1),joint_pos=m['physical_q'].flatten(0,1),root_link_lin_vel_w=m['linear'].flatten(0,1))
    class Scene(dict):pass
    env.scene=Scene(robot=NS(data=data));env.scene.terrain=NS(env_origins=torch.zeros(n,3))
    values={}
    for name in terms:
        term=cfg['rewards'][name]
        values[name]=ns[name](env,**term['params']).reshape(250,256)*term['weight']
    # Independent arithmetic oracle for this saved-state evaluation adapter.
    u=((m['frontier']-math.radians(260))/math.radians(70)).clamp(0,1)
    gate=u*u*(3-2*u)*m['head_latch']
    q=m['root_quat'];tiltsq=2*(q[:,:,1]**2+q[:,:,2]**2)
    h=torch.exp(-((m['height']-.112)/.04)**2)
    pose=torch.exp(-m['physical_q'][:,:,[0,1,2,3,4,9,10,11,12,13]].square().mean(-1)/.4**2)
    oracle={
      'roulade_landing_composite':4*h*torch.exp(-tiltsq/.4**2)*pose*gate,
      'roulade_upright_after_roll':1.5*(1-tiltsq).clamp_min(0)*gate,
      'roulade_height_after_roll':h*gate,
      'roulade_landing_sharp':2*torch.exp(-tiltsq/.3**2)*torch.exp(-((m['height']-.112)/.015)**2)*gate,
      'roulade_stand_tax':-5*(.112-m['height']).clamp_min(0)*gate}
    errors={k:float((values[k]-v).abs().max()) for k,v in oracle.items()};assert max(errors.values())<2e-5
    for group in range(5):
        mask=torch.arange(256)%8<4 if group==0 else torch.arange(256)%8==group+3
        means={k:float(v[-25:,mask].mean()) for k,v in values.items()}
        rows.append(dict(stage=stage,group='standing' if group==0 else f'midroll{[90,180,270,330][group-1]}',trials=int(mask.sum()),
          gate_mean=float(gate[-25:,mask].mean()),weighted_landing_terms_tail_mean=means,
          selected_terms_sum=sum(means.values()),independent_arithmetic_max_errors=errors))
out=dict(status='SAVED_STATE_PURE_TERM_AUDIT_COMPLETE',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=rows,
  scope='Six actual native pure landing terms at saved original50Hz boundary, weighted before control-dt scaling. Native state update intentionally disabled because saved accum/frontier/latch already reflect that boundary. No training/inference/physics run and no production modification.',
  finding='Eligible low folded postures continue receiving positive upright/height landing rewards; the broad composite and sharp height term are near zero at these states. This is evidence for a reachable reward basin, not a causal proof that one coefficient explains the full training failure.',
  limitations=['Not total reward or return: action-rate, torque-rate and other penalties are not reconstructed from missing channels.',
    'No blanket recommendation to increase tax; any next recipe must preserve rollout discovery and provide reachable late-stage transitions.',
    'Head HOME is checked by evaluation but absent from the original leg-only landing pose score. Active roll head motion must remain free.'])
p=R/'roulade-history-landing-reward-audit-001.json';assert not p.exists();p.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps(rows,indent=2))
