"""Actual no-reset successor oracle including standing-gated body commands."""
from pathlib import Path
import json,torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.custom_v2.velstand import TASK
from mjlab_microduck.custom_v2.terminal import SC0090TerminalCriticObserver
r=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04')
torch.set_num_threads(2);results=[]
for boundary in ('ordinary','commands','push'):
 cfg=load_env_cfg(TASK);cfg.scene.num_envs=8;cfg.seed=2026101619
 cfg.auto_reset=False;cfg.curriculum={};cfg.observations['actor'].enable_corruption=False
 with torch.inference_mode():
  env=ManagerBasedRlEnv(cfg,device='cuda:1')
  try:
   env.reset();observer=SC0090TerminalCriticObserver(env)
   if boundary=='commands':
    for term in env.command_manager._terms.values():term.time_left.zero_()
   if boundary=='push':
    for timer in env.event_manager._interval_term_time_left:timer.zero_()
   captured=[];original=env.metrics_manager.compute
   def capture():
    original();qpos=env.sim.data.qpos.clone();qvel=env.sim.data.qvel.clone()
    cpu_rng=torch.get_rng_state();rng=torch.cuda.get_rng_state('cuda:1')
    states={n:{k:v.clone() for k,v in vars(term).items() if isinstance(v,torch.Tensor)} for n,term in env.command_manager._terms.items()}
    timers=[v.clone() for v in env.event_manager._interval_term_time_left]
    fast=observer.capture(env).clone();full=observer.capture(env,force_full=True).clone()
    captured.append((fast,full))
    assert torch.equal(qpos,env.sim.data.qpos) and torch.equal(qvel,env.sim.data.qvel)
    assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(rng,torch.cuda.get_rng_state('cuda:1'))
    for n,state in states.items():
     for k,v in state.items():assert torch.equal(v,getattr(env.command_manager._terms[n],k)),(n,k)
    for a,b in zip(timers,env.event_manager._interval_term_time_left):assert torch.equal(a,b)
   env.metrics_manager.compute=capture
   obs,*_=env.step(torch.zeros(8,14,device='cuda:1'))
   assert len(captured)==1
   for item in captured[0]:
    assert item.shape==(8,76)
    torch.testing.assert_close(item[:,:3],obs['critic'][:,:3],rtol=0,atol=5e-7)
    torch.testing.assert_close(item[:,3:],obs['critic'][:,3:],rtol=0,atol=0)
   results.append(dict(boundary=boundary,fast_max=float((captured[0][0]-obs['critic']).abs().max()),full_max=float((captured[0][1]-obs['critic']).abs().max()),state_and_rng_exact=True))
  finally:env.close()
out=r/'velstand-terminal-successor-001.json';assert not out.exists()
out.write_text(json.dumps(dict(status='PASS',checks=results,scope='Actual V2 VelStand 76D successor; ordinary/command-resample/push, fast/full capture, exact main state/RNG isolation'),indent=2)+'\n')
print(out,flush=True)
