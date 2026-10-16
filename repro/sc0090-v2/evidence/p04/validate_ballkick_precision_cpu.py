"""CPU-only config/precision oracle + fixed-parent source transition."""
from pathlib import Path
from dataclasses import asdict,is_dataclass
import json,hashlib,inspect,functools,math
import torch
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
from mjlab_microduck.custom_v2.ball_kick_health import HEALTH_TASK
from mjlab_microduck.custom_v2.ball_kick_precision import PRECISION_TASK,target_precision_value,ball_target_precision
from train_v2_ballkick import source_hashes
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'ballkick-target-precision-worktree';B=R/'ballkick-health-history-worktree';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(x):
 if is_dataclass(x)and not isinstance(x,type):return canonical(asdict(x))
 if isinstance(x,dict):return {str(k):canonical(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [canonical(v)for v in x]
 if isinstance(x,functools.partial):return ['partial',canonical(x.func),canonical(x.args),canonical(x.keywords)]
 if callable(x):return [x.__module__,x.__qualname__,inspect.getsource(x),[canonical(c.cell_contents)for c in(getattr(x,'__closure__',None)or())]]
 return x

def main():
 differences={}
 for play in [False,True]:
  old=canonical(load_env_cfg(HEALTH_TASK,play=play));new=canonical(load_env_cfg(PRECISION_TASK,play=play));paths=[]
  def walk(a,b,p=''):
   if type(a)!=type(b):paths.append(p);return
   if isinstance(a,dict):
    if a.keys()!=b.keys():paths.append(p);return
    for k in a:walk(a[k],b[k],p+'.'+k)
   elif isinstance(a,list):
    if a!=b:paths.append(p)
   elif a!=b:paths.append(p)
  walk(old,new);assert set(paths)<=set(['.rewards.ball_forward_velocity.func','.rewards.ball_forward_velocity.params'])and '.rewards.ball_forward_velocity.func'in paths,paths
  assert canonical(load_rl_cfg(HEALTH_TASK))==canonical(load_rl_cfg(PRECISION_TASK));differences[str(play)]=paths
 f=torch.tensor([-1.,0.,.6,.8,1.,1.2,1.4,2.,1e20],dtype=torch.float64);l=torch.tensor([-.4,-.2,0.,.2,.4],dtype=torch.float64);ff,ll=torch.meshgrid(f,l,indexing='ij');valid=torch.ones_like(ff,dtype=torch.bool);beforeff=ff.clone();beforell=ll.clone();rng=torch.get_rng_state().clone();got=target_precision_value(ff,ll,valid);want=torch.tensor([[math.exp(-((float(x)-1)/.2)**2-(float(y)/.2)**2)for y in l]for x in f]);torch.testing.assert_close(got,want.double(),atol=3e-8,rtol=1e-7)
 assert torch.equal(ff,beforeff)and torch.equal(ll,beforell)and torch.equal(torch.get_rng_state(),rng)
 assert bool((target_precision_value(ff,ll,~valid)==0).all())and bool(torch.isfinite(got).all())and bool(((got>=0)&(got<=1)).all());assert float(target_precision_value(torch.tensor(1.),torch.tensor(0.),torch.tensor(True)))==1.
 g=torch.tensor([.7,.9,1.,1.1,1.3],requires_grad=True);lat=torch.tensor([.1]*5,requires_grad=True);target_precision_value(g,lat,torch.ones(5,dtype=torch.bool)).sum().backward();assert bool((g.grad[:2]>0).all())and g.grad[2]==0 and bool((g.grad[3:]<0).all())and bool((lat.grad<0).all())
 # Exact rootdecision parent criterion: strict complete includingneutralhead,
 # maximize minimum ofnominal/measured. All original reports retained.
 scores={}
 for n in [64,128,192,256,320,384]:
  rr=[json.loads((R/f'ballkick-health-history-{n:04d}-{s}/report.json').read_text())for s in ['nominal','measured']];scores[n]=dict(strict=[x['complete_kick_and_neutral_head_trials']for x in rr],complete=[x['complete_kick_trials']for x in rr],valid=[x['healthy_valid_ever_trials_200hz']for x in rr])
 parent_node=max(scores,key=lambda n:min(scores[n]['strict']));assert parent_node==256 and scores[256]['valid']==[256,256]
 parent=R/'ballkick-health-history-allocation-001/checkpoint_000256.pt';cp=torch.load(parent,map_location='cpu',weights_only=False);meta=cp['infos']['v2'];assert meta['task']==HEALTH_TASK and meta['num_envs']==24576 and meta['completed_updates']==256 and cp['infos']['env_state']['common_step_counter']==6144
 hashes=source_hashes(W);changed=sorted(k for k in set(hashes)|set(meta['source_sha256'])if hashes.get(k)!=meta['source_sha256'].get(k));allowed=['scripts/train_v2_ballkick.py','src/mjlab_microduck/custom_v2/ball_kick_metrics.py','src/mjlab_microduck/custom_v2/ball_kick_precision.py','src/mjlab_microduck/tasks/__init__.py'];assert changed==sorted(allowed),changed
 assets=[p for p in(B/'src').rglob('*')if p.is_file()and p.suffix not in('.py','.cpp','.pyc')and '__pycache__'not in p.parts];assert all(sha(p)==sha(W/p.relative_to(B))for p in assets)
 transition=dict(status='PASS_CONFIG_REVIEW',recipe='mature-ball-target-precision-v1',task=PRECISION_TASK,parent_task=HEALTH_TASK,parent_checkpoint=str(parent),parent_checkpoint_sha256=sha(parent),from_source_sha256=meta['source_sha256'],to_source_sha256=hashes,changed_files=changed,config_differences=differences,agent_exact=True,assets_exact=len(assets),source_transition_once=True)
 (R/'ballkick-target-precision-source-transition.json').write_text(json.dumps(transition,indent=2)+'\n')
 # Saved trajectory partialcounterfactual; no additionalfullreturn claim.
 partial=[]
 for n in [256,384]:
  for stage in ['nominal','measured']:
   x=torch.load(R/f'ballkick-health-history-{n:04d}-{stage}/traces.pt',map_location='cpu',weights_only=False)['measurements'];old=12*x['ball_forward'].clamp(0,1)*x['valid_kick'];new=12*target_precision_value(x['ball_forward'],x['ball_lateral'],x['valid_kick']);partial.append(dict(node=n,stage=stage,old_positive_mean=float(old.mean()),new_positive_mean=float(new.mean()),old_tail_mean=float(old[-50:].mean()),new_tail_mean=float(new[-50:].mean()),finite=bool(torch.isfinite(new).all()),before_valid_exact0=bool((new[~x['valid_kick']]==0).all())))
 report=dict(status='PASS',source_sha256=hashes,task=PRECISION_TASK,parent_checkpoint=str(parent),parent_checkpoint_sha256=sha(parent),parent_node=256,parent_controls=meta['controls'],parent_clock=6144,parent_learning_rate=cp['optimizer_state_dict']['param_groups'][0]['lr'],config_diff=differences,agent_exact=True,assets_exact=len(assets),parent_selection_scores=scores,Gaussian_cpu_oracle=True,gradient_sign=True,valid_before0=True,inputs_rng_unchanged=True,frame='Originalresetheading: forward=vxy dot dir, lateral=vy*dirx-vx*diry; no currentheading substitute',grid_valid_rewards=(12*got).tolist(),partial_saved_reward=partial,budget=dict(new_updates=256,N=24576,T=24,E=5,M=24,new_formal_controls=150994944,old_all_formal_controls=326762496,total_cap=477757440),limitations=['Onlymatureadaptation; Gaussianfarfromtarget hasweakgradient andisnotfreshdiscoveryreward.','InstantaneousisotropicGaussian aligns scales but isnotoriginalcontinuous1s rectangle successgate; those thresholds unchanged.','Actor61has no directballposition/velocity, critic82does; cannotguaranteeclosedloopprecision forlatentinitialballuncertainty.','Old/newpartialpositivecounterfactual isnotfullreturn or evidencelearningimproves.'],script_sha256=sha(Path(__file__)))
 (R/'ballkick-target-precision-cpu-001.json').write_text(json.dumps(report,indent=2)+'\n');print('PRECISION_CPU_PASS',json.dumps({'parent':256,'score':scores[256],'changes':differences,'parent_lr':report['parent_learning_rate']}),flush=True);return report
