"""Independent CPU configuration, reward-grid and frozen parent admission."""
from pathlib import Path
from dataclasses import asdict,is_dataclass
from types import SimpleNamespace
import hashlib,json,inspect,functools,math
import torch
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
from mjlab_microduck.custom_v2.ground_pick_history import HISTORY_TASK
from mjlab_microduck.custom_v2.ground_pick_precision import PRECISION_TASK
from train_v2_ground_pick import source_hashes
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'groundpick-target-precision-worktree';B=R/'groundpick-health-history-worktree'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
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
  old=canonical(load_env_cfg(HISTORY_TASK,play=play));new=canonical(load_env_cfg(PRECISION_TASK,play=play));paths=[]
  def walk(a,b,p=''):
   if type(a)!=type(b):paths.append(p);return
   if isinstance(a,dict):
    if a.keys()!=b.keys():paths.append(p);return
    for k in a:walk(a[k],b[k],p+'.'+k)
   elif isinstance(a,list):
    if a!=b:paths.append(p)
   elif a!=b:paths.append(p)
  walk(old,new)
  assert set(paths)=={'.rewards.mouth_ground_proximity.params.std','.rewards.mouth_ground_proximity.params.target_height'},paths
  assert canonical(load_rl_cfg(HISTORY_TASK))==canonical(load_rl_cfg(PRECISION_TASK));differences[str(play)]=paths
 cfg=load_env_cfg(PRECISION_TASK);term=cfg.rewards['mouth_ground_proximity'];rows=[]
 heights=[-.005,0.,.010,.020,.025,.043,.060,.10,.20]
 for phase in [0.,.1,.25,.374,.375,.400,.424,.425,.6,.799,.8,.995]:
  for head,own in [(0,0),(0,1),(1,0),(1,1)]:
   pos=torch.zeros(len(heights),1,3,dtype=torch.float64);pos[:,0,2]=torch.tensor(heights,dtype=torch.float64)
   cmd=torch.tensor([math.cos(2*math.pi*phase),math.sin(2*math.pi*phase),0.],dtype=torch.float64).expand(len(heights),-1)
   env=SimpleNamespace(num_envs=len(heights),scene={'robot':SimpleNamespace(data=SimpleNamespace(site_pos_w=pos)),'head_impact_contact':SimpleNamespace(data=SimpleNamespace(found=torch.full((len(heights),1),bool(head)))),'self_collision':SimpleNamespace(data=SimpleNamespace(found=torch.full((len(heights),1),bool(own))))},command_manager=SimpleNamespace(get_command=lambda _:cmd))
   params=dict(term.params);params['asset_cfg']=SimpleNamespace(name='robot',site_ids=[0]);got=term.func(env,**params)
   gate=min(min(max(phase/.375,0),1),min(max((.8-phase)/(.8-.425),0),1))
   want=torch.tensor([gate*math.exp(-((h-.020)/.025)**2)*(not(head or own)) for h in heights],dtype=torch.float64)
   torch.testing.assert_close(got,want,rtol=1e-12,atol=1e-12);rows.append(dict(phase=phase,head=head,self_contact=own,error=float((got-want).abs().max())))
 h=torch.tensor([.0,.01,.02,.025,.043,.06],dtype=torch.float64,requires_grad=True);torch.exp(-((h-.02)/.025).square()).sum().backward();assert bool((h.grad[:2]>0).all()) and h.grad[2]==0 and bool((h.grad[3:]<0).all())
 parent=R/'groundpick-health-history-allocation-001/checkpoint_000512.pt';cp=torch.load(parent,map_location='cpu',weights_only=False);meta=cp['infos']['v2'];assert meta['task']==HISTORY_TASK and meta['num_envs']==24576 and meta['completed_updates']==512 and cp['infos']['env_state']['common_step_counter']==12288
 hashes=source_hashes(W);assert meta['source_sha256']==source_hashes(B)
 changed=sorted(k for k in set(hashes)|set(meta['source_sha256'])if hashes.get(k)!=meta['source_sha256'].get(k));allowed=['scripts/train_v2_ground_pick.py','src/mjlab_microduck/custom_v2/ground_pick_metrics.py','src/mjlab_microduck/custom_v2/ground_pick_precision.py','src/mjlab_microduck/tasks/__init__.py'];assert changed==sorted(allowed),changed
 assets=[p for p in(B/'src').rglob('*')if p.is_file()and p.suffix not in('.py','.cpp','.pyc')and '__pycache__'not in p.parts];assert all(sha(p)==sha(W/p.relative_to(B))for p in assets)
 # Existing evaluator logic is byte-identical except import and allowed task id.
 a=(B/'scripts/evaluate_v2_ground_pick.py').read_text();b=(W/'scripts/evaluate_v2_ground_pick.py').read_text();assert b.replace('\nfrom mjlab_microduck.custom_v2.ground_pick_precision import PRECISION_TASK','').replace('(GROUND_PICK_TASK,HISTORY_TASK,PRECISION_TASK)','(GROUND_PICK_TASK,HISTORY_TASK)')==a
 transition=dict(status='PASS_CONFIG_REVIEW',recipe='mature-groundpick-height-precision-v1',task=PRECISION_TASK,parent_task=HISTORY_TASK,parent_checkpoint=str(parent),parent_checkpoint_sha256=sha(parent),from_source_sha256=meta['source_sha256'],to_source_sha256=hashes,changed_files=changed,config_differences=differences,agent_exact=True,assets_exact=len(assets),source_transition_once=True)
 (R/'groundpick-target-precision-source-transition.json').write_text(json.dumps(transition,indent=2)+'\n')
 partial=[]
 for stage in ['nominal','measured']:
  x=torch.load(R/f'groundpick-local-endpoint-policy-{stage}-001/traces.pt',map_location='cpu',weights_only=False);m=x['measurements'];assert bool((x['initial']['origins'][:,2]==0).all())
  mask=(m['phase']>=.375)&(m['phase']<.425)&~m['head_contact']&~m['self_contact'];height=m['mouth_height'][mask]
  partial.append(dict(stage=stage,frames=int(mask.sum()),old_proximity_mean=float(torch.exp(-(height/.1).square()).mean()),new_proximity_mean=float(torch.exp(-((height-.02)/.025).square()).mean()),height_mean_m=float(height.mean())))
 report=dict(status='PASS',source_sha256=hashes,task=PRECISION_TASK,parent_checkpoint=str(parent),parent_checkpoint_sha256=sha(parent),parent_controls=meta['controls'],parent_clock=12288,parent_learning_rate=cp['optimizer_state_dict']['param_groups'][0]['lr'],config_diff=differences,agent_exact=True,assets_exact=len(assets),reward_grid_cases=rows,gradient_sign=True,original_evaluator_gates_unchanged=True,evaluator_sha256=sha(W/'scripts/evaluate_v2_ground_pick.py'),partial_saved_reward=partial,budget=dict(new_updates=256,N=24576,T=24,E=5,M=24,new_formal_controls=150994944,old_all_formal_controls=377487360,total_cap=528482304),limitations=['Mature fine-stage reward only; far-away gradient weaker than historical discovery.','Reward itself is not temporal100ms/quiet acceptance; original assessment gates retained.','No success claim from reward calculation.','Original clean_cycle includes continuousfeet, stricter than report prose; preserve originalcounts and separatelyaudit settling.'],script_sha256=sha(Path(__file__)))
 (R/'groundpick-target-precision-cpu-001.json').write_text(json.dumps(report,indent=2)+'\n');print('GROUND_PRECISION_CPU_PASS',json.dumps({'changes':differences,'partial':partial}),flush=True);return report
if __name__=='__main__':main()
