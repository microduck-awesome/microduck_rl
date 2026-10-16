"""One complete historical curriculum, continuous fresh PPO, bounded by1536 updates."""
from pathlib import Path
from dataclasses import asdict
from copy import deepcopy
from functools import wraps
from types import SimpleNamespace
import gc,hashlib,json,os,runpy,sys,time,threading,signal
R=Path(__file__).resolve().parent;W=R/'roulade-history-worktree';STATUS=R/'roulade-history-status-001.json'
assert not STATUS.exists()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def state(**kw):
 p=STATUS.with_suffix('.tmp');p.write_text(json.dumps(dict(pid=os.getpid(),updated_unix=time.time(),formal_control_cap=603979776,prior_all_roulade_formal_controls=150994944,**kw),indent=2)+'\n');p.replace(STATUS)
state(status='PRELOADING_AND_CPU_CONTRACT',formal_controls=0)
sys.path.insert(0,str(W/'scripts'));ns=runpy.run_path(str(W/'scripts/train_v2_roulade.py'),run_name='roulade_history_trainer')
import torch
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
from mjlab_microduck.custom_v2.roulade import make_roulade_env_cfg,ROULADE_TASK
from mjlab_microduck.custom_v2.roulade_history import make_history_roulade_env_cfg,HISTORY_TASK
from rsl_rl.modules.normalization import EmpiricalNormalization
torch.set_num_threads(2)
plan=read(R/'roulade-history-plan-001.json');reference=read(R/'roulade-qualification-001.json')
hashes=ns['source_hashes'](W);assert hashes==plan['source_sha256']
def verify_sources():
 assert ns['source_hashes'](W)==hashes
 for name,digest in reference['model_asset_sha256'].items():assert sha(W/name)==digest,name
verify_sources()

def clean(x):
 if callable(x):return x.__module__+'.'+x.__qualname__
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [clean(v) for v in x]
 return x

def cpu_contract():
 original=make_roulade_env_cfg();cfg=make_history_roulade_env_cfg();expected=deepcopy(original);mapping=[]
 for name,term in expected.curriculum.items():
  for key,values in term.params.items():
   if isinstance(values,list):
    for item in values:
     if isinstance(item,dict) and 'step' in item:
      old=item['step'];item['step']//=4;mapping.append(dict(term=name,field=key,historical_step=old,native_step=item['step']))
 assert clean(asdict(expected))==clean(asdict(cfg))
 assert cfg.episode_length_s==5. and cfg.decimation==4 and cfg.sim.mujoco.timestep==.005
 assert cfg.actions['joint_pos'].spike_rejection is False and cfg.actions['joint_pos'].o0_local_history
 assert cfg.events['set_roulade_state'].func is original.events['set_roulade_state'].func
 assert cfg.events['set_roulade_state'].params==original.events['set_roulade_state'].params
 assert len(cfg.rewards)==len(original.rewards)
 before=asdict(load_rl_cfg(ROULADE_TASK));after=asdict(load_rl_cfg(HISTORY_TASK));want=deepcopy(before)
 want['algorithm']['num_mini_batches']=16;want['run_name']='history_nominal'
 assert clean(want)==clean(after)
 assert after['num_steps_per_env']==24 and after['algorithm']['num_learning_epochs']==5
 assert 16384*24//16==4096*24//4==24576
 assert after['algorithm']['symmetry_cfg']['use_mirror_loss'] and not after['algorithm']['symmetry_cfg']['use_data_augmentation']
 def effects(c,clock):
  rewards=deepcopy(c.rewards);events=deepcopy(c.events)
  env=SimpleNamespace(common_step_counter=clock,device='cpu',reward_manager=SimpleNamespace(get_term_cfg=lambda n:rewards[n]),event_manager=SimpleNamespace(get_term_cfg=lambda n:events[n]))
  for term in c.curriculum.values():term.func(env,torch.tensor([0]),**deepcopy(term.params))
  return clean(dict(rewards={n:t.weight for n,t in rewards.items()},events={n:t.params for n,t in events.items()}))
 points=sorted({0,1536*24,*[max(0,row['native_step']+d) for row in mapping for d in (-1,0,1)]})
 for clock in points:assert effects(cfg,clock)==effects(original,clock*4),clock
 result=dict(status='PASS',source_sha256=hashes,assets_sha256=reference['model_asset_sha256'],reward_reset_observation_action_physics_exact_original_V2=True,
  actual_agent=after,config=asdict(cfg),curriculum_mapping=mapping,boundary_call_points=points,
  reset_semantics='Native reset-only application and strict comparison preserved; scheduled boundaries/4 do not imply identical physical activation times.',
  learning_dose=dict(native_updates=1536,historical_equivalent_updates=6144,formal_controls=603979776,optimizer_steps=122880),
  historical_scope='Pinned original Roulade reward/lesson source and full6000-update curriculum; local public ONNX hasrun_pathNone, so exact successful training lineage is not asserted.')
 ns['write_json'](R/'roulade-history-cpu-001.json',result)
 return result

active={};native_runner_class=ns['load_runner_cls']
def instrumented_class(task):
 cls=native_runner_class(task)
 def build(*args,**kwargs):
  runner=cls(*args,**kwargs);env=args[0].unwrapped
  assert env.common_step_counter==0 and runner.current_learning_iteration==0 and not runner.alg.optimizer.state
  normalizers=[]
  for role,model in [('actor',runner.alg.actor),('critic',runner.alg.critic)]:
   for name,n in model.named_modules():
    if isinstance(n,EmpiricalNormalization):
     assert int(n.count)==0 and bool((n._mean==0).all()) and bool((n._var==1).all()) and bool((n._std==1).all());normalizers.append(role+'.'+name)
  assert len(normalizers)==2
  snapshot=runner.alg.save()
  def tensor_hash(sd):
   h=hashlib.sha256()
   for k,v in sorted(sd.items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
   return h.hexdigest()
  actor_hash=tensor_hash(snapshot['actor_state_dict']);critic_hash=tensor_hash(snapshot['critic_state_dict'])
  if active['phase']=='formal':
   old=torch.load(R/'roulade-history-actual16384-001/checkpoint_000016.pt',map_location='cpu',weights_only=False)
   assert actor_hash!=tensor_hash(old['actor_state_dict']) and critic_hash!=tensor_hash(old['critic_state_dict']);del old
  ns['write_json'](active['out']/'initialization.json',dict(status='FRESH_VERIFIED',warm_start=False,new_skill_clock=0,actor_sha256=actor_hash,critic_sha256=critic_hash,normalizers=normalizers,normalizer_count=0,normalizer_mean=0,normalizer_variance=1,optimizer_entries=0,learning_rate=runner.alg.learning_rate,no_checkpoint_loaded=True,validation_weights_not_inherited=active['phase']=='formal'))
  del snapshot
  optimizer_step=runner.alg.optimizer.step;steps=[0];updates=[0]
  @wraps(optimizer_step)
  def counted_step(*a,**kw):
   value=optimizer_step(*a,**kw);steps[0]+=1;return value
  runner.alg.optimizer.step=counted_step
  native_log=runner.logger.log
  @wraps(native_log)
  def counted_log(*a,**kw):
   updates[0]+=1;assert steps[0]==updates[0]*80
   row=dict(completed_updates=updates[0],optimizer_step_calls=steps[0],native_clock=env.common_step_counter,historical_clock=env.common_step_counter*4)
   with (active['out']/'optimizer-calls.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   if updates[0] in (1,125,128,250,256,375,512,625,750,768,875,1024,1280,1500,1536):
    row['reward_weights']={n:env.reward_manager.get_term_cfg(n).weight for n in env.reward_manager._term_names}
    row['spawn_params']=env.event_manager.get_term_cfg('set_roulade_state').params
    with (active['out']/'curriculum-live.jsonl').open('a') as f:f.write(json.dumps(row,default=str)+'\n')
   if active['phase']=='formal' and updates[0]%128==0 and updates[0]<1536:
    runner.save(str(active['out']/f'checkpoint_{updates[0]:06d}.pt'))
   return native_log(*a,**kw)
  runner.logger.log=counted_log
  def forbidden_load(*a,**kw):raise AssertionError('Fresh historical Roulade cannot load a checkpoint')
  runner.load=forbidden_load
  return runner
 return build
ns['main'].__globals__['load_runner_cls']=instrumented_class

def train(phase,n,it,validation=False,qualification=None):
 verify_sources();out=R/f'roulade-history-{phase}-001';active.update(phase=phase,out=out)
 state(status='RUNNING',phase=phase,output=str(out),formal_controls=0 if validation else 'read formal status')
 argv=[str(W/'scripts/train_v2_roulade.py'),'--task',HISTORY_TASK,'--output',str(out),'--device','cuda:0','--num-envs',str(n),'--iterations',str(it)]
 if validation:argv+=['--validation-only','--force-timeouts']
 if qualification:argv+=['--qualification',str(qualification)]
 sys.argv=argv;start=time.monotonic();ns['main']();d=read(out/'report.json');assert d['status']=='ALLOCATION_COMPLETE' and d['completed_updates']==it and d['allocation_controls']==n*24*it
 assert d['fresh_initialization'] and d['inherited_controls']==0
 m=d['training_metrics'];assert m['metrics']['feedback_invalid_fraction']==0 and m['metrics']['nonfinite_fraction']==0
 assert d['contact_peak']<d['contact_capacity']
 if validation:assert max(m['oracle_max_abs'])<2e-5
 ns['write_json'](out/'shared-process-wall.json',dict(seconds=time.monotonic()-start,controls=d['allocation_controls'],imports_excluded=True))
 gc.collect();torch.cuda.empty_cache();return out,d

stop=threading.Event()
def capacity_watch():
 p=R/'roulade-history-001.log';pos=0;tail=''
 patterns=('ccd overflow','contact overflow','constraint buffer full','broadphase overflow','narrowphase overflow','nefc overflow','njmax_nnz overflow')
 while not stop.wait(2):
  if not p.exists():continue
  with p.open(errors='replace') as f:f.seek(pos);chunk=f.read();pos=f.tell()
  text=(tail+chunk).lower();tail=text[-512:]
  if any(s in text for s in patterns):
   ns['write_json'](R/'roulade-history-capacity-failure-001.json',dict(status='CAPACITY_REJECTION',pid=os.getpid()));os.kill(os.getpid(),signal.SIGTERM);return
try:
 cpu=cpu_contract();state(status='CPU_CONTRACT_PASS_WAITING_GPU0',formal_controls=0)
 deadline=time.monotonic()+1800
 while True:
  marker=read(R/'velstand-teacher-yaw-status-001.json')
  if marker['status'] in ('TEACHER_RESPONSE_DATA_COMPLETE_NOT_A_VALIDATED_MAP','STOPPED_REQUIRES_REVIEW'):
   p=Path(f"/proc/{marker['pid']}/cmdline");cmd=p.read_bytes() if p.exists() else b''
   if b'run-velstand-teacher-yaw-001.py' not in cmd:break
  assert time.monotonic()<deadline,'Prior GPU0 job exceeded bounded wait';time.sleep(2)
 torch.cuda.set_device('cuda:0');threading.Thread(target=capacity_watch,daemon=True).start()
 small,smoke=train('smoke64',64,5,True);assert smoke['onnx_max_abs_error']<2e-5 and smoke['terminal_bootstrap_samples']>=32
 actual,validation=train('actual16384',16384,16,True);assert validation['terminal_bootstrap_samples']>=8192
 qpath=R/'roulade-history-qualification-001.json';ns['write_json'](qpath,dict(status='PASS',source_sha256=hashes,assets_sha256=reference['model_asset_sha256'],num_envs=16384,maximum_updates=1536,formal_control_cap=603979776,cpu=str(R/'roulade-history-cpu-001.json'),smoke=str(small/'report.json'),actual=str(actual/'report.json'),throughput=validation['throughput'],validation_inherited=False))
 out,formal=train('formal',16384,1536,qualification=qpath)
 assert formal['allocation_controls']==603979776
 script=W/'scripts/evaluate_v2_roulade.py';reports=[]
 for node in (128,256,512,768,1024,1280,1536):
  checkpoint=out/f'checkpoint_{node:06d}.pt';assert checkpoint.is_file()
  for stage in ('nominal','measured'):
   dest=R/f'roulade-history-{node:04d}-{stage}-001';state(status='EVALUATING_CURVE',phase=dest.name,formal_controls=603979776)
   sys.argv=[str(script),'--checkpoint',str(checkpoint),'--output',str(dest),'--device','cuda:0','--stage',stage]
   runpy.run_path(str(script),run_name='__main__');assert read(dest/'report.json')['status']=='EVALUATED_NOT_HARDWARE_ACCEPTANCE';reports.append(str(dest/'report.json'));gc.collect();torch.cuda.empty_cache()
 state(status='HISTORICAL_CURRICULUM1536_COMPLETE_REVIEW_CURVE',formal_controls=603979776,all_roulade_formal_controls=754974720,reports=reports)
except BaseException as exc:
 state(status='STOPPED_REQUIRES_DIAGNOSIS',error=repr(exc));raise
finally:stop.set()
