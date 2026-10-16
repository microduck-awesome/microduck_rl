"""One fresh lineage, frozen production source, bounded qualification/formal/eval."""
import os,sys,json,time,gc,hashlib,traceback,threading,signal,importlib.util,ast
from pathlib import Path
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'sitstand-history-base-worktree';os.chdir(W);sys.path.insert(0,str(W/'scripts'));os.environ.pop('MICRODUCK_WARM_START',None)
state=R/'sitstand-history-base-pipeline-status.json';log=R/'sitstand-history-base-pipeline-001.log';stop=threading.Event();patterns=('ccd overflow','contact overflow','constraint buffer full','broadphase overflow','narrowphase overflow','height field collision overflow','nefc overflow','njmax_nnz overflow')
worker_start=time.monotonic();worker_unix=time.time()
def write(x):
 x.update(pid=os.getpid(),updated_unix=time.time());t=state.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(state)
def watch():
 while not stop.wait(2):
  if log.exists() and any(p in log.read_text(errors='replace').lower() for p in patterns):
   (R/'sitstand-history-base-capacity-rejection.json').write_text(json.dumps(dict(status='CAPACITY_GUARD_REJECTED',pid=os.getpid())));os.kill(os.getpid(),signal.SIGTERM);return
threading.Thread(target=watch,daemon=True).start();write(dict(status='IMPORTING',formal_controls=0,prior_all_sitstand_formal_controls=226492416));print('WORKER_PID',os.getpid(),flush=True)
import torch
import train_v2_sitstand as train
from mjlab_microduck.custom_v2.sitstand_history import HISTORY_TASK as RELEASE_TASK,make_history_sitstand_env_cfg,curriculum_mapping
from rsl_rl.modules.normalization import EmpiricalNormalization

def tensor_hash(sd):
 h=hashlib.sha256()
 for k,v in sorted(sd.items()):h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def cpu_copy(x):
 if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:cpu_copy(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return type(x)(cpu_copy(v)for v in x)
 return x

def cpu_contract():
 from dataclasses import asdict
 from copy import deepcopy
 from types import SimpleNamespace
 from mjlab_microduck.tasks import mdp
 from mjlab_microduck.custom_v2.sitstand_clear import make_clear_sitstand_env_cfg
 from mjlab.tasks.registry import load_rl_cfg
 hashes=train.source_hashes(W)
 spec=importlib.util.spec_from_file_location('historical_c832147',R/'sitstand-history-c832147-cfg.py');historical=importlib.util.module_from_spec(spec);spec.loader.exec_module(historical)
 old=historical.make_microduck_sitstand_env_cfg();cfg=make_history_sitstand_env_cfg();clear=make_clear_sitstand_env_cfg()
 def clean(x):
  if callable(x):return x.__module__+'.'+x.__qualname__
  if isinstance(x,dict):return {k:clean(v)for k,v in x.items()}
  if isinstance(x,(tuple,list)):return [clean(v)for v in x]
  return x
 assert set(cfg.rewards)==set(old.rewards),(set(cfg.rewards)-set(old.rewards),set(old.rewards)-set(cfg.rewards))
 reward_rows=[]
 for name,term in cfg.rewards.items():
  reference=old.rewards[name];assert term.weight==reference.weight and term.func is reference.func,name
  p1,p2=deepcopy(term.params),deepcopy(reference.params)
  for p in(p1,p2):
   for key in('sit_overrides','sit_z','stand_z','asset_cfg'):p.pop(key,None)
  assert clean(p1)==clean(p2),(name,clean(p1),clean(p2))
  reward_rows.append(dict(name=name,weight=term.weight,func=term.func.__name__,parameter_parity_except_declared_V2_geometry_and_selectors=True))
 assert cfg.episode_length_s==old.episode_length_s==12.
 assert cfg.commands['twist'].resampling_time_range==old.commands['twist'].resampling_time_range==(3.5,6.5)
 assert cfg.commands['twist'].ramp_s==old.commands['twist'].ramp_s==2.
 assert type(cfg.commands['head_pose']) is type(old.commands['head_pose']) is mdp.UniformPoseCommandCfg
 assert cfg.commands['head_pose'].zero_command_prob==old.commands['head_pose'].zero_command_prob==0.
 assert cfg.commands['head_pose'].resampling_time_range==(2.,5.)
 expected=deepcopy(clear);expected.episode_length_s=12.;expected.commands['head_pose']=deepcopy(cfg.commands['head_pose']);expected.rewards.pop('neck_strain_leaned')
 for name in('seated_head_jerk','seated_tip'):expected.events.pop(name)
 for name in('seated_head_jerk_prob','seated_tip_range'):expected.curriculum.pop(name)
 for term in expected.curriculum.values():
  for stages in term.params.values():
   if isinstance(stages,list):
    for v in stages:
     if isinstance(v,dict)and 'step'in v:v['step']//=4
 assert clean(asdict(expected))==clean(asdict(cfg)),'Unexpected cfg difference'
 assert set(cfg.curriculum)==set(old.curriculum)
 mapping=curriculum_mapping(cfg);checks=[]
 # Compare actual native curriculum call effects at both sides of every boundary.
 def effects(config,clock):
  e=SimpleNamespace(common_step_counter=clock,device='cpu');rew=deepcopy(config.rewards);ev=deepcopy(config.events);cmd={n:SimpleNamespace(cfg=deepcopy(c))for n,c in config.commands.items()}
  e.reward_manager=SimpleNamespace(get_term_cfg=lambda name:rew[name]);e.event_manager=SimpleNamespace(get_term_cfg=lambda name:ev[name]);e.command_manager=SimpleNamespace(get_term=lambda name:cmd[name])
  outputs={}
  for name,t in config.curriculum.items():outputs[name]=clean(t.func(e,torch.tensor([0]),**deepcopy(t.params)).tolist())
  return clean(dict(outputs=outputs,rewards={n:t.weight for n,t in rew.items()},eventparams={n:t.params for n,t in ev.items()},headranges=cmd['head_pose'].cfg.ranges))
 reference=deepcopy(cfg)
 for t in reference.curriculum.values():
  for stages in t.params.values():
   if isinstance(stages,list):
    for v in stages:
     if isinstance(v,dict)and 'step'in v:v['step']*=4
 points=sorted({0,7680,*[max(0,r['native_step']+d)for r in mapping for d in(-1,0,1)]})
 for clock in points:
  assert effects(cfg,clock)==effects(reference,clock*4),clock;checks.append(clock)
 for name,term in cfg.curriculum.items():
  new,orig=deepcopy(term.params),deepcopy(old.curriculum[name].params)
  for key,stages in new.items():
   if isinstance(stages,list):
    for v in stages:
     if isinstance(v,dict)and 'step'in v:v['step']*=4
  if name=='head_pose_range':
   for a,b in zip(new['range_stages'],orig['range_stages']):
    assert a['step']==b['step']
    for (lo,hi),(olo,ohi)in zip(a['ranges'],b['ranges']):assert olo<=lo<=hi<=ohi
  else:assert clean(new)==clean(orig),(name,new,orig)
 agent=asdict(load_rl_cfg(RELEASE_TASK));original=asdict(historical.MicroduckSitStandRlCfg)
 for k in ('actor','critic','num_steps_per_env'):assert clean(agent[k])==clean(original[k]),k
 algo=deepcopy(agent['algorithm']);algo['class_name']=original['algorithm']['class_name'];algo['num_mini_batches']=4;assert clean(algo)==clean(original['algorithm'])
 assert agent['algorithm']['num_mini_batches']==16 and agent['algorithm']['num_learning_epochs']==5
 assert 16384*24//16==4096*24//4==24576
 for play in(False,True):
  c=make_history_sitstand_env_cfg(play=play);assert c.episode_length_s==12. and c.actions['joint_pos'].spike_rejection is False and c.actions['joint_pos'].o0_local_history
 report=dict(status='PASS',source_sha256=hashes,known_prior_actor_hashes=[],history_commit='c832147',history_cfg_sha256=hashlib.sha256((R/'sitstand-history-c832147-cfg.py').read_bytes()).hexdigest(),reward_rows=reward_rows,curriculum_mapping=mapping,native_boundary_call_points=checks,command_seconds=dict(posture_dwell=[3.5,6.5],head_dwell=[2,5],slew=2,episode=12),head_sampling='historical UniformPose; zero_probability0',actual_agent=agent,formal_controls125829120=True,scope='Actual historical config with current qualified mdp implementation; V2 geometry/physics adaptation retained; no claim of equal rollout refresh',reset_boundary='Native curriculum compute on reset, strict > vs >= unchanged; actual activation may lag configured step until next reset',historical_evidence='documented CLI + actual source, no raw W&B config locally',no_new_head_L1_EMA=True)
 train.write_json(R/'sitstand-history-base-cpu-001.json',report);print('CPU_HISTORICAL_CONTRACT_PASS',flush=True);return report

active={};native_runner_cls=train.load_runner_cls

def instrumented_runner_cls(task):
 original=native_runner_cls(task)
 def build(*args,**kwargs):
  assert '--resume' not in sys.argv and '--source-transition' not in sys.argv
  cpu_rng=torch.get_rng_state().clone();runner=original(*args,**kwargs);env=args[0].unwrapped
  assert env.common_step_counter==0 and runner.current_learning_iteration==0 and len(runner.alg.optimizer.state)==0
  saved=runner.alg.save();normalizers={}
  for role,module in [('actor',runner.alg.actor),('critic',runner.alg.critic)]:
   for name,n in module.named_modules():
    if isinstance(n,EmpiricalNormalization):
     assert int(n.count)==0 and (n._mean==0).all() and (n._var==1).all() and (n._std==1).all()
     normalizers[role+'.'+name]=dict(count=int(n.count),mean0=True,var1=True,std1=True)
  assert len(normalizers)==2,normalizers
  initial_actor=tensor_hash(saved['actor_state_dict']);initial_critic=tensor_hash(saved['critic_state_dict']);assert initial_actor not in cpu['known_prior_actor_hashes']
  if active['label']=='allocation':
   validation=torch.load(R/'sitstand-history-base-actual16384-001/checkpoint_000016.pt',map_location='cpu',weights_only=False)
   assert initial_actor!=tensor_hash(validation['actor_state_dict']) and initial_critic!=tensor_hash(validation['critic_state_dict']);del validation
  out=active['out'];torch.save(dict(model=cpu_copy(saved),common_step_counter=0,learning_iteration=0,cpu_rng_after_model_creation=torch.get_rng_state(),cuda_rng_after_model_creation=torch.cuda.get_rng_state('cuda:1')),out/'initialization.pt')
  train.write_json(out/'initialization.json',dict(status='FRESH_VERIFIED',clock=0,completed_updates=0,inherited_controls=0,optimizer_state_entries=0,learning_rate=runner.alg.learning_rate,actor_sha256=initial_actor,critic_sha256=initial_critic,normalizers=normalizers,no_checkpoint_loaded=True,not_validation_weights=active['label']=='allocation',initialization_state_sha256=hashlib.sha256((out/'initialization.pt').read_bytes()).hexdigest(),scope='Observer after native runner construction; snapshot does not change RNG/state. Random initialization distinct from all448/384/256/128 ancestors.'))
  native_optimizer_step=runner.alg.optimizer.step;steps=[0];updates=[0]
  def counted_step(*a,**kw):
   result=native_optimizer_step(*a,**kw);steps[0]+=1;return result
  runner.alg.optimizer.step=counted_step
  native_logger=runner.logger.log
  def counted_log(*a,**kw):
   updates[0]+=1;assert steps[0]==updates[0]*80
   row=dict(update=updates[0],optimizer_step_calls=80,total_optimizer_step_calls=steps[0],native_clock=env.common_step_counter,historical_equivalent_clock=env.common_step_counter*4,learning_rate=runner.alg.learning_rate)
   with(out/'native-optimizer-step-callback.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
   if active['label']=='allocation'and updates[0]%64==0 and updates[0]<320:
    runner.save(str(out/f'checkpoint_{updates[0]:06d}.pt'))
   if updates[0]in(1,64,125,128,188,192,250,256,313,320):
    row['live_reward_weights']={n:env.reward_manager.get_term_cfg(n).weight for n in env.reward_manager._term_names}
    row['head_ranges']=env.command_manager.get_term('head_pose').cfg.ranges
    with(out/'curriculum-live.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
   return native_logger(*a,**kw)
  runner.logger.log=counted_log
  def forbidden_load(*args,**kwargs):raise AssertionError('Fresh allocation must never call runner.load')
  runner.load=forbidden_load
  return runner
 return build
train.load_runner_cls=instrumented_runner_cls

def run_train(label,n,it,validation=False,forced=False,qualification=None):
 out=R/('sitstand-history-base-'+label+'-001');active.update(label=label,out=out);argv=['train','--task',RELEASE_TASK,'--output',str(out),'--device','cuda:1','--seed','2026101611','--num-envs',str(n),'--iterations',str(it)]
 if validation:argv+=['--validation-only']
 if forced:argv+=['--force-timeouts']
 if qualification:argv+=['--qualification',str(qualification)]
 write(dict(status='RUNNING',phase=label,output=str(out),formal_controls=0 if validation else 'read formal status',prior_all_sitstand_formal_controls=226492416));print('PHASE_START',label,flush=True);t=time.monotonic();sys.argv=argv;train.main();wall=time.monotonic()-t
 rep=json.loads((out/'report.json').read_text());run=json.loads((out/'run.json').read_text());assert rep['status']=='ALLOCATION_COMPLETE' and rep['completed_updates']==it and rep['fresh_initialization'] and rep['inherited_controls']==0 and rep['common_step_counter']==it*24 and run['arguments']['resume']is None
 train.write_json(out/'shared-process-phase-wall.json',dict(seconds_including_environment_and_validation=wall,imports_shared=True,controls=rep['allocation_controls'],controls_per_second=rep['allocation_controls']/wall));print('PHASE_DONE',label,wall,flush=True);gc.collect();torch.cuda.empty_cache();return out,rep

def validate(out,rep,timeout_min):
 assert rep['terminal_bootstrap_samples']>=timeout_min and rep['contact_peak']<rep['contact_capacity'];m=rep['training_metrics'];assert m['metrics']['feedback_not_ready_fraction']==0 and m['metrics']['nonfinite_fraction']==0 and max(m['oracle_max_abs'])<2e-6
 diag=[json.loads(line)['metrics']for line in(out/'ppo-diagnostics.jsonl').read_text().splitlines()];assert all(d['PPO/kl_mean']<=.5 and d['PPO/clip_fraction']<=.8 for d in diag)

try:
 cpu=cpu_contract();smoke,s=run_train('smoke',64,5,True,True);validate(smoke,s,32);assert s['onnx_max_abs_error']<2e-5 and s['onnx_actor_normalizer_unchanged']
 actual,r=run_train('actual16384',16384,16,True,True);validate(actual,r,8192)
 qualification=R/'sitstand-history-base-qualification-001.json';train.write_json(qualification,dict(status='PASS',task=RELEASE_TASK,num_envs=16384,source_sha256=r['source_sha256'],fresh_only=True,smoke=str(smoke/'report.json'),actual_N=str(actual/'report.json'),cpu=str(R/'sitstand-history-base-cpu-001.json'),throughput=r['throughput'],throughput_target_met=r['throughput']['median_control_per_second']>=100000,performance_policy='Measured performance is reported; correctness guards hard. No100k-only blocker.',formal_parent=None,formal_control_cap=125829120,all_history_formal_cap=352321536))
 out,f=run_train('allocation',16384,320,qualification=qualification);assert f['allocation_controls']==125829120 and f['controls']==125829120;validate(out,f,1)
 spec=importlib.util.spec_from_file_location('fresh_release_eval',W/'scripts/evaluate_v2_sitstand_clock_fixed.py');evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
 for node in(64,128,192,256,320):
  checkpoint=out/f'checkpoint_{node:06d}.pt';assert checkpoint.is_file()
  metadata=torch.load(checkpoint,map_location='cpu',weights_only=False)['infos']['v2'];assert metadata['completed_updates']==node
  for stage in('nominal','measured'):
   result=R/(f'sitstand-history-base-{node:04d}-'+stage);write(dict(status='RUNNING',phase=f'independent-{node}-{stage}',output=str(result),formal_controls=125829120,all_history_formal_controls=352321536));sys.argv=['evaluate','--checkpoint',str(checkpoint),'--output',str(result),'--device','cuda:1','--stage',stage,'--num-envs','256','--seed','2603'];evaluator.main();gc.collect();torch.cuda.empty_cache()
 assert train.source_hashes(W)==cpu['source_sha256'];write(dict(status='SITSTAND_FRESH_STAGE_BUDGET_COMPLETE_NO_FURTHER_TRAINING',completed_updates=320,formal_controls=125829120,all_history_formal_controls=352321536,checkpoint=str(out/'checkpoint_000320.pt'),worker_wall_s=time.monotonic()-worker_start,worker_start_unix=worker_unix))
except BaseException as e:
 write(dict(status='FAILED_REQUIRES_DIAGNOSIS',exception=repr(e)));traceback.print_exc();raise
finally:stop.set()
