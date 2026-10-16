"""One fresh lineage, frozen production source, bounded qualification/formal/eval."""
import os,sys,json,time,gc,hashlib,traceback,threading,signal,importlib.util,ast
from functools import wraps
from pathlib import Path
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'groundpick-health-history-worktree';os.chdir(W);sys.path.insert(0,str(W/'scripts'));os.environ.pop('MICRODUCK_WARM_START',None)
state=R/'groundpick-health-history-pipeline-status.json';log=R/'groundpick-health-history-pipeline-001.log';stop=threading.Event();patterns=('ccd overflow','contact overflow','constraint buffer full','broadphase overflow','narrowphase overflow','height field collision overflow','nefc overflow','njmax_nnz overflow')
worker_start=time.monotonic();worker_unix=time.time()
def write(x):
 x.update(pid=os.getpid(),updated_unix=time.time());t=state.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(state)
def watch():
 while not stop.wait(2):
  if log.exists() and any(p in log.read_text(errors='replace').lower() for p in patterns):
   (R/'groundpick-health-history-capacity-rejection.json').write_text(json.dumps(dict(status='CAPACITY_GUARD_REJECTED',pid=os.getpid())));os.kill(os.getpid(),signal.SIGTERM);return
threading.Thread(target=watch,daemon=True).start();write(dict(status='IMPORTING',formal_controls=0,prior_all_groundpick_formal_controls=75497472));print('WORKER_PID',os.getpid(),flush=True)
import torch
import train_v2_ground_pick as train
from mjlab_microduck.custom_v2.ground_pick_history import HISTORY_TASK as RELEASE_TASK
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
 cpu=json.loads((R/'groundpick-health-history-cpu-003.json').read_text());assert cpu['status']=='CPU_CONTRACT_PASS_GPU_NOT_QUALIFIED' and train.source_hashes(W)==cpu['source_sha256'];cpu['known_prior_actor_hashes']=[]
 print('CPU_EXHAUSTIVE_CFG_SOURCE_REUSED_PASS',flush=True);return cpu

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
   validation=torch.load(R/'groundpick-health-history-actual24576-001/checkpoint_000016.pt',map_location='cpu',weights_only=False)
   assert initial_actor!=tensor_hash(validation['actor_state_dict']) and initial_critic!=tensor_hash(validation['critic_state_dict']);del validation
  out=active['out'];torch.save(dict(model=cpu_copy(saved),common_step_counter=0,learning_iteration=0,cpu_rng_after_model_creation=torch.get_rng_state(),cuda_rng_after_model_creation=torch.cuda.get_rng_state('cuda:0')),out/'initialization.pt')
  train.write_json(out/'initialization.json',dict(status='FRESH_VERIFIED',clock=0,completed_updates=0,inherited_controls=0,optimizer_state_entries=0,learning_rate=runner.alg.learning_rate,actor_sha256=initial_actor,critic_sha256=initial_critic,normalizers=normalizers,no_checkpoint_loaded=True,not_validation_weights=active['label']=='allocation',initialization_state_sha256=hashlib.sha256((out/'initialization.pt').read_bytes()).hexdigest(),scope='Observer after native runner construction; snapshot does not change RNG/state. Fresh actor/critic, empty Adam and normalizer statistics, no checkpoint load.'))
  native_optimizer_step=runner.alg.optimizer.step;steps=[0];updates=[0]
  def counted_step(*a,**kw):
   result=native_optimizer_step(*a,**kw);steps[0]+=1;return result
  runner.alg.optimizer.step=counted_step
  native_logger=runner.logger.log
  @wraps(native_logger)
  def counted_log(*a,**kw):
   updates[0]+=1;assert steps[0]==updates[0]*120
   row=dict(update=updates[0],optimizer_step_calls=120,total_optimizer_step_calls=steps[0],native_clock=env.common_step_counter,historical_equivalent_clock=env.common_step_counter*6,learning_rate=runner.alg.learning_rate)
   with(out/'native-optimizer-step-callback.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
   if active['label']=='allocation'and updates[0]%64==0 and updates[0]<512:
    runner.save(str(out/f'checkpoint_{updates[0]:06d}.pt'))
   if updates[0]in(1,64,128,192,256,320,384,448,512):
    row['low_reset_exposure_HOME_LOW']=env._gp_reset_exposure.detach().cpu().tolist()
    row['live_curriculum_parameters']={name:env.event_manager.get_term_cfg(name).params for name in ('randomize_com','randomize_head_com')}
    row['live_reward_weights']={n:env.reward_manager.get_term_cfg(n).weight for n in env.reward_manager._term_names}
    with(out/'curriculum-live.jsonl').open('a')as f:f.write(json.dumps(row,default=str)+'\n')
   return native_logger(*a,**kw)
  runner.logger.log=counted_log
  def forbidden_load(*args,**kwargs):raise AssertionError('Fresh allocation must never call runner.load')
  runner.load=forbidden_load
  return runner
 return build
train.load_runner_cls=instrumented_runner_cls

def run_train(label,n,it,validation=False,forced=False,qualification=None):
 out=R/('groundpick-health-history-'+label+'-001');active.update(label=label,out=out);argv=['train','--task',RELEASE_TASK,'--output',str(out),'--device','cuda:0','--seed','2026101636','--num-envs',str(n),'--iterations',str(it)]
 if validation:argv+=['--validation-only']
 if forced:argv+=['--force-timeouts']
 if qualification:argv+=['--qualification',str(qualification)]
 write(dict(status='RUNNING',phase=label,output=str(out),formal_controls=0 if validation else 'read formal status',prior_all_groundpick_formal_controls=75497472));print('PHASE_START',label,flush=True);t=time.monotonic();sys.argv=argv;train.main();wall=time.monotonic()-t
 rep=json.loads((out/'report.json').read_text());run=json.loads((out/'run.json').read_text());assert rep['status']=='ALLOCATION_COMPLETE' and rep['completed_updates']==it and rep['fresh_initialization'] and rep['inherited_controls']==0 and rep['common_step_counter']==it*24 and run['arguments']['resume']is None
 train.write_json(out/'shared-process-phase-wall.json',dict(seconds_including_environment_and_validation=wall,imports_shared=True,controls=rep['allocation_controls'],controls_per_second=rep['allocation_controls']/wall));print('PHASE_DONE',label,wall,flush=True);gc.collect();torch.cuda.empty_cache();return out,rep

def validate(out,rep,timeout_min):
 assert rep['terminal_bootstrap_samples']>=timeout_min and rep['contact_peak']<rep['contact_capacity'];m=rep['training_metrics'];assert m['metrics']['feedback_not_ready_fraction']==0 and m['metrics']['nonfinite_fraction']==0 and max(m['oracle_max_abs'])<2e-6
 diag=[json.loads(line)['metrics']for line in(out/'ppo-diagnostics.jsonl').read_text().splitlines()];assert all(d['PPO/kl_mean']<=.5 and d['PPO/clip_fraction']<=.8 for d in diag)

try:
 cpu_deadline=time.monotonic()+600
 while not (R/'groundpick-health-history-cpu-003.json').exists():
  assert time.monotonic()<cpu_deadline,'CPU qualification unavailable; no GPU action'
  time.sleep(2)
 cpu=cpu_contract()
 write(dict(status='WAITING_GPU0_ROULADE_COMPLETE',formal_controls=0))
 handoff=R/'roulade-history-status-001.json';deadline=time.monotonic()+7200
 while True:
  prior=json.loads(handoff.read_text())
  if prior['status']=='HISTORICAL_CURRICULUM1536_COMPLETE_REVIEW_CURVE'and not Path('/proc/344162').exists():break
  if prior['status']=='STOPPED_REQUIRES_DIAGNOSIS':raise RuntimeError('Roulade predecessor failed; no automatic GPU takeover')
  assert time.monotonic()<deadline,'Bounded predecessor wait expired'
  time.sleep(2)
 torch.cuda.set_device('cuda:0')
 write(dict(status='RUNNING',phase='real-substep-terminal-validation',formal_controls=0))
 spec=importlib.util.spec_from_file_location('ground_history_gpu_validation',R/'validate_groundpick_history_gpu_001.py');gv=importlib.util.module_from_spec(spec);spec.loader.exec_module(gv)
 gpu_report=R/'groundpick-health-history-substep-terminal-001.json';sys.argv=['validate','--output',str(gpu_report)];gv.main();gr=json.loads(gpu_report.read_text());assert gr['status']=='PASS' and gr['actual_reset_only_subclass_verified'];gc.collect();torch.cuda.empty_cache()
 smoke,s=run_train('smoke',64,5,True,True);validate(smoke,s,32);assert s['onnx_max_abs_error']<2e-5 and s['onnx_actor_normalizer_unchanged']
 actual,r=run_train('actual24576',24576,16,True,True);validate(actual,r,12288)
 qualification=R/'groundpick-health-history-qualification-001.json';train.write_json(qualification,dict(status='PASS',task=RELEASE_TASK,num_envs=24576,source_sha256=r['source_sha256'],fresh_only=True,smoke=str(smoke/'report.json'),actual_N=str(actual/'report.json'),cpu=str(R/'groundpick-health-history-cpu-003.json'),real_substep_terminal=str(gpu_report),throughput=r['throughput'],throughput_target_met=r['throughput']['median_control_per_second']>=100000,performance_policy='Measured performance is reported; correctness guards hard. No100k-only blocker.',formal_parent=None,formal_control_cap=301989888,all_history_formal_cap=377487360))
 out,f=run_train('allocation',24576,512,qualification=qualification);assert f['allocation_controls']==301989888 and f['controls']==301989888;validate(out,f,1)
 spec=importlib.util.spec_from_file_location('fresh_release_eval',W/'scripts/evaluate_v2_ground_pick.py');evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
 for node in(128,256,384,512):
  checkpoint=out/f'checkpoint_{node:06d}.pt';assert checkpoint.is_file()
  metadata=torch.load(checkpoint,map_location='cpu',weights_only=False)['infos']['v2'];assert metadata['completed_updates']==node
  for stage in('nominal','measured'):
   result=R/(f'groundpick-health-history-{node:04d}-'+stage);write(dict(status='RUNNING',phase=f'independent-{node}-{stage}',output=str(result),formal_controls=301989888,all_history_formal_controls=377487360));sys.argv=['evaluate','--checkpoint',str(checkpoint),'--output',str(result),'--device','cuda:0','--stage',stage,'--num-envs','256','--seed','2607'];evaluator.main();gc.collect();torch.cuda.empty_cache()
 assert train.source_hashes(W)==cpu['source_sha256'];write(dict(status='GROUNDPICK_HEALTH_HISTORY_COMPLETE_NO_AUTOMATIC_CONTINUATION',completed_updates=512,formal_controls=301989888,all_history_formal_controls=377487360,checkpoint=str(out/'checkpoint_000512.pt'),worker_wall_s=time.monotonic()-worker_start,worker_start_unix=worker_unix))
except BaseException as e:
 write(dict(status='FAILED_REQUIRES_DIAGNOSIS',exception=repr(e)));traceback.print_exc();raise
finally:stop.set()
