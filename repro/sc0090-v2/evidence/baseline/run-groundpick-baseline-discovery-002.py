"""Bounded B0 baseline: common random initialization, admission,512 updates,transferbattery."""
import os,sys,json,time,gc,hashlib,traceback,threading,signal,importlib.util
from pathlib import Path
from functools import wraps
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/baseline');W=R/'groundpick-route-worktree';P=R.parent/'p04';os.chdir(W);sys.path.insert(0,str(W/'scripts'));os.environ.pop('MICRODUCK_WARM_START',None)
STATE=R/'groundpick-baseline-discovery-status-002.json';LOG=R/'groundpick-baseline-discovery-002.log';assert not STATE.exists();started=time.monotonic();started_unix=time.time();stop=threading.Event()
def write(**kw):
 p=STATE.with_suffix('.tmp');p.write_text(json.dumps(dict(pid=os.getpid(),updated_unix=time.time(),B0_formal_cap=301989888,**kw),indent=2)+'\n');p.replace(STATE)
def watch():
 patterns=('ccd overflow','contact overflow','constraint buffer full','broadphase overflow','narrowphase overflow','height field collision overflow','nefc overflow','njmax_nnz overflow')
 while not stop.wait(2):
  if LOG.exists()and any(p in LOG.read_text(errors='replace').lower()for p in patterns):os.kill(os.getpid(),signal.SIGTERM);return
threading.Thread(target=watch,daemon=True).start();write(status='IMPORTING_B0',formal_controls=0);print('WORKER_PID',os.getpid(),flush=True)
import torch
import train_v2_ground_pick as train
from mjlab_microduck.custom_v2.ground_pick_baseline import BASELINE_TASK
import validate_groundpick_baseline_cpu as cpucheck
from rsl_rl.modules.normalization import EmpiricalNormalization
torch.set_num_threads(2)
COMMON=P/'groundpick-health-history-allocation-001/initialization.pt';COMMON_SHA='d6984a69f78d53ee27dfbc843be4d91f6d0b0040cecd0a9f8ee99a912a8e8fbe'
assert hashlib.sha256(COMMON.read_bytes()).hexdigest()==COMMON_SHA
common=torch.load(COMMON,map_location='cpu',weights_only=False);assert common['common_step_counter']==common['learning_iteration']==0 and not common['model']['optimizer_state_dict']['state']
active={};native_runner=train.load_runner_cls

def tensorhash(sd):
 h=hashlib.sha256()
 for k,v in sorted(sd.items()):h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def cpu_copy(v):
 if isinstance(v,torch.Tensor):return v.detach().cpu().clone()
 if isinstance(v,dict):return {k:cpu_copy(x)for k,x in v.items()}
 if isinstance(v,(list,tuple)):return type(v)(cpu_copy(x)for x in v)
 return v

def build_runner(task):
 original=native_runner(task)
 def build(*args,**kwargs):
  runner=original(*args,**kwargs);env=args[0].unwrapped;out=active['out']
  assert env.common_step_counter==0 and runner.current_learning_iteration==0 and not runner.alg.optimizer.state
  assert not hasattr(env,'_v2_joint_feedback') and env._v2_ground_pick_baseline
  # Common untrained random weights, zero normalization; never use a learnedP04 checkpoint.
  runner.alg.actor.load_state_dict(common['model']['actor_state_dict'],strict=True);runner.alg.critic.load_state_dict(common['model']['critic_state_dict'],strict=True)
  state=runner.alg.save()
  for key in ('actor_state_dict','critic_state_dict'):train.exact(state[key],common['model'][key])
  assert not state['optimizer_state_dict']['state'];norms={}
  for role,module in [('actor',runner.alg.actor),('critic',runner.alg.critic)]:
   for name,n in module.named_modules():
    if isinstance(n,EmpiricalNormalization):assert int(n.count)==0 and bool((n._mean==0).all()) and bool((n._std==1).all());norms[role+'.'+name]=int(n.count)
  assert len(norms)==2
  if active['label']=='allocation':
   qualified=torch.load(R/'groundpick-baseline-actual24576-001/checkpoint_000016.pt',map_location='cpu',weights_only=False)
   assert tensorhash(state['actor_state_dict'])!=tensorhash(qualified['actor_state_dict']);del qualified
  torch.save(dict(model=cpu_copy(state),clock=0,completed_updates=0,cpu_rng_after_model_creation=torch.get_rng_state(),cuda_rng_after_model_creation=torch.cuda.get_rng_state('cuda:1')),out/'initialization.pt')
  train.write_json(out/'initialization.json',dict(status='COMMON_UNTRAINED_RANDOM_INITIALIZATION_VERIFIED',source=str(COMMON),source_sha256=COMMON_SHA,clock=0,completed_updates=0,inherited_controls=0,optimizer_state_entries=0,normalizer_counts=norms,actor_sha256=tensorhash(state['actor_state_dict']),critic_sha256=tensorhash(state['critic_state_dict']),learning_rate=runner.alg.learning_rate,no_trained_checkpoint_loaded=True,no_validation_weights_loaded=True,physics_rng_not_claimed_paired=True))
  calls=[0];updates=[0];native_step=runner.alg.optimizer.step
  def step(*a,**kw):result=native_step(*a,**kw);calls[0]+=1;return result
  runner.alg.optimizer.step=step;native_log=runner.logger.log
  @wraps(native_log)
  def log(*a,**kw):
   updates[0]+=1;assert calls[0]==updates[0]*120 and env.common_step_counter==updates[0]*24
   row=dict(update=updates[0],optimizer_step_calls=120,total_optimizer_step_calls=calls[0],native_clock=env.common_step_counter,learning_rate=runner.alg.learning_rate,unix=time.time())
   with(out/'native-optimizer-step-callback.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
   if active['label']=='allocation'and updates[0]%64==0 and updates[0]<512:runner.save(str(out/f'checkpoint_{updates[0]:06d}.pt'))
   if updates[0]in(1,64,128,192,256,320,384,448,512):
    row['low_reset_exposure_HOME_LOW']=env._gp_reset_exposure.cpu().tolist();row['live_reward_weights']={n:env.reward_manager.get_term_cfg(n).weight for n in env.reward_manager._term_names};row['live_event_params']={n:env.event_manager.get_term_cfg(n).params for n in ('randomize_com','randomize_head_com','push_robot')}
    with(out/'curriculum-live.jsonl').open('a')as f:f.write(json.dumps(row,default=str)+'\n')
   return native_log(*a,**kw)
  runner.logger.log=log
  def forbidden(*a,**kw):raise AssertionError('B0 must not call trainedcheckpoint runner.load')
  runner.load=forbidden;return runner
 return build
train.load_runner_cls=build_runner

def run(label,n,updates,validation=False,qualification=None):
 out=R/f'groundpick-baseline-{label}-001';active.update(label=label,out=out)
 argv=['train','--task',BASELINE_TASK,'--device','cuda:1','--seed','2026101636','--num-envs',str(n),'--iterations',str(updates),'--output',str(out)]
 if validation:argv+=['--validation-only','--force-timeouts']
 if qualification:argv+=['--qualification',str(qualification)]
 write(status='RUNNING',phase=label,output=str(out),formal_controls=0 if validation else 'read allocation status');sys.argv=argv;t=time.monotonic();train.main();wall=time.monotonic()-t;rep=json.loads((out/'report.json').read_text())
 assert rep['status']=='ALLOCATION_COMPLETE' and rep['completed_updates']==updates and rep['allocation_controls']==n*24*updates and rep['inherited_controls']==0 and rep['common_step_counter']==updates*24
 train.write_json(out/'shared-process-phase-wall.json',dict(seconds_including_environment_and_validation=wall,controls=rep['allocation_controls'],controls_per_second=rep['allocation_controls']/wall,imports_shared=True));gc.collect();torch.cuda.empty_cache();return out,rep

def qualify(out,rep,minimum_timeouts):
 assert rep['terminal_bootstrap_samples']>=minimum_timeouts and rep['contact_peak']<rep['contact_capacity'];m=rep['training_metrics'];assert m['feedback_fault_metric_applicable'] is False and m['sensor_chain']=='baseline-official-noise-delay' and m['metrics']['feedback_not_ready_fraction'] is None and m['metrics']['nonfinite_fraction']==0 and max(m['oracle_max_abs'])<2e-6
 ds=[json.loads(x)['metrics']for x in(out/'ppo-diagnostics.jsonl').read_text().splitlines()];assert all(x['PPO/kl_mean']<=.5 and x['PPO/clip_fraction']<=.8 for x in ds)

try:
 cpu=cpucheck.main();assert cpu['status']=='PASS';write(status='CPU_PASS_WAITING_TERMINAL_SCRIPT',formal_controls=0)
 deadline=time.time()+600;gp_script=R/'validate_groundpick_baseline_gpu.py'
 while not gp_script.exists():
  assert time.time()<deadline
  time.sleep(2)
 assert not Path('/proc/581916').exists() and json.loads((P/'sitstand-near-teacher-lr3e6-pipeline-status.json').read_text())['status']=='SITSTAND_NEAR_TEACHER_COMPLETE_NO_AUTOMATIC_CONTINUATION'
 torch.cuda.set_device('cuda:1');write(status='RUNNING',phase='terminal-validation',formal_controls=0)
 spec=importlib.util.spec_from_file_location('baseline_ground_gpu_validation',gp_script);gv=importlib.util.module_from_spec(spec);spec.loader.exec_module(gv);gp=R/'groundpick-baseline-terminal-001.json';sys.argv=['validate','--output',str(gp)];gv.main();assert json.loads(gp.read_text())['status']=='PASS';gc.collect();torch.cuda.empty_cache()
 so,s=run('smoke',64,5,True);qualify(so,s,32);assert s['onnx_max_abs_error']<2e-5 and s['onnx_actor_normalizer_unchanged']
 ao,a=run('actual24576',24576,16,True);qualify(ao,a,12288)
 q=R/'groundpick-baseline-qualification-001.json';train.write_json(q,dict(status='PASS',task=BASELINE_TASK,num_envs=24576,source_sha256=a['source_sha256'],cpu=str(R/'groundpick-baseline-cpu-001.json'),terminal=str(gp),smoke=str(so/'report.json'),actual=str(ao/'report.json'),throughput=a['throughput'],common_untrained_initialization=str(COMMON),common_untrained_initialization_sha256=COMMON_SHA,formal_controls_cap=301989888,route='B0 only; no automaticDR/P04 stage',validation_weights_inherited=False))
 out,formal=run('allocation',24576,512,qualification=q);qualify(out,formal,1)
 import evaluate_v2_ground_pick_route as evaluator
 for node in(128,256,384,512):
  for runtime,stage in [('baseline','nominal'),('p04','nominal'),('p04','measured')]:
   dest=R/f'groundpick-baseline-{node:04d}-{runtime}-{stage}';cp=out/f'checkpoint_{node:06d}.pt';assert cp.exists();write(status='EVALUATING',phase=f'{node}-{runtime}-{stage}',output=str(dest),formal_controls=301989888);sys.argv=['evaluate','--checkpoint',str(cp),'--runtime',runtime,'--stage',stage,'--num-envs','256','--seed','2607','--device','cuda:1','--output',str(dest)];evaluator.main();gc.collect();torch.cuda.empty_cache()
 assert train.source_hashes(W)==cpu['source_sha256'];write(status='BASELINE_B0_GROUNDPICK_COMPLETE_REVIEW_BEFORE_B1',completed_updates=512,formal_controls=301989888,worker_wall_s=time.monotonic()-started,start_unix=started_unix)
except BaseException:
 write(status='FAILED_REQUIRES_DIAGNOSIS',exception=traceback.format_exc());traceback.print_exc();raise
finally:stop.set()
