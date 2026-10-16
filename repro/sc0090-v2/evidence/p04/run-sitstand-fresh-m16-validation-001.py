"""One fresh actual-N16 minibatch diagnostic; formal training is forbidden."""
import os,sys,json,time,hashlib,traceback,threading,signal
from pathlib import Path
from dataclasses import asdict
from functools import wraps
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'sitstand-transition-release-worktree';os.chdir(W);sys.path.insert(0,str(W/'scripts'));os.environ.pop('MICRODUCK_WARM_START',None)
out=R/'sitstand-fresh-m16-actual16384-001';status=R/'sitstand-fresh-m16-validation-status.json';log=R/'sitstand-fresh-m16-validation-001.log';stop=threading.Event();started=time.monotonic()
def write(v):
 v.update(pid=os.getpid(),updated_unix=time.time());p=status.with_suffix('.tmp');p.write_text(json.dumps(v,indent=2)+'\n');p.replace(status)
def watch():
 patterns=('ccd overflow','contact overflow','constraint buffer full','broadphase overflow','narrowphase overflow','height field collision overflow','nefc overflow','njmax_nnz overflow')
 while not stop.wait(2):
  if log.exists() and any(p in log.read_text(errors='replace').lower()for p in patterns):
   (R/'sitstand-fresh-m16-capacity-rejection.json').write_text(json.dumps(dict(status='CAPACITY_GUARD_REJECTED',pid=os.getpid())));os.kill(os.getpid(),signal.SIGTERM);return
threading.Thread(target=watch,daemon=True).start();write(dict(status='IMPORTING',formal_controls_added=0));print('WORKER_PID',os.getpid(),flush=True)
import torch
import train_v2_sitstand as train
from mjlab_microduck.custom_v2.sitstand_transition_release import RELEASE_TASK

native_cfg=train.load_rl_cfg;original=asdict(native_cfg(RELEASE_TASK));expected=dict(original)
def agent_cfg(task):
 cfg=native_cfg(task);assert cfg.num_steps_per_env==24 and cfg.algorithm.num_mini_batches==4 and cfg.algorithm.num_learning_epochs==5;cfg.algorithm.num_mini_batches=16;return cfg
train.load_rl_cfg=agent_cfg
native_runner=train.load_runner_cls;counter={'steps':0,'updates':0};records=[]

def observed_runner(task):
 original_cls=native_runner(task)
 def construct(*args,**kwargs):
  runner=original_cls(*args,**kwargs);env=args[0].unwrapped;assert env.common_step_counter==0 and runner.current_learning_iteration==0 and not runner.alg.optimizer.state
  old_init=torch.load(R/'sitstand-fresh-release-actual16384-001/initialization.pt',map_location='cpu',weights_only=False)
  actual=runner.alg.save();train.exact(actual,old_init['model']);assert torch.equal(torch.get_rng_state(),old_init['cpu_rng_after_model_creation']) and torch.equal(torch.cuda.get_rng_state('cuda:1'),old_init['cuda_rng_after_model_creation'])
  train.write_json(out/'initialization-comparison.json',dict(status='EXACT',reference=str(R/'sitstand-fresh-release-actual16384-001/initialization.pt'),model_optimizer_lr_exact=True,cpu_and_active_cuda_rng_exact=True,clock=0,normalizer_count=0,physical_initial_tensor_pairing='Not previously captured in M4; do not claim complete state pairing.',no_checkpoint_loaded=True))
  step=runner.alg.optimizer.step
  @wraps(step)
  def counted_step(*a,**kw):
   result=step(*a,**kw);counter['steps']+=1;return result
  runner.alg.optimizer.step=counted_step
  update=runner.alg.update
  @wraps(update)
  def counted_update(*a,**kw):
   before=counter['steps'];result=update(*a,**kw);delta=counter['steps']-before;counter['updates']+=1;assert delta==80,delta
   row=dict(update=counter['updates'],optimizer_step_calls=delta,total_optimizer_step_calls=counter['steps'],learning_rate=runner.alg.learning_rate);records.append(row)
   with (out/'native-optimizer-step-callback.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
   return result
  runner.alg.update=counted_update
  def forbidden(*a,**kw):raise AssertionError('Validation must start fresh')
  runner.load=forbidden
  return runner
 return construct
train.load_runner_cls=observed_runner
try:
 torch.set_num_threads(2);hashes=train.source_hashes(W);prior=json.loads((R/'sitstand-fresh-release-qualification-001.json').read_text());assert hashes==prior['source_sha256']
 modified=asdict(agent_cfg(RELEASE_TASK));check=json.loads(json.dumps(modified,default=str));base=json.loads(json.dumps(original,default=str));check['algorithm']['num_mini_batches']=4;assert check==base
 batch=16384*24;assert batch%16==0 and batch//16==4096*24//4==24576
 chunks=torch.arange(batch).reshape(16,-1);assert chunks.shape==(16,24576) and torch.equal(chunks.flatten(),torch.arange(batch))
 m4=torch.load(R/'sitstand-fresh-release-actual16384-001/checkpoint_000016.pt',map_location='cpu',weights_only=False);assert set(int(v['step'])for v in m4['optimizer_state_dict']['state'].values())=={320};del m4
 train.write_json(R/'sitstand-fresh-m16-cpu-001.json',dict(status='PASS',source_sha256=hashes,extra_diagnostic_source=str(Path(__file__)),extra_diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),base_agent=original,actual_agent=modified,only_agent_difference='algorithm.num_mini_batches:4to16',native_batch=393216,optimizer_batch=24576,epochs=5,updates=16,expected_optimizer_step_calls=1280,M4_native_Adam_step=320,formal_controls_added=0,scope='Same implementation; no extra64 smoke needed. Existing fresh64 software checks retained; actualN16 checks this optimizer setting.'))
 print('CPU_M16_CONTRACT_PASS',flush=True);write(dict(status='RUNNING',phase='actual16384-M16-validation',output=str(out),formal_controls_added=0));sys.argv=['train','--task',RELEASE_TASK,'--output',str(out),'--device','cuda:1','--seed','2026101611','--num-envs','16384','--iterations','16','--validation-only'];t=time.monotonic();train.main();wall=time.monotonic()-t
 report=json.loads((out/'report.json').read_text());run=json.loads((out/'run.json').read_text());assert report['completed_updates']==16 and report['status']=='ALLOCATION_COMPLETE' and report['fresh_initialization'] and report['inherited_controls']==0 and report['controls']==6291456
 assert run['allocation_kind']=='validation' and run['agent']['algorithm']['num_mini_batches']==16 and counter=={'steps':1280,'updates':16}
 chk=torch.load(out/'checkpoint_000016.pt',map_location='cpu',weights_only=False);native_steps=sorted(set(int(v['step'])for v in chk['optimizer_state_dict']['state'].values()));assert native_steps==[1280]
 assert report['terminal_bootstrap_samples']>0 and report['contact_peak']<report['contact_capacity'];metrics=report['training_metrics'];assert metrics['metrics']['feedback_not_ready_fraction']==0 and metrics['metrics']['nonfinite_fraction']==0 and max(metrics['oracle_max_abs'])<2e-6
 assert train.source_hashes(W)==hashes
 train.write_json(out/'extra-execution-source.json',dict(worker=str(Path(__file__)),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_sha256=hashes,actual_agent=run['agent'],only_behavior_delta='M4toM16',optimizer_step_callback_total=1280,nativeAdamsteps=native_steps,wall_s_including_build_and_learning=wall,worker_wall_s_including_imports=time.monotonic()-started));(out/'executed-diagnostic-worker.py').write_bytes(Path(__file__).read_bytes())
 write(dict(status='M16_VALIDATION_COMPLETE_NO_FORMAL_AUTHORIZATION',diagnostic_controls=6291456,formal_controls_added=0,optimizer_steps=1280,native_Adam_steps=native_steps,output=str(out),throughput=report['throughput'],worker_wall_s=time.monotonic()-started));print('M16_COMPLETE',report['throughput'],flush=True)
except BaseException as e:
 write(dict(status='FAILED_REQUIRES_DIAGNOSIS',exception=repr(e),counter=counter));traceback.print_exc();raise
finally:stop.set()
