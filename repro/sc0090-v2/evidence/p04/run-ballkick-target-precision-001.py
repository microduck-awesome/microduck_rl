"""One mature exact256parent; CPU+terminal+smoke+actualN ->256PPO ->originalbattery."""
import os,sys,json,time,hashlib,gc,threading,signal,traceback,importlib.util
from pathlib import Path
from functools import wraps
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'ballkick-target-precision-worktree';os.chdir(W);sys.path.insert(0,str(W/'scripts'));os.environ.pop('MICRODUCK_WARM_START',None)
status=R/'ballkick-target-precision-pipeline-status.json';log=R/'ballkick-target-precision-pipeline-001.log';assert not status.exists();start_time=time.monotonic();stop=threading.Event();PARENT=R/'ballkick-health-history-allocation-001/checkpoint_000256.pt';TRANSITION=R/'ballkick-target-precision-source-transition.json'
def write(**kw):
 p=status.with_suffix('.tmp');p.write_text(json.dumps(dict(pid=os.getpid(),updated_unix=time.time(),new_formal_cap=150994944,all_BallKick_formal_cap=477757440,**kw),indent=2)+'\n');p.replace(status)
def watch():
 patterns=('ccd overflow','contact overflow','constraint buffer full','broadphase overflow','narrowphase overflow','height field collision overflow','nefc overflow','njmax_nnz overflow')
 while not stop.wait(2):
  if log.exists()and any(p in log.read_text(errors='replace').lower()for p in patterns):
   (R/'ballkick-target-precision-capacity-rejection.json').write_text(json.dumps(dict(status='CAPACITY_REJECTED',pid=os.getpid())));os.kill(os.getpid(),signal.SIGTERM);return
threading.Thread(target=watch,daemon=True).start();write(status='IMPORTING',formal_controls_added=0);print('PID',os.getpid(),flush=True)
import torch
import train_v2_ballkick as train
from mjlab_microduck.custom_v2.ball_kick_precision import PRECISION_TASK
import validate_ballkick_precision_cpu as cpu_check
active={};native_runner=train.load_runner_cls

def instrumented_runner_cls(task):
 original=native_runner(task)
 def build(*a,**kw):
  runner=original(*a,**kw);env=a[0].unwrapped;out=active['out'];native_load=runner.load;loaded=[False];calls=[0];updates=[0]
  @wraps(native_load)
  def load(path,*args,**kwargs):
   assert Path(path).resolve()==PARENT
   infos=native_load(path,*args,**kwargs);cp=torch.load(PARENT,map_location='cpu',weights_only=False);actual=runner.alg.save()
   for key in ('actor_state_dict','critic_state_dict','optimizer_state_dict'):train.exact(actual[key],cp[key])
   assert runner.alg.learning_rate==cp['optimizer_state_dict']['param_groups'][0]['lr'];assert env.common_step_counter==6144
   loaded[0]=True;train.write_json(out/'mature-load-observer.json',dict(status='EXACT_ORIGINAL_PARENT256',parent=str(PARENT),parent_sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),clock=env.common_step_counter,actor_critic_normalizer_PPOAdam_exact=True,learning_rate=runner.alg.learning_rate,validation_weights_not_loaded=True));return infos
  runner.load=load;native_step=runner.alg.optimizer.step
  def counted_step(*a,**kw):
   result=native_step(*a,**kw);calls[0]+=1;return result
  runner.alg.optimizer.step=counted_step;native_log=runner.logger.log
  @wraps(native_log)
  def counted_log(*a,**kw):
   assert loaded[0];updates[0]+=1;assert calls[0]==updates[0]*120;assert env.common_step_counter==6144+updates[0]*24
   row=dict(allocation_update=updates[0],completed_updates=256+updates[0],optimizer_step_calls=120,total_optimizer_step_calls=calls[0],native_clock=env.common_step_counter,learning_rate=runner.alg.learning_rate)
   with(out/'native-optimizer-step-callback.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
   if active['label']=='allocation'and updates[0]%64==0 and updates[0]<256:runner.save(str(out/f'checkpoint_{256+updates[0]:06d}.pt'))
   if updates[0]in(1,64,128,192,256):
    row['live_reward_weights']={n:env.reward_manager.get_term_cfg(n).weight for n in env.reward_manager._term_names};row['live_event_params']={n:env.event_manager.get_term_cfg(n).params for n in('randomize_com','randomize_head_com','push_robot')}
    with(out/'curriculum-live.jsonl').open('a')as f:f.write(json.dumps(row,default=str)+'\n')
   return native_log(*a,**kw)
  runner.logger.log=counted_log;return runner
 return build
train.load_runner_cls=instrumented_runner_cls

def run_train(label,n,it,validation=False,qualification=None):
 out=R/f'ballkick-target-precision-{label}-001';active.update(label=label,out=out)
 argv=['train','--task',PRECISION_TASK,'--device','cuda:1','--seed','2026101632','--num-envs',str(n),'--iterations',str(it),'--resume',str(PARENT),'--source-transition',str(TRANSITION),'--output',str(out)]
 if validation:argv+=['--validation-only','--force-timeouts']
 if qualification:argv+=['--qualification',str(qualification)]
 write(status='RUNNING',phase=label,output=str(out),formal_controls_added=0 if validation else 'read allocation/status.json');print('PHASE_START',label,flush=True);begin=time.monotonic();sys.argv=argv;train.main();elapsed=time.monotonic()-begin
 rep=json.loads((out/'report.json').read_text());assert rep['status']=='ALLOCATION_COMPLETE'and rep['completed_updates']==256+it and rep['allocation_controls']==n*24*it and rep['common_step_counter']==6144+24*it and not rep['fresh_initialization']
 assert rep['inherited_controls']==150994944 and rep['resume_checkpoint']==str(out/f'checkpoint_{256+it:06d}.pt')
 train.write_json(out/'shared-process-phase-wall.json',dict(seconds_including_environment_and_validation=elapsed,imports_shared=True,controls=rep['allocation_controls'],controls_per_second=rep['allocation_controls']/elapsed));gc.collect();torch.cuda.empty_cache();return out,rep

def qualify(out,rep,minimum_timeouts):
 assert rep['terminal_bootstrap_samples']>=minimum_timeouts and rep['contact_peak']<rep['contact_capacity'];m=rep['training_metrics'];assert m['metrics']['feedback_not_ready_fraction']==m['metrics']['nonfinite_fraction']==0 and max(m['oracle_max_abs'])<2e-6
 ds=[json.loads(x)['metrics']for x in(out/'ppo-diagnostics.jsonl').read_text().splitlines()];assert all(x['PPO/kl_mean']<=.5 and x['PPO/clip_fraction']<=.8 for x in ds)
try:
 cpu=cpu_check.main();assert cpu['status']=='PASS'
 assert json.loads((R/'ballkick-health-extra-nodes-status-001.json').read_text())['status']=='BALLKICK_EXTRA_NODES_EVALUATED'and not Path('/proc/467044').exists()
 write(status='RUNNING',phase='terminal-validation',formal_controls_added=0)
 spec=importlib.util.spec_from_file_location('precision_gpu_qualification',R/'validate_ballkick_precision_gpu.py');gv=importlib.util.module_from_spec(spec);spec.loader.exec_module(gv);gp=R/'ballkick-target-precision-terminal-001.json';sys.argv=['validate','--output',str(gp)];gv.main();assert json.loads(gp.read_text())['status']=='PASS';gc.collect();torch.cuda.empty_cache()
 so,s=run_train('smoke',64,5,True);qualify(so,s,32);assert s['onnx_max_abs_error']<2e-5 and s['onnx_actor_normalizer_unchanged']
 ao,a=run_train('actual24576',24576,16,True);qualify(ao,a,12288)
 qp=R/'ballkick-target-precision-qualification-001.json';train.write_json(qp,dict(status='PASS',task=PRECISION_TASK,num_envs=24576,source_sha256=a['source_sha256'],parent=str(PARENT),parent_sha256=cpu['parent_checkpoint_sha256'],source_transition=str(TRANSITION),cpu=str(R/'ballkick-target-precision-cpu-001.json'),terminal=str(gp),smoke=str(so/'report.json'),actual=str(ao/'report.json'),throughput=a['throughput'],throughput_target_met=a['throughput']['median_control_per_second']>=100000,new_formal_controls=150994944,all_history_formal_cap=477757440,validation_weights_inherited=False))
 out,formal=run_train('allocation',24576,256,qualification=qp);qualify(out,formal,1);assert formal['controls']==301989888
 import evaluate_v2_ballkick as evaluator
 for node in(320,384,448,512):
  for stage in('nominal','measured'):
   dest=R/f'ballkick-target-precision-{node:04d}-{stage}';cp=out/f'checkpoint_{node:06d}.pt';assert cp.exists();write(status='EVALUATING',phase=f'{node}-{stage}',formal_controls_added=150994944,output=str(dest));sys.argv=['evaluate','--checkpoint',str(cp),'--output',str(dest),'--device','cuda:1','--stage',stage,'--seed','2026101633','--num-envs','256'];evaluator.main();gc.collect();torch.cuda.empty_cache()
 assert train.source_hashes(W)==cpu['source_sha256'];write(status='BALLKICK_TARGET_PRECISION_COMPLETE_NO_AUTOMATIC_CONTINUATION',completed_updates=512,formal_controls_added=150994944,all_history_formal_controls=477757440,worker_wall_s=time.monotonic()-start_time)
except BaseException as e:write(status='FAILED_REQUIRES_DIAGNOSIS',error=repr(e));traceback.print_exc();raise
finally:stop.set()
