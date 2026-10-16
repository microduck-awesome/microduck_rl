"""Exact mature320 continuation at actual live difficulty; one bounded320 allocation."""
import os,sys,json,time,hashlib,traceback,threading,signal,gc,importlib.util
from pathlib import Path
from dataclasses import asdict
from copy import deepcopy
from functools import wraps
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'sitstand-history-hold320-worktree';P=R/'sitstand-history-base-allocation-002/checkpoint_000320.pt';os.chdir(W);sys.path.insert(0,str(W/'scripts'));os.environ.pop('MICRODUCK_WARM_START',None)
state=R/'sitstand-history-hold320-pipeline-status.json';log=R/'sitstand-history-hold320-pipeline-001.log';stop=threading.Event();started=time.monotonic();start_unix=time.time()
def write(v):
 v.update(pid=os.getpid(),updated_unix=time.time());tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(state)
def watch():
 patterns=('ccd overflow','contact overflow','constraint buffer full','broadphase overflow','narrowphase overflow','height field collision overflow','nefc overflow','njmax_nnz overflow')
 while not stop.wait(2):
  if log.exists()and any(v in log.read_text(errors='replace').lower()for v in patterns):
   (R/'sitstand-history-hold320-capacity-rejection.json').write_text(json.dumps(dict(status='CAPACITY_REJECTED',pid=os.getpid())));os.kill(os.getpid(),signal.SIGTERM);return
threading.Thread(target=watch,daemon=True).start();write(dict(status='IMPORTING',formal_controls_added=0));print('WORKER_PID',os.getpid(),flush=True)
import torch
import train_v2_sitstand as train
from mjlab_microduck.custom_v2.sitstand_history import HISTORY_TASK,make_history_sitstand_env_cfg
from mjlab_microduck.custom_v2.sitstand_hold320 import HOLD_TASK,make_hold_sitstand_env_cfg,LIVE_REWARD_WEIGHTS,LIVE_HEAD_RANGES,LIVE_EVENT_PARAMS

def clean(x):
 if callable(x):return x.__module__+'.'+x.__qualname__
 if isinstance(x,dict):return {k:clean(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v)for v in x]
 return x

def cpu_contract():
 hashes=train.source_hashes(W);parent=torch.load(P,map_location='cpu',weights_only=False);meta=parent['infos']['v2'];receipt=json.loads((R/'sitstand-hold320-live-parent.json').read_text());assert meta['completed_updates']==320 and parent['infos']['env_state']['common_step_counter']==7680 and meta['controls']==125829120
 assert hashlib.sha256(P.read_bytes()).hexdigest()==receipt['parent_sha256'];assert LIVE_REWARD_WEIGHTS==receipt['live']['live_reward_weights']and clean(LIVE_HEAD_RANGES)==receipt['live']['head_ranges']
 original=make_history_sitstand_env_cfg();hold=make_hold_sitstand_env_cfg();expected=deepcopy(original)
 for k,v in LIVE_REWARD_WEIGHTS.items():expected.rewards[k].weight=v
 expected.commands['head_pose'].ranges=deepcopy(LIVE_HEAD_RANGES)
 for k,v in LIVE_EVENT_PARAMS.items():expected.events[k].params.update(deepcopy(v))
 expected.curriculum={};assert clean(asdict(expected))==clean(asdict(hold))
 assert clean(asdict(make_history_sitstand_env_cfg(play=True)))==clean(asdict(make_hold_sitstand_env_cfg(play=True)))
 assert clean(asdict(train.load_rl_cfg(HISTORY_TASK)))==clean(asdict(train.load_rl_cfg(HOLD_TASK)))
 assert hold.episode_length_s==12. and hold.commands['twist'].ramp_s==2. and hold.commands['head_pose'].zero_command_prob==0.
 assert hold.rewards['action_rate_l2'].weight==-.6 and hold.rewards['rise_speed'].weight==0. and not hold.curriculum
 # Actual native scalar outputs uniquely identify the event stage; no clock-based recomputation.
 event_evidence=[]
 for name,term in original.curriculum.items():
  if name not in('com_range','head_com_range','push_magnitude'):continue
  key='push_stages'if name=='push_magnitude'else'range_stages';observed=receipt['curriculum_scalars'][name]
  matches=[v for v in term.params[key]if (max(abs(v['velocity_range']['x'][0]),abs(v['velocity_range']['x'][1]))if name=='push_magnitude'else v['range'])==observed];assert len(matches)==1
  expected_value=matches[0]['velocity_range']if name=='push_magnitude'else(-observed,observed);event=term.params['event_name'];field='velocity_range'if name=='push_magnitude'else'ranges';assert clean(LIVE_EVENT_PARAMS[event][field])==clean(expected_value);event_evidence.append(dict(curriculum=name,actual_scalar=observed,unique_stage=matches[0],frozen_event=event))
 transition=dict(kind='actual_live320_curriculum_hold',parent_task=HISTORY_TASK,target_task=HOLD_TASK,parent_source_sha256=meta['source_sha256'],new_source_sha256=hashes,parent_checkpoint_sha256=receipt['parent_sha256'],parent_live_receipt=str(R/'sitstand-hold320-live-parent.json'));train.write_json(W/'source-transition.json',transition)
 train.write_json(R/'sitstand-history-hold320-cpu-001.json',dict(status='PASS',source_sha256=hashes,actual_live=receipt['live'],event_evidence=event_evidence,train_cfg_only_authorized_hold_changes=True,play_cfg_exactly_unchanged=True,agent_cfg_exactly_unchanged=True,native_clock_preserved=7680,parent_Adam_steps=sorted({int(v['step'])for v in parent['optimizer_state_dict']['state'].values()}),parent_LR=parent['optimizer_state_dict']['param_groups'][0]['lr'],resume_only=True,formal_parent=str(P),source_transition=str(W/'source-transition.json')))
 print('CPU_EXACT_HOLD_CONTRACT_PASS',flush=True);return hashes

active={};native_runner=train.load_runner_cls

def instrumented_runner(task):
 cls=native_runner(task)
 def build(*a,**kw):
  runner=cls(*a,**kw);env=a[0].unwrapped;out=active['out'];assert '--resume'in sys.argv
  original_load=runner.load;loaded=[False];steps=[0];updates=[0]
  @wraps(original_load)
  def load(*args,**kwargs):
   assert Path(args[0]).resolve()==P.resolve();infos=original_load(*args,**kwargs);parent=torch.load(P,map_location='cpu',weights_only=False);actual=runner.alg.save()
   for k in('actor_state_dict','critic_state_dict','optimizer_state_dict'):train.exact(actual[k],parent[k])
   assert env.common_step_counter==7680 and runner.alg.learning_rate==parent['optimizer_state_dict']['param_groups'][0]['lr']
   loaded[0]=True;train.write_json(out/'mature-load-observer.json',dict(status='EXACT_PARENT320',actor_critic_normalizer_Adam_exact=True,learning_rate=runner.alg.learning_rate,clock=env.common_step_counter,parent=str(P),validation_weights_not_inherited=True));return infos
  runner.load=load
  original_step=runner.alg.optimizer.step
  @wraps(original_step)
  def counted_step(*args,**kwargs):
   assert loaded[0];r=original_step(*args,**kwargs);steps[0]+=1;return r
  runner.alg.optimizer.step=counted_step
  original_log=runner.logger.log
  @wraps(original_log)
  def log_row(*args,**kwargs):
   updates[0]+=1;assert steps[0]==updates[0]*80;node=320+updates[0]
   actual_weights={n:env.reward_manager.get_term_cfg(n).weight for n in env.reward_manager._term_names};assert actual_weights==LIVE_REWARD_WEIGHTS
   assert clean(env.command_manager.get_term('head_pose').cfg.ranges)==clean(LIVE_HEAD_RANGES)
   for n,p in LIVE_EVENT_PARAMS.items():
    actual=env.event_manager.get_term_cfg(n).params
    for k,v in p.items():assert clean(actual[k])==clean(v)
   assert env.common_step_counter==node*24
   row=dict(completed_updates=node,allocation_updates=updates[0],allocation_optimizer_steps=steps[0],expected_total_Adam_steps=25600+steps[0],clock=env.common_step_counter,learning_rate=runner.alg.learning_rate,live_curriculum_fixed=True)
   with(out/'native-optimizer-step-callback.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
   if active['label']=='allocation'and updates[0]%64==0 and updates[0]<320:runner.save(str(out/f'checkpoint_{node:06d}.pt'))
   return original_log(*args,**kwargs)
  runner.logger.log=log_row
  return runner
 return build
train.load_runner_cls=instrumented_runner

def run(label,n,it,validation=False,forced=False,qualification=None):
 out=R/f'sitstand-history-hold320-{label}-001';active.update(label=label,out=out);args=['train','--task',HOLD_TASK,'--output',str(out),'--device','cuda:1','--seed','2026101611','--num-envs',str(n),'--iterations',str(it),'--resume',str(P),'--source-transition',str(W/'source-transition.json')]
 if validation:args+=['--validation-only']
 if forced:args+=['--force-timeouts']
 if qualification:args+=['--qualification',str(qualification)]
 write(dict(status='RUNNING',phase=label,output=str(out),formal_parent=str(P),formal_controls_added=0 if validation else'read allocation status',all_history_before=352321536));sys.argv=args;t=time.monotonic();train.main();wall=time.monotonic()-t;rep=json.loads((out/'report.json').read_text());assert rep['status']=='ALLOCATION_COMPLETE'and rep['completed_updates']==320+it and rep['common_step_counter']==(320+it)*24 and rep['inherited_controls']==125829120
 train.write_json(out/'shared-process-phase-wall.json',dict(seconds=wall,allocation_controls=rep['allocation_controls'],controls_per_second=rep['allocation_controls']/wall,scope='Includes environment construction; imports shared'));print('PHASE_DONE',label,flush=True);gc.collect();torch.cuda.empty_cache();return out,rep

def validate(out,rep,min_timeout):
 assert rep['terminal_bootstrap_samples']>=min_timeout and rep['contact_peak']<rep['contact_capacity'];m=rep['training_metrics']['metrics'];assert m['feedback_not_ready_fraction']==0 and m['nonfinite_fraction']==0;assert max(rep['training_metrics']['oracle_max_abs'])<2e-6
 ds=[json.loads(v)['metrics']for v in(out/'ppo-diagnostics.jsonl').read_text().splitlines()];assert all(v['PPO/kl_mean']<=.5 and v['PPO/clip_fraction']<=.8 for v in ds)

try:
 hashes=cpu_contract();smoke,s=run('smoke',64,5,True,True);validate(smoke,s,32);assert s['onnx_max_abs_error']<2e-5
 actual,a=run('actual16384',16384,16,True,True);validate(actual,a,8192)
 q=R/'sitstand-history-hold320-qualification-001.json';train.write_json(q,dict(status='PASS',source_sha256=hashes,task=HOLD_TASK,num_envs=16384,parent=str(P),parent_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),cpu=str(R/'sitstand-history-hold320-cpu-001.json'),smoke=str(smoke/'report.json'),actual=str(actual/'report.json'),throughput=a['throughput'],throughput_not_correctness_blocker=True,new_formal_controls_cap=125829120,all_history_cap=478150656))
 out,f=run('allocation',16384,320,qualification=q);validate(out,f,1);assert f['allocation_controls']==125829120 and f['controls']==251658240
 spec=importlib.util.spec_from_file_location('hold_eval',W/'scripts/evaluate_v2_sitstand_clock_fixed.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
 for node in(448,576,640):
  cp=out/f'checkpoint_{node:06d}.pt';assert cp.is_file()
  for condition in('nominal','measured'):
   dst=R/f'sitstand-history-hold320-{node:04d}-{condition}';write(dict(status='RUNNING',phase=f'eval-{node}-{condition}',output=str(dst),formal_controls_added=125829120,all_history_formal_controls=478150656));sys.argv=['eval','--checkpoint',str(cp),'--output',str(dst),'--device','cuda:1','--stage',condition,'--num-envs','256','--seed','2603'];ev.main();gc.collect();torch.cuda.empty_cache()
 assert train.source_hashes(W)==hashes;write(dict(status='SITSTAND_HOLD320_COMPLETE_NO_AUTOMATIC_THIRD_STAGE',completed_updates=640,formal_controls_added=125829120,lineage_controls=251658240,all_history_formal_controls=478150656,worker_wall_s=time.monotonic()-started,start_unix=start_unix,checkpoint=str(out/'checkpoint_000640.pt')))
except BaseException as e:write(dict(status='FAILED_REQUIRES_DIAGNOSIS',exception=repr(e)));traceback.print_exc();raise
finally:stop.set()
