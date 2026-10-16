"""Exact mature640 continuation, near-STAND teacher and SIT anchor; one bounded320 allocation."""
import os,sys,json,time,hashlib,traceback,threading,signal,gc,importlib.util
from pathlib import Path
from dataclasses import asdict
from copy import deepcopy
from functools import wraps
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'sitstand-near-teacher-lr3e6-worktree';P=R/'sitstand-history-hold320-allocation-001/checkpoint_000640.pt';TEACHER=R/'head2-micro-exposure-allocation-001/checkpoint_001856.pt';os.chdir(W);sys.path.insert(0,str(W/'scripts'));os.environ.pop('MICRODUCK_WARM_START',None)
state=R/'sitstand-near-teacher-lr3e6-pipeline-status.json';log=R/'sitstand-near-teacher-lr3e6-pipeline-001.log';stop=threading.Event();started=time.monotonic();start_unix=time.time()
def write(v):
 v.update(pid=os.getpid(),updated_unix=time.time());tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(state)
def watch():
 patterns=('ccd overflow','contact overflow','constraint buffer full','broadphase overflow','narrowphase overflow','height field collision overflow','nefc overflow','njmax_nnz overflow')
 while not stop.wait(2):
  if log.exists()and any(v in log.read_text(errors='replace').lower()for v in patterns):
   (R/'sitstand-near-teacher-lr3e6-capacity-rejection.json').write_text(json.dumps(dict(status='CAPACITY_REJECTED',pid=os.getpid())));os.kill(os.getpid(),signal.SIGTERM);return
threading.Thread(target=watch,daemon=True).start();write(dict(status='IMPORTING',formal_controls_added=0));print('WORKER_PID',os.getpid(),flush=True)
import torch
import train_v2_sitstand as train
from mjlab_microduck.custom_v2.sitstand_near import NEAR_TASK,make_near_sitstand_env_cfg,make_near_sitstand_agent
from mjlab_microduck.custom_v2.sitstand_teacher import near_stand_mask,balanced_batches
from mjlab_microduck.custom_v2.sitstand_hold320 import HOLD_TASK,make_hold_sitstand_env_cfg,LIVE_REWARD_WEIGHTS,LIVE_HEAD_RANGES,LIVE_EVENT_PARAMS

def clean(x):
 if callable(x):return x.__module__+'.'+x.__qualname__
 if isinstance(x,dict):return {k:clean(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v)for v in x]
 return x

def cpu_contract():
 hashes=train.source_hashes(W);parent=torch.load(P,map_location='cpu',weights_only=False);meta=parent['infos']['v2'];assert meta['completed_updates']==640 and parent['infos']['env_state']['common_step_counter']==15360 and meta['controls']==251658240
 assert hashlib.sha256(P.read_bytes()).hexdigest()=='90b00e9867030d72bd97bf7a91490761c18306b754e2fc8aaeb80cdb3dc85ec5'
 for play in(False,True):
  old=make_hold_sitstand_env_cfg(play=play);new=make_near_sitstand_env_cfg(play=play);assert len(new.scene.sensors)==len(old.scene.sensors)+1
  assert new.scene.sensors[-1].name=='diagnostic_head_ground';new.scene.sensors=old.scene.sensors
  assert clean(asdict(old))==clean(asdict(new)),('env changed',play)
 old=asdict(train.load_rl_cfg(HOLD_TASK));new=asdict(make_near_sitstand_agent());changed=set(new['algorithm'])-set(old['algorithm']);assert changed=={'bc_cfg'}
 core=deepcopy(new);core['algorithm'].pop('bc_cfg');core['algorithm']['class_name']=old['algorithm']['class_name'];assert clean(core)==clean(old)
 assert new['algorithm']['bc_cfg']['learning_rate']==3e-6
 assert old['num_steps_per_env']==24 and old['algorithm']['num_mini_batches']==16 and old['algorithm']['num_learning_epochs']==5
 # Independent scalar truth table: boundaries are strict; no head-contact condition.
 import itertools,math
 cases=list(itertools.product((0.,1.),(0.,.019,.02,1.),(.1,.10001),(.0,math.pi/6),(.199,.2),(False,True),(False,True),(False,True)))
 data=torch.tensor(cases,dtype=torch.float64);actual=near_stand_mask(*data.unbind(-1));expected=torch.tensor([f==0 and abs(al-f)<.02 and h>.1 and ti<math.pi/6 and le<.2 and ft and rd and fi for f,al,h,ti,le,ft,rd,fi in cases]);assert torch.equal(actual,expected)
 for count in(1,63,64,511,512,1536,393216):
  chunks=balanced_batches(torch.arange(count),4,512);assert torch.equal(torch.cat(chunks),torch.arange(count));assert len(chunks)==min(4,max(1,count//512));assert max(map(len,chunks))-min(map(len,chunks))<=1
 for teacher,iteration in((TEACHER,1856),(P,640)):
  c=torch.load(teacher,map_location='cpu',weights_only=False);assert c['infos']['v2']['completed_updates']==iteration
  for name in('src/mjlab_microduck/sc0090_v2/contract.py','src/mjlab_microduck/sc0090_v2/measurement/stream.py','src/mjlab_microduck/actuator/sc0090_v5.py'):assert c['infos']['v2']['source_sha256'][name]==hashes[name]
 transition=dict(kind='near_stand_teacher_with_sit_anchor',parent_task=HOLD_TASK,target_task=NEAR_TASK,parent_source_sha256=meta['source_sha256'],new_source_sha256=hashes,parent_checkpoint_sha256=hashlib.sha256(P.read_bytes()).hexdigest());train.write_json(W/'source-transition.json',transition)
 train.write_json(R/'sitstand-near-teacher-lr3e6-cpu-001.json',dict(status='PASS',source_sha256=hashes,gate_truth_table_cases=len(cases),train_play_only_diagnostic_sensor_added=True,agent_only_BC_added=True,teacher_contract='Real raw61, stand twist/body zero and head preserved; independent teacher normalization; no obs/cache/physics mutation',mini_batch_partition='Balanced chunk sizes, not branch resampling',parent_LR=parent['optimizer_state_dict']['param_groups'][0]['lr'],parent_clock=15360,parent_controls=251658240,formal_parent=str(P),stand_teacher=str(TEACHER),new_formal_controls_cap=125829120,all_history_cap=603979776))
 print('CPU_NEAR_TEACHER_CONTRACT_PASS',flush=True);return hashes

active={};native_runner=train.load_runner_cls

def instrumented_runner(task):
 cls=native_runner(task)
 def build(*a,**kw):
  runner=cls(*a,**kw);env=a[0].unwrapped;out=active['out'];assert '--resume'in sys.argv
  original_load=runner.load;loaded=[False];steps=[0];bcsteps=[0];updates=[0];oracle=[0]
  @wraps(original_load)
  def load(*args,**kwargs):
   assert Path(args[0]).resolve()==P.resolve();infos=original_load(*args,**kwargs);parent=torch.load(P,map_location='cpu',weights_only=False);actual=runner.alg.save()
   for k in('actor_state_dict','critic_state_dict','optimizer_state_dict'):train.exact(actual[k],parent[k])
   assert env.common_step_counter==15360 and runner.alg.learning_rate==parent['optimizer_state_dict']['param_groups'][0]['lr'];assert not runner.alg.bc_optimizer.state and runner.alg.bc_update_count==0
   loaded[0]=True;train.write_json(out/'mature-load-observer.json',dict(status='EXACT_PARENT640',actor_critic_normalizer_Adam_exact=True,learning_rate=runner.alg.learning_rate,clock=env.common_step_counter,parent=str(P),BC_Adam_fresh=True,validation_weights_not_inherited=True));return infos
  runner.load=load
  original_step=runner.alg.optimizer.step;original_bc=runner.alg.bc_optimizer.step
  @wraps(original_step)
  def counted_step(*args,**kwargs):
   assert loaded[0];r=original_step(*args,**kwargs);steps[0]+=1;return r
  @wraps(original_bc)
  def counted_bc(*args,**kwargs):
   assert loaded[0];r=original_bc(*args,**kwargs);bcsteps[0]+=1;return r
  runner.alg.optimizer.step=counted_step;runner.alg.bc_optimizer.step=counted_bc
  if active['label']!='allocation':
   from mjlab_microduck.custom_v2.sitstand_metrics import SitStandMeasurements
   reader=SitStandMeasurements(env,oracle=False);original_act=runner.alg.act
   @wraps(original_act)
   def checked_act(obs):
    i=runner.alg.storage.step;x=reader.read();robot=env.scene['robot'];q=robot.data.joint_pos[:,reader.ids]-robot.data.default_joint_pos[:,reader.ids]
    feet=env.scene['feet_ground_contact'].data.found.reshape(env.num_envs,2,-1).any(-1).all(-1)
    actual=original_act(obs);ctx=runner.alg.teacher_context[i]
    expected=torch.stack((x['flag'],x['alpha'],x['height'],x['tilt'],q[:,[0,1,2,3,4,9,10,11,12,13]].abs().mean(-1),feet,x['feedback_ready'],x['finite']&torch.isfinite(obs['actor']).all(-1)),-1)
    assert torch.equal(ctx,expected);oracle[0]+=1;return actual
   runner.alg.act=checked_act
  original_log=runner.logger.log
  @wraps(original_log)
  def log_row(*args,**kwargs):
   updates[0]+=1;assert steps[0]==updates[0]*80;node=640+updates[0]
   assert {n:env.reward_manager.get_term_cfg(n).weight for n in env.reward_manager._term_names}==LIVE_REWARD_WEIGHTS
   assert clean(env.command_manager.get_term('head_pose').cfg.ranges)==clean(LIVE_HEAD_RANGES)
   for n,p in LIVE_EVENT_PARAMS.items():
    actual=env.event_manager.get_term_cfg(n).params
    for k,v in p.items():assert clean(actual[k])==clean(v)
   assert env.common_step_counter==node*24;runner.alg.assert_teachers_unchanged();context=runner.alg.last_bc_context_audit
   assert context['first_clock']==(node-1)*24 and context['last_clock']==node*24-1 and context['frames']==env.num_envs*24
   assert context['effective_stand_frames']==0 or context['effective_stand_frames']>=64
   assert context['effective_stand_frames']+context['sit_anchor_frames']+context['ignored_stand_frames']==context['frames']
   row=dict(completed_updates=node,allocation_updates=updates[0],allocation_optimizer_steps=steps[0],allocation_BC_steps=bcsteps[0],clock=env.common_step_counter,learning_rate=runner.alg.learning_rate,live_curriculum_fixed=True,preaction_context_oracle_calls=oracle[0],context=context)
   with(out/'native-optimizer-step-callback.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
   if active['label']=='allocation'and updates[0]in(128,256):runner.save(str(out/f'checkpoint_{node:06d}.pt'))
   return original_log(*args,**kwargs)
  runner.logger.log=log_row
  return runner
 return build

train.load_runner_cls=instrumented_runner

def run(label,n,it,validation=False,forced=False,qualification=None):
 out=R/f'sitstand-near-teacher-lr3e6-{label}-001';active.update(label=label,out=out);args=['train','--task',NEAR_TASK,'--output',str(out),'--device','cuda:1','--seed','2026101611','--num-envs',str(n),'--iterations',str(it),'--resume',str(P),'--source-transition',str(W/'source-transition.json'),'--standing-teacher',str(TEACHER),'--sit-anchor',str(P)]
 if validation:args+=['--validation-only']
 if forced:args+=['--force-timeouts']
 if qualification:args+=['--qualification',str(qualification)]
 write(dict(status='RUNNING',phase=label,output=str(out),formal_parent=str(P),formal_controls_added=0 if validation else'read allocation status',all_history_before=478150656));sys.argv=args;t=time.monotonic();train.main();wall=time.monotonic()-t;rep=json.loads((out/'report.json').read_text());assert rep['status']=='ALLOCATION_COMPLETE'and rep['completed_updates']==640+it and rep['common_step_counter']==(640+it)*24 and rep['inherited_controls']==251658240
 train.write_json(out/'shared-process-phase-wall.json',dict(seconds=wall,allocation_controls=rep['allocation_controls'],controls_per_second=rep['allocation_controls']/wall,scope='Includes environment construction; imports shared'));print('PHASE_DONE',label,flush=True);gc.collect();torch.cuda.empty_cache();return out,rep

def validate(out,rep,min_timeout):
 assert rep['terminal_bootstrap_samples']>=min_timeout and rep['contact_peak']<rep['contact_capacity'];m=rep['training_metrics']['metrics'];assert m['feedback_not_ready_fraction']==0 and m['nonfinite_fraction']==0;assert max(rep['training_metrics']['oracle_max_abs'])<2e-6
 bs=[json.loads(v)for v in(out/'bc-diagnostics.jsonl').read_text().splitlines()];assert all(v['losses']['expert_bc_kl_mean']<=.5 and v['losses']['expert_bc_steps']<=20 for v in bs);assert rep['teacher_parameters_normalizers_unchanged']
 ds=[json.loads(v)['metrics']for v in(out/'ppo-diagnostics.jsonl').read_text().splitlines()];assert all(v['PPO/kl_mean']<=.5 and v['PPO/clip_fraction']<=.8 for v in ds)

try:
 assert not Path('/proc/475752').exists() and json.loads((R/'ballkick-target-precision-pipeline-status.json').read_text())['status']=='BALLKICK_TARGET_PRECISION_COMPLETE_NO_AUTOMATIC_CONTINUATION'
 hashes=cpu_contract();smoke,s=run('smoke',64,5,True,True);validate(smoke,s,32);assert s['onnx_max_abs_error']<2e-5
 actual,a=run('actual16384',16384,16,True,True);validate(actual,a,8192)
 q=R/'sitstand-near-teacher-lr3e6-qualification-001.json';train.write_json(q,dict(status='PASS',source_sha256=hashes,task=NEAR_TASK,num_envs=16384,parent=str(P),parent_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),cpu=str(R/'sitstand-near-teacher-lr3e6-cpu-001.json'),smoke=str(smoke/'report.json'),actual=str(actual/'report.json'),throughput=a['throughput'],throughput_not_correctness_blocker=True,new_formal_controls_cap=125829120,all_history_cap=603979776))
 out,f=run('allocation',16384,320,qualification=q);validate(out,f,1);assert f['allocation_controls']==125829120 and f['controls']==377487360
 spec=importlib.util.spec_from_file_location('hold_eval',W/'scripts/evaluate_v2_sitstand_clock_fixed.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
 for node in(768,896,960):
  cp=out/f'checkpoint_{node:06d}.pt';assert cp.is_file()
  for condition in('nominal','measured'):
   dst=R/f'sitstand-near-teacher-lr3e6-{node:04d}-{condition}';write(dict(status='RUNNING',phase=f'eval-{node}-{condition}',output=str(dst),formal_controls_added=125829120,all_history_formal_controls=603979776));sys.argv=['eval','--checkpoint',str(cp),'--output',str(dst),'--device','cuda:1','--stage',condition,'--num-envs','256','--seed','2603'];ev.main();gc.collect();torch.cuda.empty_cache()
 assert train.source_hashes(W)==hashes;write(dict(status='SITSTAND_NEAR_TEACHER_COMPLETE_NO_AUTOMATIC_CONTINUATION',completed_updates=960,formal_controls_added=125829120,lineage_controls=377487360,all_history_formal_controls=603979776,worker_wall_s=time.monotonic()-started,start_unix=start_unix,checkpoint=str(out/'checkpoint_000960.pt')))
except BaseException as e:write(dict(status='FAILED_REQUIRES_DIAGNOSIS',exception=repr(e)));traceback.print_exc();raise
finally:stop.set()
