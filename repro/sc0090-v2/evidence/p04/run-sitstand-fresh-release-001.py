"""One fresh lineage, frozen production source, bounded qualification/formal/eval."""
import os,sys,json,time,gc,hashlib,traceback,threading,signal,importlib.util,ast
from pathlib import Path
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'sitstand-transition-release-worktree';os.chdir(W);sys.path.insert(0,str(W/'scripts'));os.environ.pop('MICRODUCK_WARM_START',None)
state=R/'sitstand-fresh-release-pipeline-status.json';log=R/'sitstand-fresh-release-pipeline-001.log';stop=threading.Event();patterns=('ccd overflow','contact overflow','constraint buffer full','broadphase overflow','narrowphase overflow','height field collision overflow','nefc overflow','njmax_nnz overflow')
worker_start=time.monotonic();worker_unix=time.time()
def write(x):
 x.update(pid=os.getpid(),updated_unix=time.time());t=state.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(state)
def watch():
 while not stop.wait(2):
  if log.exists() and any(p in log.read_text(errors='replace').lower() for p in patterns):
   (R/'sitstand-fresh-release-capacity-rejection.json').write_text(json.dumps(dict(status='CAPACITY_GUARD_REJECTED',pid=os.getpid())));os.kill(os.getpid(),signal.SIGTERM);return
threading.Thread(target=watch,daemon=True).start();write(dict(status='IMPORTING',formal_controls=0,prior_all_sitstand_formal_controls=176160768));print('WORKER_PID',os.getpid(),flush=True)
import torch
import train_v2_sitstand as train
from mjlab_microduck.custom_v2.sitstand_transition_release import RELEASE_TASK
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
 hashes=train.source_hashes(W);oldq=json.loads((R/'sitstand-transition-release-qualification-001.json').read_text());assert hashes==oldq['source_sha256']
 source=(W/'scripts/train_v2_sitstand.py').read_text();tree=ast.parse(source);main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main');resume=next(n for n in ast.walk(main)if isinstance(n,ast.If)and ast.unparse(n.test)=='args.resume')
 assert 'assert len(runner.alg.optimizer.state) == 0' in '\n'.join(ast.unparse(n)for n in resume.orelse)
 assert all('runner.load' not in ast.unparse(n)for n in resume.orelse)
 assert 'start = inherited = 0' in source and 'lineage = None' in source and 'assert start_clock == start * agent.num_steps_per_env' in source
 # Review ancestry CPU-only; no actor/normalizer or optimizer gets loaded.
 paths=[R/'sitstand-transition-release-allocation-001/checkpoint_000448.pt'];history=[];known_hashes=[]
 while paths:
  p=paths.pop();ck=torch.load(p,map_location='cpu',weights_only=False);m=ck['infos']['v2'];known_hashes.append(tensor_hash(ck['actor_state_dict']));lineage=m['lineage'];history.append(dict(checkpoint=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),completed_updates=m['completed_updates'],task=m['task'],parent=lineage['checkpoint']if lineage else None));
  if lineage:paths.append(Path(lineage['checkpoint']))
  del ck
 assert [v['completed_updates']for v in history]==[448,384,256,128],history
 report=dict(status='PASS',source_sha256=hashes,history=history,all_prior_correct_target_runs_inherit_wrong_target128=True,fresh_branch='start/inherited0; lineageNone; emptyAdam; no resumeCLI; no runner.load',no_source_mutation=True,frozen_reward_qualification=str(R/'sitstand-transition-release-qualification-001.json'),known_prior_actor_hashes=known_hashes)
 train.write_json(R/'sitstand-fresh-release-cpu-001.json',report);print('CPU_FRESH_CONTRACT_PASS',flush=True);return report

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
   validation=torch.load(R/'sitstand-fresh-release-actual16384-001/checkpoint_000016.pt',map_location='cpu',weights_only=False)
   assert initial_actor!=tensor_hash(validation['actor_state_dict']) and initial_critic!=tensor_hash(validation['critic_state_dict']);del validation
  out=active['out'];torch.save(dict(model=cpu_copy(saved),common_step_counter=0,learning_iteration=0,cpu_rng_after_model_creation=torch.get_rng_state(),cuda_rng_after_model_creation=torch.cuda.get_rng_state('cuda:1')),out/'initialization.pt')
  train.write_json(out/'initialization.json',dict(status='FRESH_VERIFIED',clock=0,completed_updates=0,inherited_controls=0,optimizer_state_entries=0,learning_rate=runner.alg.learning_rate,actor_sha256=initial_actor,critic_sha256=initial_critic,normalizers=normalizers,no_checkpoint_loaded=True,not_validation_weights=active['label']=='allocation',initialization_state_sha256=hashlib.sha256((out/'initialization.pt').read_bytes()).hexdigest(),scope='Observer after native runner construction; snapshot does not change RNG/state. Random initialization distinct from all448/384/256/128 ancestors.'))
  def forbidden_load(*args,**kwargs):raise AssertionError('Fresh allocation must never call runner.load')
  runner.load=forbidden_load
  return runner
 return build
train.load_runner_cls=instrumented_runner_cls

def run_train(label,n,it,validation=False,forced=False,qualification=None):
 out=R/('sitstand-fresh-release-'+label+'-001');active.update(label=label,out=out);argv=['train','--task',RELEASE_TASK,'--output',str(out),'--device','cuda:1','--seed','2026101611','--num-envs',str(n),'--iterations',str(it)]
 if validation:argv+=['--validation-only']
 if forced:argv+=['--force-timeouts']
 if qualification:argv+=['--qualification',str(qualification)]
 write(dict(status='RUNNING',phase=label,output=str(out),formal_controls=0 if validation else 'read formal status',prior_all_sitstand_formal_controls=176160768));print('PHASE_START',label,flush=True);t=time.monotonic();sys.argv=argv;train.main();wall=time.monotonic()-t
 rep=json.loads((out/'report.json').read_text());run=json.loads((out/'run.json').read_text());assert rep['status']=='ALLOCATION_COMPLETE' and rep['completed_updates']==it and rep['fresh_initialization'] and rep['inherited_controls']==0 and rep['common_step_counter']==it*24 and run['arguments']['resume']is None
 train.write_json(out/'shared-process-phase-wall.json',dict(seconds_including_environment_and_validation=wall,imports_shared=True,controls=rep['allocation_controls'],controls_per_second=rep['allocation_controls']/wall));print('PHASE_DONE',label,wall,flush=True);gc.collect();torch.cuda.empty_cache();return out,rep

def validate(out,rep,timeout_min):
 assert rep['terminal_bootstrap_samples']>=timeout_min and rep['contact_peak']<rep['contact_capacity'];m=rep['training_metrics'];assert m['metrics']['feedback_not_ready_fraction']==0 and m['metrics']['nonfinite_fraction']==0 and max(m['oracle_max_abs'])<2e-6
 diag=[json.loads(line)['metrics']for line in(out/'ppo-diagnostics.jsonl').read_text().splitlines()];assert all(d['PPO/kl_mean']<=.5 and d['PPO/clip_fraction']<=.8 for d in diag)

try:
 cpu=cpu_contract();smoke,s=run_train('smoke',64,5,True,True);validate(smoke,s,32);assert s['onnx_max_abs_error']<2e-5 and s['onnx_actor_normalizer_unchanged']
 actual,r=run_train('actual16384',16384,16,True);validate(actual,r,1)
 qualification=R/'sitstand-fresh-release-qualification-001.json';train.write_json(qualification,dict(status='PASS',task=RELEASE_TASK,num_envs=16384,source_sha256=r['source_sha256'],fresh_only=True,smoke=str(smoke/'report.json'),actual_N=str(actual/'report.json'),cpu=str(R/'sitstand-fresh-release-cpu-001.json'),throughput=r['throughput'],throughput_target_met=r['throughput']['median_control_per_second']>=100000,performance_policy='Measured performance is reported; correctness guards hard. No100k-only blocker.',formal_parent=None,formal_control_cap=50331648,all_history_formal_cap=226492416))
 out,f=run_train('allocation',16384,128,qualification=qualification);assert f['allocation_controls']==50331648 and f['controls']==50331648
 spec=importlib.util.spec_from_file_location('fresh_release_eval',W/'scripts/evaluate_v2_sitstand_clock_fixed.py');evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
 for stage in('nominal','measured'):
  result=R/('sitstand-fresh-release-0128-'+stage);write(dict(status='RUNNING',phase='independent-'+stage,output=str(result),formal_controls=50331648,all_history_formal_controls=226492416));sys.argv=['evaluate','--checkpoint',str(out/'checkpoint_000128.pt'),'--output',str(result),'--device','cuda:1','--stage',stage,'--num-envs','256','--seed','2603'];evaluator.main();gc.collect();torch.cuda.empty_cache()
 assert train.source_hashes(W)==cpu['source_sha256'];write(dict(status='SITSTAND_FRESH_STAGE_BUDGET_COMPLETE_NO_FURTHER_TRAINING',completed_updates=128,formal_controls=50331648,all_history_formal_controls=226492416,checkpoint=str(out/'checkpoint_000128.pt'),worker_wall_s=time.monotonic()-worker_start,worker_start_unix=worker_unix))
except BaseException as e:
 write(dict(status='FAILED_REQUIRES_DIAGNOSIS',exception=repr(e)));traceback.print_exc();raise
finally:stop.set()
