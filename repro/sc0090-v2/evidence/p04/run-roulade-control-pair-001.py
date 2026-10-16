"""Control qualification and one64-update allocation; root marker required."""
from pathlib import Path
from dataclasses import asdict
import ast,copy,gc,hashlib,json,math,os,re,runpy,sys,time,types
R=Path(__file__).resolve().parent;W=R/'roulade-control-worktree';status=R/'roulade-control-pipeline-status.json';assert not status.exists()
PARENT=R/'roulade-endpoint-allocation-001/checkpoint_000256.pt'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2,default=str)+'\n')
def state(**kw):
 tmp=status.with_suffix('.tmp');save(tmp,dict(updated_unix=time.time(),pid=os.getpid(),**kw));tmp.replace(status)
def normalized(v):
 if isinstance(v,dict):return {k:normalized(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [normalized(x) for x in v]
 if isinstance(v,str):
  v=re.sub(r' at 0x[0-9a-f]+','',v)
  for p in (R/'roulade-endpoint-worktree',W):v=v.replace(str(p),'<NUMERICAL_SOURCE>')
 return v
try:
 state(status='IMPORTING_FOR_ADMISSION',formal_controls=0)
 sys.path.insert(0,str(W/'scripts'));ns=runpy.run_path(str(W/'scripts/train_v2_roulade.py'),run_name='roulade_control_trainer')
 import torch
 from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
 from mjlab_microduck.custom_v2.roulade import ROULADE_TASK
 parent=torch.load(PARENT,map_location='cpu',weights_only=False);meta=parent['infos']['v2'];oldrun=read(PARENT.parent/'run.json');hashes=ns['source_hashes'](W)
 changed={k for k in set(hashes)|set(meta['source_sha256']) if hashes.get(k)!=meta['source_sha256'].get(k)};assert changed=={'scripts/train_v2_roulade.py'}
 cfg=load_env_cfg(ROULADE_TASK);cfg.seed=oldrun['arguments']['seed'];cfg.scene.num_envs=16384
 assert normalized(json.loads(json.dumps(asdict(cfg),default=str)))==normalized(oldrun['environment'])
 agent=load_rl_cfg(ROULADE_TASK);agent.logger='tensorboard';agent.upload_model=False;agent.save_interval=64
 assert normalized(json.loads(json.dumps(asdict(agent),default=str)))==normalized(oldrun['agent'])
 assert meta['completed_updates']==256 and meta['controls']==100663296 and parent['infos']['env_state']['common_step_counter']==6144
 # CPU contract checks with explicit fake CUDA state storage; no GPU0 context.
 source=(W/'scripts/train_v2_roulade.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='restore_active_training_rng');fnsource=ast.get_source_segment(source,node)
 cpu=[meta['torch_rng'].clone()];cuda={0:torch.zeros_like(meta['cuda_rng']),1:torch.ones_like(meta['cuda_rng'])};calls=[]
 class Private:
  def __init__(self,device):self.device=torch.device(device)
  def set_state(self,s):
   if self.device.type=='cpu':torch.Generator(device='cpu').set_state(s)
   elif s.numel()!=meta['cuda_rng'].numel():raise ValueError('bad synthetic native CUDA byte size')
 def setcpu(s):cpu[0]=s.clone();calls.append('cpu')
 def setcuda(s,device):cuda[torch.device(device).index]=s.clone();calls.append(str(device))
 fake=types.SimpleNamespace(device=torch.device,uint8=torch.uint8,equal=torch.equal,Generator=Private,get_rng_state=lambda:cpu[0].clone(),set_rng_state=setcpu,cuda=types.SimpleNamespace(device_count=lambda:2,set_rng_state=setcuda,get_rng_state=lambda device:cuda[torch.device(device).index].clone()))
 context={'torch':fake,'hashlib':hashlib};exec(fnsource,context);fn=context['restore_active_training_rng'];tests=[]
 for target in ('cuda:0','cuda:1'):
  old_inactive=cuda[1-int(target[-1])].clone();calls.clear();v=fn(meta,oldrun,target)
  assert torch.equal(cpu[0],meta['torch_rng']) and torch.equal(cuda[int(target[-1])],meta['cuda_rng'])
  assert torch.equal(cuda[1-int(target[-1])],old_inactive);assert calls==['cpu',target];tests.append('exact_'+target)
 for name,badmeta,badrun,target in [('ambiguous_target',meta,oldrun,'cuda'),('source_conflict',meta,{**oldrun,'arguments':{**oldrun['arguments'],'device':'cuda:1'}},'cuda:1'),('ambiguous_source',meta,{**oldrun,'arguments':{**oldrun['arguments'],'device':'cuda'}},'cuda:1'),('bad_dtype',{**meta,'cuda_rng':meta['cuda_rng'].float()},oldrun,'cuda:1')]:
  before=(cpu[0].clone(),{k:v.clone() for k,v in cuda.items()});calls.clear()
  try:fn(badmeta,badrun,target);raise AssertionError('bad provenance accepted')
  except ValueError:pass
  assert not calls and torch.equal(cpu[0],before[0]) and all(torch.equal(cuda[k],v) for k,v in before[1].items());tests.append('reject_'+name)
 manifest=dict(status='PASS_CONFIG_REVIEW',recipe='roulade-control-cross-device-resume-v1',task=ROULADE_TASK,parent_checkpoint=str(PARENT),parent_checkpoint_sha256=sha(PARENT),from_source_sha256=meta['source_sha256'],to_source_sha256=hashes,changed_files=sorted(changed),environment_equal=True,agent_equal=True,source_device='cuda:0',target_device='cuda:1')
 transition=R/'roulade-control-source-transition-001.json';save(transition,manifest);ns['validate_source_transition'](meta,hashes,PARENT,transition,ROULADE_TASK)
 save(R/'roulade-control-cpu-001.json',dict(status='PASS_CPU_ONLY',environment_exact=True,actor_critic_PPO_reward_physics_exact=True,changed_files=sorted(changed),CPU_rng_contract_cases=tests,parent_updates=256,parent_clock=6144,parent_lr=parent['optimizer_state_dict']['param_groups'][0]['lr'],source_sha256=hashes,helper_ast_sha256=hashlib.sha256(fnsource.encode()).hexdigest()))
 state(status='CPU_PASS_RUNNING_NATIVE_RNG_ORACLE',formal_controls=0)
 torch.cuda.set_device('cuda:1');restore=ns['restore_active_training_rng'];result=restore(meta,oldrun,'cuda:1');g=torch.Generator(device='cuda:1');g.set_state(meta['cuda_rng'])
 draws=[]
 for name,func in [('normal',lambda generator=None:torch.randn(257,device='cuda:1',generator=generator)),('uniform',lambda generator=None:torch.rand(257,device='cuda:1',generator=generator)),('permutation',lambda generator=None:torch.randperm(257,device='cuda:1',generator=generator))]:
  actual,expected=func(),func(g);assert torch.equal(actual,expected);draws.append(dict(kind=name,values=actual.numel(),bitwise_equal=True))
 assert torch.equal(torch.cuda.get_rng_state('cuda:1'),g.get_state());restore(meta,oldrun,'cuda:1')
 save(R/'roulade-control-native-rng-001.json',dict(status='PASS',restoration=result,draw_oracles=draws,scope='Actual saved cuda0 stream mapped to cuda1; default stream draws match independent native CUDA1 generator from identical saved bytes; no GPU0 allocations/streams accessed'))
 del parent;gc.collect();base=['--device','cuda:1','--resume',str(PARENT),'--source-transition',str(transition)]
 validation=[]
 for name,n,updates in [('roulade-control-mature64-001',64,5),('roulade-control-actual16384-001',16384,16)]:
  state(status='RUNNING_VALIDATION',phase=name,formal_controls=0);started=time.monotonic()
  sys.argv=[str(W/'scripts/train_v2_roulade.py'),'--output',str(R/name),'--num-envs',str(n),'--iterations',str(updates),*base,'--validation-only','--force-timeouts']
  ns['main']();v=read(R/name/'report.json');assert v['status']=='ALLOCATION_COMPLETE' and v['completed_updates']==256+updates and v['stop_reason'] is None
  assert v['terminal_bootstrap_samples']>=n//2 and v['contact_peak']<v['contact_capacity']
  metrics=v['training_metrics']['metrics'];assert metrics['feedback_invalid_fraction']==metrics['nonfinite_fraction']==0
  if n==64:assert v['onnx_max_abs_error']<2e-5 and v['terminal_validation']['status']=='PASS'
  rng=read(R/name/'resume-rng.json');assert rng['source_device']=='cuda:0' and rng['target_device']=='cuda:1' and rng['active_stream_mapped'] and rng['source_active_cuda_rng_sha256']==rng['target_restored_cuda_rng_sha256']
  save(R/name/'outer-phase-wall.json',dict(wall_seconds=time.monotonic()-started,allocation_kind='validation',scope='Shared process imports excluded; validation discarded, formal always parent256'))
  validation.append(v);gc.collect();torch.cuda.empty_cache()
 actual=validation[-1];assert actual['throughput']['median_control_per_second']>100000
 patterns=('ccd overflow','contact overflow','constraint buffer full','broadphase overflow','narrowphase overflow','height field collision overflow','nefc overflow','njmax_nnz overflow')
 log=R/'roulade-control-pair-001.log';assert not any(p in log.read_text().lower() for p in patterns)
 assets=read(R/'roulade-endpoint-qualification.json')['model_asset_sha256']
 for rel,value in assets.items():assert sha(W/rel)==value
 assert hashes==ns['source_hashes'](W)
 qualification=dict(status='PASS',task=ROULADE_TASK,worktree=str(W),source_sha256=hashes,model_asset_sha256=assets,num_envs=16384,parent_checkpoint=str(PARENT),parent_checkpoint_sha256=sha(PARENT),source_transition=str(transition),CPU_report=str(R/'roulade-control-cpu-001.json'),rng_report=str(R/'roulade-control-native-rng-001.json'),smoke_report=str(R/'roulade-control-mature64-001/report.json'),actual_report=str(R/'roulade-control-actual16384-001/report.json'),formal_allocation_updates=64,formal_allocation_controls=25165824,paired_additional_controls=50331648,roulade_full_history_cap=150994944,scope='Pure PPO continuation control; no teacher/reward/curriculum changes; cross-device byte-exact RNG restore; not bitwise trajectory pairing acrossGPU')
 qp=R/'roulade-control-qualification-001.json';save(qp,qualification)
 marker=R/'roulade-pair-ready-001.json';state(status='ADMISSION_PASSED_WAITING_ROOT_PAIR_MARKER',qualification=str(qp),qualification_sha256=sha(qp),formal_controls=0)
 print('CONTROL_QUALIFIED_WAITING_ROOT',str(qp),sha(qp),flush=True)
 while not marker.exists():time.sleep(2)
 ready=read(marker);assert ready['status']=='PASS_BOTH_ARMS' and ready['additional_controls']==50331648
 assert ready['teacher_qualification_sha256']==sha(R/'roulade-teacher-qualification-001.json') and ready['control_qualification_sha256']==sha(qp)
 assert hashes==ns['source_hashes'](W)
 name='roulade-control-allocation-001';state(status='ROOT_PAIR_MARKER_ACCEPTED_STARTING_FORMAL',phase=name,formal_controls=0);start=time.monotonic()
 sys.argv=[str(W/'scripts/train_v2_roulade.py'),'--output',str(R/name),'--num-envs','16384','--iterations','64',*base,'--qualification',str(qp)];ns['main']();v=read(R/name/'report.json')
 assert v['completed_updates']==320 and v['allocation_controls']==25165824 and v['controls']==125829120 and v['stop_reason'] is None
 save(R/name/'outer-phase-wall.json',dict(wall_seconds=time.monotonic()-start,allocation_kind='formal',controls=25165824,scope='Formal phase in shared-import qualification process; initialization/learning/final save included, prior admission excluded'))
 assert not any(p in log.read_text().lower() for p in patterns)
 for stage in ('nominal','measured'):
  ename='roulade-control-0320-'+stage;state(status='FROZEN_EVALUATION',phase=ename,formal_controls=25165824);gc.collect();torch.cuda.empty_cache()
  sys.argv=[str(W/'scripts/evaluate_v2_roulade.py'),'--checkpoint',str(R/name/'checkpoint_000320.pt'),'--output',str(R/ename),'--device','cuda:1','--stage',stage]
  runpy.run_path(str(W/'scripts/evaluate_v2_roulade.py'),run_name='__main__')
 state(status='CONTROL_BUDGET_COMPLETE_NO_FURTHER_TRAINING',formal_controls=25165824,parent_controls=100663296,control_lineage_controls=125829120,paired_stage_cap=50331648)
except BaseException as exc:
 state(status='STOPPED_REQUIRES_REVIEW',error=repr(exc));raise
