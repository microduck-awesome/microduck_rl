"""One bounded CPU / mature64 / actual16384 qualification sequence, no formal PPO."""
from pathlib import Path
from dataclasses import asdict
import gc,hashlib,json,math,os,re,runpy,sys,time
r=Path(__file__).resolve().parent;w=r/'roulade-landing-teacher-worktree'
status=r/'roulade-teacher-admission-status.json';assert not status.exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,value):p.write_text(json.dumps(value,indent=2,default=str)+'\n')
def state(**kw):
 p=status.with_suffix('.tmp');save(p,dict(updated_unix=time.time(),pid=os.getpid(),formal_controls=0,**kw));p.replace(status)
try:
 state(status='IMPORTING')
 sys.path.insert(0,str(w/'scripts'));ns=runpy.run_path(str(w/'scripts/train_v2_roulade.py'),run_name='roulade_teacher_trainer')
 import torch
 from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
 from mjlab_microduck.custom_v2.roulade import ROULADE_TASK
 from mjlab_microduck.custom_v2.roulade_teacher import landing_mask,balanced_batches
 parentpath=r/'roulade-endpoint-allocation-001/checkpoint_000256.pt';parent=torch.load(parentpath,map_location='cpu',weights_only=False);meta=parent['infos']['v2'];oldrun=read(parentpath.parent/'run.json');hashes=ns['source_hashes'](w)
 changed={k for k in set(hashes)|set(meta['source_sha256']) if hashes.get(k)!=meta['source_sha256'].get(k)}
 assert changed==ns['TRANSITION_FILES']
 roots=[str(r/'roulade-endpoint-worktree'),str(w)]
 def normalized(value):
  if isinstance(value,dict):return {k:normalized(v) for k,v in value.items()}
  if isinstance(value,list):return [normalized(v) for v in value]
  if isinstance(value,str):
   value=re.sub(r' at 0x[0-9a-f]+','',value)
   for prefix in roots:value=value.replace(prefix,'<NUMERICAL_SOURCE>')
  return value
 cfg=load_env_cfg(ROULADE_TASK);cfg.seed=oldrun['arguments']['seed'];cfg.scene.num_envs=16384
 assert normalized(json.loads(json.dumps(asdict(cfg),default=str)))==normalized(oldrun['environment'])
 agent=load_rl_cfg(ROULADE_TASK);agent.logger='tensorboard';agent.upload_model=False;agent.save_interval=64
 new=json.loads(json.dumps(asdict(agent),default=str));bc=new['algorithm'].pop('bc_cfg');new['algorithm']['class_name']=oldrun['agent']['algorithm']['class_name']
 assert normalized(new)==normalized(oldrun['agent'])
 for start,accum,latch,expected in [(299,300,1,False),(300,260,1,True),(360,259,1,False),(360,360,0,False),(330,330,1,True),(720,100,1,False)]:
  actual=landing_mask(torch.tensor(math.radians(start),dtype=torch.float64),torch.tensor(math.radians(accum),dtype=torch.float64),torch.tensor(latch))
  assert bool(actual)==expected
 for count in (1,3,511,512,513,1536,16384*24):
  batches=balanced_batches(torch.arange(count),4,512)
  assert torch.equal(torch.cat(batches),torch.arange(count)) and len(batches)<=4
  assert count<512 or min(map(len,batches))>=512
  assert max(map(len,batches))-min(map(len,batches))<=1
 manifest=dict(status='PASS_CONFIG_REVIEW',recipe='frozen-landing-teacher-v1',task=ROULADE_TASK,parent_checkpoint=str(parentpath),parent_checkpoint_sha256=sha(parentpath),parent_run_sha256=sha(parentpath.parent/'run.json'),from_source_sha256=meta['source_sha256'],to_source_sha256=hashes,changed_files=sorted(changed),equivalent_source_roots=roots)
 mp=r/'roulade-teacher-source-transition-002.json';assert not mp.exists();save(mp,manifest)
 ns['validate_source_transition'](meta,hashes,parentpath,mp,ROULADE_TASK)
 save(r/'roulade-teacher-cpu-002.json',dict(status='PASS_CPU_ONLY',environment_exact=True,only_bc_algorithm_and_configuration_changed=True,gate_cases=6,batch_cases=7,source_sha256=hashes,scope='Actor61/critic74, rewards/reset/curricula/physics/PPO unchanged. Privileged pre-action history gates teacher labels only. Actual collector/reset/storage alignment and optimizer behavior await smoke.'))
 del parent;gc.collect()
 base=['--device','cuda:0','--resume',str(parentpath),'--source-transition',str(mp),'--standing-teacher',str(r/'head2-micro-exposure-allocation-001/checkpoint_001856.pt'),'--roll-anchor',str(parentpath),'--validation-only']
 reports=[]
 for name,n,updates,forced in [('roulade-teacher-mature64-002',64,5,True),('roulade-teacher-actual16384-002',16384,16,True)]:
  state(status='RUNNING_VALIDATION',phase=name);start=time.monotonic()
  sys.argv=[str(w/'scripts/train_v2_roulade.py'),'--output',str(r/name),'--num-envs',str(n),'--iterations',str(updates),*base,*(['--force-timeouts'] if forced else [])]
  ns['main']();d=read(r/name/'report.json')
  assert d['status']=='ALLOCATION_COMPLETE' and d['completed_updates']==256+updates and d['stop_reason'] is None
  assert d['terminal_bootstrap_samples']>=n//2 and d['teacher_weights_normalizers_unchanged']
  if n==64:assert d['onnx_max_abs_error']<2e-5
  assert d['contact_peak']<d['contact_capacity']
  context=torch.load(r/name/'last-bc-context.pt',map_location='cpu',weights_only=False)
  data=context['context'];expected=(torch.rad2deg(data[...,0].double())>=300-1e-5)&(torch.rad2deg(data[...,1].double())>=260-1e-5)&(data[...,2]==1)
  assert int(expected.sum())==d['last_bc_context_audit']['eligible_frames']
  assert context['observations']['actor'].shape==(24,n,61) and context['observations']['critic'].shape==(24,n,74)
  assert d['last_bc_context_audit']['last_clock']-d['last_bc_context_audit']['first_clock']==23
  save(r/name/'context-oracle.json',dict(status='PASS_SAVED_ACTUAL_CONTEXT',independent_degree_gate_count=int(expected.sum()),actor_shape=[24,n,61],critic_shape=[24,n,74],native_timeout_samples=d['terminal_bootstrap_samples'],scope='Saved pre-action context and native PPO storage shape/order; no privileged fields in actor or critic. Pure boundary tests separately cover reset/stale-frontier classification.'))
  save(r/name/'outer-phase-wall.json',dict(wall_seconds=time.monotonic()-start,scope='Per-case after shared imports; validation only'))
  reports.append(str(r/name/'report.json'));gc.collect();torch.cuda.empty_cache()
 state(status='ADMISSION_REVIEW_READY',reports=reports,proposed_formal_controls=25165824)
except BaseException as exc:
 state(status='STOPPED_REQUIRES_REVIEW',error=repr(exc));raise
