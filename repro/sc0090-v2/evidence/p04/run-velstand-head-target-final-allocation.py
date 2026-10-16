"""Last128 VelStand allocation after exact-source admission; no third allocation."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,time
r=Path(__file__).resolve().parent;w=r/'velstand-head-target-worktree';planfile=r/'velstand-head-target-final-allocation-plan.json'
assert not planfile.exists()
plan=dict(status='PREPARING',created_unix=time.time(),num_envs=24576,steps_per_env=24,start_update=128,end_update=256,allocation_updates=128,allocation_controls=75497472,stage_maximum_controls=150994944,automatic_further_allocation=False,phases=[],scope='Single evidence-based walk-head BC target change with32-update label ramp after abrupt switch failed admission. Joint learning; no inference action override. All task metrics share the stage cap.')
compat='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat'
environment=dict(os.environ,SC0090_CUDA_COMPAT_DIR=compat,PYTHONFAULTHANDLER='1')
patterns=('ccd overflow','broadphase overflow','narrowphase overflow','nefc overflow','njmax_nnz overflow','height field collision overflow','contact overflow','contact buffer overflow','constraint buffer full')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def save():
 temp=planfile.with_suffix('.tmp');temp.write_text(json.dumps(plan,indent=2)+'\n');temp.replace(planfile)
def run(name,args):
 logpath=r/(name+'.log');assert not logpath.exists()
 phase=dict(name=name,arguments=args,start_unix=time.time());plan['phases'].append(phase);plan['status']=name.upper();save()
 with logpath.open('w') as log:
  proc=subprocess.Popen([str(w/'scripts/v2_python.sh'),*args],cwd=w,env=environment,stdout=log,stderr=subprocess.STDOUT)
  phase['pid']=proc.pid;save();rejected=False
  while proc.poll() is None:
   found=[p for p in patterns if p in logpath.read_text(errors='replace').lower()]
   if found and not rejected:rejected=True;phase['capacity_rejection']=found;proc.send_signal(signal.SIGTERM);save()
   time.sleep(2)
 phase.update(returncode=proc.returncode,wall_seconds=time.time()-phase['start_unix']);save()
 assert proc.returncode==0 and not rejected,phase
 return phase
try:
 assert read(r/'velstand-head-target-admission-plan-003.json')['status']=='PASS_REVIEW_BEFORE_FORMAL'
 cpu=read(r/'velstand-head-target-cpu-003.json');assert cpu['status']=='PASS'
 smoke=read(r/'velstand-head-target-smoke-003/report.json');resume=read(r/'velstand-head-target-resume64-003/report.json');actual=read(r/'velstand-head-target-actual24576-003/report.json')
 hashes=actual['source_sha256'];assert hashes==cpu['source_sha256']==smoke['source_sha256']==resume['source_sha256']
 for name,digest in hashes.items():assert sha(w/name)==digest,name
 for report,end in ((smoke,133),(resume,135),(actual,144)):
  assert report['status']=='ALLOCATION_COMPLETE' and report['completed_updates']==end and report['stop_reason'] is None
  assert report['training_metrics']['metrics']['feedback_not_ready_fraction']==0 and report['training_metrics']['metrics']['nonfinite_fraction']==0
  assert max(report['training_metrics']['oracle_max_abs'])<2e-7
  assert report['contact_peak']<report['contact_capacity']
  for bc in report['bc_diagnostics']:assert bc['expert_bc_kl_mean']<=.5 and bc['expert_bc_steps']<=20
 assert smoke['terminal_bootstrap_samples']>=32 and actual['terminal_bootstrap_samples']>=12288
 assert smoke['onnx_max_abs_error']<2e-5 and resume['onnx_max_abs_error']<2e-5
 previous=read(r/'velstand-v8-qualification.json')
 for name,digest in previous['assets_sha256'].items():assert sha(w/name)==digest,name
 # Successor/physical code and all other sources unchanged by explicit three-file manifest.
 transition=read(r/'velstand-head-target-transition-003.json')
 for name in ('src/mjlab_microduck/custom_v2/terminal.py','src/mjlab_microduck/custom_v2/ppo.py','src/mjlab_microduck/sc0090_v2/measurement/stream.py'):
  assert hashes[name]==previous['source_sha256'][name]
 inputs=['velstand-head-target-cpu-003.json','velstand-head-target-transition-003.json','velstand-head-target-smoke-003/report.json','velstand-head-target-resume64-003/report.json','velstand-head-target-actual24576-003/report.json','velstand-v8-qualification.json','head-counterfactual-healthy-review.json']
 qual=dict(status='PASS',created_unix=time.time(),num_envs=24576,source_sha256=hashes,assets_sha256=previous['assets_sha256'],inputs_sha256={name:sha(r/name) for name in inputs},throughput=actual['throughput'],throughput_target_met=actual['throughput']['aggregate_control_per_second']>=100000,throughput_target_controls_per_second=100000,teacher_contract=actual['lineage']['teachers'],source_transition=transition,controls_per_update=589824,steps_per_env=24,allocation_controls=75497472,stage_maximum_controls=150994944,scope='Actual128 state migrated exactly except declared four walking-head BC labels. Physics/obs/reward/terminal sources unchanged, cfg preconstruction equality. Mature64 smoke/resume and actualN passed; artificial half-batch next-step timeouts are validation only. Throughput limit disclosed; no inference override or skill/real acceptance.')
 q=r/'velstand-head-target-qualification.json';assert not q.exists();q.write_text(json.dumps(qual,indent=2)+'\n');plan.update(qualification=str(q),status='QUALIFIED');save()
 out=r/'velstand-head-target-allocation-001'
 phase=run(out.name,['scripts/train_v2_velstand.py','--output',str(out),'--device','cuda:1','--num-envs','24576','--iterations','128','--resume',str(r/'velstand-v8-allocation-001/checkpoint_000128.pt'),'--source-transition',str(r/'velstand-head-target-transition-003.json'),'--walking-teacher',str(r/'walking-yaw-fine-allocation-002/checkpoint_000512.pt'),'--standing-teacher',str(r/'head2-micro-exposure-allocation-001/checkpoint_001856.pt'),'--qualification',str(q)])
 trained=read(out/'report.json');assert trained['status']=='ALLOCATION_COMPLETE' and trained['completed_updates']==256 and trained['allocation_controls']==75497472
 wall=dict(**phase,actual_controls=trained['allocation_controls'],full_process_control_per_second=trained['allocation_controls']/phase['wall_seconds'],scope='Entire formal subprocess including imports, cfg, archive, physics init, save/export')
 (out/'actual-wall-time.json').write_text(json.dumps(wall,indent=2)+'\n');plan.update(actual_allocation_controls=trained['allocation_controls'],actual_stage_controls=150994944);save()
 checkpoint=out/'checkpoint_000256.pt'
 for stage in ('nominal','measured'):
  run('velstand-headtarget-0256-commands-'+stage,[str(r/'evaluate_v2_velstand_commands.py'),'--checkpoint',str(checkpoint),'--output',str(r/('velstand-headtarget-0256-commands-'+stage)),'--device','cuda:1','--stage',stage,'--repeats','8','--seeds','2026101612','2026101613'])
  run('velstand-headtarget-0256-survey-'+stage,[str(r/'evaluate_v2_velstand_survey.py'),'--checkpoint',str(checkpoint),'--output',str(r/('velstand-headtarget-0256-survey-'+stage)),'--stage',stage])
 plan['status']='STAGE_BUDGET_CLOSED_EVALUATIONS_COMPLETE_REVIEW_REQUIRED';save()
except BaseException as error:
 plan.update(status='FAILED',error=repr(error));save();raise
