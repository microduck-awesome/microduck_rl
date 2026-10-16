"""Bounded admission ->128-update integration -> independent final evidence.

No automatic second allocation. Validation/evaluation controls are not formal.
"""
from pathlib import Path
import hashlib,json,os,signal,subprocess,time
r=Path(__file__).resolve().parent;w=r/'velstand-v8-repaired-worktree';planfile=r/'velstand-first-allocation-plan.json'
assert not planfile.exists()
plan=dict(status='PREPARING',created_unix=time.time(),num_envs=24576,steps_per_env=24,first_updates=128,first_controls=75497472,stage_maximum_updates=256,stage_maximum_controls=150994944,automatic_second_allocation=False,phases=[],scope='V2 integration, not all skills or hardware acceptance; head stability/turning/stop/recovery share this budget')
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
 smoke=read(r/'velstand-v8-smoke-005/report.json');assert smoke['status']=='ALLOCATION_COMPLETE' and smoke['completed_updates']==5
 assert smoke['terminal_bootstrap_samples']==32 and smoke['onnx_max_abs_error']<2e-5
 run('velstand-final-admission-001',[str(r/'velstand-final-admission-cell.py')])
 assert read(r/'velstand-final-admission-plan.json')['status']=='PASS'
 actual=read(r/'velstand-v8-actual24576-002/report.json');resume=read(r/'velstand-v8-resume64-001/report.json')
 assert actual['status']=='ALLOCATION_COMPLETE' and actual['completed_updates']==16
 assert actual['terminal_bootstrap_samples']==12288
 assert resume['status']=='ALLOCATION_COMPLETE' and resume['completed_updates']==7 and resume['onnx_max_abs_error']<2e-5
 hashes=actual['source_sha256'];assert hashes==smoke['source_sha256']==resume['source_sha256']
 for name,digest in hashes.items():assert sha(w/name)==digest,name
 assert actual['training_metrics']['metrics']['feedback_not_ready_fraction']==0
 assert max(actual['training_metrics']['oracle_max_abs'])<2e-7
 assert actual['contact_peak']<actual['contact_capacity']
 for record in actual['bc_diagnostics']:assert record['expert_bc_kl_mean']<=.5 and record['expert_bc_steps']<=20
 inputs=['velstand-v8-cpu-boundaries-002.json','velstand-balanced-batch-cpu.json','velstand-terminal-successor-002.json','velstand-v8-smoke-005/report.json','velstand-v8-resume64-001/report.json','velstand-v8-actual24576-002/report.json','velstand-teacher-physical-004/report.json','velstand-teacher-physical-004-review.json']
 for name in inputs[:3]:assert read(r/name)['status']=='PASS'
 physics=read(r/'velstand-teacher-physical-004/report.json');assert physics['status']=='PHYSICAL_SURVEY_COMPLETE'
 assert max(physics['oracle_max_abs'])<2e-7 and len(physics['additional_oracles'])==2
 for row in physics['rows']:
  assert row['feedback_invalid_trials']==0 and row['nonfinite_trials']==0
  if row['case'] in ('idle','prone','crouch','sit'):assert row['tail_healthy_hold_trials']==16
 # Exercise new evaluation task/command gates on a validation checkpoint first.
 evalout=r/'velstand-command-admission-001'
 run('velstand-command-admission-001',[str(r/'evaluate_v2_velstand_commands.py'),'--checkpoint',str(r/'velstand-v8-actual24576-002/checkpoint_000016.pt'),'--output',str(evalout),'--device','cuda:1','--stage','nominal','--repeats','2','--seeds','2026101622'])
 ev=read(evalout/'report.json');assert ev['status']=='EVALUATION_COMPLETE'
 assert sum(v['trials'] for v in ev['records'])==42 and sum(v['feedback_invalid_trials'] for v in ev['records'])==0
 inputs.append('velstand-command-admission-001/report.json')
 assets={str(p.relative_to(w)):sha(p) for p in sorted((w/'src/mjlab_microduck/custom_v2/assets').rglob('*')) if p.is_file()}
 param='src/mjlab_microduck/actuator/params/sc0090_v5_m6.json';assets[param]=sha(w/param)
 original=read(r/'walking-final-qualification.json')
 for name,digest in original['assets_sha256'].items():assert assets[name]==digest,name
 for name,digest in hashes.items():assert sha(w/name)==digest,name
 qual=dict(status='PASS',created_unix=time.time(),num_envs=24576,source_sha256=hashes,assets_sha256=assets,inputs_sha256={name:sha(r/name) for name in inputs},throughput=actual['throughput'],teacher_contract=actual['lineage']['teachers'],controls_per_update=589824,steps_per_env=24,first_allocation_controls=75497472,stage_maximum_controls=150994944,scope='Small/actual-N and native GPU resume/timeout/reward/ONNX passed. One teacher physical survey, including2/16 fast-forward walking failures, is not full skill or real acceptance. Large-N timeout test schedules half batch to the next time boundary; not a full20s episode longevity test.')
 q=r/'velstand-v8-qualification.json';assert not q.exists();q.write_text(json.dumps(qual,indent=2)+'\n')
 plan.update(qualification=str(q),admitted_throughput=actual['throughput'],status='QUALIFIED');save()
 out=r/'velstand-v8-allocation-001'
 phase=run('velstand-v8-allocation-001',['scripts/train_v2_velstand.py','--output',str(out),'--device','cuda:1','--num-envs','24576','--iterations','128','--warm-start',str(r/'walking-yaw-fine-allocation-002/checkpoint_000512.pt'),'--walking-teacher',str(r/'walking-yaw-fine-allocation-002/checkpoint_000512.pt'),'--standing-teacher',str(r/'head2-micro-exposure-allocation-001/checkpoint_001856.pt'),'--qualification',str(q)])
 trained=read(out/'report.json');assert trained['status']=='ALLOCATION_COMPLETE' and trained['completed_updates']==128 and trained['allocation_controls']==75497472
 wall=dict(**phase,actual_controls=trained['allocation_controls'],full_process_control_per_second=trained['allocation_controls']/phase['wall_seconds'],scope='Complete formal subprocess including imports, physics init, export/save')
 (out/'actual-wall-time.json').write_text(json.dumps(wall,indent=2)+'\n');plan['actual_formal_controls']=trained['allocation_controls'];save()
 checkpoint=out/'checkpoint_000128.pt'
 for stage in ('nominal','measured'):
  run('velstand-0128-commands-'+stage,[str(r/'evaluate_v2_velstand_commands.py'),'--checkpoint',str(checkpoint),'--output',str(r/('velstand-0128-commands-'+stage)),'--device','cuda:1','--stage',stage,'--repeats','8','--seeds','2026101612','2026101613'])
  run('velstand-0128-survey-'+stage,[str(r/'evaluate_v2_velstand_survey.py'),'--checkpoint',str(checkpoint),'--output',str(r/('velstand-0128-survey-'+stage)),'--stage',stage])
 plan['status']='FIRST_ALLOCATION_AND_EVALUATIONS_COMPLETE_REVIEW_REQUIRED';save()
except BaseException as error:
 plan.update(status='FAILED',error=repr(error));save();raise
