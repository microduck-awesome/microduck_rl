"""Bounded unchanged-v8 continuation after mature actual-N validation.

Head-label alternatives are closed; never load their validation weights.
"""
from pathlib import Path
import hashlib,json,os,signal,subprocess,time
r=Path(__file__).resolve().parent;w=r/'velstand-v8-repaired-worktree';planfile=r/'velstand-v8-final-allocation-plan.json'
assert not planfile.exists()
plan=dict(status='PREPARING',created_unix=time.time(),num_envs=24576,steps_per_env=24,start_update=128,end_update=256,allocation_updates=128,allocation_controls=75497472,stage_maximum_controls=150994944,automatic_further_allocation=False,phases=[],scope='Unchanged qualified v8 source; last stage allocation only after mature actualN correctness. Camera/stop/turn/recovery all share the cap; no head-label variant or inference override.')
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
 small=read(r/'velstand-v8-mature64-control-001/report.json')
 assert small['status']=='STOPPED' and small['completed_updates']==129 and small['stop_reason']=='post_bc_kl_guard'
 previous=read(r/'velstand-v8-qualification.json');parent=r/'velstand-v8-allocation-001/checkpoint_000128.pt'
 for name,digest in previous['source_sha256'].items():assert sha(w/name)==digest,name
 teachers=['--walking-teacher',str(r/'walking-yaw-fine-allocation-002/checkpoint_000512.pt'),'--standing-teacher',str(r/'head2-micro-exposure-allocation-001/checkpoint_001856.pt')]
 run('velstand-v8-mature24576-001',['scripts/train_v2_velstand.py','--output',str(r/'velstand-v8-mature24576-001'),'--device','cuda:1','--num-envs','24576','--iterations','16','--resume',str(parent),*teachers,'--validation-only','--force-timeouts'])
 actual=read(r/'velstand-v8-mature24576-001/report.json')
 assert actual['status']=='ALLOCATION_COMPLETE' and actual['completed_updates']==144 and actual['stop_reason'] is None
 assert actual['allocation_controls']==9437184 and actual['source_sha256']==previous['source_sha256']
 assert actual['terminal_bootstrap_samples']>=12288 and actual['contact_peak']<actual['contact_capacity']
 assert actual['training_metrics']['metrics']['feedback_not_ready_fraction']==0 and actual['training_metrics']['metrics']['nonfinite_fraction']==0
 assert max(actual['training_metrics']['oracle_max_abs'])<2e-7
 for bc in actual['bc_diagnostics']:assert bc['expert_bc_kl_mean']<=.5 and bc['expert_bc_steps']<=20
 for name,digest in previous['assets_sha256'].items():assert sha(w/name)==digest,name
 qual=dict(previous);qual.update(created_unix=time.time(),mature_actual_N=str(r/'velstand-v8-mature24576-001/report.json'),mature_actual_N_sha256=sha(r/'velstand-v8-mature24576-001/report.json'),mature64_failure=str(r/'velstand-v8-mature64-control-001/report.json'),throughput=actual['throughput'],scope=previous['scope']+' Mature128 actualN16 continuation validated; mature64 original and two head-label alternatives all triggered postBC guard. Original64 initial smoke was previously passed. No source or teacher change. Do not equate smallN failure with a head-label causal effect; head-label alternatives closed without formal.')
 q=r/'velstand-v8-mature-qualification.json';assert not q.exists();q.write_text(json.dumps(qual,indent=2)+'\n');plan.update(status='MATURE_ACTUAL_N_QUALIFIED',qualification=str(q));save()
 out=r/'velstand-v8-allocation-002'
 phase=run(out.name,['scripts/train_v2_velstand.py','--output',str(out),'--device','cuda:1','--num-envs','24576','--iterations','128','--resume',str(parent),*teachers,'--qualification',str(q)])
 trained=read(out/'report.json');assert trained['status']=='ALLOCATION_COMPLETE' and trained['completed_updates']==256 and trained['allocation_controls']==75497472
 wall=dict(**phase,actual_controls=trained['allocation_controls'],full_process_control_per_second=trained['allocation_controls']/phase['wall_seconds'],scope='Entire formal subprocess including imports/config/archive/init/save/export')
 (out/'actual-wall-time.json').write_text(json.dumps(wall,indent=2)+'\n');plan.update(actual_allocation_controls=trained['allocation_controls'],actual_stage_controls=150994944);save()
 checkpoint=out/'checkpoint_000256.pt'
 for stage in ('nominal','measured'):
  run('velstand-0256-commands-'+stage,[str(r/'evaluate_v2_velstand_commands.py'),'--checkpoint',str(checkpoint),'--output',str(r/('velstand-0256-commands-'+stage)),'--device','cuda:1','--stage',stage,'--repeats','8','--seeds','2026101612','2026101613'])
  run('velstand-0256-survey-'+stage,[str(r/'evaluate_v2_velstand_survey.py'),'--checkpoint',str(checkpoint),'--output',str(r/('velstand-0256-survey-'+stage)),'--stage',stage])
 plan['status']='STAGE_BUDGET_CLOSED_EVALUATIONS_COMPLETE_REVIEW_REQUIRED';save()
except BaseException as error:
 plan.update(status='FAILED',error=repr(error));save();raise
