"""Last stage128, native-invalid-count attribution, serial final evaluation."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,time
r=Path(__file__).resolve().parent;w=r/'velstand-v8-repaired-worktree';planfile=r/'velstand-v8-final-audited-plan.json'
assert not planfile.exists()
plan=dict(status='PREPARING',created_unix=time.time(),num_envs=24576,steps_per_env=24,start_update=128,end_update=256,allocation_controls=75497472,stage_maximum_controls=150994944,automatic_further_allocation=False,phases=[],scope='Unchanged v8 source. Correct native-invalid counts terminate their episode; unexplained faults/nonfinite or >32 invalid controls stop and save the whole allocation. No head-label override or extra stage budget.')
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
 previous=read(r/'velstand-v8-qualification.json');actual=read(r/'velstand-v8-mature24576-001/report.json');diag=read(r/'velstand-v8-mature-feedback-001/report.json');capture=read(r/'velstand-v8-mature-feedback-001/feedback-capture-0.json')
 for report in (actual,diag):
  assert report['status']=='ALLOCATION_COMPLETE' and report['completed_updates']==144 and report['stop_reason'] is None
  assert report['source_sha256']==previous['source_sha256'] and report['allocation_controls']==9437184
  assert report['terminal_bootstrap_samples']>=12288 and report['contact_peak']<report['contact_capacity']
  assert max(report['training_metrics']['oracle_max_abs'])<2e-7
  assert round(report['training_metrics']['metrics']['feedback_not_ready_fraction']*9437184)==1
  assert max(x['expert_bc_kl_mean'] for x in report['bc_diagnostics'])<=.5
 assert diag['training_metrics']['diagnostic_nonfinite_control_samples']==0 and capture['worlds_with_feedback_failure']==1
 for name,digest in previous['source_sha256'].items():assert sha(w/name)==digest,name
 for name,digest in previous['assets_sha256'].items():assert sha(w/name)==digest,name
 q=r/'velstand-v8-mature-qualified-native-range.json';assert not q.exists()
 qual=dict(previous);qual.update(created_unix=time.time(),throughput=actual['throughput'],actual_mature_N=str(r/'velstand-v8-mature24576-001/report.json'),readonly_fault_diagnostic=str(r/'velstand-v8-mature-feedback-001/report.json'),fault_capture_sha256=sha(r/'velstand-v8-mature-feedback-001/feedback-first-faults-0.pt'),formal_hook_sha256=sha(r/'train-velstand-v8-final-with-fault-audit.py'),explained_fault='head_yaw native count1026 >1023, valid transaction, source status-1, reason13, no numeric/nonfinite error. This simulated read exceeds the declared feedback contract; no claim about unknown physical sensor saturation/wrap behavior.',maximum_native_range_invalid_controls=32,scope=previous['scope']+' Two mature actualN16 validations completed; each1 invalid native-count sample. Formal sidecar checks every bad lane, preserves original per-episode termination, and stops/saves on unexplained/nonfinite or >32 invalid controls. All counters/causes disclosed; never call training feedback-free.')
 q.write_text(json.dumps(qual,indent=2)+'\n');plan['qualification']=str(q);save()
 out=r/'velstand-v8-allocation-002'
 phase=run(out.name,[str(r/'train-velstand-v8-final-with-fault-audit.py')])
 trained=read(out/'report.json');assert trained['status']=='ALLOCATION_COMPLETE' and trained['completed_updates']==256 and trained['allocation_controls']==75497472
 assert trained['training_metrics']['diagnostic_unexplained_invalid_lanes']==0 and trained['training_metrics']['diagnostic_nonfinite_control_samples']==0
 wall=dict(**phase,actual_controls=trained['allocation_controls'],full_process_control_per_second=trained['allocation_controls']/phase['wall_seconds'],scope='Entire formal subprocess including imports/config/archive/init/save/export and read-only fault diagnostics')
 (out/'actual-wall-time.json').write_text(json.dumps(wall,indent=2)+'\n');plan.update(actual_allocation_controls=trained['allocation_controls'],actual_stage_controls=150994944);save()
 checkpoint=out/'checkpoint_000256.pt'
 for stage in ('nominal','measured'):
  run('velstand-0256-commands-'+stage,[str(r/'evaluate_v2_velstand_commands.py'),'--checkpoint',str(checkpoint),'--output',str(r/('velstand-0256-commands-'+stage)),'--device','cuda:1','--stage',stage,'--repeats','8','--seeds','2026101612','2026101613'])
  run('velstand-0256-survey-'+stage,[str(r/'evaluate_v2_velstand_survey.py'),'--checkpoint',str(checkpoint),'--output',str(r/('velstand-0256-survey-'+stage)),'--stage',stage])
 plan['status']='STAGE_BUDGET_CLOSED_EVALUATIONS_COMPLETE_REVIEW_REQUIRED';save()
except BaseException as error:
 plan.update(status='FAILED',error=repr(error));save();raise
