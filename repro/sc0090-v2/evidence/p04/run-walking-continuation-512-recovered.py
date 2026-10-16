"""One bounded same-source continuation, admitted by completed384 batteries."""
import hashlib,json,os,pathlib,signal,subprocess,time
r=pathlib.Path(__file__).resolve().parent;src=r/'walking-yaw-fine-worktree';parent=r/'walking-yaw-fine-allocation-001/checkpoint_000384.pt';out=r/'walking-yaw-fine-allocation-002';p=r/'walking-continuation-512-recovery-plan.json'
assert not p.exists() and not out.exists()
plan=dict(status='WAITING_384_COMMON_EVALUATIONS',created_unix=time.time(),parent=str(parent),planned_updates=128,planned_end_update=512,planned_controls=100663296,num_envs=32768,steps_per_env=24,reason='Nominal321/384 and measured321 retain336/336 while yaw oscillation/positive turning improve; continue native curriculum with no source/parameter edit; plateaued rise-head refinement stays stopped',native_stage500=dict(action_rate_weight=-.2,standing_fraction=.05,head_ranges=[.17,.17,.21,.047],body_and_head_com_m=.005),scope='Bounded continuation to assess native stage500; distinct from single-factor attribution and skill acceptance; no automatic next training segment',phases=[])
def save():
 t=p.with_suffix('.tmp');t.write_text(json.dumps(plan,indent=2)+'\n');t.replace(p)
save()
try:
 recovery=json.loads((r/'walking-yaw-fine-evaluation-recovery.json').read_text());assert recovery['status']=='PASS_CLEAN_INDEPENDENT_REEVALUATION' and recovery['repeated_returncode']==0
 plan['recovered_evaluation']=recovery
 evidence=[]
 for stage in ('nominal','measured'):
  reportpath=(r/'walking-yaw-fine-000384-measured-attempt2/report.json') if stage=='measured' else (r/'walking-yaw-fine-000384-nominal/report.json');d=json.loads(reportpath.read_text());assert d['status']=='EVALUATION_COMPLETE'
  trials=sum(x['trials'] for x in d['records']);survived=sum(x['survived_trials'] for x in d['records']);faults=sum(x['feedback_invalid_trials'] for x in d['records'])
  assert trials==336 and survived>=334 and faults==0,(stage,trials,survived,faults)
  evidence.append(dict(stage=stage,trials=trials,survived=survived,faults=faults,report_sha256=hashlib.sha256(reportpath.read_bytes()).hexdigest()))
 old=json.loads((r/'walking-yaw-fine-allocation-001/report.json').read_text());assert old['status']=='ALLOCATION_COMPLETE' and old['completed_updates']==384
 q=r/'walking-yaw-fine-qualification.json';qual=json.loads(q.read_text());assert qual['status']=='PASS' and qual['num_envs']==32768
 for name,digest in qual['source_sha256'].items():assert hashlib.sha256((src/name).read_bytes()).hexdigest()==digest,name
 original=json.loads((r/'walking-final-qualification.json').read_text())
 for name,digest in original['assets_sha256'].items():assert hashlib.sha256((src/name).read_bytes()).hexdigest()==digest,name
 plan.update(status='QUALIFIED_SAME_SOURCE',evidence=evidence,parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),qualification=str(q));save()
 cmd=[str(src/'scripts/v2_python.sh'),'scripts/train_v2_walking.py','--task','Mjlab-Velocity-Flat-MicroDuck-V2-YawFine-Repaired-S0-O0-R1','--device','cuda:1','--num-envs','32768','--iterations','128','--resume',str(parent),'--qualification',str(q),'--output',str(out)]
 phase=dict(name='formal384_to512',arguments=cmd,start_unix=time.time());plan['phases'].append(phase)
 env=dict(os.environ,SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat')
 logpath=r/'walking-yaw-fine-allocation-002.log';patterns=('ccd overflow','broadphase overflow','narrowphase overflow','nefc overflow','njmax_nnz overflow','height field collision overflow','contact overflow','contact buffer overflow','constraint buffer full');rejected=False
 with logpath.open('w') as log:
  proc=subprocess.Popen(cmd,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT);phase['pid']=proc.pid;plan['status']='FORMAL_RUNNING';save()
  while proc.poll() is None:
   contents=logpath.read_text(errors='replace').lower();found=[x for x in patterns if x in contents]
   if found and not rejected:rejected=True;phase['capacity_rejection']=found;proc.send_signal(signal.SIGTERM);save()
   time.sleep(2)
 phase.update(returncode=proc.returncode,wall_seconds=time.time()-phase['start_unix']);save()
 assert proc.returncode==0 and not rejected,phase
 state=json.loads((out/'status.json').read_text());assert state['status']=='ALLOCATION_COMPLETE' and state['completed_updates']==512
 wall={**phase,'actual_controls':state['allocation_controls'],'full_process_control_per_second':state['allocation_controls']/phase['wall_seconds'],'scope':'Entire formal process including imports/init/save and any concurrent read-only evaluations'}
 (out/'actual-wall-time.json').write_text(json.dumps(wall,indent=2)+'\n');plan.update(status='FORMAL_COMPLETE_EVALUATIONS_SEPARATE',actual_controls=state['allocation_controls']);save()
except BaseException as error:
 plan.update(status='FAILED',error=repr(error));save();raise
