"""Final shared BallKick budget: admission -> exact parent85 ->85 updates -> unchanged evals."""
import json,os,time,hashlib,subprocess,signal
from pathlib import Path
r=Path(__file__).resolve().parent;w=r/'ballkick-finish-worktree';parent=r/'ballkick-right-allocation-001/checkpoint_000085.pt';transition=r/'ballkick-finish-source-transition.json';qualification=r/'ballkick-finish-qualification.json'
env=dict(os.environ,SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat');env.pop('MICRODUCK_WARM_START',None)
patterns=('ccd overflow','contact overflow','constraint buffer full','broadphase overflow','narrowphase overflow','height field collision overflow','nefc overflow','njmax_nnz overflow')
def state(**kw):
 p=r/'ballkick-finish-pipeline-status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(updated_unix=time.time(),**kw),indent=2)+'\n');t.replace(p)
def run(args,name,formal=False):
 log=r/(name+'.log');start=time.monotonic()
 with log.open('w') as f:
  child=subprocess.Popen(['scripts/v2_python.sh',*args],cwd=w,env=env,stdout=f,stderr=subprocess.STDOUT);state(status='RUNNING',phase=name,pid=child.pid,log=str(log));print(name,child.pid,flush=True);stop=None
  while child.poll() is None:
   if not stop and any(x in log.read_text().lower() for x in patterns):stop='capacity_log'
   if stop:child.send_signal(signal.SIGTERM);state(status='SAFE_STOP_REQUESTED',phase=name,pid=child.pid,reason=stop);child.wait();break
   time.sleep(2)
  assert child.returncode==0 and not stop,(name,child.returncode,stop)
 result=json.loads((r/name/'report.json').read_text());wall=time.monotonic()-start
 (r/name/'outer-process-wall.json').write_text(json.dumps(dict(wall_seconds=wall,allocation_kind='formal' if formal else ('validation' if 'training_metrics' in result else 'evaluation'),controls=result.get('allocation_controls',0),control_per_second=result.get('allocation_controls',0)/wall),indent=2)+'\n');return result
try:
 state(status='WAITING_CPU_AND_MATURE_SMOKE',pid=58312)
 report=r/'ballkick-finish-smoke-001/report.json';log=r/'ballkick-finish-cpu-smoke-001.log';start=time.monotonic()
 while not report.exists() or Path('/proc/58312').exists():
  assert time.monotonic()-start<900,'mature smoke timeout'
  if log.exists():assert 'Traceback' not in log.read_text(),str(log)
  time.sleep(2)
 assert not any(x in log.read_text().lower() for x in patterns)
 cpu=json.loads((r/'ballkick-finish-cpu-001.json').read_text());smoke=json.loads(report.read_text())
 assert cpu['status']=='PASS' and cpu['inputs_rng_unchanged'] and cpu['pre_valid_cost_zero']
 assert smoke['status']=='ALLOCATION_COMPLETE' and smoke['completed_updates']==90 and smoke['allocation_controls']==7680 and not smoke['fresh_initialization']
 assert smoke['terminal_bootstrap_samples']>=32 and smoke['onnx_max_abs_error']<2e-5
 base=['scripts/train_v2_ballkick.py','--device','cuda:1','--resume',str(parent),'--source-transition',str(transition)]
 actual=run([*base,'--output',str(r/'ballkick-finish-actual24576-001'),'--num-envs','24576','--iterations','16','--validation-only'],'ballkick-finish-actual24576-001')
 assert actual['completed_updates']==101 and actual['allocation_controls']==9437184 and actual['terminal_bootstrap_samples']>0 and actual['stop_reason'] is None
 metrics=actual['training_metrics']['metrics'];assert metrics['feedback_not_ready_fraction']==metrics['nonfinite_fraction']==0
 assert max(actual['training_metrics']['oracle_max_abs'])<2e-5
 assert actual['source_sha256']==cpu['source_sha256']==smoke['source_sha256']
 for rel,sha in actual['source_sha256'].items():assert hashlib.sha256((w/rel).read_bytes()).hexdigest()==sha,rel
 oldq=json.loads((r/'ballkick-qualification-001.json').read_text())
 for rel,sha in oldq.get('model_asset_sha256',{}).items():assert hashlib.sha256((w/rel).read_bytes()).hexdigest()==sha,rel
 assert hashlib.sha256((w/'scripts/evaluate_v2_ballkick.py').read_bytes()).hexdigest()==oldq['evaluator_script_sha256']
 q=dict(status='PASS',task=oldq['task'],worktree=str(w),source_sha256=actual['source_sha256'],model_asset_sha256=oldq.get('model_asset_sha256',{}),evaluator_script_sha256=oldq['evaluator_script_sha256'],num_envs=24576,steps_per_env=24,formal_allocation_updates=85,formal_allocation_controls=50135040,formal_inherited_controls=50135040,stage_final_controls=100270080,stage_max_controls=100663296,unused_controls=393216,max_behavior_recipes=2,source_transition=str(transition),parent_checkpoint=str(parent),cpu_report=str(r/'ballkick-finish-cpu-001.json'),smoke_report=str(report),actual_report=str(r/'ballkick-finish-actual24576-001/report.json'),throughput=actual['throughput'],throughput_target_met=actual['throughput']['aggregate_control_per_second']>=100000,scope='Final direction/quiet costs only; old physics/observations/contact logic/terminal critic and original independent evaluation unchanged; no skill/hardware acceptance')
 qualification.write_text(json.dumps(q,indent=2)+'\n')
 out=r/'ballkick-finish-allocation-001'
 formal=run([*base,'--output',str(out),'--num-envs','24576','--iterations','85','--qualification',str(qualification)],'ballkick-finish-allocation-001',True)
 assert formal['completed_updates']==170 and formal['allocation_controls']==50135040 and formal['controls']==100270080 and not formal['fresh_initialization']
 for stage in ('nominal','measured'):
  name='ballkick-finish-0170-'+stage
  run(['scripts/evaluate_v2_ballkick.py','--checkpoint',str(out/'checkpoint_000170.pt'),'--output',str(r/name),'--device','cuda:1','--stage',stage],name)
 state(status='BALLKICK_STAGE_COMPLETE_NO_FURTHER_TRAINING',formal_updates=170,formal_controls=100270080,stage_max_controls=100663296,unused_controls=393216,scope='Two fixed recipes and original evals complete; no further right or left training in this budget; judge ball/body/head/camera separately')
except Exception as e:
 state(status='FAILED_REVIEW_REQUIRED',error=repr(e));raise
