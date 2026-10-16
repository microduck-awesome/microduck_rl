"""Fixed final recipe and stage budget, stop on admission/numerical failures."""
import json,os,time,hashlib,subprocess,signal
from pathlib import Path
r=Path(__file__).resolve().parent;w=r/'roulade-endpoint-worktree';parent=r/'roulade-allocation-001/checkpoint_000128.pt';transition=r/'roulade-endpoint-source-transition.json';qualification=r/'roulade-endpoint-qualification.json'
env=dict(os.environ,SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat');env.pop('MICRODUCK_WARM_START',None)
patterns=('ccd overflow','contact overflow','constraint buffer full','broadphase overflow','narrowphase overflow','height field collision overflow','nefc overflow','njmax_nnz overflow')
def state(**kw):
 p=r/'roulade-endpoint-pipeline-status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(updated_unix=time.time(),**kw),indent=2)+'\n');t.replace(p)
def run(args,name,formal=False):
 log=r/(name+'.log');started=time.monotonic()
 with log.open('w') as f:
  child=subprocess.Popen(['scripts/v2_python.sh',*args],cwd=w,env=env,stdout=f,stderr=subprocess.STDOUT);state(status='RUNNING',phase=name,pid=child.pid,log=str(log));print(name,child.pid,flush=True);stop=None
  while child.poll() is None:
   if not stop and any(x in log.read_text().lower() for x in patterns):stop='capacity_log'
   sp=r/name/'status.json'
   if not stop and sp.exists():
    try:
     metrics=json.loads(sp.read_text())['training_metrics']['metrics']
     if metrics['feedback_invalid_fraction']>0:stop='new_feedback_requires_attribution'
     elif metrics['nonfinite_fraction']>0:stop='nonfinite_physics'
    except (KeyError,json.JSONDecodeError):pass
   if stop:
    child.send_signal(signal.SIGTERM);state(status='SAFE_STOP_REQUESTED',phase=name,pid=child.pid,reason=stop);child.wait();break
   time.sleep(2)
  assert child.returncode==0 and not stop,(name,child.returncode,stop)
 result=json.loads((r/name/'report.json').read_text());wall=time.monotonic()-started
 (r/name/'outer-process-wall.json').write_text(json.dumps(dict(wall_seconds=wall,allocation_kind='formal' if formal else ('validation' if 'training_metrics' in result else 'evaluation'),controls=result.get('allocation_controls',0),control_per_second=result.get('allocation_controls',0)/wall),indent=2)+'\n');return result
def wait_report(path,log,timeout=300):
 start=time.monotonic()
 while not path.exists():
  assert time.monotonic()-start<timeout,('prerequisite_timeout',path)
  if log.exists():assert 'Traceback' not in log.read_text(),str(log)
  time.sleep(2)
 return json.loads(path.read_text())
try:
 state(status='WAITING_CPU_AND_PHYSICAL_REPORTS')
 cpu=wait_report(r/'roulade-endpoint-cpu-001.json',r/'roulade-endpoint-cpu-001.log')
 physics=wait_report(r/'roulade-endpoint-physical-001/report.json',r/'roulade-endpoint-physical-001.log')
 assert cpu['status']=='PASS' and cpu['play_equal'] and cpu['rl_numeric_equal']
 assert physics['results'][0]['healthy_standing_last1s']==16
 assert sum(x['healthy_standing_last1s'] for x in physics['results'][1:])>=100
 assert min(x['healthy_standing_last1s'] for x in physics['results'][1:])>=12
 assert all(x['full3s_fault_trials']==x['full3s_nonfinite_trials']==x['self_contact_seen_trials']==0 for x in physics['results'])
 base=['scripts/train_v2_roulade.py','--device','cuda:0','--resume',str(parent),'--source-transition',str(transition)]
 smoke=run([*base,'--output',str(r/'roulade-endpoint-smoke-001'),'--num-envs','64','--iterations','5','--validation-only','--force-timeouts'],'roulade-endpoint-smoke-001')
 assert smoke['completed_updates']==133 and smoke['allocation_controls']==7680 and not smoke['fresh_initialization']
 assert smoke['terminal_bootstrap_samples']>=32 and smoke['terminal_validation']['status']=='PASS' and smoke['onnx_max_abs_error']<2e-5
 actual=run([*base,'--output',str(r/'roulade-endpoint-actual16384-001'),'--num-envs','16384','--iterations','16','--validation-only'],'roulade-endpoint-actual16384-001')
 assert actual['completed_updates']==144 and actual['allocation_controls']==6291456 and actual['terminal_bootstrap_samples']>0 and actual['stop_reason'] is None
 assert actual['throughput']['median_control_per_second']>100000
 assert actual['training_metrics']['metrics']['feedback_invalid_fraction']==actual['training_metrics']['metrics']['nonfinite_fraction']==0
 assert actual['source_sha256']==cpu['source_sha256']==smoke['source_sha256']
 for rel,sha in actual['source_sha256'].items():assert hashlib.sha256((w/rel).read_bytes()).hexdigest()==sha,rel
 oldq=json.loads((r/'roulade-qualification-001.json').read_text())
 for rel,sha in oldq['model_asset_sha256'].items():assert hashlib.sha256((w/rel).read_bytes()).hexdigest()==sha,rel
 q=dict(status='PASS',task=oldq['task'],worktree=str(w),source_sha256=actual['source_sha256'],model_asset_sha256=oldq['model_asset_sha256'],num_envs=16384,formal_allocation_updates=128,stage_max_controls=100663296,formal_inherited_controls=50331648,source_transition=str(transition),parent_checkpoint=str(parent),physics_report=str(r/'roulade-endpoint-physical-001/report.json'),cpu_report=str(r/'roulade-endpoint-cpu-001.json'),smoke_report=str(r/'roulade-endpoint-smoke-001/report.json'),actual_report=str(r/'roulade-endpoint-actual16384-001/report.json'),scope='Final endpoint exposure recipe; inherited reverse-curriculum latch/progress not counted as full roll; original full battery unchanged')
 qualification.write_text(json.dumps(q,indent=2)+'\n')
 out=r/'roulade-endpoint-allocation-001'
 formal=run([*base,'--output',str(out),'--num-envs','16384','--iterations','128','--qualification',str(qualification)],'roulade-endpoint-allocation-001',True)
 assert formal['completed_updates']==256 and formal['allocation_controls']==50331648 and formal['controls']==100663296 and not formal['fresh_initialization']
 for stage in ('nominal','measured'):
  name='roulade-endpoint-0256-'+stage
  run(['scripts/evaluate_v2_roulade.py','--checkpoint',str(out/'checkpoint_000256.pt'),'--output',str(r/name),'--device','cuda:0','--stage',stage],name)
 state(status='STAGE_BUDGET_COMPLETE_NO_FURTHER_TRAINING',formal_updates=256,formal_controls=100663296,stage_max_controls=100663296,scope='Both original full evals complete; determine acceptance from real body/head/camera behavior; no automatic next recipe')
except Exception as e:
 state(status='FAILED_REVIEW_REQUIRED',error=repr(e));raise
