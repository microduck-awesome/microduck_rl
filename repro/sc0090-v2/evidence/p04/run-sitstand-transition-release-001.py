"""Import once; fixed-parent CPU/smoke/actual-N/formal/eval bounded pipeline."""
import os,sys,json,time,gc,hashlib,traceback,threading,signal,importlib.util
from pathlib import Path
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'sitstand-transition-release-worktree';os.chdir(W);sys.path.insert(0,str(W/'scripts'));os.environ.pop('MICRODUCK_WARM_START',None)
state=R/'sitstand-transition-release-pipeline-status.json';log=R/'sitstand-transition-release-pipeline-001.log';patterns=('ccd overflow','contact overflow','constraint buffer full','broadphase overflow','narrowphase overflow','height field collision overflow','nefc overflow','njmax_nnz overflow');stop=threading.Event()
def write(x):
 x.update(pid=os.getpid(),updated_unix=time.time());t=state.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(state)
def watch():
 while not stop.wait(2):
  if log.exists() and any(p in log.read_text(errors='replace').lower() for p in patterns):
   (R/'sitstand-transition-release-capacity-rejection.json').write_text(json.dumps(dict(status='CAPACITY_GUARD_REJECTED',pid=os.getpid())));os.kill(os.getpid(),signal.SIGTERM);return
threading.Thread(target=watch,daemon=True).start();write(dict(status='IMPORTING',formal_controls=0));print('WORKER_PID',os.getpid(),flush=True)
import torch
import train_v2_sitstand as train
from mjlab_microduck.custom_v2.sitstand_transition_release import RELEASE_TASK
parent=R/'sitstand-lesson4-allocation-001/checkpoint_000384.pt';transition=W/'source-transition.json';task=RELEASE_TASK

def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def run_train(label,n,it,validation=False,forced=False,qualification=None):
 out=R/('sitstand-transition-release-'+label+'-001');argv=['train_v2_sitstand.py','--task',task,'--output',str(out),'--device','cuda:1','--num-envs',str(n),'--iterations',str(it),'--resume',str(parent),'--source-transition',str(transition)]
 if validation:argv+=['--validation-only']
 if forced:argv+=['--force-timeouts']
 if qualification:argv+=['--qualification',str(qualification)]
 write(dict(status='RUNNING',phase=label,output=str(out),formal_controls=0 if validation else 'read formal status'));print('PHASE_START',label,flush=True);t=time.monotonic();sys.argv=argv;train.main();wall=time.monotonic()-t
 report=json.loads((out/'report.json').read_text());assert report['status']=='ALLOCATION_COMPLETE' and report['completed_updates']==384+it
 (out/'shared-process-phase-wall.json').write_text(json.dumps(dict(seconds_including_environment_and_validation=wall,imports_shared=True,controls=report['allocation_controls'],controls_per_second=report['allocation_controls']/wall),indent=2)+'\n')
 print('PHASE_DONE',label,wall,flush=True);gc.collect();torch.cuda.empty_cache();return out,report
try:
 write(dict(status='RUNNING',phase='CPU contract and native RNG oracle'));cpu=load(W/'tests/test_v2_sitstand_transition_release.py','release_cpu').run()
 smoke,s=run_train('smoke',64,5,True,True);assert s['terminal_bootstrap_samples']>=32 and s['onnx_max_abs_error']<2e-5
 assert max(s['training_metrics']['oracle_max_abs'])<2e-6
 actual,r=run_train('actual16384',16384,16,True);assert r['terminal_bootstrap_samples']>0 and r['contact_peak']<r['contact_capacity'];assert r['training_metrics']['metrics']['feedback_not_ready_fraction']==0 and r['training_metrics']['metrics']['nonfinite_fraction']==0 and max(r['training_metrics']['oracle_max_abs'])<2e-6
 assert r['throughput']['median_control_per_second']>=100000,r['throughput']
 qualification=R/'sitstand-transition-release-qualification-001.json';train.write_json(qualification,dict(status='PASS',task=task,num_envs=16384,source_sha256=r['source_sha256'],smoke=str(smoke/'report.json'),actual_N=str(actual/'report.json'),cpu=str(R/'sitstand-transition-release-cpu-001.json'),source_transition=str(transition),formal_parent=str(parent),formal_control_cap=25165824))
 out,f=run_train('allocation',16384,64,qualification=qualification);assert f['allocation_controls']==25165824 and f['controls']==176160768
 evaluator=load(W/'scripts/evaluate_v2_sitstand_clock_fixed.py','release_evaluator')
 for stage in ('nominal','measured'):
  result=R/('sitstand-transition-release-0448-'+stage);write(dict(status='RUNNING',phase='independent-'+stage,output=str(result),formal_controls=f['controls']));sys.argv=['evaluate','--checkpoint',str(out/'checkpoint_000448.pt'),'--output',str(result),'--device','cuda:1','--stage',stage,'--num-envs','256','--seed','2603'];evaluator.main();gc.collect();torch.cuda.empty_cache()
 write(dict(status='SITSTAND_STAGE_BUDGET_COMPLETE_NO_FURTHER_TRAINING',completed_updates=448,formal_controls=f['controls'],additional_controls=f['allocation_controls'],checkpoint=str(out/'checkpoint_000448.pt')))
except BaseException as e:
 write(dict(status='FAILED_REQUIRES_DIAGNOSIS',exception=repr(e)));traceback.print_exc();raise
finally:stop.set()
