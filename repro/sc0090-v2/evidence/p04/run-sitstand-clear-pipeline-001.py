import json,os,subprocess,time,signal
from pathlib import Path
root=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');work=root/'sitstand-clear-pose-worktree'
env=dict(os.environ,SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat');env.pop('MICRODUCK_WARM_START',None)
task='Mjlab-SitStand-Flat-MicroDuck-V2-ClearSeat-Repaired-S0-O0-R1';parent=root/'sitstand-allocation-001/checkpoint_000128.pt';transition=work/'source-transition.json'
patterns=('ccd overflow','contact overflow','constraint buffer full','broadphase overflow','narrowphase overflow','height field collision overflow','nefc overflow','njmax_nnz overflow')
statepath=root/'sitstand-clear-pipeline-status.json'
def write(value):
 tmp=statepath.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(statepath)
def run(args,logname,phase):
 logpath=root/logname;started=time.monotonic()
 with logpath.open('w') as log:
  child=subprocess.Popen(['scripts/v2_python.sh',*args],cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT)
  write(dict(status='RUNNING',phase=phase,pid=child.pid,updated_unix=time.time(),log=str(logpath)));print(phase,'PID',child.pid,flush=True)
  rejected=False
  while child.poll() is None:
   text=logpath.read_text().lower()
   if not rejected and any(x in text for x in patterns):child.send_signal(signal.SIGTERM);rejected=True
   time.sleep(2)
  assert child.returncode==0 and not rejected,(phase,child.returncode,rejected)
 return time.monotonic()-started
smoke=root/'sitstand-clear-smoke-001/report.json'
while not smoke.exists():
 if not Path('/proc/3788221').exists():raise RuntimeError('clear target smoke stopped without report')
 time.sleep(2)
s=json.loads(smoke.read_text());assert s['status']=='ALLOCATION_COMPLETE' and s['completed_updates']==133 and s['terminal_bootstrap_samples']>=32 and s['onnx_max_abs_error']<2e-5
run(['scripts/train_v2_sitstand.py','--task',task,'--output',str(root/'sitstand-clear-actual16384-001'),'--device','cuda:0','--num-envs','16384','--iterations','32','--validation-only','--resume',str(parent),'--source-transition',str(transition)],'sitstand-clear-actual16384-001.log','actual-N-validation')
r=json.loads((root/'sitstand-clear-actual16384-001/report.json').read_text());assert r['status']=='ALLOCATION_COMPLETE' and r['completed_updates']==160 and r['terminal_bootstrap_samples']>0
assert r['contact_peak']<r['contact_capacity'] and r['training_metrics']['metrics']['nonfinite_fraction']==0 and max(r['training_metrics']['oracle_max_abs'])<2e-6
assert r['throughput']['median_control_per_second']>=100000,r['throughput']
qualification=root/'sitstand-clear-qualification-001.json';qualification.write_text(json.dumps(dict(status='PASS',num_envs=16384,source_sha256=r['source_sha256'],task=task,throughput=r['throughput'],pose_evidence=str(root/'sitstand-clear-pose-batch-002/report.json'),source_transition=str(transition),smoke=str(smoke),actual_N=str(root/'sitstand-clear-actual16384-001/report.json')),indent=2)+'\n')
out=root/'sitstand-clear-allocation-001'
wall=run(['scripts/train_v2_sitstand.py','--task',task,'--output',str(out),'--device','cuda:0','--num-envs','16384','--iterations','128','--resume',str(parent),'--source-transition',str(transition),'--qualification',str(qualification)],'sitstand-clear-allocation-001.log','formal-128-to-256')
report=json.loads((out/'report.json').read_text());assert report['completed_updates']==256
(out/'outer-process-wall.json').write_text(json.dumps(dict(wall_seconds_including_imports=wall,controls=report['allocation_controls'],controls_per_second=report['allocation_controls']/wall),indent=2)+'\n')
for stage in ('nominal','measured'):
 run(['scripts/evaluate_v2_sitstand.py','--checkpoint',str(out/'checkpoint_000256.pt'),'--output',str(root/('sitstand-clear-0256-'+stage)),'--device','cuda:0','--stage',stage,'--num-envs','256','--seed','2603'],'sitstand-clear-0256-'+stage+'.log','independent-'+stage)
write(dict(status='EVALUATION_COMPLETE_DECISION_REQUIRED',completed_updates=256,checkpoint=str(out/'checkpoint_000256.pt'),updated_unix=time.time(),formal_controls=report['allocation_controls'],total_sitstand_formal_controls=report['controls'],scope='Read actual bidirectional/hold results; no blind continuation'))
print('Formal and independent evaluations complete; inspect behavior now',flush=True)
