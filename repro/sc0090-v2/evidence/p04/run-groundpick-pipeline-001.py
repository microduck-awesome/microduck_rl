import json,os,subprocess,time,signal
from pathlib import Path
root=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');work=root/'groundpick-repaired-worktree'
env=dict(os.environ,SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat');env.pop('MICRODUCK_WARM_START',None)
task='Mjlab-GroundPick-Flat-MicroDuck-V2-Repaired-S0-O0-R1'
patterns=('ccd overflow','contact overflow','constraint buffer full','broadphase overflow','narrowphase overflow','height field collision overflow','nefc overflow','njmax_nnz overflow')
statepath=root/'groundpick-pipeline-status.json'
def write(value):
 p=statepath.with_suffix('.tmp');p.write_text(json.dumps(value,indent=2)+'\n');p.replace(statepath)
def run(args,logname,phase):
 logpath=root/logname;start=time.monotonic()
 with logpath.open('w') as log:
  child=subprocess.Popen(['scripts/v2_python.sh',*args],cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT)
  write(dict(status='RUNNING',phase=phase,pid=child.pid,log=str(logpath),updated_unix=time.time()));print(phase,'PID',child.pid,flush=True);bad=False
  while child.poll() is None:
   text=logpath.read_text().lower()
   if not bad and any(p in text for p in patterns):bad=True;child.send_signal(signal.SIGTERM)
   time.sleep(2)
  assert child.returncode==0 and not bad,(phase,child.returncode,bad)
 return time.monotonic()-start
smoke=root/'groundpick-smoke-001/report.json'
while not smoke.exists():
 log=(root/'groundpick-smoke-001.log').read_text()
 if 'Traceback (most recent call last)' in log:raise RuntimeError('GroundPick smoke failed; preserve before repair')
 time.sleep(2)
s=json.loads(smoke.read_text());assert s['status']=='ALLOCATION_COMPLETE' and s['completed_updates']==5 and s['fresh_initialization']
assert s['terminal_bootstrap_samples']>=32 and s['onnx_max_abs_error']<2e-5 and max(s['training_metrics']['oracle_max_abs'])<2e-6
# Full-CAD N16384 already fits capacity; 48*24 exceeds this skill's20s natural timeout.
run(['scripts/train_v2_ground_pick.py','--task',task,'--output',str(root/'groundpick-actual16384-001'),'--device','cuda:0','--num-envs','16384','--iterations','48','--validation-only'],'groundpick-actual16384-001.log','actual-N-48')
r=json.loads((root/'groundpick-actual16384-001/report.json').read_text());assert r['status']=='ALLOCATION_COMPLETE' and r['completed_updates']==48 and r['fresh_initialization']
assert r['terminal_bootstrap_samples']>=16384 and r['contact_peak']<r['contact_capacity']
assert r['training_metrics']['metrics']['nonfinite_fraction']==0 and max(r['training_metrics']['oracle_max_abs'])<2e-6
assert r['throughput']['median_control_per_second']>=100000,r['throughput']
qualification=root/'groundpick-qualification-001.json';qualification.write_text(json.dumps(dict(status='PASS',task=task,num_envs=16384,source_sha256=r['source_sha256'],smoke=str(smoke),actual_N=str(root/'groundpick-actual16384-001/report.json'),terminal=str(root/'groundpick-terminal-validation-001.json'),physical_reach=str(root/'groundpick-reach-hold-002/report.json'),throughput=r['throughput'],formal_stage_max_updates=256,formal_stage_max_controls=100663296,first_allocation_updates=128,first_allocation_controls=50331648),indent=2)+'\n')
out=root/'groundpick-allocation-001'
wall=run(['scripts/train_v2_ground_pick.py','--task',task,'--output',str(out),'--device','cuda:0','--num-envs','16384','--iterations','128','--qualification',str(qualification)],'groundpick-allocation-001.log','formal-fresh-128')
r=json.loads((out/'report.json').read_text());assert r['completed_updates']==128 and r['controls']==50331648 and r['fresh_initialization']
(out/'outer-process-wall.json').write_text(json.dumps(dict(wall_seconds_including_imports=wall,controls=r['allocation_controls'],controls_per_second=r['allocation_controls']/wall),indent=2)+'\n')
for stage in ('nominal','measured'):
 run(['scripts/evaluate_v2_ground_pick.py','--checkpoint',str(out/'checkpoint_000128.pt'),'--output',str(root/('groundpick-0128-'+stage)),'--device','cuda:0','--stage',stage],'groundpick-0128-'+stage+'.log','independent-'+stage)
write(dict(status='EVALUATION_COMPLETE_DECISION_REQUIRED',completed_updates=128,total_groundpick_formal_controls=50331648,stage_max_controls=100663296,updated_unix=time.time(),scope='No automatic second allocation; evaluate full-cycle pickup/return/body/head-camera and identify bounded next decision'))
