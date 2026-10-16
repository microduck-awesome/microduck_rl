import os,json,subprocess,time,signal
from pathlib import Path
r=Path(__file__).resolve().parent;w=r/'roulade-repaired-prep-worktree';env=dict(os.environ,SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat');env.pop('MICRODUCK_WARM_START',None)
patterns=('ccd overflow','contact overflow','constraint buffer full','broadphase overflow','narrowphase overflow','height field collision overflow','nefc overflow','njmax_nnz overflow')
def write(**kw):
 p=r/'roulade-pipeline-status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(updated_unix=time.time(),**kw),indent=2)+'\n');t.replace(p)
def run(args,name):
 log=r/(name+'.log');start=time.monotonic()
 with log.open('w') as f:
  p=subprocess.Popen(['scripts/v2_python.sh',*args],cwd=w,env=env,stdout=f,stderr=subprocess.STDOUT);write(status='VALIDATING',phase=name,pid=p.pid,log=str(log));print(name,p.pid,flush=True);bad=False
  while p.poll() is None:
   if not bad and any(x in log.read_text().lower() for x in patterns):bad=True;p.send_signal(signal.SIGTERM)
   time.sleep(2)
  assert p.returncode==0 and not bad,(name,p.returncode,bad)
 report=json.loads((r/name/'report.json').read_text());(r/name/'outer-process-wall.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-start,allocation_kind='validation'),indent=2)+'\n');return report
p=r/'roulade-physical-004/report.json';write(status='WAITING_PHYSICAL_REPORT',phase='roulade-physical-004',pid=4045441)
while not p.exists():
 if not Path('/proc/4045441').exists():raise RuntimeError('Physical004 ended without report; refusing smoke')
 time.sleep(2)
a=json.loads(p.read_text());assert all(x['full3s_nonfinite_trials']==0 and x['full3s_fault_trials']==0 for x in a['results']);assert a['results'][0]['healthy_standing_last1s']==16;assert max(a['reward_state_oracle_max_abs'])<2e-5
smoke=run(['scripts/train_v2_roulade.py','--output',str(r/'roulade-smoke-001'),'--device','cuda:0','--num-envs','64','--iterations','5','--validation-only','--force-timeouts'],'roulade-smoke-001')
assert smoke['completed_updates']==5 and smoke['terminal_bootstrap_samples']>=32 and smoke['terminal_validation']['status']=='PASS' and smoke['onnx_max_abs_error']<2e-5
actual=run(['scripts/train_v2_roulade.py','--output',str(r/'roulade-actual16384-001'),'--device','cuda:0','--num-envs','16384','--iterations','16','--validation-only'],'roulade-actual16384-001')
assert actual['completed_updates']==16 and actual['terminal_bootstrap_samples']>=16384
write(status='VALIDATION_RESULTS_REQUIRE_REVIEW',phase='actual16384_complete',source=actual['source_sha256'],throughput=actual['throughput'],formal_updates=0,scope='No formal launch before source qualification, fixed budget and frozen evaluation readiness')
