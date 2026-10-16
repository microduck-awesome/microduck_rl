import json,os,subprocess,time,signal
from pathlib import Path
root=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');work=root/'sitstand-repaired-worktree'
env=dict(os.environ,SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat')
for name,pid in [('sitstand-training-smoke-cap64',3674077),('sitstand-physical-feasibility-cap64',3674018)]:
 while not (root/name/'report.json').exists():
  if not Path('/proc/'+str(pid)).exists():raise RuntimeError(name+' ended without PASS')
  time.sleep(5)
 log=(root/(name+'.log')).read_text().lower();assert 'ccd overflow' not in log and 'contact overflow' not in log
s=json.loads((root/'sitstand-training-smoke-cap64/report.json').read_text());assert s['status']=='ALLOCATION_COMPLETE'
assert s['terminal_bootstrap_samples']>=32 and s['onnx_max_abs_error']<2e-5
subprocess.run(['/home/l40-vm01/data2/experiments/microduck-v2-20261015/venv-p04/bin/python',str(root/'compare-sitstand-capacity.py')],env=env,check=True)
print('Capacity and 64x5 verified; entering actual N=32768',flush=True)
logpath=root/'sitstand-actual32768-002.log';cmd=['scripts/v2_python.sh','scripts/train_v2_sitstand.py','--output',str(root/'sitstand-actual32768-002'),'--device','cuda:0','--num-envs','32768','--iterations','32','--validation-only']
with logpath.open('w') as log:
 child=subprocess.Popen(cmd,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT);print('actual-N PID',child.pid,flush=True)
 rejected=False
 while child.poll() is None:
  content=logpath.read_text().lower()
  if not rejected and any(t in content for t in ('ccd overflow','contact overflow','constraint buffer full','broadphase overflow','narrowphase overflow','height field collision overflow','nefc overflow','njmax_nnz overflow')):
   child.send_signal(signal.SIGTERM);rejected=True
  time.sleep(2)
 assert child.returncode==0 and not rejected
r=json.loads((root/'sitstand-actual32768-002/report.json').read_text())
assert r['status']=='ALLOCATION_COMPLETE' and r['completed_updates']==32 and r['terminal_bootstrap_samples']>0
assert r['contact_peak']<r['contact_capacity'] and r['training_metrics']['metrics']['nonfinite_fraction']==0
assert max(r['training_metrics']['oracle_max_abs'])<2e-6
assert r['throughput']['median_control_per_second']>=100000,r['throughput']
q=dict(status='PASS',task='Mjlab-SitStand-Flat-MicroDuck-V2-Repaired-S0-O0-R1',num_envs=32768,source_sha256=r['source_sha256'],smoke=str(root/'sitstand-training-smoke-cap64/report.json'),actual_N=str(root/'sitstand-actual32768-002/report.json'),throughput=r['throughput'],terminal_bootstrap_samples=r['terminal_bootstrap_samples'],capacity_comparison=str(root/'sitstand-capacity-comparison.json'),measured_policy_qualification=False)
(root/'sitstand-training-qualification-002.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q),flush=True)
