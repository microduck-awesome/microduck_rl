import json,os,subprocess,time
from pathlib import Path
root=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04')
work=root/'sitstand-repaired-worktree'
env=dict(os.environ,SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat')
smoke=root/'sitstand-training-smoke-001/report.json'
while not smoke.exists():
 if not Path('/proc/3642489').exists():raise RuntimeError('64x5 stopped without report; inspect retained log')
 time.sleep(5)
s=json.loads(smoke.read_text());assert s['status']=='ALLOCATION_COMPLETE' and s['completed_updates']==5
assert s['terminal_bootstrap_samples']>=32 and s['onnx_max_abs_error']<2e-5
print('64x5 verified; entering actual N=32768, T=24, 32 updates',flush=True)
cmd=['scripts/v2_python.sh','scripts/train_v2_sitstand.py','--output',str(root/'sitstand-actual32768-001'),'--device','cuda:0','--num-envs','32768','--iterations','32','--validation-only']
with (root/'sitstand-actual32768-001.log').open('w') as log:
 child=subprocess.Popen(cmd,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT)
 print('actual-N PID',child.pid,flush=True);assert child.wait()==0
r=json.loads((root/'sitstand-actual32768-001/report.json').read_text())
assert r['status']=='ALLOCATION_COMPLETE' and r['completed_updates']==32 and r['terminal_bootstrap_samples']>0
assert r['contact_peak']<r['contact_capacity']
assert r['training_metrics']['metrics']['nonfinite_fraction']==0
assert max(r['training_metrics']['oracle_max_abs'])<2e-6
q=dict(status='PASS',task=r.get('task','Mjlab-SitStand-Flat-MicroDuck-V2-Repaired-S0-O0-R1'),num_envs=32768,source_sha256=r['source_sha256'],smoke=str(smoke),actual_N=str(root/'sitstand-actual32768-001/report.json'),throughput=r['throughput'],terminal_bootstrap_samples=r['terminal_bootstrap_samples'],measured_policy_qualification=False)
(root/'sitstand-training-qualification-001.json').write_text(json.dumps(q,indent=2)+'\n')
print(json.dumps(q),flush=True)
