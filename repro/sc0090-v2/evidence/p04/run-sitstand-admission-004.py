import json,time
from pathlib import Path
root=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');out=root/'sitstand-actual16384-001'
while not (out/'report.json').exists():
 if not Path('/proc/3689654').exists():raise RuntimeError('Actual16384 ended without complete report')
 time.sleep(2)
r=json.loads((out/'report.json').read_text());assert r['status']=='ALLOCATION_COMPLETE' and r['completed_updates']==32
assert r['terminal_bootstrap_samples']>0 and r['contact_peak']<r['contact_capacity']
assert r['training_metrics']['metrics']['nonfinite_fraction']==0
assert max(r['training_metrics']['oracle_max_abs'])<2e-6
patterns=('ccd overflow','contact overflow','constraint buffer full','broadphase overflow','narrowphase overflow','height field collision overflow','nefc overflow','njmax_nnz overflow')
log=(root/'sitstand-actual16384-001.log').read_text().lower();assert not any(p in log for p in patterns)
assert r['throughput']['median_control_per_second']>=100000,r['throughput']
q=dict(status='PASS',task='Mjlab-SitStand-Flat-MicroDuck-V2-Repaired-S0-O0-R1',num_envs=16384,source_sha256=r['source_sha256'],smoke=str(root/'sitstand-training-smoke-cap64/report.json'),actual_N=str(out/'report.json'),throughput=r['throughput'],terminal_bootstrap_samples=r['terminal_bootstrap_samples'],capacity_comparison=str(root/'sitstand-capacity-comparison.json'),log_guard_patterns=patterns,measured_policy_qualification=False)
(root/'sitstand-training-qualification-003.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q),flush=True)
