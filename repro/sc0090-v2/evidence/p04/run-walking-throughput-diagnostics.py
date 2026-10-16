"""Bounded fresh walking qualification; stop at the first adequate environment count."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04')
SOURCE=ROOT/'walking-repaired-prep-worktree'
TASK='Mjlab-Velocity-Flat-MicroDuck-V2-Repaired-S0-O0-R1'
PLAN=ROOT/'walking-throughput-diagnostics-plan.json'
assert not PLAN.exists()
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(value):
    temp=PLAN.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(PLAN)
smoke=ROOT/'walking-repaired-smoke-diag-64/report.json'
# Wait for the exact prior validation process to release its context, not just
# for a report written before env.close().
def matching_smoke_processes():
    found=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            argv=(proc/'cmdline').read_bytes().split(b'\0')
            status=(proc/'stat').read_text().rsplit(')',1)[1].split()[0]
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if b'scripts/smoke_v2_ppo.py' in argv and str(smoke.parent).encode() in argv and status!='Z':
            found.append(int(proc.name))
    return found
deadline=time.monotonic()+1800
while matching_smoke_processes():
    if time.monotonic()>deadline:raise TimeoutError('Diagnostic smoke did not finish')
    time.sleep(2)

assert read(smoke)['status']=='PASS' and read(smoke)['terminal_bootstrap_samples']>=32
hashes={str(p.relative_to(SOURCE)):sha(p) for p in [*SOURCE.glob('src/**/*.py'),SOURCE/'scripts/smoke_v2_ppo.py',SOURCE/'scripts/v2_python.sh']}
plan=dict(status='RUNNING',created_unix=time.time(),task=TASK,source=str(SOURCE),device='cuda:1',
    environment_counts=[16384,32768],updates_per_candidate=16,steps_per_env=24,
    purpose='Previous8192 completed at69893 controls/s and lacked optimizer diagnostics. With the fresh diagnostic wiring fixed, test16384 then32768, stop at the first qualifying N. Preserve the old attempt; no repeated8192 speed-only run.',
    scope='Fresh initialized validation only; no inherited standing normalizers, no formal walking policy or learned-skill claim.',
    optimizer_admission=dict(max_update_mean_kl=.5,max_clip_fraction=.8,last5_median_kl=.1,last5_median_clip_fraction=.5),
    source_sha256=hashes,smoke_report_sha256=sha(smoke),phases=[],selected_num_envs=None,
    next_gate='Define and verify physical command/stride/contact/posture/head evaluation and a bounded formal budget before long walking training.')
save(plan)
env=dict(os.environ,SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat')
try:
    for n in plan['environment_counts']:
        for name,value in hashes.items():assert sha(SOURCE/name)==value,name
        out=ROOT/f'walking-throughput-diag-{n}';assert not out.exists()
        args=[str(SOURCE/'scripts/v2_python.sh'),'scripts/smoke_v2_ppo.py','--task',TASK,'--device','cuda:1',
            '--num-envs',str(n),'--iterations','16','--compiled-measurement','--fused-registers',
            '--output-dir',str(out),'--report',str(out/'report.json')]
        phase=dict(num_envs=n,arguments=args,start_unix=time.time());plan['phases'].append(phase)
        with (ROOT/f'walking-throughput-diag-{n}.log').open('w') as log:
            process=subprocess.Popen(args,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT)
            phase['pid']=process.pid;save(plan);code=process.wait()
        phase.update(returncode=code,wall_seconds=time.time()-phase['start_unix']);save(plan)
        if code:raise RuntimeError(f'N={n} smoke failed; inspect before further work')
        report=read(out/'report.json');assert report['status']=='PASS' and report['steps_per_env']==24
        rows=[json.loads(line)['metrics']for line in (out/'ppo-diagnostics.jsonl').read_text().splitlines()]
        assert len(rows)==16
        kl=[r['PPO/kl_mean']for r in rows];clip=[r['PPO/clip_fraction']for r in rows]
        quality=dict(max_update_mean_kl=max(kl),max_clip_fraction=max(clip),
            last5_median_kl=statistics.median(kl[-5:]),last5_median_clip_fraction=statistics.median(clip[-5:]))
        phase.update(quality=quality,throughput=report['throughput'],report_sha256=sha(out/'report.json'))
        phase['optimizer_pass']=all(quality[k]<=limit for k,limit in plan['optimizer_admission'].items())
        phase['speed_pass']=report['throughput']['median_end_to_end_steps_per_s']>100000
        save(plan)
        if not phase['optimizer_pass']:
            plan['status']='OPTIMIZER_ADMISSION_FAILED';save(plan);break
        if phase['speed_pass']:
            plan.update(status='SELECTED_N_VALIDATION_ONLY',selected_num_envs=n);save(plan);break
    else:
        plan['status']='NO_N_PASSED_THROUGHPUT';save(plan)
except Exception as error:
    plan.update(status='FAILED',error=repr(error));save(plan);raise
