import os,sys,subprocess,time,json
from pathlib import Path
b=Path(__file__).resolve().parent
root=b/'rise-head-micro-exposure-worktree';os.chdir(root)
env=os.environ.copy();env['SC0090_CUDA_COMPAT_DIR']='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat'
status=b/'head2-micro1856-legs-only-queue-status.json'
started=time.time()
common=[str(b/'audit_fresh_p04_legs_noise_v3.py'),'--task','Mjlab-StandUp-Flat-MicroDuck-V2-Fresh-M4-HeadStdCap001-MicroExposure025-Repaired-S0-O0-R1','--checkpoint',str(b/'head2-micro-exposure-allocation-001/checkpoint_001856.pt'),'--device','cuda:0','--repeats','8','--randomization-stage','nominal','--reward-stage','2','--reward-reference-run',str(b/'head2-micro-exposure-allocation-001/run.json'),'--noise-mode','legs']
def run(name,args):
    with (b/(name+'.log')).open('w') as log:
        proc=subprocess.Popen(['scripts/v2_python.sh',*args],env=env,stdout=log,stderr=subprocess.STDOUT)
        status.write_text(json.dumps(dict(status='RUNNING',stage=name,pid=proc.pid,queue_pid=os.getpid(),started=started),indent=2))
        print(name,proc.pid,flush=True)
        rc=proc.wait()
        if rc:raise SystemExit(str(name)+' failed '+str(rc))
mean='head2-micro1856-fullreward-mean'
legs='head2-micro1856-fullreward-legs-gaussian'
run(mean,[*common,'--output',str(b/mean)])
run(legs,[*common,'--sampled-actions','--initial-state-reference',str(b/mean),'--output',str(b/legs)])
for name in (mean,legs):
    run(name+'-summary',['/home/l40-vm01/data2/experiments/microduck-v2-20261015/baseline/head-foundation-worktree/scripts/summarize_v2_head_saved.py','--reports',str(b/name/'report.json'),'--device','cpu','--output',str(b/(name+'-summary'))])
status.write_text(json.dumps(dict(status='COMPLETE',started=started,finished=time.time(),wall_seconds=time.time()-started),indent=2))
print('COMPLETE',flush=True)
