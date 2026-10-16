from pathlib import Path
import os,subprocess,json,time,hashlib
root=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04/rise-head-micro-exposure-worktree');base=root/'runs-v2'
os.chdir(root);env=os.environ.copy();env['SC0090_CUDA_COMPAT_DIR']='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat'
task='Mjlab-StandUp-Flat-MicroDuck-V2-Fresh-M4-HeadStdCap001-MicroExposure025-Repaired-S0-O0-R1'
parent=base/'head2-micro-exposure-allocation-001/resume_001856.pt'
qualification=json.loads((base/'head2-micro-exposure-qualification.json').read_text());assert qualification['status']=='PASS'
for name,digest in qualification['source_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
def run(label,args):
 started=time.time();log=base/(label+'.log')
 with log.open('x') as f:
  p=subprocess.Popen(['scripts/v2_python.sh',*map(str,args)],env=env,stdout=f,stderr=subprocess.STDOUT)
  (base/(label+'-process.json')).write_text(json.dumps(dict(pid=p.pid,started_unix=started,argv=args),indent=2,default=str))
  print(label,p.pid,flush=True);code=p.wait()
 end=time.time();(base/(label+'-wall.json')).write_text(json.dumps(dict(returncode=code,started_unix=started,ended_unix=end,wall_s=end-started),indent=2)+'\n')
 if code:raise RuntimeError((label,code))

common=['--task',task,'--device','cuda:0','--head-precision','--head-precision-stage','2','--freeze-normalizers']
label='head2-micro-control-allocation-001'
run(label,['scripts/train_v2_standup.py',*common,'--resume',str(parent),'--num-envs','32768','--iterations','64','--eval-interval','64','--eval-samples','128','--preflight',str(base/'head2-micro-exposure-bench-32768/report.json'),'--output',str(base/label)])
checkpoint=base/label/'checkpoint_001920.pt'
run('head2-micro-control-1920-head-measured',['scripts/evaluate_v2_head.py','--task',task,'--checkpoint',str(checkpoint),'--device','cuda:0','--repeats','8','--randomization-stage','measured','--output',str(base/'head2-micro-control-1920-head-measured')])
run('head2-micro-control-1920-head-summary',['/home/l40-vm01/data2/experiments/microduck-v2-20261015/baseline/head-foundation-worktree/scripts/summarize_v2_head_saved.py','--reports',str(base/'head2-micro-control-1920-head-measured/report.json'),'--device','cpu','--output',str(base/'head2-micro-control-1920-head-summary')])
run('head2-micro-control-1920-measured',['scripts/evaluate_v2_fresh_sweep.py','--checkpoint',str(checkpoint),'--device','cuda:0','--samples','128','--stages','10','--randomization-stage','measured','--output',str(base/'head2-micro-control-1920-measured')])
print('ALL_COMPLETE',flush=True)
