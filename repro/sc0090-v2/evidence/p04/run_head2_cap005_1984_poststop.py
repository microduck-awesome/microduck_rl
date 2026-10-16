from pathlib import Path
import os,json,time,subprocess
b=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');r=b/'rise-head-cap005-worktree';os.chdir(r)
env=os.environ.copy();env['SC0090_CUDA_COMPAT_DIR']='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat'
head=b/'head2-cap005-1984-head-measured';hp=3488969
while Path(f'/proc/{hp}').exists():time.sleep(2)
assert (head/'report.json').exists(),'Retained head process ended without report'
meta=json.loads((b/'head2-cap005-1984-head-measured-process.json').read_text())
(b/'head2-cap005-1984-head-measured-wall.json').write_text(json.dumps(dict(returncode=None,status='REPORT_COMPLETE_AFTER_PARENT_QUEUE_REPLAN',started_unix=meta['started_unix'],ended_unix=time.time(),wall_s=time.time()-meta['started_unix'],limitation='Original parent queue terminated after safe formal stop; exit code unavailable. Saved report and frozen-state/oracle checks used for validation.'),indent=2)+'\n')
def run(label,args):
 start=time.time()
 with (b/(label+'.log')).open('x') as f:
  p=subprocess.Popen(['scripts/v2_python.sh',*map(str,args)],env=env,stdout=f,stderr=subprocess.STDOUT);(b/(label+'-process.json')).write_text(json.dumps(dict(pid=p.pid,started_unix=start,argv=list(map(str,args))),indent=2)+'\n');print(label,p.pid,flush=True);code=p.wait()
 (b/(label+'-wall.json')).write_text(json.dumps(dict(returncode=code,started_unix=start,ended_unix=time.time(),wall_s=time.time()-start),indent=2)+'\n');assert code==0,(label,code)
run('head2-cap005-1984-head-summary',['/home/l40-vm01/data2/experiments/microduck-v2-20261015/baseline/head-foundation-worktree/scripts/summarize_v2_head_saved.py','--reports',str(head/'report.json'),'--device','cpu','--output',str(b/'head2-cap005-1984-head-summary')])
run('head2-cap005-1984-measured',['scripts/evaluate_v2_fresh_sweep.py','--checkpoint',str(b/'head2-cap005-allocation-002/checkpoint_001984.pt'),'--device','cuda:0','--samples','128','--stages','10','--randomization-stage','measured','--output',str(b/'head2-cap005-1984-measured')])
subprocess.run(['scripts/v2_python.sh','scripts/analyze_v2_standing_order.py',str(b/'head2-cap005-1984-measured/stage-10')],env=env,check=True)
print('POSTSTOP_EVALUATION_COMPLETE',flush=True)
