from pathlib import Path
import os,subprocess,json,time,hashlib
b=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');r=b/'rise-head-cap005-worktree';os.chdir(r)
env=os.environ.copy();env['SC0090_CUDA_COMPAT_DIR']='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat'
g=json.loads((b/'head2-cap005-continuation-guard.json').read_text());assert g['status']=='PASS'
assert hashlib.sha256((r/g['modified_file']).read_bytes()).hexdigest()==g['after_sha256']
for n,h in g['unchanged_compute_sources'].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
assert hashlib.sha256(Path(g['preflight']).read_bytes()).hexdigest()==g['preflight_sha256']
task='Mjlab-StandUp-Flat-MicroDuck-V2-Fresh-M4-HeadStdCap005-MicroExposure025-Repaired-S0-O0-R1'
parent=b/'head2-cap005-allocation-001/resume_001920.pt';out=b/'head2-cap005-allocation-002'
old=json.loads((parent.parent/'completed-zero-lineage.json').read_text())
lineage=dict(root_random_initialization=old.get('root_random_initialization'),parent_checkpoint=str(parent),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),selected_parent_controls=old['candidate_lineage_controls'],actual_prior_controls=old['actual_cumulative_across_fresh_experiments'],unselected_prior_controls=old['actual_cumulative_across_fresh_experiments']-old['candidate_lineage_controls'],prior_segments=old['completed_segments'],planned_updates=128,planned_controls=128*32768*24,planned_end=2048,reason='Bounded continuation of matched cap005 current/final head gains; keep micro1856, no new factor',qualification_guard=str(b/'head2-cap005-continuation-guard.json'))
(b/'head2-cap005-allocation002-zero-lineage.json').write_text(json.dumps(lineage,indent=2)+'\n')
def run(label,args):
 started=time.time()
 with (b/(label+'.log')).open('x') as f:
  p=subprocess.Popen(['scripts/v2_python.sh',*map(str,args)],env=env,stdout=f,stderr=subprocess.STDOUT)
  meta=dict(pid=p.pid,started_unix=started,argv=args);(b/(label+'-process.json')).write_text(json.dumps(meta,indent=2,default=str)+'\n');print(label,p.pid,flush=True)
  code=p.wait()
 wall=dict(returncode=code,started_unix=started,ended_unix=time.time(),wall_s=time.time()-started);(b/(label+'-wall.json')).write_text(json.dumps(wall,indent=2)+'\n')
 if code:raise RuntimeError((label,code))
 return wall
common=['--task',task,'--device','cuda:0','--head-precision','--head-precision-stage','2','--freeze-normalizers']
wall=run(out.name,['scripts/train_v2_standup.py',*common,'--resume',str(parent),'--num-envs','32768','--iterations','128','--eval-interval','64','--eval-samples','128','--preflight',g['preflight'],'--output',str(out)])
s=json.loads((out/'status.json').read_text());stop=s['completed_updates'];controls=(stop-1920)*32768*24
lineage.update(actual_stop=stop,actual_allocation_controls=controls,actual_cumulative_across_fresh_experiments=lineage['actual_prior_controls']+controls,candidate_lineage_controls=lineage['selected_parent_controls']+controls,other_experiments_or_unselected_tails=lineage['unselected_prior_controls'],completed_segments=lineage['prior_segments']+[dict(output=out.name,start=1920,actual_stop=stop,actual_controls=controls)])
(out/'completed-zero-lineage.json').write_text(json.dumps(lineage,indent=2)+'\n')
(out/'actual-wall-time.json').write_text(json.dumps(dict(**wall,actual_formal_controls=controls,complete_process_control_steps_per_s=controls/wall['wall_s'],scope='Popen through exit including initialization/PPO/formal recovery, external head excluded'),indent=2)+'\n')
for d in sorted(out.glob('evaluation_*_evidence')):subprocess.run(['scripts/v2_python.sh','scripts/analyze_v2_standing_order.py',str(d)],env=env,check=True)
for node in [1984,2048]:
 cp=out/f'checkpoint_{node:06d}.pt';label=f'head2-cap005-{node}-head-measured'
 run(label,['scripts/evaluate_v2_head.py','--task',task,'--checkpoint',str(cp),'--device','cuda:0','--repeats','8','--randomization-stage','measured','--output',str(b/label)])
 run(f'head2-cap005-{node}-head-summary',['/home/l40-vm01/data2/experiments/microduck-v2-20261015/baseline/head-foundation-worktree/scripts/summarize_v2_head_saved.py','--reports',str(b/label/'report.json'),'--device','cpu','--output',str(b/f'head2-cap005-{node}-head-summary')])
label='head2-cap005-2048-measured';run(label,['scripts/evaluate_v2_fresh_sweep.py','--checkpoint',str(out/'checkpoint_002048.pt'),'--device','cuda:0','--samples','128','--stages','10','--randomization-stage','measured','--output',str(b/label)])
subprocess.run(['scripts/v2_python.sh','scripts/analyze_v2_standing_order.py',str(b/label/'stage-10')],env=env,check=True)
print('ALL_COMPLETE',flush=True)
