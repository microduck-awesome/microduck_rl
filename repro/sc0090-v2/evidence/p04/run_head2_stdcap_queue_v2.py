from pathlib import Path
import os,subprocess,json,time,hashlib,tarfile
root=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04/rise-head-stdcap-worktree');base=root/'runs-v2'
os.chdir(root);env=os.environ.copy();env['SC0090_CUDA_COMPAT_DIR']='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat'
task='Mjlab-StandUp-Flat-MicroDuck-V2-Fresh-M4-HeadStdCap001-Repaired-S0-O0-R1'
parent=base/'head2-late-entropy-allocation-001/resume_001728.pt'
assert json.loads((base/'head2-stdcap-unit-tests.json').read_text())['status']=='PASS'
sources=[*root.glob('src/**/*.py'),root/'scripts/train_v2_standup.py',root/'scripts/smoke_v2_ppo.py',root/'tests/test_v2_head_std_cap.py']
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
(base/'head2-stdcap-source.json').write_text(json.dumps(dict(worktree=str(root),task=task,source_sha256=hashes,parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest()),indent=2)+'\n')
def run(label,args):
 started=time.time();log=base/(label+'.log')
 with log.open('x') as f:
  p=subprocess.Popen(['scripts/v2_python.sh',*map(str,args)],env=env,stdout=f,stderr=subprocess.STDOUT)
  (base/(label+'-process.json')).write_text(json.dumps(dict(pid=p.pid,started_unix=started,argv=args),indent=2,default=str))
  print(label,p.pid,flush=True);code=p.wait()
 end=time.time();(base/(label+'-wall.json')).write_text(json.dumps(dict(returncode=code,started_unix=started,ended_unix=end,wall_s=end-started),indent=2)+'\n')
 if code:raise RuntimeError((label,code))
common=['--task',task,'--device','cuda:0','--head-precision','--head-precision-stage','2','--freeze-normalizers']
for n,it,label in [(64,5,'head2-stdcap-smoke-v2-64'),(32768,16,'head2-stdcap-bench-v2-32768')]:
 run(label,['scripts/smoke_v2_ppo.py',*common,'--mature-resume',str(parent),'--compiled-measurement','--fused-registers','--reset-graph-chunk','19','--num-envs',str(n),'--iterations',str(it),'--output-dir',str(base/label),*(['--force-timeouts'] if n==64 else [])])
 report=json.loads((base/label/'report.json').read_text());assert report['status']=='PASS' and report['first_minibatch_distribution_consistent_all_updates']
 if n==32768:assert report['throughput']['median_end_to_end_steps_per_s']>100000
assert hashes=={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
(base/'head2-stdcap-qualification.json').write_text(json.dumps(dict(status='PASS',source_sha256=hashes,smoke='head2-stdcap-smoke-v2-64/report.json',throughput='head2-stdcap-bench-v2-32768/report.json',unit='head2-stdcap-unit-tests.json'),indent=2)+'\n')
label='head2-stdcap-allocation-001'
run(label,['scripts/train_v2_standup.py',*common,'--adapt-from',str(parent),'--num-envs','32768','--iterations','64','--eval-interval','64','--eval-samples','128','--preflight',str(base/'head2-stdcap-bench-v2-32768/report.json'),'--output',str(base/label)])
checkpoint=base/label/'checkpoint_001792.pt'
run('head2-stdcap-1792-head-measured',['scripts/evaluate_v2_head.py','--task',task,'--checkpoint',str(checkpoint),'--device','cuda:0','--repeats','8','--randomization-stage','measured','--output',str(base/'head2-stdcap-1792-head-measured')])
run('head2-stdcap-1792-head-summary',['/home/l40-vm01/data2/experiments/microduck-v2-20261015/baseline/head-foundation-worktree/scripts/summarize_v2_head_saved.py','--reports',str(base/'head2-stdcap-1792-head-measured/report.json'),'--device','cpu','--output',str(base/'head2-stdcap-1792-head-summary')])
run('head2-stdcap-1792-measured',['scripts/evaluate_v2_fresh_sweep.py','--checkpoint',str(checkpoint),'--device','cuda:0','--samples','128','--stages','10','--randomization-stage','measured','--output',str(base/'head2-stdcap-1792-measured')])
print('ALL_COMPLETE',flush=True)
