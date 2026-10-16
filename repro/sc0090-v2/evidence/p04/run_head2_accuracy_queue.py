from pathlib import Path
import os,subprocess,json,time,hashlib,tarfile,statistics
root=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04/rise-head-accuracy-worktree');base=root/'runs-v2'
os.chdir(root);env=os.environ.copy();env['SC0090_CUDA_COMPAT_DIR']='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat'
task='Mjlab-StandUp-Flat-MicroDuck-V2-Fresh-M4-HeadStdCap001-MicroExposure025-AccuracyBonus05-Repaired-S0-O0-R1'
parent=base/'head2-micro-exposure-allocation-001/resume_001856.pt'
for attempt in range(150):
 if (base/'head2-accuracy-unit-tests.json').exists():break
 if 'Traceback' in (base/'head2-accuracy-unit-tests.log').read_text():raise RuntimeError('CPU/GPU sampling test failed')
 time.sleep(1)
assert json.loads((base/'head2-accuracy-unit-tests.json').read_text())['status']=='PASS'
sources=[*root.glob('src/**/*.py'),root/'scripts/train_v2_standup.py',root/'scripts/smoke_v2_ppo.py',root/'tests/test_v2_head_std_cap.py',root/'tests/test_v2_head_accuracy.py']
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
(base/'head2-accuracy-source.json').write_text(json.dumps(dict(worktree=str(root),task=task,source_sha256=hashes,parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest()),indent=2)+'\n')
def run(label,args):
 started=time.time();log=base/(label+'.log')
 with log.open('x') as f:
  p=subprocess.Popen(['scripts/v2_python.sh',*map(str,args)],env=env,stdout=f,stderr=subprocess.STDOUT)
  (base/(label+'-process.json')).write_text(json.dumps(dict(pid=p.pid,started_unix=started,argv=args),indent=2,default=str))
  print(label,p.pid,flush=True);code=p.wait()
 end=time.time();(base/(label+'-wall.json')).write_text(json.dumps(dict(returncode=code,started_unix=started,ended_unix=end,wall_s=end-started),indent=2)+'\n')
 if code:raise RuntimeError((label,code))
common=['--task',task,'--device','cuda:0','--head-precision','--head-precision-stage','2','--freeze-normalizers']
for n,it,label in [(64,5,'head2-accuracy-smoke-64'),(32768,16,'head2-accuracy-bench-32768')]:
 run(label,['scripts/smoke_v2_ppo.py',*common,'--mature-resume',str(parent),'--compiled-measurement','--fused-registers','--reset-graph-chunk','19','--num-envs',str(n),'--iterations',str(it),'--output-dir',str(base/label),*(['--force-timeouts'] if n==64 else [])])
 report=json.loads((base/label/'report.json').read_text());assert report['status']=='PASS' and report['first_minibatch_distribution_consistent_all_updates']
 if n==32768:assert report['throughput']['median_end_to_end_steps_per_s']>100000
metrics=[json.loads(line)['metrics'] for line in (base/'head2-accuracy-bench-32768/ppo-diagnostics.jsonl').read_text().splitlines()]
assert len(metrics)==16
limits=dict(max_update_kl_mean=.5,max_update_clip_fraction=.8,last5_median_kl_mean=.1,last5_median_clip_fraction=.5)
observed=dict(max_update_kl_mean=max(m['PPO/kl_mean'] for m in metrics),max_update_clip_fraction=max(m['PPO/clip_fraction'] for m in metrics),last5_median_kl_mean=statistics.median(m['PPO/kl_mean'] for m in metrics[-5:]),last5_median_clip_fraction=statistics.median(m['PPO/clip_fraction'] for m in metrics[-5:]))
passed=all(observed[k]<=v for k,v in limits.items())
(base/'head2-accuracy-optimizer-admission.json').write_text(json.dumps(dict(status='PASS' if passed else 'FAIL',scope='actual32768 mature sameparent 16update preliminary stability guard, not learned behavior',limits=limits,observed=observed,updates=metrics),indent=2)+'\n')
assert passed, ('Actual-N optimizer too aggressive; formal segment blocked',observed)
assert hashes=={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
(base/'head2-accuracy-qualification.json').write_text(json.dumps(dict(status='PASS',source_sha256=hashes,smoke='head2-accuracy-smoke-64/report.json',throughput='head2-accuracy-bench-32768/report.json',unit='head2-accuracy-unit-tests.json'),indent=2)+'\n')
label='head2-accuracy-allocation-001'
run(label,['scripts/train_v2_standup.py',*common,'--adapt-from',str(parent),'--num-envs','32768','--iterations','64','--eval-interval','64','--eval-samples','128','--preflight',str(base/'head2-accuracy-bench-32768/report.json'),'--output',str(base/label)])
checkpoint=base/label/'checkpoint_001920.pt'
run('head2-accuracy-1920-head-measured',['scripts/evaluate_v2_head.py','--task',task,'--checkpoint',str(checkpoint),'--device','cuda:0','--repeats','8','--randomization-stage','measured','--output',str(base/'head2-accuracy-1920-head-measured')])
run('head2-accuracy-1920-head-summary',['/home/l40-vm01/data2/experiments/microduck-v2-20261015/baseline/head-foundation-worktree/scripts/summarize_v2_head_saved.py','--reports',str(base/'head2-accuracy-1920-head-measured/report.json'),'--device','cpu','--output',str(base/'head2-accuracy-1920-head-summary')])
run('head2-accuracy-1920-measured',['scripts/evaluate_v2_fresh_sweep.py','--checkpoint',str(checkpoint),'--device','cuda:0','--samples','128','--stages','10','--randomization-stage','measured','--output',str(base/'head2-accuracy-1920-measured')])
print('ALL_COMPLETE',flush=True)
