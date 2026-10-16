from pathlib import Path
import os,subprocess,json,time,hashlib,tarfile,statistics
root=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04/rise-head-cap005-worktree');base=root/'runs-v2'
os.chdir(root);env=os.environ.copy();env['SC0090_CUDA_COMPAT_DIR']='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat'
task='Mjlab-StandUp-Flat-MicroDuck-V2-Fresh-M4-HeadStdCap005-MicroExposure025-Repaired-S0-O0-R1'
parent=base/'head2-micro-exposure-allocation-001/resume_001856.pt'
gate=json.loads((base/'head2-cap005-rng-change.json').read_text())
assert gate['status']=='PASS' and gate['proof']['legacy_same_device_all_streams_bitwise']
for name,digest in gate['sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
while not ((base/'head2-micro-control-1920-head-summary/report.json').exists() and (base/'head2-micro-control-1920-measured/report.json').exists() and (base/'head2-micro-control-1920-measured/stage-10/standing-order.json').exists()):
 for label in ['head2-micro-control-allocation-001','head2-micro-control-1920-head-measured','head2-micro-control-1920-head-summary','head2-micro-control-1920-measured']:
  wall=base/(label+'-wall.json')
  if wall.exists():assert json.loads(wall.read_text())['returncode']==0, 'Control failed; inspect before next arm: '+label
 time.sleep(2)
print('CONTROL_COMPLETE_STARTING_CAP005_QUALIFICATION',flush=True)
sources=[*root.glob('src/**/*.py'),root/'scripts/train_v2_standup.py',root/'scripts/smoke_v2_ppo.py',root/'tests/test_v2_head_std_cap.py',root/'tests/test_v2_head_cap005.py']
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
(base/'head2-cap005-source.json').write_text(json.dumps(dict(worktree=str(root),task=task,source_sha256=hashes,parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest()),indent=2)+'\n')
def run(label,args):
 started=time.time();log=base/(label+'.log')
 with log.open('x') as f:
  p=subprocess.Popen(['scripts/v2_python.sh',*map(str,args)],env=env,stdout=f,stderr=subprocess.STDOUT)
  (base/(label+'-process.json')).write_text(json.dumps(dict(pid=p.pid,started_unix=started,argv=args),indent=2,default=str))
  print(label,p.pid,flush=True);code=p.wait()
 end=time.time();(base/(label+'-wall.json')).write_text(json.dumps(dict(returncode=code,started_unix=started,ended_unix=end,wall_s=end-started),indent=2)+'\n')
 if code:raise RuntimeError((label,code))
run('head2-cap005-unit-tests',['tests/test_v2_head_cap005.py','--devices','cpu','cuda:0','--report',str(base/'head2-cap005-unit-tests.json')])
assert json.loads((base/'head2-cap005-unit-tests.json').read_text())['status']=='PASS'
common=['--task',task,'--device','cuda:0','--head-precision','--head-precision-stage','2','--freeze-normalizers']
for n,it,label in [(64,5,'head2-cap005-smoke-64'),(32768,16,'head2-cap005-bench-32768')]:
 run(label,['scripts/smoke_v2_ppo.py',*common,'--mature-resume',str(parent),'--compiled-measurement','--fused-registers','--reset-graph-chunk','19','--num-envs',str(n),'--iterations',str(it),'--output-dir',str(base/label),*(['--force-timeouts'] if n==64 else [])])
 report=json.loads((base/label/'report.json').read_text());assert report['status']=='PASS' and report['first_minibatch_distribution_consistent_all_updates']
 if n==32768:assert report['throughput']['median_end_to_end_steps_per_s']>100000
metrics=[json.loads(line)['metrics'] for line in (base/'head2-cap005-bench-32768/ppo-diagnostics.jsonl').read_text().splitlines()]
assert len(metrics)==16
limits=dict(max_update_kl_mean=.5,max_update_clip_fraction=.8,last5_median_kl_mean=.1,last5_median_clip_fraction=.5)
observed=dict(max_update_kl_mean=max(m['PPO/kl_mean'] for m in metrics),max_update_clip_fraction=max(m['PPO/clip_fraction'] for m in metrics),last5_median_kl_mean=statistics.median(m['PPO/kl_mean'] for m in metrics[-5:]),last5_median_clip_fraction=statistics.median(m['PPO/clip_fraction'] for m in metrics[-5:]))
passed=all(observed[k]<=v for k,v in limits.items())
(base/'head2-cap005-optimizer-admission.json').write_text(json.dumps(dict(status='PASS' if passed else 'FAIL',scope='actual32768 mature sameparent 16update preliminary stability guard, not learned behavior',limits=limits,observed=observed,updates=metrics),indent=2)+'\n')
assert passed, ('Actual-N optimizer too aggressive; formal segment blocked',observed)
assert hashes=={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
(base/'head2-cap005-qualification.json').write_text(json.dumps(dict(status='PASS',source_sha256=hashes,smoke='head2-cap005-smoke-64/report.json',throughput='head2-cap005-bench-32768/report.json',unit='head2-cap005-unit-tests.json'),indent=2)+'\n')
label='head2-cap005-allocation-001'
run(label,['scripts/train_v2_standup.py',*common,'--adapt-from',str(parent),'--num-envs','32768','--iterations','64','--eval-interval','64','--eval-samples','128','--preflight',str(base/'head2-cap005-bench-32768/report.json'),'--output',str(base/label)])
checkpoint=base/label/'checkpoint_001920.pt'
run('head2-cap005-1920-head-measured',['scripts/evaluate_v2_head.py','--task',task,'--checkpoint',str(checkpoint),'--device','cuda:0','--repeats','8','--randomization-stage','measured','--output',str(base/'head2-cap005-1920-head-measured')])
run('head2-cap005-1920-head-summary',['/home/l40-vm01/data2/experiments/microduck-v2-20261015/baseline/head-foundation-worktree/scripts/summarize_v2_head_saved.py','--reports',str(base/'head2-cap005-1920-head-measured/report.json'),'--device','cpu','--output',str(base/'head2-cap005-1920-head-summary')])
run('head2-cap005-1920-measured',['scripts/evaluate_v2_fresh_sweep.py','--checkpoint',str(checkpoint),'--device','cuda:0','--samples','128','--stages','10','--randomization-stage','measured','--output',str(base/'head2-cap005-1920-measured')])
print('ALL_COMPLETE',flush=True)
