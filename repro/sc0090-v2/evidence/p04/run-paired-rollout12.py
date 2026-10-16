"""Equal-control-budget mature rollout horizon experiment with measured admission."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04')
SOURCE=ROOT/'paired-rollout12-worktree'
TASK='Mjlab-StandUp-Flat-MicroDuck-V2-PairedAdapt-Catch2sPitch10-LateEntropy005-HeadStdCap001-Rollout12-Repaired-S0-O0-R1'
PARENT=ROOT/'paired-stdcap-allocation-002/resume_004146.pt'
OUTPUT=ROOT/'paired-rollout12-allocation-001'
PLAN=ROOT/'paired-rollout12-plan.json'
TIMING=ROOT/'paired-rollout12-timing.json'
HELPERS=ROOT.parent/'baseline/head-foundation-worktree/scripts'
ENV=dict(os.environ,SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat')

def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def wait(path):
    deadline=time.monotonic()+3600
    while not path.exists():
        if time.monotonic()>deadline:raise TimeoutError(path)
        time.sleep(5)

assert not OUTPUT.exists()
config=read(ROOT/'paired-rollout12-config.json');assert config['status']=='PASS'
files=[*SOURCE.glob('src/**/*.py'),SOURCE/'scripts/train_v2_standup.py',SOURCE/'scripts/smoke_v2_ppo.py',SOURCE/'scripts/v2_python.sh',SOURCE/'scripts/check_paired_rollout12.py']
source_hashes={str(path.relative_to(SOURCE)):digest(path) for path in files}
plan=dict(status='WAITING_FOR_FRONTIER_DIAGNOSTIC',created_unix=time.time(),task=TASK,source=str(SOURCE),
    parent=str(PARENT),parent_sha256=digest(PARENT),source_sha256=source_hashes,
    control='paired-stdcap-allocation-003: same 4146 parent, T24, 128 updates, 100663296 controls',
    experimental='T12, 256 updates to 4402, 100663296 controls, 32768 environments; no other behavioral parameter changed',
    config_report=str(ROOT/'paired-rollout12-config.json'),config_sha256=digest(ROOT/'paired-rollout12-config.json'),
    restore='Exact actor/critic/normalizers/Adam/LR/common-step clock, new native episodes; no LR transition, no curriculum reset',
    expected_parent_clock=99504,expected_parent_learning_rate=5.062500000000001e-5,
    optimizer_admission=dict(max_update_mean_kl=.5,max_clip_fraction=.8,last5_median_kl=.1,last5_median_clip_fraction=.5),
    decision='One bounded equal-sample ablation; compare sustained head accuracy, full recovery, negative-pitch catch and complete wall time. No automatic continuation or deployment.')
save(PLAN,plan)
wait(ROOT/'paired-cap-pitch10-frontier-4146-4274/report.json')
assert read(ROOT/'paired-cap-pitch10-frontier-4146-4274-plan.json')['status']=='COMPLETE'
timing=dict(pipeline_start_unix=time.time(),phases=[],scope='Includes this recipe preflight, formal training and post-training head/common evaluations. Frozen T24/control and frontier-diagnostic costs are separate.')

def check_source():
    for path,sha in source_hashes.items():assert digest(SOURCE/path)==sha,path

def run(label,args,log):
    phase=dict(label=label,start_unix=time.time(),arguments=list(map(str,args)))
    timing['phases'].append(phase)
    with log.open('w') as stream:
        process=subprocess.Popen([str(SOURCE/'scripts/v2_python.sh'),*map(str,args)],cwd=SOURCE,env=ENV,stdout=stream,stderr=subprocess.STDOUT)
        phase['pid']=process.pid;save(TIMING,timing);print(label,process.pid,flush=True)
        code=process.wait()
    phase.update(returncode=code,end_unix=time.time());phase['wall_seconds']=phase['end_unix']-phase['start_unix']
    if label=='formal' and (OUTPUT/'status.json').exists():
        phase['controls']=read(OUTPUT/'status.json')['allocation_controls']
        phase['complete_process_interactions_per_s']=phase['controls']/phase['wall_seconds']
        save(OUTPUT/'actual-wall-time.json',phase)
    save(TIMING,timing)
    if code:
        plan.update(status='FAILED',failed_phase=label,returncode=code);save(PLAN,plan)
        raise subprocess.CalledProcessError(code,args)

check_source()
subprocess.run(['git','rev-parse','HEAD'],cwd=SOURCE,check=True,stdout=subprocess.DEVNULL)
plan['status']='PREFLIGHT_RUNNING';save(PLAN,plan)
for label,n,iterations in [('mature-smoke',64,5),('throughput-32768',32768,16)]:
    out=ROOT/f'paired-rollout12-{label}'
    args=['scripts/smoke_v2_ppo.py','--task',TASK,'--device','cuda:1','--num-envs',n,'--iterations',iterations,
        '--compiled-measurement','--fused-registers','--head-precision','--head-precision-stage','5',
        '--mature-resume',PARENT,'--freeze-normalizers','--output-dir',out,'--report',out/'report.json']
    if n==64:args+=['--force-timeouts']
    run(label,args,ROOT/f'paired-rollout12-{label}.log')
    report=read(out/'report.json')
    assert report['status']=='PASS' and report['steps_per_env']==12
    assert report['initialization']['actor_critic_normalizers_optimizer_exact']
    assert report['initialization']['inherited_completed_updates']==4146
    assert report['initialization']['curriculum_clock']==99504
    assert report['initialization']['learning_rate']==plan['expected_parent_learning_rate']
    assert report['curriculum_clock_advance_verified'] and report['curriculum_clock_before_learning']==99504
    assert report['curriculum_clock_after_learning']==99504+iterations*12
    assert report['first_minibatch_distribution_consistent_all_updates']
    assert report['normalization_unchanged_all_updates']
    if n==64:assert report['terminal_bootstrap_samples']>0
    check_source()

out=ROOT/'paired-rollout12-throughput-32768'
rows=[json.loads(line)['metrics'] for line in (out/'ppo-diagnostics.jsonl').read_text().splitlines()]
assert len(rows)==16
kl=[r['PPO/kl_mean'] for r in rows];clip=[r['PPO/clip_fraction'] for r in rows]
quality=dict(max_update_mean_kl=max(kl),max_clip_fraction=max(clip),last5_median_kl=statistics.median(kl[-5:]),last5_median_clip_fraction=statistics.median(clip[-5:]))
quality['median_control_steps_per_s']=read(out/'report.json')['throughput']['median_end_to_end_steps_per_s']
quality['status']='PASS' if all(quality[k]<=v for k,v in plan['optimizer_admission'].items()) and quality['median_control_steps_per_s']>100000 else 'FAIL'
save(ROOT/'paired-rollout12-optimizer-admission.json',quality)
if quality['status']!='PASS':
    plan.update(status='ADMISSION_FAILED',optimizer=quality);save(PLAN,plan)
    raise RuntimeError(quality)
plan.update(status='FORMAL_RUNNING',optimizer=quality);save(PLAN,plan)
run('formal',['scripts/train_v2_standup.py','--task',TASK,'--device','cuda:1','--num-envs','32768',
    '--iterations','256','--eval-interval','128','--eval-samples','128','--head-precision','--head-precision-stage','5',
    '--freeze-normalizers','--preflight',out/'report.json','--adapt-from',PARENT,'--output',OUTPUT],
    ROOT/'paired-rollout12-allocation-001.log')
status=read(OUTPUT/'status.json')
assert status['completed_updates']==4402 and status['allocation_controls']==100663296
assert status['common_step_counter']==99504+256*12
for evidence in sorted(OUTPUT.glob('evaluation_*_evidence')):
    for script in ['analyze_v2_standing_order.py','analyze_standing_pitch_bins.py']:
        run(evidence.name+script,[SOURCE/'scripts'/script,evidence],OUTPUT/(evidence.name+'-'+script+'.log'))
for iteration in [4352,4402]:
    head=ROOT/f'paired-rollout12-{iteration}-head-measured'
    run(f'head{iteration}',['scripts/evaluate_v2_head_normalizer.py','--task',TASK,'--checkpoint',OUTPUT/f'checkpoint_{iteration:06d}.pt',
        '--output',head,'--device','cuda:1','--repeats','8','--randomization-stage','measured'],ROOT/f'paired-rollout12-{iteration}-head-measured.log')
    run(f'head-summary{iteration}',[HELPERS/'summarize_v2_head_saved.py','--device','cpu','--reports',head/'report.json',
        '--output',ROOT/f'paired-rollout12-{iteration}-head-summary'],ROOT/f'paired-rollout12-{iteration}-head-summary.log')
common=ROOT/'paired-rollout12-4402-common-measured'
run('common4402',[HELPERS/'evaluate_common_p04_candidates.py','--checkpoints',OUTPUT/'checkpoint_004402.pt','--labels','rollout12-4402',
    '--output',common,'--device','cuda:1','--samples','128'],ROOT/'paired-rollout12-4402-common-measured.log')
for script in ['analyze_v2_standing_order.py','analyze_standing_pitch_bins.py']:
    run('common-'+script,[SOURCE/'scripts'/script,common/'rollout12-4402'],common/(script+'.log'))
timing['pipeline_end_unix']=time.time();timing['pipeline_wall_seconds']=timing['pipeline_end_unix']-timing['pipeline_start_unix'];save(TIMING,timing)
plan.update(status='ALLOCATION_AND_EVALUATIONS_COMPLETE',goal_complete=False);save(PLAN,plan)
