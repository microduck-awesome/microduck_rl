"""Bounded T24 baseline-rooted independent head accuracy bonus experiment."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04')
SOURCE=ROOT/'paired-head-accuracy-worktree'
TASK='Mjlab-StandUp-Flat-MicroDuck-V2-PairedAdapt-Catch2sPitch10-LateEntropy005-HeadStdCap001-AccuracyBonus05-Repaired-S0-O0-R1'
PARENT=ROOT/'paired-stdcap-allocation-003/resume_004274.pt'
OUTPUT=ROOT/'paired-head-accuracy-allocation-001'
PLAN=ROOT/'paired-head-accuracy-plan.json'
TIMING=ROOT/'paired-head-accuracy-timing.json'
HELPERS=ROOT.parent/'baseline/head-foundation-worktree/scripts'
ENV=dict(os.environ,SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat')

def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)

assert not OUTPUT.exists() and not PLAN.exists()
plan=dict(status='WAITING_FOR_UNIT_AND_NEGATIVE_PITCH_EVALUATION',created_unix=time.time(),task=TASK,
    source=str(SOURCE),parent=str(PARENT),parent_sha256=digest(PARENT),
    only_behavioral_change='Add independent bounded command-axis physical precision reward, weight 0.5; retain existing Head5 tracking and all other reward/curriculum/PPO/feedback settings.',
    num_envs=32768,rollout_steps=24,updates=64,new_controls=50331648,start_update=4274,end_update=4338,
    initial_clock=102576,final_clock=104112,inherited_learning_rate=2.2500000000000005e-5,
    restore='Exact actor/critic/normalizers/Adam/LR/real clock, native new episodes. No learning-rate transition.',
    rationale='T12 lost current head accuracy without clear whole-negative-pitch advantage; retain T24 and test direct command-axis precision reward while preserving successful catch curriculum.',
    comparison_limit='Static 4274 parent and earlier T24 learning history are references, not a new matched-parent no-bonus 64-update control. Do not infer strict causal benefit from parent comparison alone.',
    optimizer_admission=dict(max_update_mean_kl=.5,max_clip_fraction=.8,last5_median_kl=.1,last5_median_clip_fraction=.5),
    decision='One 64-update allocation, then full head/standing/recovery assessment. No automatic additional training or deployment.')
save(PLAN,plan)
deadline=time.monotonic()+1800
while True:
    unit=ROOT/'paired-head-accuracy-unit-tests.json'
    frontier=ROOT/'paired-negative-pitch-t24-t12-plan.json'
    unit_log=ROOT/'paired-head-accuracy-unit-tests.log'
    if unit_log.exists() and 'Traceback' in unit_log.read_text():raise RuntimeError('Unit checks failed')
    if unit.exists() and read(frontier)['status']=='COMPLETE':break
    if read(frontier)['status']=='FAILED':raise RuntimeError('Preceding negative-pitch evaluation failed')
    if time.monotonic()>deadline:raise TimeoutError('Waiting for unit/frontier')
    time.sleep(5)
assert read(unit)['status']=='PASS'
files=[*SOURCE.glob('src/**/*.py'),SOURCE/'scripts/train_v2_standup.py',SOURCE/'scripts/smoke_v2_ppo.py',SOURCE/'scripts/check_paired_head_accuracy.py',SOURCE/'scripts/v2_python.sh']
hashes={str(path.relative_to(SOURCE)):digest(path) for path in files}
plan.update(source_sha256=hashes,unit_report_sha256=digest(unit),status='PREFLIGHT_RUNNING');save(PLAN,plan)
timing=dict(pipeline_start_unix=time.time(),phases=[],scope='Includes this recipe preflight, formal training, two head evaluations and common measured recovery. Prior training and negative-pitch evaluation excluded.')

def check_source():
    for path,sha in hashes.items():assert digest(SOURCE/path)==sha,path

def run(label,args):
    phase=dict(label=label,start_unix=time.time(),arguments=list(map(str,args)))
    timing['phases'].append(phase)
    with (ROOT/f'paired-head-accuracy-{label}.log').open('w') as log:
        process=subprocess.Popen([str(SOURCE/'scripts/v2_python.sh'),*map(str,args)],cwd=SOURCE,env=ENV,stdout=log,stderr=subprocess.STDOUT)
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
for label,n,iterations in [('mature-smoke',64,5),('throughput-32768',32768,16)]:
    out=ROOT/f'paired-head-accuracy-{label}'
    args=['scripts/smoke_v2_ppo.py','--task',TASK,'--device','cuda:1','--num-envs',n,'--iterations',iterations,
        '--compiled-measurement','--fused-registers','--head-precision','--head-precision-stage','5',
        '--mature-resume',PARENT,'--freeze-normalizers','--output-dir',out,'--report',out/'report.json']
    if n==64:args+=['--force-timeouts']
    run(label,args)
    report=read(out/'report.json')
    assert report['status']=='PASS' and report['steps_per_env']==24
    assert report['initialization']['actor_critic_normalizers_optimizer_exact']
    assert report['initialization']['inherited_completed_updates']==4274
    assert report['initialization']['curriculum_clock']==102576
    assert report['initialization']['learning_rate']==plan['inherited_learning_rate']
    assert report['curriculum_clock_advance_verified']
    assert report['curriculum_clock_after_learning']==102576+iterations*24
    assert report['first_minibatch_distribution_consistent_all_updates']
    assert report['normalization_unchanged_all_updates']
    assert report['accuracy_oracle']['status']=='PASS' and report['accuracy_oracle']['calls']>=iterations*24
    if n==64:assert report['terminal_bootstrap_samples']>0
    check_source()

out=ROOT/'paired-head-accuracy-throughput-32768'
rows=[json.loads(line)['metrics'] for line in (out/'ppo-diagnostics.jsonl').read_text().splitlines()]
assert len(rows)==16
kl=[row['PPO/kl_mean'] for row in rows];clip=[row['PPO/clip_fraction'] for row in rows]
quality=dict(max_update_mean_kl=max(kl),max_clip_fraction=max(clip),last5_median_kl=statistics.median(kl[-5:]),last5_median_clip_fraction=statistics.median(clip[-5:]))
quality['median_control_steps_per_s']=read(out/'report.json')['throughput']['median_end_to_end_steps_per_s']
quality['status']='PASS' if all(quality[k]<=v for k,v in plan['optimizer_admission'].items()) and quality['median_control_steps_per_s']>100000 else 'FAIL'
save(ROOT/'paired-head-accuracy-optimizer-admission.json',quality)
if quality['status']!='PASS':
    plan.update(status='ADMISSION_FAILED',optimizer=quality);save(PLAN,plan)
    raise RuntimeError(quality)
plan.update(status='FORMAL_RUNNING',optimizer=quality);save(PLAN,plan)
run('formal',['scripts/train_v2_standup.py','--task',TASK,'--device','cuda:1','--num-envs','32768',
    '--iterations','64','--eval-interval','64','--eval-samples','128','--head-precision','--head-precision-stage','5',
    '--freeze-normalizers','--preflight',out/'report.json','--adapt-from',PARENT,'--output',OUTPUT])
status=read(OUTPUT/'status.json')
assert status['completed_updates']==4338 and status['allocation_controls']==50331648
assert status['common_step_counter']==104112
plan['status']='POST_TRAINING_EVALUATIONS';save(PLAN,plan)
for evidence in sorted(OUTPUT.glob('evaluation_*_evidence')):
    for script in ['analyze_v2_standing_order.py','analyze_standing_pitch_bins.py']:
        run(evidence.name+script,[SOURCE/'scripts'/script,evidence])
for iteration in [4288,4338]:
    head=ROOT/f'paired-head-accuracy-{iteration}-head-measured'
    run(f'head{iteration}',['scripts/evaluate_v2_head_normalizer.py','--task',TASK,'--checkpoint',OUTPUT/f'checkpoint_{iteration:06d}.pt',
        '--output',head,'--device','cuda:1','--repeats','8','--randomization-stage','measured'])
    run(f'head-summary{iteration}',[HELPERS/'summarize_v2_head_saved.py','--device','cpu','--reports',head/'report.json',
        '--output',ROOT/f'paired-head-accuracy-{iteration}-head-summary'])
common=ROOT/'paired-head-accuracy-4338-common-measured'
run('common4338',[HELPERS/'evaluate_common_p04_candidates.py','--checkpoints',OUTPUT/'checkpoint_004338.pt','--labels','accuracy4338',
    '--output',common,'--device','cuda:1','--samples','128'])
for script in ['analyze_v2_standing_order.py','analyze_standing_pitch_bins.py']:
    run('common-'+script,[SOURCE/'scripts'/script,common/'accuracy4338'])
timing['pipeline_end_unix']=time.time();timing['pipeline_wall_seconds']=timing['pipeline_end_unix']-timing['pipeline_start_unix'];save(TIMING,timing)
plan.update(status='ALLOCATION_AND_EVALUATIONS_COMPLETE',goal_complete=False);save(PLAN,plan)
