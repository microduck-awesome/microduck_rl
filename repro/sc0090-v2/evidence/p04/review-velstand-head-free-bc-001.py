"""Compare the bounded leg-only walking-BC candidate to retained references."""
from pathlib import Path
import hashlib,json,math,runpy,subprocess,sys
import numpy as np
import torch
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(2)
aggregate=runpy.run_path(str(R/'compare-velstand-camera-pair.py'))['aggregate']
camera_metrics=runpy.run_path(str(R/'velstand-camera-worktree/src/mjlab_microduck/custom_v2/head_metrics.py'))['camera_metrics']
groups=dict(neutral_idle=lambda n,p:n=='idle',
    straight=lambda n,p:n.startswith(('forward','lateral','diagonal')) and p in ('move','reverse'),
    turn=lambda n,p:n.startswith(('yaw','curve')) and p in ('move','reverse'),
    walking_head=lambda n,p:n.startswith('walking_head') and p in ('move','reverse'),
    stop=lambda n,p:n!='idle' and p=='stop',return_to_idle=lambda n,p:n!='idle' and p=='return')
marker=read(R/'velstand-head-free-bc-pipeline-status-001.json')
assert marker['status']=='BOUNDED_HEAD_BC_STAGE_COMPLETE_REVIEW_REQUIRED'
result=dict(status='HEAD_BC_BUDGET_COMPLETE_REVIEW_REQUIRED',formal_additional_controls=37748736,
    results={},initial_comparisons=[],evidence={},source_sha256=sha(Path(__file__)),
    scope='One new fixed64 from original256, compared with retained original mild64 from same parent; control320 and parent256 also retained. Not simultaneous reruns. No validation checkpoint inherited. Initial equality tested, no bitwise trajectory claim. Upright-window camera is supplemental and does not loosen original failure/command gates.')
prefixes=dict(parent256='velstand-0256',full_bc_mild='velstand-camera-mild-0320',
    full_bc_control='velstand-camera-control-0320',walking_leg_bc='velstand-head-free-bc-0320')
for stage in ('nominal','measured'):
    stage_result={};initials={}
    for arm,prefix in prefixes.items():
        directory=R/f'{prefix}-commands-{stage}';report=read(directory/'report.json')
        residual_path=R/f'{prefix}-camera-world-residual-{stage}.json'
        if not residual_path.exists():
            subprocess.run([sys.executable,str(R/'audit-camera-world-residual.py'),
                '--directory',str(directory),'--output',str(residual_path)],check=True,stdout=subprocess.DEVNULL)
        residual=read(residual_path);assert residual['input_report_sha256']==sha(directory/'report.json')
        survey=read(R/f'{prefix}-survey-{stage}/report.json')
        rows=report['records']
        data=dict(trials=sum(x['trials'] for x in rows),survived=sum(x['survived_trials'] for x in rows),
            feedback_invalid=sum(x['feedback_invalid_trials'] for x in rows),
            groups={name:aggregate(rows,residual['rows'],f) for name,f in groups.items()},
            survey={row['case']:{k:v for k,v in row.items() if k not in ('weighted_reward_rates','tail_weighted_reward_rates')} for row in survey['rows']},
            upright_camera={})
        healthy={g:[] for g in groups}
        for seed in (2026101612,2026101613):
            trace=torch.load(directory/f'trace-{seed}.pt',weights_only=False,map_location='cpu')
            initials[(arm,seed)]=trace['initial'];x=trace['measurements']
            seed_rows=[row for row in rows if row['seed']==seed]
            for i,row in enumerate(seed_rows):
                sl=slice(i*8,(i+1)*8)
                for phase,a,b in [('idle',0,50),('move',50,150),('stop',150,200),('reverse',200,300),('return',300,400)]:
                    gate=((x['height'][a:b,sl]>=.102)&(x['tilt'][a:b,sl]<=math.radians(20))&x['finite'][a:b,sl]&x['feedback_ready'][a:b,sl]).all(0)&~x['failed'][:b,sl].any(0)
                    cm=camera_metrics(trace['camera'][a*4:b*4+1,sl])
                    for g,select in groups.items():
                        if select(row['name'],phase):healthy[g].append((gate,cm))
        for g,items in healthy.items():
            mask=torch.cat([item[0] for item in items]);values={k:torch.cat([item[1][k] for item in items]) for k in items[0][1]}
            data['upright_camera'][g]=dict(trial_phase_count=len(mask),eligible_trial_phase_count=int(mask.sum()),
                criteria='whole phase height>=.102m/tilt<=20deg/finite/feedback/no prior walking failure; no requirement for double support in motion',
                metrics={k:float(v[mask].mean()) for k,v in values.items()} if mask.any() else None)
        stage_result[arm]=data
        result['evidence'][f'{stage}/{arm}']=dict(checkpoint_sha256=report['checkpoint_sha256'],
            commands_sha256=sha(directory/'report.json'),survey_sha256=sha(R/f'{prefix}-survey-{stage}/report.json'),residual_sha256=sha(residual_path))
    for seed in (2026101612,2026101613):
        a=initials[('walking_leg_bc',seed)];b=initials[('full_bc_mild',seed)]
        fields={k:dict(exact=torch.equal(a[k],b[k]),max_abs=float((a[k]-b[k]).abs().max())) for k in a}
        result['initial_comparisons'].append(dict(stage=stage,seed=seed,fields=fields))
        assert all(x['exact'] for x in fields.values()),'Initial-state mismatch needs review'
    result['results'][stage]=stage_result
train=read(R/'velstand-head-free-bc-allocation-001/report.json')
result['training']={k:train[k] for k in ('completed_updates','allocation_controls','throughput','feedback_audit','camera_audit','terminal_bootstrap_samples','contact_peak','contact_capacity')}
out=R/'velstand-head-free-bc-review-001.json';assert not out.exists()
out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({stage:{arm:dict(survived=x['survived'],straight_rms=x['groups']['straight']['camera_rms_per_trial_mean_rad_s'],
    turn_rms=x['groups']['turn']['camera_rms_per_trial_mean_rad_s'],head_rms=x['groups']['walking_head']['camera_rms_per_trial_mean_rad_s'],
    stop_rms=x['groups']['stop']['camera_rms_per_trial_mean_rad_s']) for arm,x in vals.items()} for stage,vals in result['results'].items()},indent=2))
