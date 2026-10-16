"""Close the complete fixed Roulade budget using saved trajectories, CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib, importlib.util, json, math, re, time
from pathlib import Path
import torch

torch.set_num_threads(2)
R = Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04')
W = R / 'roulade-history-worktree'
A = R / 'roulade-history-formal-001'
OUT = R / 'roulade-history-full-curve-review-001.json'
assert not OUT.exists()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
marker = read(R/'roulade-history-status-001.json')
assert marker['status'] == 'HISTORICAL_CURRICULUM1536_COMPLETE_REVIEW_CURVE'
assert not Path('/proc/344162').exists()
formal = read(A/'report.json')
assert formal['completed_updates'] == 1536 and formal['allocation_controls'] == 603979776
calls = [json.loads(line) for line in (A/'optimizer-calls.jsonl').read_text().splitlines()]
assert len(calls) == 1536
assert all(x['completed_updates'] == i+1 and x['optimizer_step_calls'] == (i+1)*80 for i,x in enumerate(calls))
spec = importlib.util.spec_from_file_location('saved_camera_metrics', W/'src/mjlab_microduck/custom_v2/head_metrics.py')
head = importlib.util.module_from_spec(spec); spec.loader.exec_module(head)
LEGS = [0,1,2,3,4,9,10,11,12,13]
refs, cfg_refs, rows, retained, source_checks = {}, {}, [], [], []
def scrub(x):
    if isinstance(x, dict): return {k:scrub(v) for k,v in x.items()}
    if isinstance(x, list): return [scrub(v) for v in x]
    if isinstance(x, str): return re.sub(r' at 0x[0-9a-fA-F]+', '', x)
    return x
def summary(v, mask):
    if not bool(mask.any()): return None
    return {k:dict(mean_over_trials=float(x[mask].mean()), p95_over_trials=float(torch.quantile(x[mask],.95))) for k,x in v.items()}
for node in [128,256,512,768,1024,1280,1536]:
    cp = A/f'checkpoint_{node:06d}.pt'
    checkpoint = torch.load(cp,map_location='cpu',weights_only=False)
    meta = checkpoint['infos']['v2']
    adam = sorted({int(x['step']) for x in checkpoint['optimizer_state_dict']['state'].values() if 'step' in x})
    normalizers = {name:int(checkpoint[f'{name}_state_dict']['obs_normalizer.count']) for name in ['actor','critic']}
    assert meta['completed_updates'] == node and meta['controls'] == node*16384*24
    assert adam == [node*80] and set(normalizers.values()) == {meta['controls']}
    assert all(sha(W/k) == v for k,v in meta['source_sha256'].items())
    retained.append(dict(node=node,checkpoint=str(cp),sha256=sha(cp),controls=meta['controls'],adam_steps=adam[0],normalizer_counts=normalizers))
    del checkpoint
    for stage in ['nominal','measured']:
        p = R/f'roulade-history-{node:04d}-{stage}-001'
        report = read(p/'report.json')
        assert report['checkpoint_sha256'] == retained[-1]['sha256']
        assert report['num_envs'] == 256 and report['seconds'] == 5 and report['actor_normalizer_unchanged']
        assert report['script_sha256'] == sha(p/'executed-script.py') == sha(W/'scripts/evaluate_v2_roulade.py')
        cfg = scrub(read(p/'config.json'))
        if stage not in cfg_refs: cfg_refs[stage] = cfg
        assert cfg == cfg_refs[stage], (node,stage,'config changed')
        trace = torch.load(p/'traces.pt',map_location='cpu',weights_only=False)
        m, initial = trace['measurements'], trace['initial']
        if stage not in refs: refs[stage] = {k:v.clone() for k,v in initial.items()}
        pairing = {k:dict(exact=torch.equal(v,refs[stage][k]),max_abs_difference=float((v.double()-refs[stage][k].double()).abs().max())) for k,v in initial.items()}
        parts = dict(both_feet=m['feet'].all(-1),no_head=~m['head_contact'],no_self=~m['self_contact'],
                     height=(m['height']-.112).abs()<.012,tilt=m['tilt']<math.radians(15),
                     linear_speed=m['linear'].norm(dim=-1)<.05,angular_speed=m['omega'].norm(dim=-1)<.5,
                     leg_pose=m['physical_q'][:,:,LEGS].abs().mean(-1)<.2,feedback=m['feedback_ready'],finite=m['finite'])
        body = torch.stack(list(parts.values())).all(0)
        head_neutral = m['physical_q'][:,:,5:9].abs().amax(-1)<.1
        assert torch.equal(body & head_neutral,m['stable'])
        land,neutral = body[-25:].all(0),m['stable'][-25:].all(0)
        moving=(m['omega'][:,:,1].abs()>.5)&(m['support'].int().cumsum(0)>0)
        support=(m['support']&moving).sum(0)/moving.sum(0).clamp_min(1)
        flat=torch.where(moving,m['lateral_axis_z'].abs(),0).amax(0)
        geometric=trace['geometric_unwrapped_rad'][-1]+initial['spawn_angle'] >= math.radians(345)
        eligible=m['head_latch'][-1]&(m['frontier'][-1]>=math.radians(330))
        core=land&eligible&geometric&m['finite'].all(0)&m['feedback_ready'].all(0)&(support>=.95)&(flat<math.sin(math.pi/4))
        camera=head.camera_metrics(trace['camera_quat_w'][-101:],dt=.005,exposure_s=.02)
        idx=torch.arange(256); groups=[]
        for gi,original in enumerate(report['results']):
            mask=idx%8<4 if gi==0 else idx%8==gi+3
            for key,value in [('core_completion',core),('strict_neutral_head_completion',core&neutral),('body_landing_tail500ms',land),('neutral_head_landing_tail500ms',neutral)]:
                assert int(value[mask].sum()) == original[key],(node,stage,gi,key)
            row=dict(original)
            row.update(landing_component_tail500ms_trials={k:int(v[-25:,mask].all(0).sum()) for k,v in parts.items()},
                       eligible_trials=int(eligible[mask].sum()),actual_geometric_trials=int(geometric[mask].sum()),
                       tail_height_m=float(m['height'][-25:,mask].mean()),tail_tilt_deg=float(torch.rad2deg(m['tilt'][-25:,mask]).mean()),
                       tail_linear_mps=float(m['linear'][-25:,mask].norm(dim=-1).mean()),tail_angular_radps=float(m['omega'][-25:,mask].norm(dim=-1).mean()),
                       tail_leg_mae_rad=float(m['physical_q'][-25:,mask][:,:,LEGS].abs().mean()),
                       tail_physical_joint_mean_rad=m['physical_q'][-25:,mask].mean((0,1)).tolist(),
                       tail_self_sample_fraction=float(m['self_contact'][-25:,mask].float().mean()),
                       camera_tail_all_trials=summary(camera,mask),camera_tail_all_denominator=int(mask.sum()),
                       camera_tail_body_healthy_cpu=summary(camera,mask&land),camera_tail_body_healthy_denominator=int((mask&land).sum()))
            groups.append(row)
        rows.append(dict(node=node,stage=stage,report_sha256=sha(p/'report.json'),trace_sha256=sha(p/'traces.pt'),
                         original_config_equal_to128=True,initial_pairing_with128=pairing,groups=groups))
        del trace,m,initial
wall=read(A/'shared-process-wall.json')
out=dict(status='FULL1536_CLOSED_COMPLETE_SKILL_UNQUALIFIED',created_unix=time.time(),
    formal_new_controls=603979776,prior_formal_controls=150994944,all_roulade_formal_controls=754974720,
    fixed_curve_diagnostic_controls=14*256*250,native_adam_steps=122880,native_clock=36864,
    native_report_wall_seconds=formal['wall_seconds'],native_report_controls_per_second=formal['full_process_control_per_second'],
    formal_phase_shared_process_wall=wall,formal_phase_controls_per_second=603979776/wall['seconds'],
    steady_throughput=formal['throughput'],training_metrics=formal['training_metrics'],
    curriculum_live=[json.loads(x) for x in (A/'curriculum-live.jsonl').read_text().splitlines()],
    retained_checkpoints=retained,rows=rows,source_hashes_verified=True,
    runtime_qualified=False,hardware_tested=False,
    limitations=['Full formal phase excludes Python imports, qualification, and final evaluation; not full-worker throughput.',
      'Small Ground007/008 diagnostics shared GPU0 during part of the run, so this is not an exclusive-device benchmark.',
      '200Hz camera motion includes intentional roll motion. Healthy landing denominators are separate.',
      'Contact and body gates retained at original50Hz reward boundary; no saved200Hz contact history.',
      'Precredited midroll frontier and latch are not standing-origin full-roll success.',
      'Repeated seeds across nodes are a fixed learning curve, not independent replication.',
      'No next formal allocation is implied by this review.'])
OUT.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(status=out['status'],formal_controls=out['formal_new_controls'],formal_phase_controls_per_second=out['formal_phase_controls_per_second'],
    all_initial_fields_equal=all(v['exact'] for row in rows for v in row['initial_pairing_with128'].values()),
    latest=[dict(stage=row['stage'],groups=[{k:g[k] for k in ['origin','spawn_pitch_deg','core_completion','body_landing_tail500ms','tail_height_m','tail_tilt_deg','tail_self_sample_fraction','landing_component_tail500ms_trials']} for g in row['groups']]) for row in rows if row['node']==1536]),indent=2))
