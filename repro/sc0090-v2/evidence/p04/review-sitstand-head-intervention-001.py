"""CPU-only review of the bounded640 external-head intervention."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,math,re,time,importlib.util
from pathlib import Path
import torch,mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04')
W=R/'sitstand-history-hold320-worktree'
torch.set_num_threads(2)
out=R/'sitstand-head-intervention-closeout-001.json';assert not out.exists()
marker=json.loads((R/'sitstand-head-intervention-status-001.json').read_text())
assert marker['status']=='FOUR_FIXED_DIAGNOSTICS_COMPLETE_REVIEW_REQUIRED'
assert not Path(f'/proc/{marker["pid"]}').exists()
spec=importlib.util.spec_from_file_location('camera_functions',W/'src/mjlab_microduck/custom_v2/head_metrics.py')
cam=importlib.util.module_from_spec(spec);spec.loader.exec_module(cam)
model=mujoco.MjModel.from_xml_path(str(W/'src/mjlab_microduck/custom_v2/assets/robot.xml'))
names=['left_hip_yaw','left_hip_roll','left_hip_pitch','left_knee','left_ankle','neck_pitch','head_pitch','head_yaw','head_roll','right_hip_yaw','right_hip_roll','right_hip_pitch','right_knee','right_ankle']
ids=[int(model.joint(n).qposadr[0])for n in names];legs=[0,1,2,3,4,9,10,11,12,13]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def clean(x):
    if isinstance(x,dict):return {k:clean(v)for k,v in x.items()}
    if isinstance(x,list):return [clean(v)for v in x]
    if isinstance(x,str):return re.sub(r' at 0x[0-9a-fA-F]+','',x)
    return x
def aggregate(values,mask):
    if not bool(mask.any()):return None
    return {k:dict(mean_over_trials=float(v[mask].mean()),p95_over_trials=float(torch.quantile(v[mask],.95)))for k,v in values.items()}
rows=[];pairing={};refs={};configs={};data={};total=0
for arm in ['policy','command_when_standing']:
    for stage in ['nominal','measured']:
        p=R/f'sitstand-head-intervention-{arm}-{stage}-001'
        report=json.loads((p/'report.json').read_text());trace=torch.load(p/'traces.pt',map_location='cpu',weights_only=False)
        counts=json.loads((p/'diagnostic-count.json').read_text());assert counts['completed_controls']==38400;total+=38400
        cfg=clean(json.loads((p/'eval-config.json').read_text()))
        if stage not in configs:configs[stage]=cfg
        assert cfg==configs[stage]
        m=trace['traces'];obs=trace['actor_obs_preaction'];policy=trace['raw_policy_actions'];applied=trace['actions_preprocess'];contacts=trace['contacts200']
        assert m.shape==(600,64,14) and contacts.shape==(2400,64,4)
        assert torch.equal(policy[:,:,legs],applied[:,:,legs])
        stand=obs[:,:,48]==0
        expected=policy[:,:,5:9] if arm=='policy' else torch.where(stand[:,:,None],obs[:,:,51:55],policy[:,:,5:9])
        assert torch.equal(applied[:,:,5:9],expected)
        assert report['actor_normalizer_unchanged'] and report['numerical_source_verified']
        assert max(report['oracle_max_abs'])<2e-6
        assert bool(torch.isfinite(trace['qpos_boundary']).all()&torch.isfinite(trace['qvel_boundary']).all())
        initial=trace['initial']
        if stage not in refs:refs[stage]={k:v.clone()for k,v in initial.items()}
        pairing[f'{arm}-{stage}']={k:dict(exact=torch.equal(v,refs[stage][k]),max_abs=float((v-refs[stage][k]).abs().max()))for k,v in initial.items()}
        q=trace['qpos_boundary'][1:,:,ids];mode=trace['mode'];data[arm,stage]=trace
        for mode_id,name in enumerate(['stand_to_stand','stand_to_sit','sit_to_stand','sit_to_sit']):
            select=mode==mode_id
            for phase in range(3):
                lo,hi=phase*200,(phase+1)*200;tail=slice(hi-25,hi)
                body=m[tail,select,4].bool().all(0);flag=int(obs[lo,select,48][0]);assert bool((obs[lo:hi,select,48]==flag).all())
                physical=contacts[(hi-25)*4:hi*4,select]
                no_bad=~physical[:,:,2:].any((0,2));both_feet=physical[:,:,:2].all((0,2))
                strict=body&no_bad&(both_feet if flag==0 else torch.ones_like(both_feet))
                head1deg=(m[tail,select,7]<math.pi/180).all(0)
                camera=cam.camera_metrics(trace['camera_quat_w'][(hi-25)*4:hi*4+1,select],dt=.005,exposure_s=.02)
                original=report['groups'][name][phase];assert int(body.sum())==original['tail_hold']
                zero=slice(lo+100,lo+150)
                rows.append(dict(arm=arm,stage=stage,start_group=name,phase=phase,command='STAND'if flag==0 else'SIT',trials=int(select.sum()),
                  tail_original_body_hold=int(body.sum()),tail_extra_contact_screen_body_hold=int(strict.sum()),
                  tail_body_and_head1deg=int((body&head1deg).sum()),tail_contactscreen_body_and_head1deg=int((strict&head1deg).sum()),
                  head_step_reached=original['head_step_reached'],head_step_tail_band_hold=original['head_step_tail_band_hold'],
                  tail_height_m=float(m[tail,select,0].mean()),tail_tilt_deg=float(torch.rad2deg(m[tail,select,1]).mean()),
                  tail_head_mean_abs_error_deg=torch.rad2deg(m[tail,select,10:14]).mean((0,1)).tolist(),
                  tail_physical_head_mean_deg=torch.rad2deg(q[tail,select,5:9]).mean((0,1)).tolist(),
                  zero_head_window_body_hold=int(m[zero,select,4].bool().all(0).sum()),zero_head_window_tilt_deg=float(torch.rad2deg(m[zero,select,1]).mean()),
                  zero_head_window_neck_deg=float(torch.rad2deg(q[zero,select,5]).mean()),
                  any_tilt_over45_trials=int((m[lo:hi,select,1]>math.pi/4).any(0).sum()),
                  whole_phase_feedback_fault_trials=int(m[lo:hi,select,5].bool().any(0).sum()),
                  tail_head_ground_contact_trials=int(physical[:,:,2].any(0).sum()),tail_self_contact_trials=int(physical[:,:,3].any(0).sum()),
                  tail_continuous_both_feet_trials=int(both_feet.sum()),phase_head_force_peak_N=float(trace['head_force200'][lo*4:hi*4,select].max()),
                  camera_tail_all=aggregate(camera,torch.ones_like(body)),camera_tail_original_body_healthy=aggregate(camera,body),
                  camera_tail_extra_contactscreen_healthy=aggregate(camera,strict)))
assert total==153600
prefix={}
for stage in ['nominal','measured']:
    a,b=data['policy',stage],data['command_when_standing',stage]
    # Both policies are unmodified during firstSIT phase; separate runs may
    # still have small simulator differences, which must not be hidden.
    select=a['mode']%2==1
    dq=(a['qpos_boundary'][:201,select]-b['qpos_boundary'][:201,select]).abs()
    prefix[stage]=dict(never_intervened_first4s_trials=int(select.sum()),initial_qpos_equal=bool((dq[0]==0).all()),
      first_control_qpos_max_difference=float(dq[1].max()),full_prefix_qpos_max_difference=float(dq.max()),
      raw_policy_action_max_difference=float((a['raw_policy_actions'][:200,select]-b['raw_policy_actions'][:200,select]).abs().max()))
fig,axes=plt.subplots(4,2,figsize=(13,10),sharex=True);times=torch.arange(600).numpy()*.02
for col,stage in enumerate(['nominal','measured']):
    for arm,label,color in [('policy','Frozen640 policy','#b64b43'),('command_when_standing','External head target on STAND','#1677a6')]:
        trace=data[arm,stage];sel=trace['mode']==0;m=trace['traces'][:,sel];q=trace['qpos_boundary'][1:,sel][:,:,ids]
        values=[m[:,:,0]*1000,torch.rad2deg(m[:,:,1]),torch.rad2deg(q[:,:,5]),torch.rad2deg(q[:,:,8])]
        for ri,v in enumerate(values):
            axes[ri,col].plot(times,v.mean(1).numpy(),label=label,color=color,lw=1.4)
            axes[ri,col].fill_between(times,torch.quantile(v,.1,dim=1).numpy(),torch.quantile(v,.9,dim=1).numpy(),alpha=.16,color=color)
    for ri,ax in enumerate(axes[:,col]):
        ax.axvspan(4,8,color='gray',alpha=.10);ax.grid(alpha=.2);ax.set_ylabel(['Height (mm)','Body tilt (deg)','Neck (deg)','Head roll (deg)'][ri])
    axes[0,col].set_title(stage+' / HOME start / STAND-SIT-STAND')
    axes[1,col].axhline(15,ls='--',color='gray',lw=.8)
    axes[-1,col].set_xlabel('Time (s)');axes[0,col].legend(fontsize=8)
fig.suptitle('Frozen640 diagnostic: mean and 10–90% range across16 trials\nExternal head replacement is not a trained policy; gray = SIT command')
fig.tight_layout();figure=R/'sitstand-head-intervention-curve-001.png';fig.savefig(figure,dpi=140);plt.close(fig)
result=dict(status='BOUNDED_HEAD_INTERVENTION_REVIEW_COMPLETE',created_unix=time.time(),diagnostic_controls=total,new_formal_controls=0,
   parent_sha256=report['checkpoint_sha256'],source_unchanged=True,original_body_and_head_gates_retained=True,
   initial_pairing=pairing,unmodified_SIT_prefix_comparison=prefix,rows=rows,figure=str(figure),
   limitations=['This is an external action intervention, not singlepolicy training success or deployment advice.',
    'After head replacement actual feedback and hence subsequent policy leg outputs may change; this is not fixedlegtrajectory replay.',
    'Separate simulator runs may diverge even before intervention. Initial pairing and untouchedprefix differences are reported, not assumed.',
    'Contactscreen is additional to original50Hz body gate;200Hz contacthistory does not certify200Hz body state.',
    'Bothfeet added only to standing hold screen. SIT can use shin support; do not impose walkingfeet gate on healthy sitting.',
    'Camera activecommand motion and posturetransitions are intentional; only comparable quiet windows assess stability.',
    'Small diagnostic shared GPU0 with Groundformal; no exclusive throughput claim.'])
out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(status=result['status'],initial_pairing=pairing,prefix=prefix,home_start_first_stand=[r for r in rows if r['start_group']=='stand_to_stand'and r['phase']==0]),indent=2))
