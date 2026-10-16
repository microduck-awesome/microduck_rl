"""CPU closeout of the local fullbody teacher handoff, preserving failed receipts."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,time,hashlib,importlib.util,re
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'sitstand-history-hold320-worktree'
torch.set_num_threads(2)
marker=json.loads((R/'sitstand-near-stand-teacher-status-002.json').read_text())
assert marker['status']=='FOUR_LOCAL_TEACHER_DIAGNOSTICS_COMPLETE'and not Path(f'/proc/{marker["pid"]}').exists()
spec=importlib.util.spec_from_file_location('native_camera_metrics',W/'src/mjlab_microduck/custom_v2/head_metrics.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
def norm(x):
    if isinstance(x,dict):return {k:norm(v)for k,v in x.items()}
    if isinstance(x,list):return [norm(v)for v in x]
    return re.sub(r' at 0x[0-9a-fA-F]+','',x)if isinstance(x,str)else x
def aggregate(values,mask):
    if not bool(mask.any()):return None
    return {k:dict(mean_over_trials=float(v[mask].mean()),p95_over_trials=float(torch.quantile(v[mask],.95)))for k,v in values.items()}
rows=[];data={};refs={};configs={};total=0;pairing={};prefix={}
for arm in ['policy','near_stand_teacher']:
    for stage in ['nominal','measured']:
        suffix='001'if arm=='policy'and stage=='nominal'else'002'
        p=R/f'sitstand-near-stand-teacher-{arm}-{stage}-{suffix}';rep=json.loads((p/'report.json').read_text());t=torch.load(p/'traces.pt',map_location='cpu',weights_only=False)
        assert json.loads((p/'diagnostic-count.json').read_text())['completed_controls']==12800;total+=12800
        cfg=norm(json.loads((p/'config.json').read_text()))
        if stage not in configs:configs[stage]=cfg
        assert cfg==configs[stage]
        if stage not in refs:refs[stage]=t['initial']
        pairing[f'{arm}-{stage}']={k:dict(exact=torch.equal(v,refs[stage][k]),max_abs=float((v-refs[stage][k]).abs().max()))for k,v in t['initial'].items()}
        raw,adapted=t['actor_obs_preaction'],t['teacher_obs_preaction'];assert raw.shape==(200,64,61)
        assert torch.equal(raw[:,:,:48],adapted[:,:,:48])and torch.equal(raw[:,:,51:55],adapted[:,:,51:55])
        assert bool((adapted[:,:,48:51]==0).all()&(adapted[:,:,55:61]==0).all())
        eligible=t['eligible_at2s'];selected=t['mode']==0;assert int(eligible.sum())==16 and torch.equal(eligible,selected)
        want=t['student_actions'].clone()
        if arm=='near_stand_teacher':want[100:,eligible]=t['teacher_actions'][100:,eligible]
        assert torch.equal(t['actions'],want)
        m=t['measurements'];assert bool(m['finite'].all()&m['feedback_ready'].all())
        assert rep['actors_and_normalizers_unchanged']and max(rep['oracle_max_abs'])<2e-6
        body=m['stable'][-25:].all(0);c=t['contacts200'][-100:];clean=~c[:,:,2:].any((0,2))&c[:,:,:2].all((0,2));head=m['physical_q'][-25:,:,5:9].abs().amax(-1).lt(.1).all(0)
        assert rep['tail_body']==int(body[selected].sum())and rep['tail_body_head_HOME']==int((body&clean&head)[selected].sum())
        values=h.camera_metrics(t['camera_quat_w'][-101:],dt=.005,exposure_s=.02)
        headforce=t['head_force200'][200:400,selected]
        row=dict(rep);row.pop('source_sha256')
        row.update(report_path=str(p/'report.json'),trace_sha256=hashlib.sha256((p/'traces.pt').read_bytes()).hexdigest(),
                   saved_raw_preaction_at_switch_head_previous_action=t['actor_obs_preaction'][100,selected,39:43].mean(0).tolist(),
                   before_switch_1to2s_head_force_mean_N=float(headforce.mean()),
                   after_switch_tilt_max_deg=float(torch.rad2deg(m['tilt'][100:,selected]).max()),
                   after_switch_self_contact_trials=int(t['contacts200'][400:,selected,3].any(0).sum()),
                   tail_all_head_axes1deg=int((m['physical_q'][-25:,selected,5:9].abs().amax(-1)<torch.pi/180).all(0).sum()),
                   camera_tail_all=aggregate(values,selected),camera_tail_healthy_denominator=int((selected&body&clean&head).sum()),
                   camera_tail_healthy=aggregate(values,selected&body&clean&head))
        rows.append(row);data[arm,stage]=t
for stage in ['nominal','measured']:
    a,b=data['policy',stage],data['near_stand_teacher',stage];sel=a['mode']==0
    q=(a['qpos_boundary'][:101,sel]-b['qpos_boundary'][:101,sel]).abs()
    prefix[stage]=dict(scope='First2s both arms execute unchanged640',initial_qpos_exact=bool((q[0]==0).all()),
                      first_control_qpos_max_abs=float(q[1].max()),all_prefix_qpos_max_abs=float(q.max()),
                      pre_switch_last_qpos_max_abs=float(q[-1].max()))
fig,axes=plt.subplots(4,2,figsize=(12,9),sharex=True)
for ci,stage in enumerate(['nominal','measured']):
    for arm,label,color in [('policy','Frozen640 throughout','#b64b43'),('near_stand_teacher','Full1856 teacher after2s','#1677a6')]:
        t=data[arm,stage];sel=t['mode']==0;m=t['measurements'];vs=[m['height'][:,sel]*1000,torch.rad2deg(m['tilt'][:,sel]),torch.rad2deg(m['physical_q'][:,sel,5]),t['head_force200'][:,sel]]
        for ri,v in enumerate(vs):
            x=torch.arange(len(v)).numpy()*(.005 if ri==3 else .02)
            axes[ri,ci].plot(x,v.mean(-1).numpy(),label=label,color=color,lw=1.4)
            axes[ri,ci].fill_between(x,torch.quantile(v,.1,dim=-1).numpy(),torch.quantile(v,.9,dim=-1).numpy(),color=color,alpha=.15)
    for ri,ax in enumerate(axes[:,ci]):
        ax.axvline(2,ls='--',color='black',lw=.8);ax.grid(alpha=.2);ax.set_ylabel(['Height(mm)','Body tilt(deg)','Neck(deg)','Head ground force(N)'][ri])
    axes[0,ci].set_title(stage+' /16 HOME-STAND trials');axes[0,ci].legend(fontsize=8);axes[-1,ci].set_xlabel('Time(s)')
fig.suptitle('Local fullbody teacher handoff: mean and10–90% range\nFrozen640 until2s; actual feedback/history/previousaction retained')
fig.tight_layout();figure=R/'sitstand-near-stand-teacher-curve-001.png';fig.savefig(figure,dpi=140);plt.close(fig)
assert total==51200
result=dict(status='LOCAL_NEAR_STAND_TEACHER_DEMONSTRATED_NOT_FULL_SITSTAND',new_diagnostic_controls=total,new_formal_controls=0,
  all_current_head_and_teacher_diagnostic_controls=153600+total,initial_pairing=pairing,unmodified_prefix=prefix,rows=rows,figure=str(figure),
  conclusion='Fullbody1856 teacher restores32/32 qualified nearSTAND head-supported states without exceeding45deg afterswitch; head-only intervention failed. Supports only this localteacher region, not lowSIT transition or trained640 success.',
  limitations=['Teacher produces a small heightdip to96.9–97.5mm; do not describe thehandoff as having no downwardmotion.',
   'Initialpairing does not imply bitexact trajectories; actual2s preintervention differences retained.',
   'No formalPPO budget consumed; both frozenmodels and normalizers unchanged.',
   'The stable/head gates remain original; extra200Hz contactchecks and healthycamera denominators separate.',
   'Casepolicy/nominal reused fromworker001; failedsecondconstruction completed0controls; only remainingthreecases ran underdeepcopyfix.'])
p=R/'sitstand-near-stand-teacher-closeout-001.json';assert not p.exists();p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(status=result['status'],prefix=prefix,summary=[{k:r[k]for k in ['arm','stage','tail_body_head_HOME','after_switch_tilt_over45','after_switch_tilt_max_deg','after_switch_min_height','tail_head_force_mean_N','tail_all_head_axes1deg','camera_tail_healthy_denominator']}for r in rows]),indent=2))
