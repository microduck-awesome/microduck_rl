"""One bounded saved-trajectory/actor audit; no simulation, no PPO, no GPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,hashlib,math
import torch,mujoco
import torch.nn.functional as F
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'sitstand-history-hold320-worktree';torch.set_num_threads(2)
cp=R/'sitstand-history-hold320-allocation-001/checkpoint_000640.pt';c=torch.load(cp,map_location='cpu',weights_only=False);sd=c['actor_state_dict'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();before={k:v.clone()for k,v in sd.items()}
run=json.loads((R/'sitstand-history-hold320-allocation-001/run.json').read_text());assert run['agent']['actor']['activation']=='elu';assert run['agent']['actor']['hidden_dims']==[512,256,128]
model=mujoco.MjModel.from_xml_path(str(W/'src/mjlab_microduck/custom_v2/assets/robot.xml'));names=['left_hip_yaw','left_hip_roll','left_hip_pitch','left_knee','left_ankle','neck_pitch','head_pitch','head_yaw','head_roll','right_hip_yaw','right_hip_roll','right_hip_pitch','right_knee','right_ankle'];qi=[int(model.joint(n).qposadr[0])for n in names];legs=[0,1,2,3,4,9,10,11,12,13];sit=torch.zeros(14)
for k,v in run['environment']['rewards']['posture_pose_legs']['params']['sit_overrides'].items():sit[int(k)]=v
@torch.inference_mode()
def policy(x):
 x=(x-sd['obs_normalizer._mean'])/(sd['obs_normalizer._std']+.01)
 for i in [0,2,4]:x=F.elu(F.linear(x,sd[f'mlp.{i}.weight'],sd[f'mlp.{i}.bias']))
 return F.linear(x,sd['mlp.6.weight'],sd['mlp.6.bias'])
results=[]
for condition in ['nominal','measured']:
 path=R/f'sitstand-history-hold320-0640-{condition}/traces.pt';saved=torch.load(path,map_location='cpu',weights_only=False);t=saved['traces'];obs=saved['actor_obs_preaction'];act=saved['actions_preprocess'];q=saved['qpos_boundary'][1:,:,qi];mode=saved['mode'];head=t[:,:,10:]
 assert torch.equal((q[:,:,5:9]-obs[:,:,51:55]).abs(),head)
 # Use only actual recorded rawobs; never regenerate policyinputs fromtruth.
 sample=obs[::5].reshape(-1,61);expected=act[::5].reshape(-1,14);mean=policy(sample);err=float((mean-expected).abs().max());assert err<2e-5,err
 flip=sample.clone();flip[:,48]=1-flip[:,48];other=policy(flip);delta=(other-mean).reshape(120,256,14)
 a=t[:,:,8];flag=t[:,:,9];le=q[:,:,legs]-a[:,:,None]*sit[legs];dz=t[:,:,0]-(.112+a*(.075999-.112));cos=torch.cos(t[:,:,1]);pose=torch.exp(-le.square().mean(-1)/.4**2);h=((t[:,:,0]-.075)/.025).clamp(0,1)
 terms=dict(pose=4*torch.exp(-(le/.5).square()).mean(-1),pose_l1=-le.abs().mean(-1),height=torch.exp(-(dz/.04).square()),height_sharp=torch.exp(-(dz/.015).square()),height_l1=-6*dz.abs(),head=.75*torch.exp(-(head/.5).square()).mean(-1),composite=3*torch.exp(-(dz/.03).square())*torch.exp(-(1-cos)/.4**2)*pose*torch.exp(-head.square().mean(-1)/.4**2),upright=2.5*cos,upright_tall=1.5*cos*h.square()*(3-2*h))
 groups=[]
 for m,name in enumerate(['stand_to_stand','stand_to_sit','sit_to_stand','sit_to_sit']):
  sel=mode==m
  for phase in range(3):
   sl=slice(phase*200+175,(phase+1)*200);f=flag[sl,sel];assert bool((a[sl,sel]==f).all());leg=(q[sl][:,sel][:,:,legs]-f[:,:,None]*sit[legs]).abs().mean(-1);height=(t[sl,sel,0]-(.112+f*(.075999-.112))).abs()<.012;tilt=t[sl,sel,1]<math.pi/12;leg_ok=leg<.2
   groups.append(dict(start_group=name,phase=phase,command=int(f[0,0]),trials=int(sel.sum()),tail_body_hold=int(t[sl,sel,4].bool().all(0).sum()),height_gate_all=int(height.all(0).sum()),tilt_gate_all=int(tilt.all(0).sum()),leg_gate_all=int(leg_ok.all(0).sum()),height_tilt_leg_all=int((height&tilt&leg_ok).all(0).sum()),height_m=float(t[sl,sel,0].mean()),tilt_deg=float(torch.rad2deg(t[sl,sel,1]).mean()),leg_mae_rad=float(leg.mean()),physical_q_mean_rad=q[sl][:,sel].mean((0,1)).tolist(),raw_mean_action_mean_rad=act[sl][:,sel].mean((0,1)).tolist(),head_mean_abs_error_deg=torch.rad2deg(head[sl][:,sel]).mean((0,1)).tolist(),known9_reward_rate={k:float(v[sl,sel].mean())for k,v in terms.items()},known9_sum=float(sum(terms.values())[sl,sel].mean())))
 # Zero-head commanded standing fromHOME is already an endpoint failure.
 sel=mode==0;zero_window=slice(100,150)
 home=dict(trials=int(sel.sum()),zero_head_assert=bool((obs[zero_window,sel,51:55]==0).all()),zero_head_body_hold=int(t[zero_window,sel,4].bool().all(0).sum()),initial_raw_action=act[0,sel].mean(0).tolist(),late_raw_action=act[zero_window][:,sel].mean((0,1)).tolist(),late_physical_q=q[zero_window][:,sel].mean((0,1)).tolist(),late_height_m=float(t[zero_window,sel,0].mean()),late_tilt_deg=float(torch.rad2deg(t[zero_window,sel,1]).mean()),late_head_mae_deg=torch.rad2deg(head[zero_window][:,sel]).mean((0,1)).tolist(),same_obs_flag_flip_action_abs_mean_rad=delta[:30,sel].abs().mean((0,1)).tolist())
 results.append(dict(condition=condition,trace_sha256=sha(path),actor_saved_action_parity_max_abs=err,normalizer_flag_mean=float(sd['obs_normalizer._mean'][0,48]),normalizer_flag_std=float(sd['obs_normalizer._std'][0,48]),same_rawobs_flag_flip_action_abs_mean_rad=delta.abs().mean((0,1)).tolist(),from_HOME_zero_head_stand=home,groups=groups))
for k,v in before.items():assert torch.equal(v,sd[k])
oldteacher=json.loads((R/'sitstand-teacher-probe-closeout-002.json').read_text())
out=dict(status='BOUNDED_CPU_STAND_ENDPOINT_AND_COMMAND_AUDIT_COMPLETE',new_gpu_controls=0,new_formal_controls=0,checkpoint=str(cp),checkpoint_sha256=sha(cp),actor_normalizer_unchanged=True,results=results,existing_teacher_evidence=oldteacher['summary'],scope='Recorded640 actualrawobs and realq at qualified rewardboundary; CPUfunctional actor inference checked againstsavedactions. Commandflip is an instantaneous sameinput counterfactual, not a newclosedloop. Nine statelessreward components only, no fullreturn claim.',limitations=['Missing full reward/torque/contacthistory prevents entire-return or dynamiccost rootcause attribution.','Existing1856teacher standing16/16 is previousexternalprobe; seated-to-standing16/16 firsttippedover, so it is not a healthy transitionteacher.','Existingteacher and640 battery are not exactsameinitial/range traces; no causalteacher superiority claim from unpairedruns.','NearHOMEteacher BC/gating is only nextcandidate, no newformalbudget authorized.'],next_minimal_candidate='If same-current-environment teacher check holds: preserve640learnedSIT, add only nearHOME/STAND branch teacheranchor; do not teach seatedtransition or alterreward. Need boundedpaired physicalcheck under actual measuredinputs, teacher0twist/rawhead mapping and safehandoff before anyPPO. This targetsalready-standing endpoint, not claim fullsit-to-stand solved.',script_sha256=sha(Path(__file__)))
(R/'sitstand640-stand-cpu-audit-001.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(status=out['status'],results=[{k:v for k,v in x.items()if k in ['condition','actor_saved_action_parity_max_abs','from_HOME_zero_head_stand','normalizer_flag_mean','normalizer_flag_std']}for x in results]),indent=2))
