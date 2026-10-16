"""One bounded saved-state audit; no environment creation, integration or policy update."""
from pathlib import Path
import json,hashlib,math
import torch,mujoco
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'sitstand-history-base-worktree';torch.set_num_threads(2)
run=json.loads((R/'sitstand-history-base-allocation-002/run.json').read_text());report=json.loads((R/'sitstand-history-base-allocation-002/report.json').read_text());model=mujoco.MjModel.from_xml_path(str(W/'src/mjlab_microduck/custom_v2/assets/robot.xml'))
names=['left_hip_yaw','left_hip_roll','left_hip_pitch','left_knee','left_ankle','neck_pitch','head_pitch','head_yaw','head_roll','right_hip_yaw','right_hip_roll','right_hip_pitch','right_knee','right_ankle'];qi=[model.joint(n).qposadr[0]for n in names];legs=[0,1,2,3,4,9,10,11,12,13];sit=torch.zeros(14)
for k,v in run['environment']['rewards']['posture_pose_legs']['params']['sit_overrides'].items():sit[int(k)]=v
results={}
for condition in('nominal','measured'):
 x=torch.load(R/f'sitstand-history-base-0320-{condition}/traces.pt',map_location='cpu',weights_only=False);t=x['traces'];a=t[:,:,8];flag=t[:,:,9];q=x['qpos_boundary'][1:,:,qi];head=t[:,:,10:]
 assert torch.equal((q[:,:,5:9]-x['actor_obs_preaction'][:,:,51:55]).abs(),head)
 err=q[:,:,legs]-a[:,:,None]*sit[legs];dz=t[:,:,0]-(.112+a*(.075999-.112));cos=torch.cos(t[:,:,1]);pose=torch.exp(-err.square().mean(-1)/.4**2);h=((t[:,:,0]-.075)/.025).clamp(0,1)
 terms=dict(pose=4*torch.exp(-(err/.5).square()).mean(-1),pose_l1=-err.abs().mean(-1),height=torch.exp(-(dz/.04).square()),height_sharp=torch.exp(-(dz/.015).square()),height_l1=-6*dz.abs(),head=.75*torch.exp(-(head/.5).square()).mean(-1),composite=3*torch.exp(-(dz/.03).square())*torch.exp(-(1-cos)/.4**2)*pose*torch.exp(-head.square().mean(-1)/.4**2),upright=2.5*cos,upright_tall=1.5*cos*h.square()*(3-2*h))
 tail=torch.zeros_like(a,dtype=torch.bool)
 for phase in range(3):tail[phase*200+175:(phase+1)*200]=True
 assert torch.equal(a[tail],flag[tail]);groups={}
 for value,label in((0,'STAND_command'),(1,'SIT_command')):
  mask=tail&(flag==value);m=mask.sum();groups[label]=dict(samples=int(m),actual_height_mean=float(t[:,:,0][mask].mean()),tilt_mean_deg=float(torch.rad2deg(t[:,:,1][mask]).mean()),head_axis_mean_error_deg=torch.rad2deg(head[mask]).mean(0).tolist(),known9_weighted_means={k:float(v[mask].mean())for k,v in terms.items()},known9_sum=float(sum(terms.values())[mask].mean()),legs_mean_abs_error=float(err[mask].abs().mean()),physical_body_stable_fraction=float(t[:,:,4][mask].mean()))
 # Body-stable SIT subset uses native original gate, not inferred from tilted low states.
 mask=tail&(flag==1)&t[:,:,4].bool();groups['SIT_body_stable_samples']=dict(samples=int(mask.sum()),known9_weighted_means={k:float(v[mask].mean())for k,v in terms.items()},known9_sum=float(sum(terms.values())[mask].mean()))
 results[condition]=groups
acc=EventAccumulator(str(sorted((R/'sitstand-history-base-allocation-002').glob('events.out.tfevents.*'))[0]),size_guidance={'scalars':0});acc.Reload();rates={}
for tag in acc.Tags()['scalars']:
 if tag.startswith('Episode_Reward/'):
  vs=acc.Scalars(tag);rates[tag]=dict(last_iteration=vs[-1].step,last_weighted_episode_rate_mean=vs[-1].value)
reference=json.loads((R/'sitstand-fresh-reward-and-update-audit-001.json').read_text())['healthy_endpoint_reference'];exp=report['training_metrics']['exposure_control_samples']
out=dict(status='CPU_SAVED_STATE_REWARD_AUDIT_COMPLETE',formal_controls_added=0,gpu_controls_added=0,source=str(W),stage320_saved_states=results,already_saved_fixed_endpoint_reference=reference,actual_stochastic_training_last_reward_rates=rates,actual_training_exposure=exp,scope='Nine stateless components reconstructed from actual saved reward-boundary z/tilt/alpha and exact-parity joints/head. Not full return, not PPO/actiongradient or paired counterfactual.',limitations=['Actual velocity-vector/torques/contact forces and all regularizers are not in frozen eval traces, so cannot establish total-reward inversion or dynamic transition cost.','Healthy HOME/SIT reference comes from existing different fixed-target physical trajectories, not identical state/action histories.','Training exposure counts near-finished ramp, not exactg1; evaluation tails actual alpha=flag exactly.'],conclusion='Healthy HOME known9 reward remains higher than failed low STAND-command states; no demonstrated global preference for low pose. Partial leg/upright/head rewards remain positive while height/fine/composite goal is missed. This supports a remaining action-discovery/credit problem as a hypothesis, not a proven cause or authority to change reward. Current authorized continuation freezes taxes and changes no reward formula.')
(R/'sitstand-history320-reward-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'groups':results,'references':reference,'exposure':exp},indent=2))
