"""One bounded CPU review; existing source/trajectories only, no simulation."""
import os,ast,hashlib,json,math,subprocess
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'groundpick-repaired-worktree'
D=Path('/home/l40-vm01/Documents/ai-research/microduck/microduck-v2-p04/docs')
torch.set_num_threads(2)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
run=read(R/'groundpick-allocation-001/run.json');report=read(R/'groundpick-allocation-001/report.json');env=run['environment'];cfg=env['rewards'];ppo=run['agent']['algorithm']
source=W/'src/mjlab_microduck/tasks/mdp.py';text=source.read_text();tree=ast.parse(text)
# Execute only the two pure tensor gate functions, no environment imports.
ns={'torch':torch};fns={}
for node in tree.body:
 if isinstance(node,ast.FunctionDef) and node.name in ('phase_pose_blend','phase_rise_gate'):
  piece=ast.get_source_segment(text,node);fns[node.name]=hashlib.sha256(piece.encode()).hexdigest();exec(piece,ns)
down=lambda p:ns['phase_pose_blend'](p,.375,.425,.8)
up=lambda p:ns['phase_rise_gate'](p,.425,.8)
assert env['scene']['entities']['robot']['init_state']['joint_pos']=={'.*':0.0}
assert cfg['mouth_ground_proximity']['params']['std']==.1 and cfg['mouth_ground_proximity']['weight']==3
assert cfg['mouth_perpendicular_to_ground']['weight']==2
inputs=[R/'groundpick-allocation-001/run.json',R/'groundpick-allocation-001/report.json',R/'groundpick-training-input-contract.json',R/'groundpick-reach-hold-002/traces.pt',R/'groundpick-reach-screen-001.json',source,W/'src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py',W/'src/mjlab_microduck/custom_v2/ground_pick.py']
observed=[]
for stage in ('nominal','measured'):
 p=R/f'groundpick-0128-{stage}/traces.pt';inputs.append(p);saved=torch.load(p,map_location='cpu',weights_only=False);x=saved['measurements'];phase=x['phase'];low=(phase>=.375)&(phase<.425)
 assert bool((saved['initial']['origins'][:,2]==0).all())
 z=x['mouth_height'][low].double();cos=x['mouth_down_cos'][low].double();q=x['physical_q'][low].double()
 mouth=3*torch.exp(-(z/.1).square());orient=2*cos;dz=-6*z/.01*torch.exp(-(z/.1).square())
 observed.append(dict(stage=stage,low_control_samples=int(low.sum()),worlds=phase.shape[1],cycles=3,repeated_worlds_not_independent=True,mouth_mean_m=float(z.mean()),down_cos_mean=float(cos.mean()),physical_head_mean_rad=q[:,5:9].mean(0).tolist(),physical_q_mean_rad=q.mean(0).tolist(),low_mouth_reward_rate_mean=float(mouth.mean()),low_orientation_reward_rate_mean=float(orient.mean()),low_two_task_terms_sum_mean=float((mouth+orient).mean()),mouth_reward_height_derivative_mean_per_m=float(dz.mean()),return_pose_and_upright_gate_max_during_low=float(up(phase)[low].max()),low_goal_samples=int(x['low_goal'][low].sum()),head_contact_low_samples=int(x['head_contact'][low].sum()),self_contact_low_samples=int(x['self_contact'][low].sum()),recorded_fields=sorted(x),source_boundary=report['training_metrics']['boundary']))
physical=torch.load(R/'groundpick-reach-hold-002/traces.pt',map_location='cpu',weights_only=False);pt=physical['trace'];pq=physical['physical_q'];group=physical['group'];refs=[]
for g in (2,3):
 sel=group==g;x=pt[100:,sel].double();q=pq[100:,sel].double();z=x[:,:,0];cos=x[:,:,1]
 neck_return=6*torch.exp(-(q[:,:,5:9]/.15).square()).mean(-1)
 # Per-head-axis derivatives include averaging over4 axes and weight6, gate=1.
 grad=-3*q[:,:,5:9]/(.15**2)*torch.exp(-(q[:,:,5:9]/.15).square())
 refs.append(dict(payload_kg=(g%2)*.03,worlds=int(sel.sum()),time_window='physical direct-target last1s of3s, reward counterfactual at low gate=1/up=0',mouth_mean_m=float(z.mean()),down_cos_mean=float(cos.mean()),physical_head_mean_rad=q[:,:,5:9].mean((0,1)).tolist(),mouth_reward_rate_mean=float((3*torch.exp(-(z/.1).square())).mean()),orientation_reward_rate_mean=float((2*cos).mean()),two_task_terms_sum_mean=float((3*torch.exp(-(z/.1).square())+2*cos).mean()),return_head_at_full_up_gate_mean=float(neck_return.mean()),return_head_absolute_derivative_per_axis_full_up_gate=grad.abs().mean((0,1)).tolist(),bothfeet_fraction=float(x[:,:,8].mean()),body_tilt_mean_deg=float(torch.rad2deg(x[:,:,2]).mean()),linear_max_mps=float(x[:,:,3].max()),omega_max_radps=float(x[:,:,4].max()),self_contact_settled_trials=int((pt[25:,sel,5]>0).any(0).sum()),head_impact_settled_trials=int((pt[25:,sel,6]>0).any(0).sum()),full_fault_trials=int(pt[:,sel,7].bool().any(0).sum())))
# Actual event logs, not initial cfg weight mistaken for executed curriculum.
ea=EventAccumulator(str(R/'groundpick-allocation-001'));ea.Reload();logged={}
for key in ea.Tags()['scalars']:
 if key.startswith('Curriculum/') or key.startswith('Episode_Reward/'):
  v=ea.Scalars(key);logged[key]={'count':len(v),'first':[v[0].step,v[0].value],'second':[v[1].step,v[1].value],'last':[v[-1].step,v[-1].value],'min':min(x.value for x in v),'max':max(x.value for x in v)}
inputs+=list((R/'groundpick-allocation-001').glob('events.out.tfevents.*'))
onnx_path=Path('/home/l40-vm01/Documents/ai-research/microduck/microduck-web/web/simulator/assets/policies/alpha_ground_pick.onnx')
import onnx
model=onnx.load(str(onnx_path));onnxmeta={v.key:v.value for v in model.metadata_props};inputs.append(onnx_path)
oldcommit=subprocess.check_output(['git','show','-s','--format=%H%n%aI%n%B','ffd0b05'],cwd=W,text=True)
controls=report['controls'];T=run['agent']['num_steps_per_env'];E=ppo['num_learning_epochs'];M=ppo['num_mini_batches'];N=run['arguments']['num_envs']
clocks=[]
for n in (24576,8192,4096):
 updates=controls//(n*T);assert updates*n*T==controls
 clocks.append(dict(num_envs=n,rollout_T=T,controls=controls,updates=updates,adam_minibatch_steps=updates*E*M,minibatch_samples=n*T//M,common_control_clock=updates*T,kind='actual' if n==N else 'arithmetic_counterfactual_not_historical_run'))
result=dict(status='BOUNDED_CPU_REVIEW_COMPLETE',formal_training_controls_added=0,gpu_controls_added=0,scope='Single review of frozen source and saved data; no training edits, no new simulation',source_path=str(W),input_sha256={str(p):sha(p) for p in inputs},pure_gate_function_sha256=fns,actual_training=dict(num_envs=N,T=T,epochs=E,minibatches=M,updates=128,controls=controls,common_step_counter=report['common_step_counter'],phase_exposure=report['training_metrics']['exposure_control_samples'],low_goal_all_sample_fraction=report['training_metrics']['metrics']['low_goal_all_sample_fraction'],initial_cfg_action_rate_weight=cfg['action_rate_l2']['weight'],logged_curricula={k:v for k,v in logged.items() if k.startswith('Curriculum/')}),command_contract=read(R/'groundpick-training-input-contract.json'),command_semantics=dict(actual_phase_segments=[0,.375,.425,.8,1],period_s=4.,inherited_rel_forward_envs=.2,rel_forward_effect='none: GroundPickPhaseCommand.compute writes cos/sin directly, _resample/_update no-op',actor_phase_slots=[48,51],actor_phase_scaling='none',actual_eval_obs_equals_native_command=True),reward_audit=dict(low_return_gate_exact_zero=True,default_joint_pose_all_zero=True,low_pose_target='none; current recipe is task-space mouth geometry',all_reward_weights_initial_cfg={k:v['weight'] for k,v in cfg.items()},saved_policy_low=observed,physical_low_reference=refs,analytical_gates=dict(down_gate_time_integral_seconds=1.7,up_gate_time_integral_seconds=1.55,down_two_term_max_cycle_return=8.5,up_legs_head_upright_max_cycle_return=24.8,interpretation='Theoretical maxima only, not observed full returns or proof of reward preference'),limits=['Old evaluation traces lack actions, previous actions, joint velocity, all joint torques, foot orientation, angular momentum and exact entire reward stack; cannot reconstruct full returns or PPO gradients.','Physical direct-target trials and policy trials are not paired initial states; partial task rewards are a source-based counterfactual, not whole-policy causal evidence.','Height reward analytic derivative is nonzero near92mm; this is not action Jacobian or PPO optimizer gradient.','Return head Gaussian can have negligible derivative at deep neck/head offsets, but no trial proves this caused failure.']),optimization_count_comparison=clocks,historical_evidence=dict(published_model_path=str(onnx_path),metadata=onnxmeta,actual_success_run_N_T_updates_curriculum='UNKNOWN: no provenance-linked historical run/config found in bounded local review; ONNX run_path=None and no N/T/update metadata',current_default_T=24,current_default_epochs=5,current_default_minibatches=4,current_default_max_iterations=20000,defaults_are_not_actual_success_run=True,history_reverted_pose_following_commit=oldcommit,design_doc='docs/superpowers/specs/2026-07-24-ground-pick-pose-following-design.md',design_doc_is_not_active_recipe_or_success_evidence=True),findings=['No demonstrated command phase/sign/scale or default HOME error.','Low hold has no return-pose reward; current low task-space objective does not prescribe neck/head branch.','All reset joint poses are HOME despite randomized phase; zero observed healthy low_goal across75,497,472 samples; successful low-state exposure absent in recorded metrics.','Current128 updates are only one third/sixth the optimizer iterations of equal-control N8192/N4096 arithmetic examples; this does not establish historical-run comparison or authorize more budget.'],one_repair_direction=dict(name='phase-consistent verified low-endpoint reverse-reset curriculum',status='hypothesis only, no source modification or training authorized',evidence='Physically healthy low endpoint exists, but current HOME-only resets and frozen trajectory never reach it',constraints='keep task-space rewards/PPO fixed, no forced joint interpolation, preserve original HOME full-cycle acceptance',next_evidence='authorized separate frozen128/64world lowstart probe at phase.375, with real observation/action/target records'),logged_episode_reward_summary=logged,script_sha256=sha(Path(__file__)))
(R/'groundpick-branch-cpu-audit-001.json').write_text(json.dumps(result,indent=2)+'\n')
rows='\n'.join(f"|{x['num_envs']}|{x['updates']}|{x['adam_minibatch_steps']}|{x['minibatch_samples']}|{x['common_control_clock']}|" for x in clocks)
md=f'''# GroundPick 一次 CPU 复核（2026-10-16）

本记录只追加诊断，不改原阶段“未合格”的结论。正式消耗仍为 **75,497,472 controls / 128 PPO 更新**；本 CPU 审阅新增正式和 GPU 交互均为0。冻结源：`{W}`。

## 可以证实的内容

- 命令是真实 `cos(2πφ), sin(2πφ), 0`，原始 actor 槽48:51与每步实际相位相同。GroundPick 的 compute 覆写原速度采样逻辑，继承字段 `rel_forward_envs=.2` 不会生成速度指令。四阶段为1.5s下探、.2s低位、1.5s回正、.8s站立，旧文件顶部“两半周期”注释不是实际配置。
- HOME 14关节为0，回正奖励取实际 default joint pose。低位φ∈[.375,.425)回正腿/头/直立三项 gate 严格为0；未发现低位阶段被 HOME 姿态项直接拉回的证据。
- 活跃方案是嘴端任务空间奖励，没有低位关节目标。已验证低位约 neck+1.99/head_pitch−.861；策略采用 neck{observed[0]['physical_head_mean_rad'][0]:.3f}/head_pitch{observed[0]['physical_head_mean_rad'][1]:.3f} 的另一分支。仅凭正负相反不能认定轴向映射错。
- 名义低位嘴高{observed[0]['mouth_mean_m']*1000:.2f}mm，measured为{observed[1]['mouth_mean_m']*1000:.2f}mm。名义低位两任务项平均{observed[0]['low_two_task_terms_sum_mean']:.4f}，真实低位参考{refs[0]['two_task_terms_sum_mean']:.4f}（空载）；这里只比较可重建分项，不冒称完整return排序。
- 嘴高Gaussian在当前误差下导数约{observed[0]['mouth_reward_height_derivative_mean_per_m']:.2f}/m，没有数值饱和。深低位的回正头Gaussian远离零时导数接近0，可能使回正难发现；未证明它就是PPO失败原因。
- 低位命令已有3,153,796采样（4.177%），但整个训练健康低位命中0。所有reset关节都是HOME，随机相位不能替代正确低位状态曝光。

## 优化与课程时钟

|N|PPO更新|Adam小批步|单批样本|control时钟|
|---:|---:|---:|---:|---:|
{rows}

只有第一行是真实run，另外两行是相同75,497,472 controls、T24/E5/M4的算术反事实，不能写成历史成功run。每个样本的epochs仍相同，但优化步数和课程时钟不同。真实clock3072未到6000/12000；action-rate的run.json初始值−2不代表全程：TensorBoard第0更新汇总−1.45，第1至127更新约−.8。COM课程全程.003。

旧发布 `alpha_ground_pick.onnx` 有XL330 default_joint_pos，`run_path=None`，没有N/T/更新数元数据。有限本地审阅未发现与该成功模型可追溯绑定的run.json，故实际历史成功N/T/更新数/课程时钟**未知**。源码默认T24、E5/M4和max_iterations20,000只能作为默认。历史commit ffd0b05明确撤销pose-following，因为头部入地且回正困难；7月24设计稿不能当当前成功配方。

## 唯一候选方向与边界

提出**相位一致的已验证低位端点反向reset课程**，先补成功低位状态覆盖，暂不同时改奖励/PPO或规定关节直线轨迹。该方向尚未训练，也没有新增正式预算。根代理已单独授权冻结128策略从该低位φ=.375运行4秒的64世界GPU诊断，用真实obs/动作/执行目标确定它是否立即离开正确分支；结果另归档。

旧保存轨迹缺实际动作、前动作、全关节速度/力矩、足姿和角动量，无法精确重建全奖栈或PPO梯度；不能据部分Gaussian推断全局奖励偏好，更不能据等交互比较宣布更新不足是根因。

机器证据：`{R/'groundpick-branch-cpu-audit-001.json'}`。输入SHA、实际分项、TensorBoard摘要、历史元数据均在JSON；原报告/检查点/训练源未更改。
'''
(D/'sc0090-v2-groundpick-cpu-review.md').write_text(md)
print(json.dumps({'status':result['status'],'policy_low':observed,'physical_low':refs,'json':str(R/'groundpick-branch-cpu-audit-001.json'),'md':str(D/'sc0090-v2-groundpick-cpu-review.md')},indent=2))
