# 起身基线 → 随机化 → 完整P04：可复用证据（只读）

本次以实际源码、run/resume记录和冻结报告核对；不把起身的可用配方直接等同其它技能可用，也不以吞吐选择最终路线。

## 真实基线定义

`baseline/paired-feedback-worktree/src/mjlab_microduck/custom_v2/microduck_v2.py:make_v2_baseline_env_cfg` 使用V2 CAD、HOME0、14策略关节+独立mouth、SC0090V5ActuatorCfg（冻结v5，P15/D30，11.1V）、200Hz物理/50Hz策略；起身factory再换全接触V2机器人。并非XL330本体或简单PD动力学。

其动作是原生连续physical `JointPositionAction`（起身scale1），没有SC0090 native-count目标转换、交易延迟或完整反馈残差档案。观察沿官方joint噪声/延迟/encoder-bias，原critic仍物理观察。故它不是完整P04，成功还必须在实际S0/O0目标中独立测。

## 三段实际入口

1. 基础动作：`baseline/head-foundation4-allocation-001/checkpoint_003506.pt`。Official-Baseline、head4、8192×24/E5/M4，从3250再256；自身恢复1792/1792。脚本源 `baseline/head-foundation-worktree`。head4还包含命令微幅曝光和精度函数，不应误称完全官方原奖励。
2. 有效DR：`baseline/paired-feedback-worktree/scripts/run_paired_feedback_trial.sh`，任务 `Mjlab-StandUp-Flat-MicroDuck-V2-PairedError-Baseline`，3506→3762、256更新、50,331,648新增控制、5120 Adam。继承N/actor/critic/Adam/LR1e-5/clock84144及head4，结果 `baseline/paired-feedback-allocation-001/checkpoint_003762.pt` SHA e753856776bb43e44d9465b4067a0b9cd2d10fe03fc8cb88ee4177ec786f93e1。
3. 实际P04续训：`p04/paired-adapt-worktree/scripts/run_paired_p04_adaptation.sh`，`train_v2_standup.py --adapt-from .../resume_003762.pt`，3762→3826、64×32768×24=50,331,648控制，E5/M4仅1280 Adam。actor/critic/N/Adam/LR7.59375e-5/clock90288/stage5/head4精确继承；移除代理噪声和encoder-bias，由真实S0/O0覆盖观察，绝不双重叠加。结果 `p04/paired-p04-adapt-allocation-001/checkpoint_003826.pt` SHA97838820e2ca7a4c9696c9b24118b217d480c8fd7f75ff5ae8530515a1e6da6c。

两段各约50M交互，但PPO优化次数差4倍，不能当等优化量。旧首P04续训使用stream SHA c678dde...（旧有效原始读数前端），后续源为native2 SHA2ffcf872...（修复反馈读数0–1023域）；新技能应沿当前已验证反馈实现，不能原封不动恢复旧缺陷源。原脚本的100k硬准入也不应不加审查复制。

## DR究竟包含什么

`paired_randomization.py` + `error_proxy.py`，bank `training-assets/repaired-paired-errors-v2/error-blocks.npz` SHA d2eaedd5e76d315765f9ceb8f3cdf93baa628d3944b0660331c693822562a0af，形状150528×16×2；每块0.32s，保持块内q/v配对和时间相关性，两通道均20ms延迟。替代原q/v独立白噪声与encoder bias；各环境电压从10.2/11.1/12.4V独立抽样并实际写入BAM vin。

误差源是冻结P04从零1024在修复S0/O0 measured条件下的448条成功恢复模型轨迹，不是新增实机数据。关节/电压混池、无在线q/v/load条件选择、块间不连续、不复现故障，且没有专门主动头命令。它是有效近似DR，不是完整S0等价模型。`paired-feedback-checks.json`有partial reset、一次控制只推进一次、真实BAM电压核验。

**历史没有独立“动作native-count量化适应”阶段。** 这段paired DR仍连续动作，量化/命令链是在切入完整P04时一并引入。后续若分离该阶段，要明确新增转换及预算，不能把它写成已成功跑过。

## 成功与保留缺口

- 随机化3762不续训即在实际raw-P04 nominal/measured各1792/1792、fault0：`p04/baseline-paired-feedback-3762-valid-raw-{nominal,measured}/report.json`。站姿先倒仍约28/256；新种子共同目标报告仍1792/1792，但25/256先倒。恢复终点合格不等于直接承接合格。
- 首完整P04适应3826仍1792/1792、fault0，站姿先倒27/256，**头部明显遗忘**：同measured协议3762→3826 small78→10/128、单128→117/128、两轴351→139/384、四轴53→4/64；末.5s保持11→4、116→115、206→125、12→1。见 `paired-adaptation-head-comparison/report.json`。不能称简单P04续训全面成功。
- 后续精确分桶承接/短片段、冻结N、head cap/精度等另有多段开发。可保留实际P04候选4466：`paired-head-accuracy-allocation-002/checkpoint_004466.pt` SHA f5a9a86b9acd16faeca12a9aefc63e204fff7fe90a7f328b6aad32ce7133159f；共同measured恢复1792/1792/fault0，普通站姿保持0倒（细负pitch仍须独立检查）。这些额外成本不能抹去归给最初64适应。

## 下一技能公平复用边界

先用当前技能同V2/fullcontact/v5/正确端点、同奖励/物理秒数/课程标尺训练B0；改变的只是动作/测量复杂度。B0本身合格后，按该技能实际轨迹覆盖决定DR/量化阶段，再用相同完整P04评测及必要续训。起身bank不自动覆盖低头载荷、坐站或滚转。

比较总取得时间应包含B0学习、DR学习/数据构建、量化与P04适应、准入和共同评测，并分别列优化步数、交互数、采样/学习wall与失败分支成本。上述旧正式run没有统一完整worker wall字段；预检131k/s不是整个路线耗时，不能从它外推。继承元数据可证paired3762基线既有739,639,296控制、首P043826累计789,970,944；不是只花后两段100M就从零完成路线。新试验从启动就统一记账，按同指标达标的时间比较，未达标标删失，不凭P04吞吐好默认所有技能采用P04。

此审阅没有新GPU/PPO、没有改冻结源码。
