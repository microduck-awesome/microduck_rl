# GroundPick 一次 CPU 复核（2026-10-16）

本记录只追加诊断，不改原阶段“未合格”的结论。正式消耗仍为 **75,497,472 controls / 128 PPO 更新**；本 CPU 审阅新增正式和 GPU 交互均为0。冻结源：`/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04/groundpick-repaired-worktree`。

## 可以证实的内容

- 命令是真实 `cos(2πφ), sin(2πφ), 0`，原始 actor 槽48:51与每步实际相位相同。GroundPick 的 compute 覆写原速度采样逻辑，继承字段 `rel_forward_envs=.2` 不会生成速度指令。四阶段为1.5s下探、.2s低位、1.5s回正、.8s站立，旧文件顶部“两半周期”注释不是实际配置。
- HOME 14关节为0，回正奖励取实际 default joint pose。低位φ∈[.375,.425)回正腿/头/直立三项 gate 严格为0；未发现低位阶段被 HOME 姿态项直接拉回的证据。
- 活跃方案是嘴端任务空间奖励，没有低位关节目标。已验证低位约 neck+1.99/head_pitch−.861；策略采用 neck-0.387/head_pitch1.126 的另一分支。仅凭正负相反不能认定轴向映射错。
- 名义低位嘴高91.89mm，measured为92.06mm。名义低位两任务项平均3.1656，真实低位参考4.9478（空载）；这里只比较可重建分项，不冒称完整return排序。
- 嘴高Gaussian在当前误差下导数约-23.69/m，没有数值饱和。深低位的回正头Gaussian远离零时导数接近0，可能使回正难发现；未证明它就是PPO失败原因。
- 低位命令已有3,153,796采样（4.177%），但整个训练健康低位命中0。所有reset关节都是HOME，随机相位不能替代正确低位状态曝光。

## 优化与课程时钟

|N|PPO更新|Adam小批步|单批样本|control时钟|
|---:|---:|---:|---:|---:|
|24576|128|2560|147456|3072|
|8192|384|7680|49152|9216|
|4096|768|15360|24576|18432|

只有第一行是真实run，另外两行是相同75,497,472 controls、T24/E5/M4的算术反事实，不能写成历史成功run。每个样本的epochs仍相同，但优化步数和课程时钟不同。真实clock3072未到6000/12000；action-rate的run.json初始值−2不代表全程：TensorBoard第0更新汇总−1.45，第1至127更新约−.8。COM课程全程.003。

旧发布 `alpha_ground_pick.onnx` 有XL330 default_joint_pos，`run_path=None`，没有N/T/更新数元数据。有限本地审阅未发现与该成功模型可追溯绑定的run.json，故实际历史成功N/T/更新数/课程时钟**未知**。源码默认T24、E5/M4和max_iterations20,000只能作为默认。历史commit ffd0b05明确撤销pose-following，因为头部入地且回正困难；7月24设计稿不能当当前成功配方。

## 唯一候选方向与边界

提出**相位一致的已验证低位端点反向reset课程**，先补成功低位状态覆盖，暂不同时改奖励/PPO或规定关节直线轨迹。该方向尚未训练，也没有新增正式预算。根代理已单独授权冻结128策略从该低位φ=.375运行4秒的64世界GPU诊断，用真实obs/动作/执行目标确定它是否立即离开正确分支；结果另归档。

旧保存轨迹缺实际动作、前动作、全关节速度/力矩、足姿和角动量，无法精确重建全奖栈或PPO梯度；不能据部分Gaussian推断全局奖励偏好，更不能据等交互比较宣布更新不足是根因。

机器证据：`/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04/groundpick-branch-cpu-audit-001.json`。输入SHA、实际分项、TensorBoard摘要、历史元数据均在JSON；原报告/检查点/训练源未更改。

## 低位物理证据口径纠正（冻结128诊断001）

此前3秒固定目标物理证据的“head impact=0”只表示超过1N部分为0，不能称无头地接触。新64世界预保持显示全部原低位门槛均通过，唯独head_contact found=1：实际0.663–0.781N；CPU确认为头壳接地。旧轨迹未保存found，原件保留；“健康低位已验证”的推断撤回。原正式阶段仍75,497,472步且未合格。唯一后续诊断002只将neck目标−.06rad增加净距，保持严格no-contact gate，不改训练源或开始PPO；实际结果另记。

证据：`groundpick-physical-qualification-erratum.json`、`groundpick-lowstart-contact-audit.json`。

## 低位诊断最终收尾：不进入新PPO

001/002各64世界，1s固定目标预保持+4s冻结128策略原生周期，共新增32,000诊断controls；**正式仍75,497,472 / 128更新，余42更新未花**。两进程均exit0，GPU1已交还，无active PID。源码SHA和actor/normalizer逐位不变，真实actor相位观测/执行目标已保存，未建新训练源。

002仅将neck固定目标减.06rad；同64保存qpos的CPU净距最小6.863mm、嘴15.912–16.376mm/down≥.999843都过几何门槛。但真实预保持至.10s时64/64重新头触地；尾200ms嘴11.79–12.20mm、头力.612–.814N。其它低位门槛全64通过，仅no-head-contact全0。因此两次起点均不合格，不能用“first exit=0”声称策略从健康低位立刻离开。

002冻结策略随后在站立phase持续500ms通过48/64，完整clean cycle0，fault0；这只是恢复诊断，不补判拾取成功。200Hz相机/20ms曝光按真正站稳末500ms子集另列在 `groundpick-lowstart-final-review.json`，低位健康分母为0。分配的30g负载在初始φ=.375按原任务尚不施力，φ=.425后才递增，不应称带载低位保持。

确认的奖励/评测谓词缺口：head impact只惩罚>1N部分，而评测拒绝任何head_contact found；约.7N头撑在该罚项成本为0。未重建全奖栈，不能因此声称全局奖励偏爱头撑，也未授权另一奖励配方。唯一端点reset候选因为动态健康准入失败而关闭；不继续姿态搜索或花剩余预算。

完整机器结果：`groundpick-lowstart-final-review.json`；当前状态：`groundpick-agent-current.json`。原report/traces/checkpoint不改。

## 003动态结果：端点探查关闭，未合格

唯一静力最近候选3s实际批次已exit0。空载32/32无头地接触，尾1s嘴8.51–12.62mm、down≥.99837、双足和速度/姿态门槛均过；但是32/32自接触。30g组嘴7.94–10.41mm、down≥.99961，同样32/32自接触，另12/32有≤.190N头地接触。两组尾1s完整low_goal都是0/32，fault和非有限均0；健康低位相机分母0，不作稳像通过声明。

保存末态CPU原CAD接触复现：全部64世界为左右各自 `ground_hip_collision ↔ foot_collision`，穿深约0.067–0.473mm；不改碰撞掩码、不忽略该接触。实际固定目标的关节与物理平衡位置有偏差，这个CPU几何候选因此没有转成健康动力学端点。实际随机质量.5888–.6128kg、body_ipos/inertia、每步COM/qpos/qvel/真实动作链都已保存。

按约定结束，不再候选扫描或追加GPU。三批共41,600诊断controls；正式仍 **75,497,472 / 128更新**，剩余42未花。GPU1释放，无active PID。`groundpick-endpoint-probe-closeout.json`是此次后续诊断最终状态，`groundpick-agent-current.json`已同步；原正式128策略保留且未合格。
