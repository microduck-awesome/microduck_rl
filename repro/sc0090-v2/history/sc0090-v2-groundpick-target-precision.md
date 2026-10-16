# GroundPick 成熟策略高度精度阶段

目标：把已学到的健康下探/回站推进到原嘴高0–25mm、向下余弦≥.95、低位连续100ms和返回站稳500ms门槛。训练中的低位reset命中不能替代完整周期评测，实机未测。

原512完整预算301,989,888已完成，空载嘴约43mm、回站有进展，四节点评测low全0。四组153,600控制的常量008短时接管也未完成低位，不能把静态端点当动态过渡教师。原43–44mm在std.10/target0高斯中仍拿82–83%的接近奖。

本段只改嘴高奖励的std .10→.025m和target0→.020m，其余健康接触门控、负orientation、所有奖励权重、V2物理/P04测量、25%低位reset、4s相位、20s回合和原课程不改。20mm在原25mm门槛内、已有008端点实证；当前43–44mm新奖约40–43%，仍有梯度。只有成熟策略使用这项精度奖励，不把远距离弱梯度配方作为从零发现动作方案。

正式原512→768，固定256更新；N24576/T24/E5/M24，每轮589,824控制/120原生PPO Adam步，新增150,994,944，全Ground累计上限528,482,304。原actor/critic/normalizer/Adam/自适应LR/RNG/clock12288精确恢复，未使用验证权重。配置初始LR.001不代表恢复时实际LR：实载7.59375e−5；clip.2、gamma.99、GAE lambda.95、entropy.01、desiredKL.01、gradclip1均保持。

CPU独立配置差异仅两参数，283资产SHA一致、432组合reward标量oracle和梯度符号通过。真实GPU六相位×fast/full successor、partial reset/载荷力矩检查通过；64×5含32timeout、ONNX最大4.77e−7。实际24576×16含12,288真timeout，7oracle误差最大5.96e−8、反馈/非有限0，容量194,403/1,572,864。验证稳态aggregate103,349.54/median103,509.10控制/s，正式阶段与整worker速度另计。

当前运行worker002/PID556944，源冻结。首worker001外置导入名错误，0GPU/PPO退出并保留；未改生产数学来绕过错误。独立子代理CPU源码复核无阻断项。

四节点576/640/704/768 × nominal/measured原12s/256世界/四载荷/推扰评测，200Hz头/自碰与相机RMS/P95/20ms曝光、全/健康分母保持。新固定seed2026101633与历史2607不同，追加原512在新seed双条件参考307,200诊断控制，等待主worker释放GPU0后执行；比较前核对初态，不跨seed直接归因。CPU收尾队列随后自动复算。

原clean_cycle实现要求整个周期双足接地，报告文字仅列无头/自碰/故障；初始落地导致第一周期clean0。保留原门槛/计数，同时独立给出接触与200Hz筛查，不静默改成功标准。当前low全0不由该差异引起。

机器材料位于实验p04目录：`groundpick-target-precision-plan-001.json`、`groundpick-target-precision-cpu-001.json`、`groundpick-target-precision-independent-review-001.json`、`groundpick-target-precision-source-and-eval.patch`、`groundpick-target-precision-qualification-001.json`。整个固定预算后再据全曲线评价效力，不凭前几轮判定成功或失败。
