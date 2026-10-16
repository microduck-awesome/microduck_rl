# SitStand 正确坐姿慢速过渡与中间腿模板释放

当前：唯一过渡模板释放64更新及nominal/measured原12s末评均结束，worker PID233673已exit0，GPU1释放。新增25,165,824、SitStand累计 **176,160,768 正式控制**，到达本次明确上限，不再自动续训。两条件四组×三phase身体tail与head到位均0，健康相机分母0，仍未合格。原384和新448保留，不能因没有feedback故障或软件PASS称技能成功。完整收尾为数据盘 `sitstand-transition-release-closeout-001.json`。

## 单批正确 HOME↔ClearSeat 慢速探针

- `sitstand-slow-roundtrip-001`：GPU1 PID209499，exit0。64世界×20s，2s/4s平滑过渡各32（每种HOME/SIT起点各16）；0–3s初始保持、3s起过渡、7–10s共同对端保持、10s反向、14–20s最终保持。诊断64,000控制，PPO0。
- 原Lesson4数值源前后SHA一致；真正R1三电压且BAM电压同步，S0/O0、原动作/native观测时序保留，1000次动作前命令断言通过。控制目标仅外置诊断，未写奖励/教师。
- 初始最后1s健康：2s HOME16/16、SIT16/16；4s HOME16/16、SIT15/16。四组过渡后/最终保持均0/16；全程head found64/64，self0，feedback0、nonfinite0。头冲击峰分别47.33/45.81/52.96/45.53N。
- 首次头触地距命令：2s两向中位0.98s，4s HOME→SIT1.57s、SIT→HOME1.52s。首次倾角>15°中位0.60/0.69/1.11/1.22s。触地前腿物理q对实际O0目标MAE中位.0207/.0279/.0187/.0229rad；不是没有发出目标。
- 原目标→O0量化最大误差.0026944rad；小目标变化未改变count约44.8%（2s）、60.0%（4s）。这是格点行为，不单凭此归因机械死区。processed target与action逐位相同。
- 初始最后1s健康子集相机RMS .02458/.002493/.02297/.002909rad/s；20ms转动跨度P95均值 .000471/.0000484/.000474/.0000702rad，分母16/16/16/15。末段健康分母全0，倒地后低相机角速不算稳像成功。

## 重心与支持边界

- 五固定几何截面无自碰；root强制直立假设下α=.5/.75双足投影COM余量−11.6/−20.3mm。该假设不是动态轨迹，不证明任务不可行。
- 重要补正：真实ClearSeat静态姿态的CPU接触重建显示两侧shin结构同时着地，原SitStand允许膝/下腿支撑。仅足接触点域COM中位约−42mm；加入非头下腿接触后约+11.7mm。不能把足域不足说成整个坐姿失去静力支持。
- 对保存全轨迹每100ms（12,800状态）做一次CPU几何重建，未积分/未新候选；以同qpos、实际随机化mass/ipos重算COM，与接触几何同边界。原GPU派生COM比qpos滞后5ms，旧混边界分析文件另存保留。CPU重建不是GPU接触逐位重放，未恢复实际压力/COP。
- 这一批只否定固定关节插值路径，不能否定可学习的重心转移/暂态腿偏离。

## 实际奖励审计与唯一修正

- 原 `posture_pose_legs`：权重4、std .5；`posture_pose_l1`：权重1。两者全程追踪alpha×SIT，无过渡完成或健康门控。`posture_composite`权重3，height/upright/pose/head乘积，pose std .4，同样追踪中间关节模板。
- 高度宽/细Gaussian权重各1，std .04/.015m；高度L1权重6，原2s线性高度slew。`posture_stillness`权重2已要求|flag−alpha|<.02、速度/高度/倾角门控，不是中途静止收益。
- 保存轨迹的10个无状态分项可重建；原5项runtime oracle通过。2s过渡pose奖励仍3.908/3.952（上限4），跌倒末态pose仍约3.97–3.99，而composite近0。没有完整角动量/扭矩/原加速度历史，因此不称完整return反向或所有失败由该项造成。
- 新配方仅在过渡释放中间腿模板：x=clip(1−|flag−alpha|/.1,0,1)，g=x²(3−2x)；Gaussian和L1乘g；composite的pose因子改为(1−g)+g×pose。native alpha完成时g=1，原三项逐位一致。高度、upright、温和升降、head、endpointstillness、PPO/DR/动作/测量均保持。
- 原384父模型，不继承验证权重；GPU0活动RNG明确映射GPU1，actor/critic/N/Adam/LR/clock精确恢复。比较clock-fixed384可观察行为改变，但无同父自然续训控制，不能声称排除额外训练贡献。

## 准入及继续规则

真实新worktree `sitstand-transition-release-worktree`。task `Mjlab-SitStand-Flat-MicroDuck-V2-ClearSeat-TransitionFree-Lesson4s-Repaired-S0-O0-R1`。CPU配置/端点逐位/边界与符号/独立oracle→成熟64×5/32timeout/官方ONNX→实际N16384×16→正式384→448单64→原256world×12s corrected-clock名义/实测电池与200Hz相机。8项容量告警、反馈、非有限、PPO护栏保持。吞吐10万目标分稳态与完整wall如实报告，不仅因轻微性能噪声无限改造或阻塞；未通过正确性不得正式。

关键记录：`sitstand-transition-release-plan-001.json`、`sitstand-transition-release-cpu-001.json`、`source-transition.json`、pipeline状态和逐段run/status/report。所有失败、原384及慢路径证据保留。真实硬件尚未可用，不宣称sim2real验收。


## 准入实测与非数值续接

CPU9项PASS（原所有MDP函数/class字节不变、只新注册3个奖励入口；512随机状态两端3项逐位等价），成熟64×5/32timeout/ONNX9.54e−7、8oracle最大5.96e−8。实际16384×16=6,291,456验证控制，16,384真实timeout，contact122119/1048576，feedback/nonfinite0，oracle最大7.45e−8，maxKL.016108/clip.229286。稳态median95858.34/aggregate94814.79，内部构建+验证77.254s；未达到10万。

原worker仅100k性能assert停止，原报告和失败状态保留。父级明确性能目标不作为唯一阻塞；续接002核对原源/CPU/smoke/actual/RNG/PPO证据并签资格，从原384父正式64，不重复准入。续接001在导入且0正式时发现外置诊断JSON需读metrics子键而非平铺，已停止留receipt，002仅修该核验读取；训练数值源未变。


正式已实际410，新增10,223,616控制，clock9840；恢复原384/9216、actor/critic/N/Adam精确、LR.0004444444444444446。反馈/非有限0；健康训练sample约.000220，不能当独立行为通过。读数快照仅供续接，实际进度以正式status为准。


## 448 最终结果与停止

- 正式64全部完成，actor/critic/N/PPO Adam/LR/clock从原384精确恢复，114,688真实timeout，feedback/nonfinite0，contact峰122274/1048576，max update KL .0167323、clip .229798，容量日志无告警。
- 正式phase含环境构建330.0748s/25,165,824=76,242.8controls/s；稳态median97,847.5、aggregate96,427.7。新续接worker含imports+正式+双条件末评约461.497s，不能把稳态当全程。
- 名义/实测各256×12s：四初态/目标组、三phase的身体末500ms保持及head同时到位全部0。站姿起点各0/64超过45°，但站目标尾倾角约22–23°且头平均误差约36°；坐目标停高约92.6–94.4mm，未到75.999mm。坐姿起点两组各64/64曾超过45°，尾高约63.3–63.9mm、tilt约66–67°。
- 两条件所有相机阶段的连续身体健康分母0，原始200Hz RMS/20ms转动跨度保留在报告中，不改判头稳。
- 相对root clock-fixed384，两条件初始qpos/qvel/raw actor obs/origins逐位相同；因此评测初态可比，但没有同父64自然续训控制，不能隔离奖励修正的全部因果贡献。新动作未达到健康保持门槛，只能说这次单配方/预算不足以解除现有失败行为，不能据此宣布任务物理不可行。
- 数值源资格结束复核逐SHA不变，原131个CAD资产相同。保留 `sitstand-lesson4-allocation-001/checkpoint_000384.pt` 与 `sitstand-transition-release-allocation-001/checkpoint_000448.pt`，两者均非合格部署候选。未追加路径、PPO或预算。
