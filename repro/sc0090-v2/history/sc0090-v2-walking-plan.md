# SC0090 V2 步行准备与训练计划

## 用户最新优先级：历史遥控品质与官方VelStand v8

2026-10-16用户澄清：头控与头部稳定性仍是重点，所有重点均有阶段总预算；平台期不无限追加。步行共同目标包括头部响应/世界坐标稳像，以及完整步幅、合理步频、双向/组合转向、启停反转、零速不持续跺脚。官方VelStand v8已锁定实际模型/3750776源码；迁移及后续训练顺序见[VelStand计划](sc0090-v2-velstand-plan.md)。

当前walking阶段已完成到512，正式累计402,653,184控制（fresh256+YawFine128+同源128）；512统一复盘步态、线速、正负转向、零速脚步、头控和200Hz相机稳定，不能每完成一段就把同一问题预算归零。保留常规头控训练；不在本段另开头部奖励变体。主动头命令与静稳阶段分开解释，转向时有意的方向变化不直接等同抖动。

已重新核对的历史证据（均为旧本体仿真，不能直接当V2已通过）：

- [完整腾空配方对照](../../microduck_rl/docs/sc0090-air-time-trial.md)：恢复0.125–0.300s、weight3、不按低速衰减，且保持零命令不奖励踏步；配合横向瞬时容差，约.11m/s时双脚落脚频率10.13→4.10/s、同足步幅2.13→5.25cm。仍有高速偏航和侧移不足，不能只摘成功数字。当前V2已保留完整腾空配方，不重复降为.075s/1.5低速衰减。
- [碎步诊断](../../microduck_rl/docs/sc0090-gait-quality.md)：只调高腾空下限，或只关熵，并未消除碎步；不能将低速速度跟踪通过当步态自然。足部事件要排除单帧接触抖动，区分双脚落脚频率和同足跨步，统一视角与1×视频。
- [成功遥控入口及保留问题](../../microduck_rl/docs/sc0090-teleop-current-comparison.md)：A_fresh@3037行走与独立站立策略、编码器交接及当前朝向再起步的组合，实现完整停稳后无再次碎步。新头控目标50更新候选的固定扰动回归25/25移动正确、25/25停稳无脚步复发。成功来源包含两策略交接，不能误记为行走单网络已经学会全部停止。
- [遥控优先级](../../microduck_rl/docs/sc0090-teleoperation-priority-2026-09-19.md)：零请求下持续5–6次/s踏步属于明确问题；左右脚开合、持续髋限位、短程走直、停车偏航/停距与重启都要保留。允许平衡微调/必要补步，不用僵硬锁腿或牺牲转向/再次启动换静止分数。
- [停止模仿失败](../../microduck_rl/docs/sc0090-original-walk-stop.md)：动作MSE降低和冻结行走主干都不能替代闭环停止验证；首次停止就倒的候选不能靠追加同配置训练宣布可用。

当前V2八秒共同电池只提供短时move/stop/reverse/return证据，零速漂移、偏航与头部误差尚在，不能替代历史34/80秒连续遥控与长时间静止检查。后续延用完整连续操作、首次失败锁存与脚步复发指标；仅在读取已有轨迹/必要补测后决定最小奖励或曝光调整，不因完整清单另开无限诊断。


更新：2026-10-16 06:28，Asia/Shanghai。fresh256已完成，YawFine256→384正在正式训练，步行全命令与实机尚未验收。

## 2026-10-16 06:22 第一段完成及下一段决定

第一段256实际完成201,326,592，整进程96575.78控制/s；65/129/193/256全部nominal/measured已完成，结果与失败详见主台账和`walking-discovery-evidence-summary.json`。256前进较好但转向、停止未满足任务，不能只看336/336生存。

后续单因素YawFine保留stock角速度项，另加历史yaw-only Gaussian2/std.35；物理奖励读取相同body yaw数据。PPO/命令/DR/head范围和归一化不变，256→384/128结束在500课程之前。新源`walking-yaw-fine-worktree`，三文件迁移清单及父SHA强制核对；12CPU/成熟64×5通过，actual32768×16已PASS（median117175.99/aggregate117079.05控制/s，max mean KL .016133/clip .234094），队列已启动正式PID3734661。已有自然20s时间边界由首段256覆盖，本次奖励变化不重跑长时间物理预检。共同评测使用原物理/命令任务；新额外奖不计入旧评测分数。没有等预算无改动控制，不声明完全因果。

当前只修转向奖励，idle缓慢前移、停止/反向承接、侧移与全部head/body命令还需独立推进。body_pose_tracking当前仍0，最终身体命令响应不能据现有zero-body电池宣布合格。

## 路线与范围

原全策略目标的下一技能。已有独立P04及基线根完整起身保留候选；GPU0的起身头控有界对照收尾后进入SitStand，GPU1承担步行。后续默认完整修复P04，基线仅起身/站立比较。此为现有从零/吞吐证据支持的工程选择，非实机路线优胜结论；仍无轮滑、P4/host/Web部署或固件工作。

实际独立git worktree：`/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04/walking-repaired-prep-worktree`，base070d606，包含实际验证的P04反馈/动力学及历史步行修复。任务`Mjlab-Velocity-Flat-MicroDuck-V2-Repaired-S0-O0-R1`及Measured。旧任务与rise factories未热改。

## 已核实的配方与依据

| 参数 | 当前值 | 意义与后续决策 |
| --- | --- | --- |
| 环境数N | 32768 | 8192约6.99万、16384约9.49万控制/s未达目标；32768足以摊薄流水线开销，但增大每次更新数据量，不能把环境越多当质量越高 |
| 采样T | 24=0.48s | 沿用本地官方步行配方；前段起身T12等预算对照无综合收益，未任意移植 |
| 批量/小批 | 786432 /196608 | 每更新5epoch×4minibatch=20次优化，样本访问3,932,160；真实学习控制只计786432 |
| PPO裁剪/目标KL | .2/.01 | 控制策略偏移；记录实际pooled KL/clip，非只看loss |
| 初始LR/调度 | .001/adaptive | fresh Adam按实际KL调节；恢复时继承真实Adam/LR，不重置至初始值 |
| gamma/lambda | .99/.95 | 保留官方回报与GAE配方；超时使用重置前successor critic，真正终止优先 |
| entropy/std | .01 /Gaussian初始1 | 发现阶段保留探索，14关节统一官方分布；不套成熟头cap或低熵 |
| actor/critic | 512/256/128 ELU，动态归一化 | actor61/privileged critic76/动作14；不载入起身近零twist统计，不冻结零计数N |
| episode/频率 | 20s；50Hz控制/200Hz物理 | 控制步吞吐不乘4当物理子步；真实episode年龄不随机伪造 |

命令固定vx±.4m/s、vy±.3m/s、yaw±1rad/s，3–8秒重采样；初始standing2%、显式原地转向15%、forward桶20%。实际曝光按执行到的命令GPU累计，不能只报告配置概率。head从零/单/双/四轴都有曝光，zero=.5；初始±.05/±.05/±.07/±.015rad。body保持小非零曝光。

保留历史完整腾空奖励weight3、窗口.125–.300s及2cm足离地目标，linear/angular tracking各2（std√.1/√.5）、upright2（std√.05）、head tracking2（std.5）。发现阶段action_rate−.1，head_bias0，body_pose_tracking0；.112m已写入身体高度参考，但当前该项权重0，不能称它已约束步行站高。head_bias仅新无门控步行入口使用当前命令重基准EMA，旧默认逐位保留，rise仍门控不rebase。

阶段时钟：500更新开始加平滑、扩大head/CoM及standing比例；600加head DC bias，后续750/1000/1250/1500/2000阶梯依原配方。首段256不会跨首次500升级，依据独立行为再判断是否及时升级或修课程，不机械跑满5000。V2 CAD/HOME0、冻结v5 M6、native count反馈、spike=False/O0局部历史、compiled/fused/CUDA graph固定。

## 已完成准入

1. 7项配置/旧奖励等价CPU测试；通用64×5、32timeout bootstrap、ONNX差1.788e−7、物理及测量graph验证通过。
2. 原8192进程成功但通用smoke没给fresh Gaussian写诊断，准入队列因此失败；旧结果/日志保留。只修接线，已有同算术/RNG诊断helper未改。fresh动态统计首批分布并非严格一致，已如实报告。
3. 原实际N32768×16 median113286.22、aggregate112768.37控制/s；加入专用指标后32768×16 median111549.79、aggregate110862.00，maxmeanKL .015473/maxclip .223129，384次线速/角速/头部reward oracle零差。8192/16384不被选用。
4. 专用trainer64×5 PASS，32真实timeout bootstrap、120oracle零差、反馈故障0。4项CPU测试锁定跌倒锁存、命令覆盖分类、持续到位/失败censor和命令切换时刻。
5. 独立评测nominal与measured各21工况×2、400控制、1600相机子步PASS，actor/normalizer冻结、奖励oracle零差。5更新诊断模型多数只是站住，指令移动/位移几乎为零，明确不是走路成功。
6. 同源resume5→7 PASS，actor/critic/N/Adam逐位恢复、真实LR/clock120→168正确、仅3072新增控制。导出84个实际nominal/measured actor观测CPU核对max7.8231e−8，normalizer已烘焙。外置核验器起初错误传入未启用的CNN/RNN配置，两个报错与修复留档；未改训练/模型/导出。

统一准入证据：`walking-final-qualification.json`，包含源/模型assets/输入报告SHA。短验证交互全部单列，正式预算审计明确按allocation_kind排除，不能用N大就算正式。

## 正式预算与当前进程

`walking-discovery-allocation-001` / process PID3625334 / session44077；fresh随机初始化、256更新上限=201,326,592计划控制。启动05:22:41，启动不等于已完成更新。`run-walking-discovery.py`外置记录真实整进程wall time；trainer内部wall字段从建env前开始，不含module imports，不能用它冒充整进程速度。

为了摊薄初始化，采用单个驻留进程。原生保存是0-based：model64/128/192的元数据分别为真实65/129/193完成更新，最终checkpoint256；按元数据计账，不按文件名猜。queue27226/`watch-walking-discovery-evaluations.py`在每点跑nominal/measured，各21工况×8×2seed，预留GPU1至少4500MiB才启动评测。只读检查点，可能与训练并发，完整wall与吞吐如实包含竞争，不宣称独占benchmark速度。

每更新finite损失、KL>.5或clip>.8触发保存停止；无自动下一段或课程硬化。根据中间行为读数可提前停止诊断，不能靠持续延长预算掩盖无位移平台。

## 行为评测与限制

使用同一`walking_metrics.py`，在reward之后、reset/forward/下次命令之前采样；派生几何双方保留明确5ms延迟，物理指标不进入actor。21工况包括正反向、侧向、原地转向、曲线、对角及边走边头控；1s idle→2s move→1s stop→2s reverse→2s return。保留checkpoint时钟对应reset/startup DR，禁interval pushes和actor人工corruption；nominal/measured分别报告，完整配置落盘。

失败从第一控制锁存（height<.08或tilt>45°或nonfinite/feedback-invalid），禁自动reset，恢复后的好帧不能抹掉失败。记录body-frame速度/yaw误差、实际世界位移、head误差/响应/1°联合生存保持、姿态、50Hz足接触/腾空/双支撑/足滑、同足相邻落地步幅与200Hz相机运动。速度持续到位带是明确的工程诊断，不冒充用户验收标准；响应失败保持censored，不填0或8s。50Hz接触力P95不是完整冲击峰值或冲量，短于20ms的接触事件可能漏采，后续成熟步态需补物理子步冲击核对。

当前验证不覆盖粗糙地形、持续推搡、跨策略切换、全部最终head范围或实机表现。先形成有效步态，再按数据补齐；不以低抖动但站住的模型宣称完成。
