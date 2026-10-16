当前：SitStand历史基础320 + Hold320到640及所有固定末评均已结束，原PID356135 exit0。576/640两条件坐姿末保持384/384，但站姿0/384、严格头响应0/672，仍未双向合格。全部SitStand正式478,150,656封段，无自动第三段。GPU1经root GroundPick005后已转Ball健康锁存准入PID397198/session30107。

## 历史阶段记录（下文旧当前行仅代表当时状态）

# SitStand 历史基础320后课程保持续段

当前真实阶段：Hold320 正式续段运行，PID356135/session49554，已到610/640（最新status），本段新增114,032,640控制；每更新实际课程冻结断言通过。原HistoryBase320及其十项末评已完成，原GPU窗口等待也已结束。正式到640后自动448/576/640双条件评测，无自动第三段。随后GPU1先交root GroundPick005，Ball GPU资格等待该窗口结束。详见 [Hold320续段](sc0090-v2-sitstand-history-hold320.md)。

授权：从正式HistoryBase320精确续320到640；N16384/T24/E5/M16不变，新增125,829,120 controls/25,600 Adam。全SitStand历史上限478,150,656；本fresh谱系末251,658,240。只有这一续段，无自动第三段，中期站立0不触发换方；数值/容量/反馈护栏保留。

源：数据盘 `sitstand-history-hold320-worktree`，任务 `Mjlab-SitStand-Flat-MicroDuck-V2-ClearSeat-HistoryHold320-Repaired-S0-O0-R1`。唯一数值变更是冻结所有后续课程到320末实际生效状态。脚本 `run-sitstand-history-hold320-001.py`，PID356135/session49554；状态` sitstand-history-hold320-pipeline-status.json`，计划` sitstand-history-hold320-plan-001.json`。

## 实际课程状态与证据

直接live记录：action_rate−.6、descent20、torque_rate−.0005、rise0及全部其它奖励原值；head范围±(.39,.39,.49,.11)。原生课程日志：COM/headCOM各.01、push幅.05；按原函数唯一匹配阶段还原对称事件范围，并保存证据，**不是按clock7680重算**（那会错误提前到action_rate−.8）。receipt为`sitstand-hold320-live-parent.json`。

训练cfg取消后续curriculum，并设置上述实际20项权重/头范围/COM和push范围；奖励函数、SIT/HOME、head UniformPose采样方式、物理秒数、冻结反馈/动力学、PPO全部不变。play cfg逐项保持原HistoryBase play配置，不把新训练难度放入独立共同评测。

原父320 actor/critic/normalizer/PPO Adam/LR/CPU与活动CUDA RNG/native clock7680精确恢复；不加载验证训练权重。CPU差分→成熟64×5→actual16384×16→连续320自动接，64更新保存384/448/512/576/640不中断物理，末448/576/640原12s nominal/measured。每个训练更新断言live manager所有权重/头范围/COM/push保持，实际optimizer.step每更新80。

## 临时GPU1交接

worker356135在导入45秒、0采样/0更新时按root窗口指令SIGSTOP，保留原进程。root借GPU1运行PID358643直行标定闭环；外置CPU watcher `wait_sitstand_hold320_gpu1_window.py` / session22901 等待指定marker终态且root /proc消失，自动SIGCONT原356135。没有重启/热改源，整个worker wall会包含等待；正式阶段计时在恢复后，等待不混成训练吞吐。

## 一次只读奖励审计

`sitstand-history320-reward-audit.json`重建实际320保存尾迹9项无状态奖励，并使用旧真实健康HOME/SIT端点参考。STAND命令尾高79–80mm、neck误差62–64°，已知9项总6.03–6.13；腿Gaussian仍2.71–2.73/4，upright约2.36/2.5、head均奖.538/.75，但composite仅.042/3。健康HOME同9项13.69、SIT12.20；现身体合原门槛SIT9项11.43。说明局部正奖仍多，不证明完整return偏好坏姿态；扭矩/全速度/接触正则未完整保存，不能重建动态全回报或定因果。实际训练stand_hold49,560,491、sit_hold47,876,290、rising13,316,221控制，站命令并非缺曝光。当前续段仍原奖励，未加入L1/EMA或其它新项。

## 准入完成与正式接续

root闭环marker完成且PID358643退出后，CPU watcher于1792133259.36自动SIGCONT原356135，receipt保留，未重启。CPU精确差分PASS；成熟64×5恢复actor/critic/N/Adam/LR/clock精确，32timeout、ONNX1.19e−6、contact476/4096、5物理oracle全0。actual16384×16：6,291,456验证控制、8192timeout、反馈/非有限0、contact108165/1048576、maxKL.01559/maxclip.22755；median92,152.72/aggregate92,000.39controls/s，未达100k。正式自动进入`allocation-001`，原320父/clock7680/LR.00038443359375，验证336权重未继承；每更新实时断言冻结难度完全保持。

## 最终640固定条件曲线

父320→448→576→640：SIT末.5s保持nom324/360/384/384、measured330/358/384/384（各384重复trial-phases）；STAND始终0/384、严格head到位与尾保持始终0/672，fault全0。四节点同条件初态qpos/qvel/rawactorobs/origins逐位exact，cfg仅资产根路径归一后相同（283资产逐SHA验证）。

640站指令尾高约92–106mm，较320的79–81mm提高，但倾角约15–20°、头误差大，未达到完整站立门槛。坐稳零头保持窗口健康分母两条件384；平均单trial相机RMS .022682/.045234rad/s，平均单trial20ms曝光P95 .000642/.001421rad。这是稳定子集角运动而非整体响应通过。

新增125,829,120、谱系251,658,240、全SitStand478,150,656。原生Adam累计51,200、N count251,658,240，clock15360。正式phase1431.033s/87,928.88controls/s；稳态88,190.04/88,536.16，未100k。196,608timeout、反馈/非有限0、contact155869/1048576，所有live课程冻结断言通过。完整worker2191.991s包括已说明的root临时借GPU1等待。source/候选384/448/512/576/640与原报告保留；`sitstand-history-hold320-closeout.json`和`...curve-review.json`归档，无自动第三阶段。
