# V2 运动相机稳像：历史证据迁移与有限试验

## 2026-10-16 10:53 相机准入故障已归因并继续；GPU1正式修正进行中

实际异常world558/step6160是head_yaw原生1033count、reason13/status−1、transaction valid，numeric/nonfinite0，发生在第一PPO更新之前。控制边界物理yaw2.7989rad、估计2.7480rad；源采样更早，估计不是真值。不夹造count、不改S0/O0或episode终止，也不将旧40°压力测试的其它故障一并归因。证据`velstand-camera-feedback-native-range-attribution.json`。

已移植历史验证过的只读native-range分类，只放行该原因进入实验计数；两真实捕获与6负夹具通过，正常有效count/坏包/数值故障均不能冒充已解释事件。每臂实际N16最多8次，正式64最多16次，未知/非有限仍即停并保存。新manifest`velstand-camera-source-transition-002.json`仅增加camera项/采样和只读故障审计；原已通过的camera算法、PPO、BC、物理不变，不重复物理/64smoke。session73340/PID76016依次执行mild/control成熟24576×16，正式0；源冻结。

GPU1 BallKick实际N16完成9,437,184验证controls、24,281timeout、五项oracle最大1.43e−6、feedback/nonfinite0；稳态86,603仍未10万。唯一修正正式已从原85精确恢复，PID70926/session54702，status120新增20,643,840，最后至170累计100,270,080后原两条件末评，不新增配方或左足训练。


更新：2026-10-16 10:29。新阶段处于准入准备，正式0。此前VelStand256阶段150,994,944已关闭并完整保留，不归零旧成本。用户最新要求继续参考运动头稳成功经验，并连续安排两张GPU。

## 新依据

历史`backup/local-work-20261015@2ca93c9`中，固定来源5102继续201更新、N8192/T24，仅改变世界相机角速cost权重0/−.1/−.3。已重新核对control/mild/strong在.1/.3m/s的六份原始200Hz NPZ SHA并独立复算RMS，均与原机器记录相符：.1命令control/mild实速.09163/.09086，相机1.84673/.60268rad/s；.3命令实速.31831/.30828，相机2.34293/.67065。轻项确有运动稳像收益。强项及轻项都有头roll增益/航向取舍，不能把历史局部改善称完整遥控或实机通过。证据`historical-camera-evidence-for-v2.json`，来源文档`microduck_rl/docs/sc0090-camera-stability-trial.md`。

实际V2 VelStand256 `run.json`没有任何camera奖励或采样metric，仅头关节跟踪、EMA偏置课程和身体角速项。这与用户强调的世界相机稳定指标存在明确目标缺口，构成一次新试验的依据，而非只因原阶段平台而续训。

## 单一试验变量与预算

拟固定父256，两臂各最多64更新（N24576/T24），新阶段总正式上限75,497,472；只比camera权重0与−.01，不扫更多权重，不改teacher标签/动作、网络、PPO、物理或原评测。若实际N变动，按controls上限换算，不能扩大总预算。两臂资格通过前不正式，验证权重不继承；保存原actor/critic/N、PPO和BC两个Adam、LR和clock6144。跨GPU恢复须显式映射原CUDA RNG并验证张量状态，不能静默丢弃Adam或归一化。

历史−.1不直接复制：V2+.30相机RMS约6.74，−.01对应粗略惩罚率约−.45，而旧2rad/s×−.1约−.4；这是量级选择，平均RMS不能代替真实各子步平方成本，最终由准入实际采样校验。保持唯一较轻候选，若不能兼顾动作则保留原父结束阶段，不无限改标签、权重或gate。

## 采样及评价

独立源`velstand-camera-worktree/custom_v2/camera_stability.py`实际位于`src/mjlab_microduck/custom_v2/`。每个200Hz物理子步读取真实世界相机角速度，计算`wx²+wy²+(wz−requested_yaw)²`，四次平方后平均；请求yaw是执行该控制区间的原命令。通过原head-bias的高度/倾角窗口减轻起身时必要的旋转成本，站立和行走仍接受稳像目标。采样是无权重metric，零权重对照也采集；不加入额外forward/sense/physics，不把真值送actor，不加动作滤波。

准入需实际V2 Jacobian×qvel独立对照世界角速、原生子步时间与奖励oracle、现场qpos/qvel/ctrl/act/动作/命令/RNG不变、重置隔离；随后精确成熟恢复、软件smoke和实际N优化/吞吐。CPU002已通过，CPU001仅`MetricsTermCfg`名称误用导致0仿真失败。物理001在首物理子步因测试记录器引用不存在的`raw_actions`属性失败，已改为实际`_raw_actions/_processed_actions`，原失败留档、正式0；物理002重跑中。它是测试代码问题，未改当前正式模型。

验收沿用相同336命令电池和恢复/静止调查，额外同时报告减请求yaw后的相机残差与原总角运动。头部主动转动仍有瞬态成本，必须看响应时间/尾误差；速度、双向转向、步幅/碎步、停车、恢复和健康相机分母同列。不得通过少走、少转或躺倒获得低抖动。若只改善个别场景而损害原已通过能力，报告取舍并保留原父，不以奖励上涨批准模型。

GPU0在Roulade末评结束后自动接物理准入；CPU导入提前完成62.36s，已实际省去该段GPU空等。GPU1继续BallKick已有阶段，不因相机准备中断其它技能。

## 2026-10-16 10:44 实际准入进度

物理003完整PASS：640控制/80子步×32world，36份独立CPU Jacobian最大1.369e−6rad/s，原始成本误差0、gated误差7.63e−6，live动作/物理/命令/RNG不变，partial reset通过。物理002在完成这些前置断言后因测试退出inference_mode再reset而失败，已把reset放回相同评测上下文，原件留档。

新trainer严格camera-only源迁移CPU通过，目标环境去除新增metric/reward后与原父配置相同；跨GPU CUDA RNG映射显式记录。64×5软件warm smoke完成7,680验证控制、32timeout、ONNX9.54e−7、feedback/nonfinite0。成熟父256的24576验证在首更新257保存停止：1/589,824反馈未就绪、非有限0，PPO KL.01295、postBC KL.12552。正式仍0；有界同源只读diagnostic PID63153/session3591正在捕获原始反馈，不能直接将这1次归为旧native-count1026，也不忽略它。待证据决定准入，不重复已通过的物理和小smoke。

## 2026-10-16 11:06 — 双卡camera固定对照正式启动

- 目标：按已核验历史经验补运动相机稳定目标，同时检查速度、步态、头控和恢复退化；不把静止或低折叠姿态当头稳成功。
- 完成：实际N24576×16两臂准入通过，各1已定位native越界/未知0/非有限0；核对当前80余源文件和全部资产SHA；资格`velstand-camera-qualification-001.json`、公共评测协议`velstand-camera-eval-protocol.json`已冻结。旧小批smoke复用限数值实现未变；新增只读审计通过真实及负例夹具验证。
- 正式：原256父模型，mild GPU0 PID93712/control GPU1 PID93799，各64更新，合计75,497,472控制交互上限；不继承准入272。训练与四项末评单进程接续，正常结束不再自动加训。
- 评测：两臂同nominal/measured各336命令条目和128初态survey；survey唯一变更为设备参数，原公式/随机化/种子不变。保存原始200Hz相机姿态及控制数据，可独立复算转向残差。
- 吞吐：准入mild99,730/control98,893控制步/s，不能称突破10万。正常速度/有效完成率优先于盲目增加负载。
- BallKick：最后170累计100,270,080完成且阶段关闭未合格；横速和身体减速有收益，但完整/身体保持/头回中为0，相机没有改善。子代理只做CPU归档。
