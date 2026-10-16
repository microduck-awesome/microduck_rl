# SitStand 历史基础配方固定复现（2026-10-16）

当前：本页基础320及后续Hold320到640均已完成；最终坐姿保持已全过原身体门槛，站立/头响应未过，详见[Hold320结果](sc0090-v2-sitstand-history-hold320.md)。没有待启动坐站训练。

## 已批准的固定预算

N16384 × T24 × 320 = **125,829,120新增正式控制**；E5/M16 = 80原生Adam/更新，共25,600。此前全部SitStand正式226,492,416保留，总上限352,321,536。一条fresh谱系，actor/critic/N/Adam/clock0，不加载旧模型或验证权重。无种子/权重扫描，不自动追加第二段。64/128/192/256仅观察节点，不能因早期无效果更换配方；保留数值、容量、反馈护栏。320物理连续，不在中间评测重建。段末对64/128/192/256/320逐个原fixed-clock12s nominal/measured评测，保留身体/头响应/200Hz世界相机和健康分母。

## 原始成功证据及差异

来源：官方固定worktree `docs/sitstand_policy.md`，实际 `git show c832147:src/mjlab_microduck/tasks/microduck_sitstand_env_cfg.py` 与 `77ffadf`。本机未发现两个成功run的原始W&B cfg，以下历史N来自文档命令，T/E/M及课程来自对应源码；不是原始训练日志逐项复原。

| 项目 | 成功基础 jngaedcs@1250 / c832147 | 此前V2 fresh128 | 本次固定复现 |
|---|---|---|---|
| N/T/E/M |4096/24/5/4|16384/24/5/4|16384/24/5/16|
| 每mini-batch |24576|98304|24576|
| 约交互/Adam |122880000/25000|50331648/2560|125829120/25600|
| 回合 |12s|4s|12s|
| 腿目标奖 |全程slewed Gaussian/L1/composite|3项过渡释放|恢复原3项，不加新奖励|
| 头命令 |UniformPose，zero_prob0|BalancedHead，zero_prob.5|恢复UniformPose0|
| neck_strain−.5 |基础段无，后1250→2250新增|提前从0存在|基础段移除|
| 坐姿/高度 |XL330旧坐姿|ClearSeat正确V2|保留ClearSeat及实测高度|
| 课程clock |1250×24≈30000|128×24=3072，全部stage0|native320×24=7680，历史等曝光clock30720|

历史后续77ffadf实机硬化是从基础1250 plain resume到2250；不是一开始就加其neck strain、seated head jerk/tip。新阶段尚未授权那一硬化阶段。基础1250存在突然头动坐姿后倾缺点，不能把历史成功写成所有指标完美。

课程只将每项step边界除4，native common_step_counter仍真实累计；例如历史500/750/1000/1250更新映射native125/187.5/250/312.5。保持原函数严格 `>` 或 `>=` 边界、原reset时才计算课程语义；实际生效可能延后到下一reset。姿态dwell3.5–6.5s、头dwell2–5s、slew2s、episode12s、物理200Hz/控制50Hz均不缩。将保存完整阶段映射、V2限位裁剪后的可达head范围及实际live manager值。总曝光和Adam剂量接近历史，但rollout刷新频率、物理模型和反馈不同，不能称严格官方等价。

## 先前诊断的准确边界

M4/M16 actual16用于数值/吞吐，M16实测1280次optimizer.step vsM4原生Adam320。四项冻结16评测共614400诊断控制、0正式；两臂在每个条件的初态qpos/qvel/rawactorobs/origins逐位相同。M4在部分站初态阶段有早期保持，M16仅首站保持；坐初态两臂均未学会。**这些是早期快照，不足以否定M16或成功历史配方**。原报告 `sitstand-fresh-m4-m16-eval-review-001.json`保留。

此前CPU曾提出head L1备选，相关JSON只是提案、没有进入生产源或训练。根据最新用户/父级决策明确撤下本轮候选，不增加L1/EMA，不把neck Gaussian远端饱和推定为本轮必须修改奖励的因果证据。

## 不变项

冻结v5、CAD、P04 S0/O0有效读数无尖峰去除、compiled/fused/graph、61actor/14action、HOME0；原physics、solver、ncon64容量和TerminalCritic修复不变。原独立12s指标/阈值不改。本次只声明V2历史基础配方复现试验，不等同实机成功。

## 语义审查与必要重试

`c832147`与当前MDP的18项明确函数/类AST比较：17项去docstring后完全相同，覆盖posture奖、SitStand/UniformPose命令和课程。head_pose_tracking多了fine_std与backlash输出侧读数；本任务实际fine_std=None、完整V2模型无passive_backlash，保留原std.5、四轴平均，用本次资格已有物理oracle核对。审计见数据盘 `sitstand-history-mdp-semantic-audit.json`及差分。配置import指向当前函数不单独作为历史实现等价证据。

全部5阶段×4头轴已同实际robot.xml限位和HOME0核对，包含320外的后续阶段；20行均在限位内。后两阶段neck负范围裁为−.4857776，最终pitch正范围裁为.904656，其余按历史/原V2适配。见 `sitstand-history-model-limits-cpu.json`，只读CPU，无额外GPU仿真。

001失败是外置counted_log没有functools.wraps，原diagnostics inspect.signature无法绑定it；首更新已采样1536验证controls、80Adam，正式0。未改生产diagnostics/PPO。002仅修外置包装并保留001日志/源码SHA/失败receipt，复用已通过CPU001和相同数值源；smoke重新必要执行。没有以该软件记录错误评价M16学习效果。

## 资格结果与正式启动

worker002 PID319428/session27653已通过全部资格并自动进入`allocation-002`连续320。64×5：32timeout、ONNX2.38e−7、contact427/4096、5物理oracle全0。actual16384×16：6,291,456验证控制、1,280实测Adam、8,192真实timeout、反馈/非有限0、contact100487/1048576、maxKL.016976/maxclip.251261；稳态中位93,913.94、合计93,390.70controls/s，未达10万，原正确性门槛全过。正式fresh空Adam/N0/clock0、LR.001，actor/critic未加载检查点且不是验证训练权重。准确实时更新/总数以`allocation-002/status.json`为准。

## 320正式完成，等待独立曲线结果

已完成125,829,120正式控制、25,600实测Adam、native clock7680；全历史正式352,321,536。formal phase（含建环境、共享import）1337.151s，94,102.38controls/s；稳态中位95,544.25/aggregate94,909.87，未达10万。196,608真实timeout，反馈/非有限0，contact峰106596/1048576；训练累计健康采样率0.0366354%，不是episode成功率。末live action_rate−.6、descent20、torque_rate−.0005、rise0，head范围±(.39,.39,.49,.11)，符合原严格边界+reset时序。64/128/192/256/320检查点均保留，当前同进程自动进行10项原12s曲线评测，无后续PPO队列。

## 最终固定条件曲线与结论

worker002已exit0，整个CPU复用/资格/正式/10评测流水线1770.198s（不含已独立保存的001失败准备）。五节点同一条件下qpos/qvel/rawactorobs/origins逐位一致，eval配置也一致。评测不加载训练env clock，预先清空curriculum并移除非allowlist重置随机化，因此曲线不是不同课程DR初态造成。10项共1,536,000诊断控制，未计正式。

| 更新 | SIT末.5s保持 nominal /384 | measured /384 | STAND两条件 /384 | 主动head到位两条件 /672 |
|---|---:|---:|---:|---:|
|64|0|0|0 / 0|0 / 0|
|128|273|233|0 / 0|0 / 0|
|192|30|25|0 / 0|0 / 0|
|256|82|83|0 / 0|0 / 0|
|320|324|330|0 / 0|0 / 0|

分母是同256场景三命令阶段中重复trial-phases，不是384独立episode。所有节点fault0，严格head末保持也0/672。320所有站命令尾高度均值约78–81mm，站目标112mm，故不能把坐稳当起立。320是目前这条谱系可保留的partial-SIT候选，双向动作与头响应仍未合格。body hold沿用原运动学门槛，不增加声称“接触全合格”。

320在坐目标、零头命令保持窗口2–3s，身体持续稳定子集nominal313/measured322 trial-phases；加权平均单trial相机角速RMS .048327/.080401rad/s，平均单trial20ms曝光P95 .001721/.003010rad。它们不是所有场景的pooled RMS/P95，也不把主动转头算成静止抖动；STAND健康分母0，不能用倒地低角速冒称头稳。头角误差和严格响应继续独立报告。

完整证据：`sitstand-history-base-curve-review.json`、`sitstand-history-base-closeout.json`、各checkpoint和10份原始report/traces。320 checkpoint原生Adam所有state.step=25600，actor/critic normalizer count各125829120，clock7680。正式总数125829120新增、352321536全历史保持。GPU1已交还root安排后续，未启动额外PPO或实机/部署。64零分之后128/320出现坐姿能力，不能把16/64早期零分当配方无效；同时320完整基础段未掌握起立，后续要依据完整证据决定，不自动加入硬化或新奖励。
