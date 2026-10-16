# V2 P04 反馈容错历史诊断台账

最后更新：2026-10-15 17:08，Asia/Shanghai。**最新用户决定：取消尖峰去除，靠随机化/策略适应。以下异常拒绝候选已退出最终主线，仅保留诊断证据，不继续其原生 C 实现，不合入训练。** 当前要求见[阶段合同](sc0090-v2-current-stage.md)。

补充历史结果：v2 CPU10项、device-v4、graph-v4和smoke-v2通过；0/20/40/80ms缺测分别443/440/110/0 of448恢复，sensor fault0/0/336/448。20ms依靠恢复后的真实双侧支撑，未发生控制级外推，不能称全部缺测已覆盖。原生C只有语法检查，没有binding/oracle/部署验证。

评测口径纠正：同种子、同N32和同协议不等于轨迹逐位配对。旧缺测实验在注入前已有物理状态差异（首控制步qpos最大约0.007839rad）；原因未定位。保留报告，不把3例恢复差直接归因缺测。

## 当时目标与边界（已被最新用户决定覆盖）

用户要求最终方案拒绝异常测量，短时缺测继续估计；区分真实加速；预测不回灌为新测量；恢复后校正；持续失联/不确定度过大才失效。查询目标仍为控制时刻前20ms，不能注入物理真值。关闭旧尖峰检测与缩短O0预热是诊断/过渡对照，不能标为最终处理方案。

当前隔离源：`runs-v2/feedback-tolerance-worktree`。从 `repair-local-o0-adapt-001/source.tar.gz` 恢复，`feedback-base-source.json`保存SHA；现有GPU0随机起点训练源未被热改。新任务 `Mjlab-StandUp-Flat-MicroDuck-V2-Bounded-O0-S0-R1`，执行版本 `o0-bounded-prediction-candidate-v1`。

## 候选机制

- 原生count、speed、load、voltage和交易start/end/available时间是唯一测量输入。保留valid、有限值、count范围、时间顺序、PID、命令确认及输出数值/限速/位置检查。
- 异常判据使用已接受真实测量的速度/加速度趋势、实际dt、测量噪声与运动不确定度；第二条符合速度/加速度边界的真实候选读数可佐证运动变化。所有门槛在冻结O0仿射映射后的物理rad单位计算。
- 原始环中保留拒绝状态，LS历史仅接收新时间戳的真实有效读数。预测不会替换原始读数、不会进入LS或特征历史，也不会刷新测量年龄。相同真实读数不因多次tick被当成多个新样本。
- 保留冻结O0 affine与真实测量LS3；查询有已完成左右支撑时插值，否则仅从最近真实测量生成的已完成输出外推至**指定查询时刻**。query是纯函数，重复查询不改变状态。
- 估计不确定度候选为 `sigma_q + sigma_v * dt + 0.5 * sigma_a * dt^2`，dt从最后真实测量到查询目标计算；同时约束wall时刻的测量年龄。短时坏包不清空整条链；长间隔恢复时重新收集5个新有效测量，没有固定220ms失明期。
- PID/命令契约故障仍是硬失效，不被测量预测掩盖。无法仅凭编码器辨别所有长期、连续且物理上看似合理的错误；此类局限必须保留，不能把运动一致性判据称为真实故障完美分类器。

## 参数状态

以下全部是**待实际轨迹/故障注入验证的候选值**，不是实机标定结论：位置噪声界.03rad、速度不确定度2rad/s、加速度不确定度200rad/s²、运动确认最大加速度2000rad/s²、速度界40rad/s、创新系数4；查询外推上限30ms、真实测量wall年龄上限50ms、位置不确定度上限.15rad。多门槛共同作用，不能保证每次都允许完整30ms预测。

## 已执行验证

- CPU协议测试：9项PASS/2.38s，覆盖恒速查询时刻解析值、预测不回灌/重复query纯函数、两次invalid/NaN/count越界/时间错序保历史、重连校正、真实加速与孤立尖峰、时长/不确定度/未来查询限制、部分reset隔离、PID硬故障。`bounded-o0-cpu-tests-v1.log`；后续诊断reason字段变更的复核为v2。
- CPU vs GPU vs编译GPU：80次×4lane，真实运动、尖峰、连续缺测、NaN与部分reset，最大输出差2.98e−8；GPU eager/compiled所有张量状态逐位一致；实际出现6次有效预测、48次无效查询（含启动/重置未就绪）。`bounded-o0-device-parity-v3/report.json`。
- 首GPU检查暴露Python时间标量触发反复编译，已在编译入口归一化为device tensor；第二轮暴露测试在各设备分别反算浮点count导致输入不同，改用CPU一次生成同一原生整数count再分发。失败v1/v2日志保留，未靠放宽误差门槛通过。
- 完整S0+O0与CUDA图：正常/高速/NaN、完整/部分/补齐reset两组均PASS，各80步、1次rollout capture/78次replay、4类reset graph，全部因果张量状态逐位一致。`bounded-o0-graph-parity-v2/report.json`。随后只添加诊断reason=11字段，v3重复完整检查并保存源码快照及SHA，要求验证期间源不变。
- 首图检查发现独立cuda:1调用在default device0下捕获错卡，导致RNG offset错误。已在隔离版ResetPrehistory/FeedbackGraph显式指定所属device和独立capture_stream；没有改运行中的GPU0源。
- 64×5实际PPO/真实timeout/ONNX PASS；ONNX CPU最大差1.94e−7，61输入/14输出与归一化契约通过。`bounded-o0-smoke-v1/report.json`。导出明确标记simulation candidate，不能让旧固件默默接受。
- 新诊断同时记录真实接受/拒绝次数、有效预测查询次数和最大不确定度；计数不包含reset预热。仅统计原source_fault无法反映新短时拒绝/容错，不能以fault0代替异常处理审计。

## 当前实验与下一验收

- GPU1 session55968：冻结baseline3756，nominal→measured，各2种子×7桶×128×8s，使用实际候选前端；输出`baseline3756-bounded-o0-{nominal,measured}`。源权重/normalizer逐位核验，不追加训练，不叠加基线旧位置噪声。
- 必须补：实际拒绝率/误拒绝与预测误差轨迹审计、受控孤立/成串异常及恢复时长、同一基线在旧/去尖峰/最终候选三者对照、头部响应和站立不倒、32768环境实际PPO吞吐。
- 必须补：与部署可用原生C/C++实现的独立oracle对照和接口接入。当前只有Python CPU/GPU一致性，**尚未证明现固件与候选一致**；不把当前legacy LS3说成此算法。
- 正式合入须携带源与参数SHA、全/部分reset和CUDA图回归、对应N/T/minibatch的smoke/吞吐证据。GPU0只沿真实从零谱系续训；基线适应模型继续作为迁移对照。

### 16:14 结果补充

- 最终诊断版本CPU复核9 passed/2.22s；完整图v3保存source.tar.gz/source_sha256，验证期间源不变，正常与压力全部张量逐位PASS。
- baseline3756→候选 nominal1772/1792、measured1709/1792，fault均0；无注入回放的控制预测次数0，不能据此证明短时外推闭环。拒绝率nominal≈0.00327%、measured≈0.0507%，详各目录tolerance-summary.json。
- session82847在20/40/80ms左膝坏包闭环注入测试，2种子×7桶×32。还需同样N32、种子的0ms对照。当前脚本仅接受20/40/80参数；待该队列退出后再扩展0，或新建独立对照脚本，避免运行期间源码SHA失真。

### 16:25 v1短缺测失败，candidate-v2修复中

- 20ms缺测出现222/448误invalid（只有left_knee，首次step101，reason2已有新真实输出），恢复224/448；40/80ms均448/448 invalid。v1短缺测能力未通过，不能用正常测量回放0fault替代这一结果。
- 根因是恢复后已有右侧真实支撑，但旧25ms跨度门槛禁止插值，退回单侧预测并触发σ上限（约.1727>.15）。v2使用有真实两端支撑的有界插值：端点估计不确定度的线性加权 + `0.5*a_uncertainty*dt_left*dt_right`；支撑间隔≤50ms。加入8rad/s速度不确定度上限。没有提高.15rad位置上限、没有无限外推、没有预测回灌。
- 新异步交易完成/40ms支撑跨度回归通过，CPU共10 passed/2.29s。代码/manifest执行版本升为candidate-v2。
- session83250正在device-v4→graph-v4→smoke-v2→同N32/同种子dropout0/20/40/80-v2；用独立目录保留v1，结果未提前写PASS。
- 最终准入另需fresh64/256真实探索动作正常数据审计，以验证新异常判据不误杀初期大动作；以及原生C/运行时、最终吞吐和完整头部/站立保持。
