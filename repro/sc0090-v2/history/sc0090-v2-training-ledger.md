# SC0090 V2 双分支训练台账

## 2026-10-16 10:53 相机准入故障已归因并继续；GPU1正式修正进行中

实际异常world558/step6160是head_yaw原生1033count、reason13/status−1、transaction valid，numeric/nonfinite0，发生在第一PPO更新之前。控制边界物理yaw2.7989rad、估计2.7480rad；源采样更早，估计不是真值。不夹造count、不改S0/O0或episode终止，也不将旧40°压力测试的其它故障一并归因。证据`velstand-camera-feedback-native-range-attribution.json`。

已移植历史验证过的只读native-range分类，只放行该原因进入实验计数；两真实捕获与6负夹具通过，正常有效count/坏包/数值故障均不能冒充已解释事件。每臂实际N16最多8次，正式64最多16次，未知/非有限仍即停并保存。新manifest`velstand-camera-source-transition-002.json`仅增加camera项/采样和只读故障审计；原已通过的camera算法、PPO、BC、物理不变，不重复物理/64smoke。session73340/PID76016依次执行mild/control成熟24576×16，正式0；源冻结。

GPU1 BallKick实际N16完成9,437,184验证controls、24,281timeout、五项oracle最大1.43e−6、feedback/nonfinite0；稳态86,603仍未10万。唯一修正正式已从原85精确恢复，PID70926/session54702，status120新增20,643,840，最后至170累计100,270,080后原两条件末评，不新增配方或左足训练。


## 2026-10-16 10:29 历史运动头稳已核验；两卡接续安排

已核验六条历史原始相机轨迹SHA并独立重算：轻相机代价在相近实速下将RMS1.8467→.60268rad/s；当前V2确无直接相机稳像奖励。新独立相机目标阶段拟两臂0/−.01、各64更新/总75,497,472，正式0，先GPU采样/恢复/PPO准入；详见[相机目标计划](sc0090-v2-camera-objective-plan.md)。已有VelStand预算和缺口不抹去。

GPU0 Roulade第二段完成，累计100,663,296，阶段封顶无第3段；两条件末评仍不能证明完整翻滚/站稳，专职做一次CPU奖励审计。GPU0已自动接相机真实采样验证，预加载62.36s与末评重叠。GPU1右足BallKick首85完成50,135,040、稳态83,386控制/s、fault/nonfinite0；两条件有效触球全256，完整/末段站稳仍0。保存轨迹发现身体直立但持续运动（线速.183、角速2.4、双足34%），尾部92–93%正球速收益付给未站稳样本，同时球横速约.45m/s。专职将接GPU1处理这两个明确目标缺口，只用原共享剩余50,528,256预算，保留原评测与来源，不先复制未健康策略到左足。


## 2026-10-16 10:08 BallKick首段准入完成并锁定正式预算

CPU配置/接触状态、实际128×5s物理、6种多实体successor/部分重置、64×5/ONNX、16384与24576各16更新及两条件32例评测入口均已完成。实际批量选择N=24576、T=24，验证稳态aggregate=82,388控制/s，尚未达到10万目标；停止扩大性能搜索，按较快已验证配置推进有限实验；完整进程吞吐单独记录，正式表现尚待实测。

GPU1即将从随机初始化执行右足85更新，共50,135,040正式控制，验证权重不继承；左右足与后续修正共享100,663,296总上限、最多两配方。原生PPO及已验证接触/几何配方冻结，反馈/非有限/优化/容量异常保存并停。末尾串行nominal/measured各256条原起点5秒评测，球速/有效踢球/身体落稳/头中性与200Hz相机稳定分别报告，无第二段自动续训。资格与精确来源见`ballkick-qualification-001.json`，运行状态见`ballkick-formal-status.json`。准入并非动作或实机验收。


## 2026-10-16 10:06 当前双卡进度与头稳证据

头部稳定性已独立整理为[逐场景记录](sc0090-v2-head-stability.md)：静止改善，运动/停车未达标，健康分母和角速度/20ms曝光口径明确。VelStand本阶段关闭，不自动追加。

GPU0 Roulade端点物理完成：112个近完成初态105个末1s身体+头中性保持，健康相机RMS均值.0162–.0206 rad/s；失败7个保留，仅证明端点可达。成熟64×5恢复/终止/ONNX通过，现actual16384×16，随后按原剩余128/50,331,648正式预算和原完整起点末评自动推进。

GPU1 BallKick16384×16完成6,291,456验证控制、8192强制timeout、fault/nonfinite0，稳态median74,237/aggregate73,764控制/s，未达10万；两条件32例评测入口均正常，5更新未学会踢球属于预期，不算技能通过。当前仅一次24576×16批量对照（PID4188668），后续监督器session98004在全部精确源码/138资产/准入证据通过后，选择较快N，自动先写明首段预算再从零训练右足；首段≤50,331,648，左右与修正共享100,663,296，无第二段自动续。任何准入失败保留状态停下诊断，源与P04测量链冻结。监督状态`ballkick-supervisor-status.json`。


## 2026-10-16 09:50 头部稳定性结论与新技能准入

VelStand 256阶段已关闭，共150,994,944正式控制；原配方末段nominal/measured命令电池均336/336，评测feedback0。独立静止调查名义末1s健康保持16/16，相机200Hz角速度RMS均值0.18350 rad/s（128时约4.75）；measured健康保持13/16，全16条RMS均值0.49435、健康13条0.19892。健康分母不同，不能只选成功轨迹宣称全通过。恢复调查prone/crouch/sit仍有未保持，行走相机约6–7 rad/s、走后停步约5 rad/s及约2.4Hz脚步复发仍未解决。头关节命令误差、身体健康保持、世界相机RMS/P95/20ms曝光跨度分别报告，不能相互替代。

本阶段预算结束不等于头稳或全技能合格。保留128/256和独立起身站立候选，停止自动第三段。末段稳态median99,679/aggregate99,029控制/s，完整进程83,256控制/s，不能报成全程超过10万。正式2次native-count非法值已归因，未知故障/非有限0；不改S0/O0/v5、不夹造测量。依据`velstand-stage-256-review.json`与`velstand-v8-final-audited-plan.json`。

GPU0专职Roulade首128已完成50,331,648，稳态aggregate110,439/median112,059控制/s，完整进程81,483；自然timeout196,608、feedback/nonfinite0，正在复核独立末评后决定余下预算。GPU1转BallKick多实体终止/训练准入。BallKick物理002已验证actor61/critic82，实际200Hz接触锁存；恒HOME与近球16/16健康，扰动HOME15/16，一条非踢球部位接触不再获得正球速奖励。来球被静止脚弹回16/16只证明接触记账，不是学会主动踢球。

BallKick修复包括隐含forward命令bucket清零、按实际CAD姿态仅抬root清地面、正确足/支撑/健康接触门槛及坏接触锁存、ball独立非有限终止。原目标1m/s及12/−4奖励权重保留。物理001的floor判据误把body0当terrain，复算2条初态穿地>1mm；原证据保留，002用明确plane geom与修复重跑通过。当前多实体successor/partial reset准入正在运行，尚无BallKick正式控制。完整计划见[踢球计划](sc0090-v2-ballkick-plan.md)。


## 2026-10-16 08:34 头部稳定实证、VelStand首128收尾与单一标签修正准入


### 2026-10-16 09:02 罕见反馈故障已归因；原配方最后128启动

只读GPU重演再次捕获1/9,437,184未就绪，非有限0：world6516、control3078、head_yaw，S0原读数1026超出原生有效0–1023；事务valid=1、front_status−1、observer reason13、numeric事件0。物理偏航2.86895rad（164.38°），最后有效估计2.78228rad，估计不等于真值。有效尖峰过滤始终关闭，本例不是有效尖峰误杀。证据`velstand-feedback-native-range-attribution.json`与完整原始pt。

这里修正的是准入解释：0未解释故障/非有限仍是要求，越过已声明原生读数范围的少量真实模拟无效读数应按既有episode终止处理；不修改CAD/v5/S0/O0，不夹造1023，不以真值补actor，也不猜硬件在范围外会饱和还是回绕。正式仍原v8冻结源/两教师/两Adam/N/LR/128课程时钟，当前正确保护保留。

成熟无sidecar验证稳态aggregate101,066、带只读捕获重演100,234控制/s，均为50Hz控制样本。最后同源128外置队列`velstand-v8-final-audited-plan.json`、session15223、训练PID4063673已启动；native-range sidecar每个坏lane核因，任何其它原因或非有限、或本段超过32次无效控制即在当前update后保存并停。32是本段实验的故障累计预算（约基于验证频率外推的4倍余量），不是硬件容错/模型通过率承诺；尚未完成不提前计正式量。总正式上限仍150,994,944，无第三段。末尾串行两条件完整命令/恢复/相机评测。

Roulade已实际128world×3s物理PASS，HOME16/16末1s站稳，相机RMS.02029rad/s；midroll恒定tuck自然倒地、没有凭此称会翻滚。镜像PPO诊断在CPU/GPU和原生update逐位对照通过，专职继续64×5→实际N。BallKick V2配置CPU与原奖励语义审计通过，仍未GPU/正式资格。


### 2026-10-16 08:55 实际批量通过优化，但1次反馈未就绪阻止正式启动

原v8成熟24576×16完成、无优化guard，post-BC meanKL最大.26643，真实successor12288、4oracle最大1.49e−7、接触峰125134/1572864。稳态median100,839/aggregate101,066控制/s，当前该条件已过10万；完整验证进程234.764s包含imports/初始化/保存，不能用稳态代替整进程。

9437184验证样本中恰有1次feedback not ready，严格0准入自动停队列，第二段正式尚未启动。复核首128正式75497472中有2次；之前独立命令/物理评测feedback0仍成立，不能把评测0推广到训练0。原因尚未知，不能先判为模拟故障/尖峰去除/真值替代。正在原冻结源上做一次有界同N16只读GPU诊断，使用既有FeedbackFaultRecorder，保存首故障的原寄存器、时间、原因码、物理q/估计q，不修改动力学/策略/反馈/RNG。还补显式非有限样本计数：WalkingTrainingSummary原schema没有GroundPick的nonfinite_fraction字段，不再跨技能假设字段存在。证据`velstand-v8-mature-qualification-review.json`，诊断session58697。


### 2026-10-16 08:50 原配方对照澄清与最后一段排程

原v8成熟父128、同64环境也在第一次更新post-BC meanKL.61469停止；因此两种头标签的.63752/.58155不能独立证明标签是失稳原因，也不能证明标签有效。新标签试验仍关闭、正式0，保留原父。现在GPU1使用完全相同的原v8源码/教师/PPO/BC参数做成熟24576×16，检查新恢复分布下的优化、真实successor、接触容量和同边界oracle；不放宽任何guard。通过后自动从原正式128而非验证144继续最后128，随后串行nominal/measured完整命令和恢复/相机调查。队列`velstand-v8-final-allocation-plan.json`/session8169，阶段总上限150,994,944、无第三段。当前尚在成熟actualN验证，不提前记第二段正式。

### 2026-10-16 08:46 头标签试验准入失败，停止扩配方

瞬时切换与唯一一次32更新渐进切换，均在成熟父128的64环境首更新触发post-BC meanKL>0.5：分别.63752/.58155，均保存129后停止，正式0控制；两次合计3072验证控制。CPU/物理配置/状态迁移与ONNX通过不能替代优化稳定性。标签试验关闭，不放宽guard、不再追加第三种头部配方。正在用原v8同父/同64设置补一个对照，不能仅凭失败先归因于新头标签。其后若继续整合，仅以原已验证24576环境的v8配方续完最后128，原总上限150,994,944不变；不把失败验证权重继承给正式。决定`velstand-head-target-decision.json`，新源明确未获准入。

### 2026-10-16 08:40 补充：健康相机对照、准入检查修复与GroundPick预算收尾

- `head-counterfactual-healthy-review.json`已用现有raw GPU tensor重算相机健康子集：整条8s未触发步行失败且camera有限，统一1.5s窗口，非严格静止保持。原walk512与请求直通的idle各8/8，RMS均值6.083→1.466rad/s；+.30前进8/8→7/8，6.959→1.778；主动头行走8/8→6/8，6.729→1.591。后两组健康子集有选择效应，且原地正转响应退化，直接覆盖动作仍被拒绝；仅支持有界地训练统一网络适应新标签。
- VelStand128 measured唯一失败已定位到forward−.40组第4例、reverse阶段5.98s，tilt首次46.77°越过45°界线，height约.11085m、feedback/finite正常。是实际姿态失败，不能归咎反馈异常或改判通过。
- 新头标签CPU8项实际checkpoint测试通过：物理cfg相同、头索引/scale/HOME0、限定标签/输入不变、原load拒绝未经声明变化、128两模型/N/两个Adam/count/LR精确、拒绝其它teacher/参数变化、实际BC保持teacher/N/std/rollout、同任务恢复。首GPU启动检查发现manager构建会修改cfg，导致把构建后的resolved cfg与父run的构建前快照比较；验证提前至建环境之前修复，原smoke001为0更新失败留存。新源重新CPU并按64×5→resume→actualN串行验证，尚不把准入当正式或头稳成功。
- GroundPick两条件完整末评已完成：每周期256例，三个周期low/complete均0；nominal return63/177/63，measured73/142/76，fault0。实际低位嘴高91–93mm，头pitch/髋roll接近错误方向限位；已有315万低位训练样本、std.10奖励梯度非数值饱和，与真实15mm可达姿态仍相差很大。阶段以75,497,472控制关闭未合格，不为用完剩42而盲续。保留原持续双足与声明clean口径并列复算，因low0不改变结论；没有推断物理不可学。
- GPU0专职转Roulade已有四项实证最小修复与准入，暂锁阶段正式总上限100,663,296、最多两种配方。实际N16384首128=50,331,648；若准入选24576则首128=75,497,472，剩余≤42，不能改N增预算。未正式不提前计。GPU1头部整合继续原阶段总150,994,944，不新增第三段。
- 新预算快照`training-budget-audit-20261016-0835.json`：baseline根39段2,629,828,608；P04随机根含legacy41段3,694,067,712；新增VelStand/GroundPick各75,497,472，无未解析账目。验证/评测单列，父量不重复加。

- VelStand首128已实际完成75,497,472控制，N24576/T24，无训练guard；稳态median85,029/aggregate83,238控制/s，完整子进程1052.007s（约71,765控制/s），仍未到10万。中间65评测与大批训练重叠发生CUDA OOM，未生成合格结果；正式训练正常结束。后续完整接触评测与大N训练串行，失败日志保留，不算评测通过。
- 128末完整命令电池nominal336/336、measured335/336，feedback均0。唯一失败位于seed2026101612、forward−.40组的reverse阶段，前idle/move/stop均8/8；未据此声称极限双向速度合格。8组独立物理调查nominal/measured末1s严格健康保持均0；idle/prone/crouch/sit已直立但持续踏步，名义相机RMS轨迹均值约4.59–4.75rad/s。直立存活不等于零速站稳/头稳。
- 已有walk512原始轨迹显示相机200Hz角速RMS约5.78–7.74、身体50Hz约1.20–1.56rad/s，头关节50Hz有限差分各轴约2.3–4.75rad/s。这是时域/量级证据，不能把不同坐标/采样的标量直接相减当贡献分解。证据`walking-head-motion-diagnostic-512.json`。
- 两个有限闭环对照：相同命令电池原walk512为167/168，直接拼站立teacher头部为109/168，直接把头部动作设为实际请求为159/168；三者feedback0。直接替换动作都未接受。后者相机数值明显降低但伴随快走/主动头/转向退化；全样本相机均值含失败，不能当健康稳像改进结论。
- 只使用VelStand剩余一个128预算做单一训练标签变化：upright walking BC的4个头标签改为原始actor中请求的HOME相对头目标；行走腿标签、完整站立/恢复teacher、PPO/物理/奖励/课程不变。仍为统一61→14网络，推理不强制覆盖。独立源`velstand-head-target-worktree`已建，精确三文件变更与显式迁移manifest正在验证；保留actor/critic/N/PPO Adam/BC Adam/LR/RNG/128课程时钟，旧父不改。64×5/实际N通过前不正式。全阶段仍最多150,994,944，不因头稳新增第三段。
- GPU0 GroundPick首128完成75,497,472，稳态aggregate100,939/median102,276控制/s，建环境至结束内部计时820.587s/92,004控制/s；外置完整进程904.344s/83,483控制/s；自然timeout successor12,183，fault/NaN0，无guard。低位目标训练累计仍0，正做两条件3周期独立末评，不提前称学会拾取。原阶段总预算100,663,296保持，N24576下至多再42更新，未自动续。
- BallKick CPU实际V2足几何已测：左右脚最前x=.0186606，球半径.035m/质量.015kg，候选中心x=.0786606、y=±.0433932，保留±.015放置噪声后HOME净距至少.010。只验证HOME静态几何，未验证扰动姿态/真实接触。官方实际球目标1.0m/s、forward权重12、overshoot−4，与过时注释.25m/s不同；正式前须检查正确踢球足接触与奖励，不能直接复制注释。专职Roulade已CPU定位腿镜像、头壳轴、tuck自穿和midroll初态深穿，未提前训练。

## 2026-10-16 07:53 VelStand准入修复、真实头稳教师与固定阶段预算

### 2026-10-16 07:55 最终批量验证补记

- 同任务GPU resume5→7通过，两套Adam/N/权重/时钟精确恢复，ONNX实际观测误差9.09e−7。
- 修正均衡BC后独占24576×16完成，全部BC更新固定5epoch×4batch=20步，post-BC meanKL最大.11975；真实timeout successor12288（人为排到下一时间边界），feedback0、reward oracle最大1.19e−7，无guard。
- 独占稳态median86983.16/aggregate87786.72控制/s，**当前VelStand尚未达到10万**。这是50Hz控制交互，不乘4冒充物理子步；与纯walking任务的>10万结果分开。已有完整接触/双教师开销，不以继续性能改造阻塞本段有界实验。
- 自动队列当前`VELSTAND-COMMAND-ADMISSION-001`；首正式128仍需新的42例评测入口准入后才启动，不能提前计完成。

- 实际GPU前置检查发现stock servo_stall将含mouth的15 actuator_force与14策略关节速度配对报错；V2新入口按相同14个关节/电机名称匹配，CAD、v5、反馈不动。独立MuJoCo transmission→DOF oracle核对堵转计数；3种普通/命令重采样/推扰×fast/full terminal successor全部逐位一致，主qpos/qvel/命令内部张量/timer/CPU与GPU RNG不变。
- 原PPO监控拒绝native BC wrapper，已用限定类型的组合adapter保留严格原PPO.update源码校验。真实两个更新的带监控/原native PPO→BC对照：两模型/N/两套Adam/LR/损失/RNG完全一致，九项CPU验证PASS。BC本身单列KL；原官方3e−4首次mean1.1713/max137.37，被guard停止，仅1次64环境验证，无正式更新。只将BC LR降到3e−5，5更新mean最大.07503，保持PPO原LR和Adam。
- 实际N又发现floor(n/4)会把1–3个余数样本独立做第五次Adam，每epoch样本权重严重失衡。已改成不超过4个均衡batch并保持min512/每样本每epoch恰好一次；8种规模CPU边界PASS。新64×5（smoke005）已exit0，32timeout、4reward oracle最大1.19e−7、ONNX64实际观测差8.3447e−7；small-N每epoch2个均衡batch，5epoch10步，BC meanKL最大.04047。正式没有使用旧余数batch配方。
- 真实128world×6s两教师调查完成。评测强制twist最初只写command、没同步is_standing_env，导致body reward gate仍随机；旧003运动/相机数据保留，body成本被004替代。004同步命令状态并独立验证真实z/roll/pitch/body gate，actor头/body零命令逐步断言通过。新004的idle、prone、crouch、sit各16/16末1s健康保持、所有fault0；静止相机RMS轨迹均值.02445rad/s，三恢复组约.02381–.02924。完整接触下forward+.30有2/16触发步行失败边界；这个父模型不是已完成VelStand，保留风险后做有界整合。步行时相机仍约5.88–6.78rad/s（含失败），不能用恢复后静稳成绩替代。
- 第一次24576×16真实批量完成、12288人为安排的时间边界bootstrap、fault0；它与teacher调查有GPU并发，median84654/aggregate78749控制/s，不是独占benchmark，不宣称已过10万。现在单进程先GPU精确resume5→7再独占24576×16复核，session4673外层PID计划已落盘；通过后自动测试42例新评测入口，再正式首128。不会为了GPU满载绕过这些检查。
- VelStand整合阶段锁定N24576/T24，首128=75,497,472正式控制，最多两个128阶段总上限150,994,944；第二段须复盘，没有自动续训。头稳、步态、转向、停车、恢复共同计入；正式父始终walking512，所有validation模型不作正式父。末128自动安排nominal/measured共同命令336例和恢复调查，先保留候选再决定剩余预算用途。当前尚未正式，计划`velstand-first-allocation-plan.json`、源`velstand-v8-repaired-worktree`。
- 坐站Lesson4已完成且独立nominal/measured所有完整保持仍0、head保持0、fault0、相机健康子集0；总150,994,944到预算关闭为未合格，保留三候选，不宣称物理不可学。GPU0转GroundPick：实际嘴端15mm目标空载/30g各16稳定；phase/load successor10案例逐位正确，仍在新技能准入，详见专职GroundPick台账。
- 预算audit v3新增跨任务warm start clock0和新技能分类，旧全部计账逐项不变。最新已提交：baseline根2,629,828,608，P04随机含legacy3,543,072,768；技能rise_stand5,619,253,248、walking402,653,184、sitstand150,994,944；VelStand/GroundPick尚0正式。

## 2026-10-16 07:30 头部稳定单列复盘；步行阶段结束，VelStand进入真实准入

- 用户再次强调头部稳定性。补完256/384/449/512共同measured报告的相机复盘，固定前进+.30m/s时每条轨迹200Hz角速RMS的均值依次6.7511/6.8955/6.9560/7.0694rad/s；20ms曝光跨度P95的轨迹均值.1997/.2050/.2030/.2036rad。这里是各试验统计量均值，不是池化RMS/P95；转向和主动头命令分开。转向误差下降不能当头稳改善，512也不是综合最优。证据`walking-head-stability-512-review.json`；健康动作与静稳/主动运动必须共同看。
- walking同源384→512已实际完成100,663,296新控制，阶段累计402,653,184到预算止步。整进程1072.836372s/93829.12控制/s；稳态median113692.17/aggregate108812.29。449/512 nominal/measured均336/336、feedback0、干净exit0。measured yaw RMS约1.011→.8392，但+.50原地转向尾均值.3912→.1320rad/s，负转−.2747→−.2640，需保留449。生存不是指令或稳像验收；不再自动延长纯walking阶段。
- GPU0 ClearSeat128→256已完成，阶段累计100,663,296；修正自干涉端点后独立12s电池所有完整尾部保持仍0，反馈0，说明可行目标本身未解决学习失败。明确诊断为长期倒地占用/健康承接曝光不足后，只允许一次12→4s训练episode的Lesson4s128到384，新增50,331,648，整个SitStand阶段上限150,994,944；末评仍12s并含实际200Hz相机。当前PID3848934正式运行，未完成不提前计账。结束后不再新增该方向配方。
- GPU1新独立`velstand-v8-repaired-worktree`，本体/执行器/反馈保持冻结V2/P04，实际v8机制适配而非直接用XL330教师。CPU八项边界验证PASS：本体合同、零速/移动/倒地/body路由、终止bootstrap组合、真实head-cap起身teacher均值等价和独立N、teacher不被训练、PPO与BC两个Adam及teacher哈希精确恢复、拒绝换teacher/缺teacher。GPU64×5刚启动PID3856970/session20549；尚未准入或正式。
- 新VelStand从V2 walking512加载actor/critic76/N/PPO Adam和真实LR，重置的是新技能课程时钟；同任务resume保留两套Adam和时钟。倒地用micro1856，零速且无body请求用站立teacher，移动用walkingteacher；body有效轴请求PPO单独学习，因为实际V2站立teacher的body奖励为0。教师专用yaw映射暂未启用；V2校准网格55命令×4×2seed完成439/440生存、fault0，有非单调与一例失稳，不把未验证逆表直接入训。
- 新stage计划先合并走/站/恢复能力，首128后评估，再至多一个有证据的128修正；精确总控制随实际N准入锁定。所有头稳追加计入同一stage，不能改run名重置预算；有效动作/姿态不退化是前提。独立终止命令状态隔离、教师物理成本/初态、实际N与评测仍待完成，未拿CPU检查冒充全部GPU通过。
- GPU0专职在当前段收尾后接GroundPick有限准入，已实际发现stock weight=0 payload reward被RewardManager跳过，负载原本没施加；将验证移动到明确step事件后的力/力矩与终止副作用，保留失败证据。不做轮滑或P4/host部署。

## 2026-10-16 06:56 用户澄清：头控与稳像仍是重点，全部重点都有阶段总预算

- 用户强调头控、尤其头部稳定性仍为重点；此前“平台期可停止”不能解释为降低其重要性。当前阶段/步行/VelStand/路线页已纠正，并同步专职。头部响应、物理头/相机运动及200Hz曝光跨度分别记录，静稳、转向/启停、主动头命令分开；不拿关节精度代替世界坐标稳像。
- walking阶段总上限锁到512：fresh256+YawFine128+当前同源128，合计402,653,184正式控制；本段至512后统一复盘所有重点，默认保留最好点并安排VelStand/其它策略，不在本段再加head/stop配方。SitStand阶段首128+ClearSeat128合计100,663,296；一个目标/高度修复后独立末评，不自动续第二段。起身专项head已结束，历史账不归零。每个run有界之外，还要按技能问题累计总成本；新诊断与新有限预算才重开，平台本身不是无限追加理由。
- 实际双卡已确认：GPU1正式PID3783182到390，新增4,718,592控制；GPU0成熟64×5至133完整PASS，32timeout/5oracle0/ONNX最大1.19e−6，之后actualN16384×32 PID3793156/session83141已自动启动。ClearSeat正式将从原128父到256，不从验证160继承。要求独立SitStand末评补实际头部/相机稳定记录，训练源与本段参数不热改。


## 2026-10-16 06:52 双卡已接续：ClearSeat准入与步行384→512

- GPU0唯一健康SIT候选hip−.7/knee−.8/ankle−.15、roll±.0873；32worlds的settled .5–3s无self-contact/max tilt6.45°、末1s实测高.075999m且静止，全3s无fault。另三候选均倒地，不抹去失败。新独立`sitstand-clear-pose-worktree`/ClearSeat任务进入CPU差分及mature64×5 PID3788221/session85034；同128父保留actor/critic/N/Adam/LR/clock/RNG，只修SIT目标/height，实际指标/评测同步读取task目标。之后16384准入→128正式自动链由专职继续，尚未计正式。
- GPU1第二次384 measured exit0（session14274），两seed每400oracle全0，336/336生存/fault0。首轮写出后−11未重现，根因未明；干净复测单独保留，不能说已找到/修复原生析构原因。严格记录逐位对比失败：10,014个报告字段不同；初始qpos/qvel/raw actor/origins及命令逐位相同，首拍线速差<=6.41e−7、角速差<=2.32e−5，闭环随后明显分离，不能当同轨迹。全部保存测量有限，原/新文件hash各自核对。一次失败严格检查与差异报告均留档，没有扩大tol伪装exact，也未为此无限重复测。
- `walking-yaw-fine-evaluation-recovery.json`明确PASS_CLEAN_INDEPENDENT_REEVALUATION，只依据干净重复完整电池作接续；原失败watch/review/continuation计划都保留。新同源384→512/128队列session92128，正式PID3783182已启动，06:51父384/本段0状态，不能把预算100,663,296算完成。源/六个资产hash再核对，未热改训练代码；原生500课程的奖励/曝光/DR同时变化已在计划披露。449/512共同评测56785、384可视化恢复83781已排队。
- 384名义零命令仍约4.312次/s双脚原始50Hz接触事件、同足stride均值.0245m，需继续解决持续踏步；原始事件未按历史双采样接触确认，不用它当严格步频。前进+.10同足stride均值.0615m，不能仅根据步幅或生存宣布整体步态合格。
- 最新预算审计实际06:44:54（文件0646）：基线根2,629,828,608/39段，P04随机含legacy3,341,746,176/36段；rise_stand5,619,253,248、walking301,989,888、sitstand50,331,648。当前新段尚未计完成。


## 2026-10-16 06:45 官方v8/历史遥控约束与两项实际问题

- 用户要求官方VelStand v8参考后训练、借鉴历史遥控步态/转向/无碎步/零速不跺脚。已重新读取固定官方manifest、3750776配置与distill代码，明确说明文档仍写v7，实际v8开启teacher yaw remap、宽倒地初态和>75°全关节教师输入mask。已写[VelStand计划](sc0090-v2-velstand-plan.md)，后续基于V2自身行走/起身教师迁移，不能复制XL330动作和响应表。
- 重新核对历史完整腾空配方的成功2×2对照及失败单改下限；当前P04已有完整.125–.300s/weight3，不做低速衰减。历史零速无跺脚成功入口依赖两策略交接；单网络停止模仿MSE下降仍可首次停下就倒。步行计划记录实际来源、短松杆/再起步/脚步复发/脚开合/持续限位要求；头控平台期不阻塞其它技能。
- SitStand实际GPU settled128/128确认两侧ground_hip_collision与同侧foot_collision约−8.30mm穿入，尽管tilt<3.75°、height.06321/fault0。早先“端点稳定”只能证明姿态保持，不能证明没有内部干涉。专职正在一次候选筛选+一批GPU保持后只修SIT目标和实测高度，保留模型与奖励/课程，原128失败不删除。
- Walking YawFine128完成100,663,296，整进程1071.370815s/93957.47控制/s；稳态median112740.44/aggregate108687.53。真实timeout bootstrap88133，contact_peak177883/cap1146880。321 nominal/measured336/336、fault0，384 nominal336/336。
- 384 measured报告写出完整336/336、fault0、两seed各400oracle零差，随后进程退出−11。没有Python traceback/内核core（ulimit0），尚不能证明具体原生析构根因。外层评测watch、review384与512准备队列均按失败停止，保留全部日志。相同冻结源/模型/两seed新目录`walking-yaw-fine-000384-measured-attempt2`已启动session14274，开启faulthandler；需exit0及原始报告/轨迹核对后才恢复后续，不删除或隐藏首轮异常。
- 已准备同源384→512/128（100,663,296预算），尚未执行；原生500阶段.action_rate−.1→−.2、零速2%→5%、head±.17/.17/.21/.047与CoM3→5mm。依据偏航RMS/正转持续改善有理由继续一次有限课程段，但不宣称负转、停止或所有指令已学会；末384/新512共同评测仍需完成。


## 2026-10-16 06:35 用户确认头控平台后转其它策略

用户明确允许：头控持续平台期就停止专项改进，训练其它策略。执行为保留已有最佳头控候选/局限，停止起身head追加；GPU0继续SitStand真实平衡/姿态修复，GPU1继续walking有界训练。新技能常规头控奖励/评测仍保留，不将理想精度作为阻塞全部技能的前置条件。已同步专职代理和当前阶段合同。

GPU0奖励审计已实际跑完mean与固定端点两臂：正确站总奖励率15.669>坏站11.513，正确坐10.533>坏坐6.143，未发现全栈排序颠倒。正确端点在该条件各16/16保持；坐端点self-collision惩罚约−3.50，正在识别实际接触对，尚未随意删惩罚。坏坐/起身轨迹长期倒地解释有效曝光不足。

GPU1中间321名义336/336、feedback0。共同口径256→321：idle vx .0627→.0173，yaw+.50的尾部yaw RMSE2.1601→1.3504rad/s，但平均yaw仅.0578，尚未准确转向；前进+.30实际.2872→.2572，收益有取舍。新`walking-yaw-fine-321-nominal-comparison.json`按同工况/seed校验，RMSE通过平方误差汇聚，未平均分位数；measured仍在队列。384共同末评后已排有界轨迹可视化queue35692，纯评测/渲染不计正式控制。


## 2026-10-16 06:28 GPU0接续空档核查与步行正式进度

- 用户更正为空闲GPU0。06:26现场GPU0利用率1%/1027MiB、无train/eval进程；SitStand128及其nominal/measured末评已完成，后续未及时接续。这是实际调度空档，不能以“子代理在做事”冒充GPU训练。已followup现有专职代理，继续独占GPU0，要求具体PID/预算和行为依据，文档汇总不阻塞接续。
- SitStand nominal/measured各256例、stand→stand/stand→sit/sit→stand/sit→sit各64，3个命令阶段所有tail_hold均0；两个坐姿初态组每组64/64 tilt>45°，站姿保持组0/64 tilt>45°但完整保持仍0。反馈故障0、oracle五项全0，问题是行为未学会，不能称测量故障或原样追加长段。先检查倒地样本占用、奖励分解与端点/过渡曝光，再最小修复。
- GPU1 YawFine实际32768×16准入已PASS：median117175.99/aggregate117079.05控制/s，max mean KL .016133/max clip .234094，16次是验证不计正式。从原256父正式128更新已启动PID3734661，06:27状态实际290、本段26,738,688控制；384和100,663,296仍是预算上限。评测队列等待321/384，继续共用旧物理评测口径。
- 台账同步当前阶段/路线页，仍不开展轮滑或部署，不把GPU占用率当策略成功。


## 2026-10-16 06:22 步行发现段完成，单因素偏航精度续训准入

- 首段fresh步行256完成201,326,592正式控制。外置整进程2084.648941s/96575.78控制/s；内部从env前计时2022.237945s/99556.33不含imports，不能混称整进程。同期独立评测和可视化采集在GPU1并发，准入独占稳态111549.79不代表整个进程速度。
- 四点nominal/measured生存：65=335/335，129=336/333，193=336/336，256=336/334，各条件分母336，全部feedback-invalid0。256 measured前进+.30实际+.278m/s，后退−.30实际−.237m/s，线速已明显形成；但+.50原地转向实际−.315rad/s，+.70曲线仅−.147rad/s，零命令仍约+.06m/s前移。未把生存或前进改善当全部步行合格。逐工况汇总`walking-discovery-evidence-summary.json`保留源SHA。
- 129的三次measured失稳在stop/reverse：曲线停下3.86s、快速前进反向4.66s、转向反向5.96s，全部实际tilt>45°且单足接触；无NaN/feedback故障，非误把低站高当摔倒。离线诊断首版漏推进case索引，被计数一致性断言拦截；已修并逐项对应336例，失败脚本/日志留档。
- 保存GPU qpos/qvel的独立65回放已完成。6工况同初始刚体对齐、固定共同斜侧相机、1×8秒、共同请求积分路径/实际轨迹/网格，选每组第一例而非最佳例。`walking-review-000065-video/walking-review.mp4`及manifest；已目视2/4/8s帧，能看到偏转和停止后移动。它是GPU轨迹的CPU运动学渲染，不是CPU动力学迁移或实机验收。
- 只优化评测的步幅汇总：逐时间步GPU布尔索引改为一次cummax/gather。840个已保存工况阶段stride计数/均值零差，实际完整stride向量逐位一致。单次诊断旧.01432s→.000435s，不能冒充整评测提速倍数。原运行评测源没改，新外置`evaluate_v2_walking_fastsummary.py`供后续使用；actor/物理/命令/奖励与其它指标未改变。
- 原walking正式controller只有finite/optimizer guard，原通用smoke才有每步nacon guard；已如实补全日志容量扫描并回扫全程，无9类已知警告命中。新yaw源再加入每物理步GPU nacon guard，不能把原段称为已有该guard。锁定Warp实际消息是broadphase/narrowphase/nefc/njmax_nnz/CCD等，已同步专职补齐。
- 根据193→256偏航回落，下一段只增加历史2026-10-01独立yaw Gaussian（weight2,std.35），保留原stock宽角速度项；不删除远误差信号、不扩大head/DR、不冻结N、不改PPO。实际新worktree`walking-yaw-fine-worktree`，任务`Mjlab-Velocity-Flat-MicroDuck-V2-YawFine-Repaired-S0-O0-R1`。12CPU PASS；64×5从256成熟父精确恢复actor/critic/N/Adam/LR，32timeout、4oracle最大1.19e−7、nacon324/2240。实际N32768×16 PID3729617；queue4348通过后只正式128到384、100,663,296计划控制，正式父仍256，不用validation272。新source-transition三文件差分、父SHA和完整新旧源SHA保留；准入未完成前计划不计正式。
- queue43177在实际321/384使用共同旧物理任务及新等价步幅汇总做nominal/measured；训练新增yaw奖励不混进共同评测分数。没有同父同128无改动控制，后续不能把全部收益归因于额外奖励。停止/侧移/头部精度仍单列，未声称该项能一并解决。
- GPU0 SitStand两个端点128组均稳定，但简单关节插值双向128/128倒；不能把目标斜坡当动作模板。真实successor32×5与状态隔离PASS。32768下ncon200 EPA申请20GB OOM、cap64又在首次step graph OOM，均0正式更新；锁定接口不能独立设CCD容量，未改求解精度/时间步/碰撞。16384×32后准入median100936.72/aggregate99037.67，真实timeout16384、5oracle全0、nacon129479/1048576、无已知容量警告。fresh正式128已完成50,331,648控制，末行为评测由专职继续，未宣称坐站学会。
- 新审计`training-budget-audit-20261016-0619.json`实际06:18:02：基线根39段2,629,828,608，P04随机根含legacy35段3,241,082,880；按技能rise_stand5,619,253,248、walking201,326,592、sitstand50,331,648。验证/评测另列，无未解账目；文件名不是采样时间。

完整技能清单仍保留。源码确认官方Spin实际使用ROLLERS本体，属于用户排除的轮滑；没有遗漏另一个双足Spin训练。仍不进行P4/host/Web部署或实机运行声明。

## 2026-10-16 05:27 完整负pitch结论与下一技能正式启动

- 用户指出的[-10,-5]承接已通过扩大75%定向站姿课程做完4594→4658/64。相同1024开发例中2→1次首次稳定前倒地，双方最终全恢复/fault0；剩余pitch−6.75894°、roll−8.57346°，.98s倒、2.24s首次满足连续稳定。平均首次稳定.75891→.75041s，但P95 .90→.92s，不能宣称全面更快。GPU离线报告`paired-negative-pitch-full-4594-4658/catch-times.json`逐项与原统计相符，含post-control/censor校验。完整共同measured另有1例supine保持回退，保留失败和旧全过候选。
- GPU0新phase+accuracy1984已两条件1792/fault0/standing0，头3(尾2)/115(114)/129(120)/8(6)，两轴退步；一次active聚合离线核对不支持直接认定mean导致整体退化，停止该奖励追加。专职起身正式累计2,732,064,768（含未选支），所选新候选1,560,281,088。GPU0随后进入SitStand准备，独立源/任务/记录，不再延长起身head试验。
- GPU0 SitStand真实物理预检已启动PID3629598/session53375：128环境、坐/站×精确/扰动，3s保持→2s过渡→3s保持→2s返回，无reset；真实successor32环境5边界queue62775排队。尚无结果，不把初始化称为物理通过。
- GPU1步行独立指标与有界controller准入PASS：64×5/32timeout/120oracle零差；4项CPU语义测试；32768×16带指标median111549.79/aggregate110862.00控制/s，maxmeanKL .015473/maxclip .223129、384oracle零差。nominal/measured各400controls×42worlds评测接线通过、actor/N冻结；5更新模型只会站住/指令位移近零，明确非会走。
- 新controller同源resume5→7精确恢复actor/critic/N/Adam/LR，clock120→168；ONNX84实际观测max7.8231e−8。外置checker错误传入未启用CNN/RNN配置，两次错误及修正留档；训练代码、模型、导出没因此修改。`walking-final-qualification.json`冻结源、assets、评测和输入SHA。
- 首次fresh步行formal PID3625334/session44077已真实提交10更新/7,864,320控制（本次读status）；单段上限256=201,326,592是计划，不计已完成。GPU1实测约85%/36.6GB。为摊薄初始化采用驻留单进程，native保存actual65/129/193及最终256，按checkpoint元数据计账。外置评测queue27226在GPU1余量≥4500MiB时逐点nominal/measured，保留并发开销；无自动下一段。
- 完整process wall由`run-walking-discovery.py`记录；controller内部wall从env构造前开始、不含module imports，不能冒充整进程耗时。后续以实际位移、速度、接触步态、头部和生存决定课程，首段不跨500更新升级。
- 正式预算审计已适配新技能metadata和explicit validation排除（旧脚本保留）。`training-budget-audit-20261016-0519.json`实际快照05:16:21：基线根39段2,629,828,608，P04随机根含legacy33段2,989,424,640，合计72段均rise_stand；当时步行尚未正式开始。预检/评测与之后步行状态单列，不按N大就计正式。

当前原全策略目标继续；基线只起身站立对照，后续默认完整P04；不轮滑、不部署、不称实机通过。详见[当前阶段](sc0090-v2-current-stage.md)与[步行计划](sc0090-v2-walking-plan.md)。

## 2026-10-16 05:08 核验

步行实际N带诊断验证已结束：16384×16 median94889.69控制/s、未达目标；32768×16 median113286.22/aggregate112768.37控制/s，max pooled KL .015961/max clip .229826，满足预声明护栏。选择32768用于下一准入，仍不是正式步行或技能通过。旧8192漏诊断失败保留；fresh诊断修正已完成64×5集成验证，动态归一化首批分布不宣称逐位一致。GPU1当前进入训练/独立评测入口准备，GPU0起身phase+accuracy段继续。

预算审计`training-budget-audit-20261016-0457.json`的实际快照为04:56:56：基线根39段2,629,828,608正式控制，P04随机根（含早期legacy）32段2,939,092,992；新phase+accuracy段当时尚无提交status，明确未计入。预检、评测、未提交状态和父点重复均不计；不是项目总GPU耗时。

最后更新：2026-10-16 04:53，Asia/Shanghai。当前聚焦起身、承接、头控与迁移；总体目标未完成。

## 最新快照（续接先读）

### 2026-10-16 04:41 全负pitch对照完成；GPU1进入修复P04步行准入，GPU0继续头控

04:53运行追记：步行8192×16进程exit0，median69893.28/aggregate69154.49控制/s，169.152s；准入外置脚本随后缺ppo-diagnostics.jsonl而失败，非物理/策略崩溃。原因是旧smoke仅在分布有exploration_metadata时安装诊断，fresh默认Gaussian没有该属性。已在独立walking源仅修改smoke接线，始终调用已验证helper ba7b2314…；capped旧严格检查保留，fresh报告实际首批分布一致性（False）且不强行冻结N。原源脚本/plan/log保存，未启动正式步行。重新64×5诊断验证PASS、32timeout、规范导出通过；新 `run-walking-throughput-diagnostics.py` / session38301从16384×16 PID3578937继续，不重复已明显不足的8192；再不够才32768。旧8192验证成本仍计入，所有新PPO诊断存在后才选N。

- 上一目标轮PROGRESS：完成4594/全桶复核并实际启动定向课程；本轮已完成4658正式/全部评测，准备并验证新步行入口、启动实际N验证，仍PROGRESS。全策略目标ACTIVE，轮滑/部署仍排除。
- 新全负pitch课程4594→4658/64已完成50,331,648控制，所选基线谱系1,444,282,368；formal578.245s/87042.11控制/s（短64未全程10万），包含本配方小/实际N预检及标准末评pipeline1339.163s。外置原KL/clip护栏未触发，全部64已检查；不自动第二段。
- 4658 nominal恢复1792/fault0/standing0；共同measured恢复1791/fault0/standing0。唯一seed1592 supine trial791于1.34s已经达到连续稳定，最后稳定标志7.5s，末.5s无稳定标志；8s物理height .112495m/tilt2.177°/lin.001643/ang.01606，本体没有末端严重倒地的几何证据，但严格连续稳定失败必须保留。未存逐帧contact原语，不能指定是哪项接触/gate导致；报告 `paired-catch-full-pitch-4658-recovery-retention.json`。不以最终姿态看着好修改判据。
- `paired-negative-pitch-full-4594-4658`已同完整[-10,-5]/roll±10、8s、1024配对结束：父4594倒2，新4658倒1，均首次稳定前，双方最终1024稳定/fault0。这是开发集、有训练预算差，非独立同父同64控制或实机验收；保留旧1792全过的4594/4466及独立micro1856，4658不是全指标替换。
- 4608 head29(尾14)/116(116)/219(188)/11(8)；4658 head38(20)/115(112)/227(194)/6(4)，final80(77)/253(233)/7(6)。新1°尾保持92/66/217/10，.5°54/37/50/2，均同时身体尾稳定；相对4594小幅到位和当前pair提高、final pair下降，保留取舍。来源`paired-catch-full-pitch-4658-absolute-precision.json`。
- GPU0同步cap005续段因1971极端KL已于1984安全停64，实际50,331,648，原128剩余未做；phase同1920父64也已完成50,331,648，maxKL .072889/maxclip .17004、第51 .011999，formal677.221s/74320.90控制/s，52准入稳态101400.43另记。phase头2(2)/109(108)/152(140)/4(1)，同父同步为2(2)/114(113)/162(148)/4(3)，无全面行为收益且更慢，不直接延长或推广所有技能。measured评测入口旧allowlist漏新ID导致0模型/0物理步拒绝，已显式补ID、失败attempt1保留并重跑；不能把遗漏当策略失败。专职下一单因素phase+独立accuracy .5同1920/64在单位/52准入流程，不自动追加。
- 两条起身/实际P04迁移已有完整通过保留候选，余下头控/边缘保持由GPU0继续；依原全策略目标，GPU1完成本轮倾斜验收后转入步行短准入。后续工程默认完整修复P04、baseline保留唯一起身站立对照，依据是已有正式>100k和独立起身能力，不是实机优胜结论，也没有宣称全头控已完结。先前可选阶段分工问题不是必需审批，不以用户无回复推导授权；原全策略范围与已完成基础迁移支持本阶段推进。
- 新独立`walking-repaired-prep-worktree`（base070d606）从已验证测量/动力学源建立，仅新增显式Repaired walking两个ID和历史head_bias可选rebase；原rise factory不改，默认函数逐位保持。旧walking入口spike=True/O0local=False的问题在新入口显式修复，compiled/fused=True，TerminalCriticPPO、HOME0、.112高度、balanced head零.5、原airtime3/.125–.300与coarse head发现配方保留。7项CPU测试PASS，未改共享依赖；source-parent与qualification保存SHA。
- 新步行64×5已PASS：32真实超时自举、61→14规范ONNX误差1.788e-7、物理graph开启、measurement1capture/478replay。新actor/N/Adam随机初始化，未沿用stand归一化；仍非学会步行或正式长训。`run-walking-throughput-sweep.py` / session7153正在8192×16，按8192→16384→32768最多三档，选达到>100k与原optimizer护栏的最小N，之后停止，先完成步行指标/预算合同再长训。完整下一步见[步行计划](sc0090-v2-walking-plan.md)。

### 2026-10-16 04:10 完整负pitch残余定位；只扩大初态覆盖的新对照已进入预检

04:17运行追记：GPU1成熟64×5 PASS（65 timeout bootstrap、accuracy oracle120次/max1.788e-7）；真实32768×20 PASS，47350个bootstrap样本，480步跨第一8s周期，首分布/冻结N/实际clock全部核验。中位130437.18、aggregate124778.08控制/s；max pooledKL .017637/maxclip .218919，末5中位 .008203/.089039，未放宽护栏。正式PID3516633/session7043已实际4595/4658、本段786,432控制，actor/critic/N/Adam恢复逐位、LR继承1.5e-5、source_changed=true仅新注册/初态范围，不是恢复失败。外置watcher82760只读已有PPO诊断，沿用max KL .5/clip .8，超限请求既有SIGTERM安全保存并标记下游失败；不改变训练数值或参数。watch初始化阶段修正半行JSON边界与失败传播，两个先前WAITING版本归档，均未向训练发送停止信号。GPU0首次deadline相位候选64×5已PASS，实际N32768×52 PID3516534在跑；1984末measured也1792/fault0/standing0，绝对.5°尾10/59/46/1、1°尾33/98/207/10，存在精度收益与相对组合取舍，仍保留1984而非称全面退化。

- 本轮完成accuracy4594/100,663,296新控制、标准两条件与1024负pitch复核、实际执行链探针、GPU稳定耗时与绝对精度补算，并建立下一单参数课程实验；PROGRESS。总体仍ACTIVE，当前只起身/站立/相关头控与迁移，无其它动作或部署。
- `paired-negative-pitch-accuracy-4274-4594`已完成：共同measured实际P04、同开发seeds1603/1604、同捕获初态、pitch[-10,-5]/roll±10、完整8s、1024例；4274与4594均2倒（全部首次稳定前）、1024最终稳定、fault0。4594失败初态为pitch−5.2623/roll−8.3986与pitch−6.7589/roll−8.5735，首次倒地.96/.76s，随后2.10/1.96s达到原连续.5s稳定；普通standing0/256确实漏掉这两类边缘样本。
- `catch-times.json`按post-step时刻GPU补算，全部计数对原报告PASS。该次4274/4594平均首次达稳定为.73852/.75885s，P95 .86/.88s；直接承接末稳定均1022/1024。4594微幅头控提高但承接没有继续改善，不能把晚期head训练当作承接已修好。与独立micro1856的0/1024仍分开保留；不是实机或最终留出验收。
- 下一对照源为真正detached git worktree `paired-catch-full-pitch-worktree`（base070d606），父为4594 resume。唯一环境变化是站姿75%定向pitch从[-10,-7]扩大至[-10,-5]，25%全±10仍保留；roll/yaw/高度/关节、2s站姿+8s其它恢复、Head5+accuracy奖励、头命令/σ、PPO/T24/N/测量均不变。站姿reset条件下[-7,-5]预期占比2.5%→32.5%，原极负区曝光会相应降低，因此也检查[-10,-7]保持，不能预报全面改善。
- 新CPU/GPU配置/采样检查均PASS：新旧train/play移除该一个参数差异后规范配置相同，PPO全部相同；评测qpos与RNG逐位不变，部分reset未选行/原roll/yaw/height/joints保留。GPU实际中段占比.324514、CPU .326429；selected .750143。全部src Python只有feedback_repair注册/参数文件有差异，trainer/smoke/其它计算源逐位一致；源与哈希见source-parent.json及plan。
- `run-paired-catch-full-pitch.py` / session7043已进入成熟64×5预检PID3501374；之后32768×20，480控制步跨过第一完整8s timeout，记录timeout bootstrap/冻结N/首分布一致/precision oracle与原KL、clip、>100k吞吐护栏。它仍不证明所有更长段优化稳定。均PASS才从原4594父正式64到4658（50,331,648计划控制），不是从预检权重续训；父实际LR1.5000000000000004e-5、clock110256，末111792，不重置LR。
- 新配方先以静态父为参照，没有新同父同64未扩区控制，因此不能把所有差异当作单参数因果胜利。4608/4658 head、完整恢复/standing/common自动；`run-negative-pitch-full-4658.py` / session92893等末评全部完成后再做同1024/8s的4594→4658配对。正式未开始前不把计划交互算实耗，无自动第二段。
- `paired-head-execution-4594`17工况×8×2已PASS，实际自身Head5/训练source、measured、actor/N冻结，每seed300 MLP/动作与每actuator1200实际motor调用，目标链/四子步count保持逐位，全部记录有限；无另算动作/电机、无真值补actor。与fresh旧nominal探针及81工况的初态随机布局不同，不作严格配对；完整记录供后续有界判断，未继续改探针接口。
- GPU0 cap005续段在第51更新1971出现一次pooledKL2.603、clip .408、梯度峰10917与return variance .041→11.410，首次分布重算仍精确且N冻结，随后KL恢复。没有逐样advantage/完整奖励记录，不能证明根因是timeout/bootstrap。按1984安全边界保存并正常退出，实际只64/50,331,648（原128剩余64未执行）；完整formal550.769s/91384.39控制/s。nominal恢复1792/fault0/standing0；head2(2)/114(113)/162(148)/4(3)，比1920略回落，不盲续。measured末评由专职继续。
- GPU0下一候选只分散第一次timeout相位；审计发现episode_length_buf还服务真实年龄门控与历史初始化，因此不会直接随机它。专职准备独立首次deadline、原物理年龄从0、正常重置后仍8s，独立RNG/初态q/v/obs逐位证明，实际N至少52更新覆盖原51风险再决定同1920父64。此为待验证方案，未把它当已修复软件根因。

### 2026-10-16 03:48 承接与头控共同保持；新完整负pitch复核已排队

03:59追记：003训练和全部标准末评已exit0，完整pipeline1371.576s；4594两条件均1792恢复/fault0/standing0，头30(22)/116(113)/204(168)/7(3)，final79(77)/267(249)/7(7)。小幅保持比4466的11提高到22，但两轴187→168/四轴13→3，故保留4338/4416/4466/4594，不自动同配方第四段。4594的1°尾保持87/69/187/11、.5°50/33/55/0，见精度审计。负pitch全桶配对PID3482413/session96616已实际运行；末后queue70588 `run-paired-head-execution-4594.py`再执行同源/自身Head5的17工况冻结实际执行链探针，沿用专职已验证SHA449406aa脚本、不改训练源，也不与fresh nominal数据当严格配对。GPU0正式首更新已验证，实际到1949/2048；resume-rng-qualification PASS/same-task-later-parent，新父活动流1f06fefe…与恢复逐位一致，旧1856流未复用，LR=继承1e-5而非重置。source_changed=true仅trainer资格修复，计算源码不变，11边界PASS。

03:55运行追记：GPU1已实际完成4594，本段100,663,296控制/所选谱系1,393,950,720；完整formal994.765s/101192.99控制每秒。4480/4544/4594均nominal恢复1792/fault0/standing0；4544头26(12)/112(111)/209(179)/15(10)，末4594 head及common仍在评测，不预写完成。GPU0新cap0051920→2048/128正式PID3472734/session67085已启动初始化；仅trainer新父RNG资格逻辑修复，11项边界检查PASS、物理/PPO/奖励等计算源hash不变，同源预检复用，不重置LR。第一次实际更新仍待核验。4466新增绝对精度报告显示1°尾保持84/64/200/18，并非所有指标比4338提高，见精度审计页；有取舍候选继续保留。

- 本轮已完成accuracy4466及cap0051920正式结果、核算预算快照并实际继续训练，属PROGRESS；总体目标仍ACTIVE。当前仍只起身/站立/相关头控与迁移，不启动其它动作、轮滑或P4/host部署。下一阶段双卡分工的可选问题尚待回复，不将无回复当作改变阶段的批准。
- GPU1 `paired-head-accuracy-allocation-002` 已4338→4466/128，新增100,663,296控制、所选谱系1,293,287,424；源/PPO/奖励/课程不变，exact恢复PASS。nominal与共同measured均1792/1792恢复、fault0、standing0/256。head小29(尾11)/当前单116(115)/两轴217(187)/四轴17(13)，final单80(78)/两轴263(245)/四轴9(8)。中间4416为25(14)/116(114)/207(179)/15(8)，微幅尾保持未单调提高，保留中间候选。
- 4466完整formal992.901s / 101383.02控制交互/s；连同两次head和共同measured的continuation pipeline1370.033s，见 `paired-head-accuracy-allocation-002-timing.json`。不是全pipeline每秒10万，也没有把200Hz物理子步当50Hz控制交互。
- GPU1已同源精确resume4466→4594/128，`paired-head-accuracy-allocation-003` / PID3455497 / session73695。03:46状态实际4550、本段66,060,288控制；计划100,663,296，common clock107184→110256。resume-verification PASS/source_changed=false，没有重新设置LR/归一化或重复准入。4480/4544/4594恢复、4544/4594头控、末common自动，不自动追加第四段。
- 用户关心完整pitch[-10,-5]直接承接：保留现有同初态1024比较（T24-4274倒2、micro1856倒0，均最终全稳定/fault0），不把它算到4466。新 `run-negative-pitch-accuracy-4594.py` / session96616等待003正式及标准评测全部完成，再同GPU1做4274→4594的512×2、完整8s、roll±10、measured实际S0/O0配对；旧开发seeds1603/1604、同原始Head5评测边界、捕获初态逐位核验。分别报首次稳定前倒、后倒、最终恢复，不以2s训练截断替代8s验收。该测试尚未完成。
- GPU0 cap0051920从micro1856同父/同64已完成50,331,648控制，nominal与measured均1792/fault0/standing0。head3(3)/116(114)/165(150)/8(3)，final114(111)/229(221)/2(2)。同预算无改动control1920为2(2)/113(112)/153(138)/6(4)：单/双轴保持较好，四轴保持较差；相比静态micro1856的两轴171(160)仍未超越。两者保留，不宣称σ降低全面胜出。cap完整formal553.053s/91006.93控制/s；实际N稳态132642另记。
- GPU0冻结执行探针 `head2-micro1856-execution-probe` 已PASS：17工况×8×2，记录实际MLP输入、actor mean、动作处理、原生命令count和每5ms真实motor输入/PWM/输出，未重复计算motor或拿真值构造actor。small ±1.719°命令已使实际目标相对idle移动约1.78–2.55°，但物理位移仅约.05–.64°；因此不能再说网络完全没读命令，也不能单凭此把根因归为静摩擦。资料 `head2-micro1856-execution-analysis.json`。专职依据cap的局部保持收益准备同配方有界128，并修复“新父活动RNG hash被错要求等于旧预检父”的续训资格判断，原预检只证明同实现，恢复必须核对本次父的实际状态；准入结果由专职页跟踪。
- 新正式预算审计 `training-budget-audit-20261016-0326.json` 的实际快照为03:24:32（文件名不是采集时刻）：67段，基线根37段2,466,250,752控制；P04随机根含旧执行30段2,822,701,056。当时accuracy002在4450、cap在1900，各1段尚运行。每段status内容与SHA来自同一次读，保存快照正文；8个验证/无正式status目录明确排除，无未解决账目。非全部GPU成本，不包含预检/评测/快照后学习；旧01:51快照保留。
- 已纠正主台账底部把8192理解成硬要求的旧表述；当前两条实际P04正式均32768×24，基线本体历史8192。预训练审核增加成熟阶段PPO/奖励/normalizer/RNG/完整耗时更新，不让历史阶段表覆盖现状。

### 2026-10-16 03:13 共同倾斜初态比较完成；基线精度候选续训，从零做同预算小噪声对照

03:16运行追记：GPU1 PID3417279/session14302已实际4380/4466、本段33,030,144控制，clock105120。GPU0无改动control全部measured验收也1792/fault0/standing0，已结束；cap005 CPU/GPU单位和成熟64×5已PASS，实际同卡rng_restore源0/目标0、mapped=false、活动流hash相同，首小N KL .45705/clip .58919保留不抹去。真实32768×16 PID3426428/session32673在跑，仍由原实际N护栏决定正式准入，不把小N通过当正式学成。

- 上一目标轮为PROGRESS（T12完成、全负pitch复核、新accuracy两卡实际开训）；本轮完成两条accuracy段行为证据、跨路线同初态比较、修复跨GPU随机流恢复并实际启动下一段，仍PROGRESS。总体目标ACTIVE，当前只起身/站立/相关头控/迁移，不扩展其它动作或部署。
- GPU1 `paired-head-accuracy-allocation-001` 4274→4338/64完成50,331,648控制，所选谱系1,192,624,128；nominal与共同measured均1792/1792、fault0，standing分别1/256（首次稳定后）和0/256。完整formal578.220s/87045.78控制每秒；包含本配方预检和末评的pipeline1278.036s另计，短64段不能声称全程10万。
- 4338头严格小20(尾9)/当前单115(112)/两轴193(170)/四轴9(7)，父4274为15(8)/110(110)/181(151)/7(5)；final大范围单75(72)/两轴238(223)/四轴9(8)。首4288为17(8)/108(105)/172(147)/4(3)。附加1°末保持小79→83、单66→66、两轴175→213、四轴14→22，全部这些通过样本同时满足身体尾段稳定；0.5°为28→41、34→33、40→60、1→1。保留原相对T90与绝对指标，未改验收线，也不说全头控已完成。资料`paired-head-accuracy-4338-absolute-comparison.json`。
- 新完整负pitch配对`paired-negative-pitch-routes-4274-1856/report.json`：共同实际measured P04、pitch[-10,-5]/roll±10、8s、512×2开发seeds1603/1604。before=T24-4274倒2/1024（均首次稳定前），after=独立micro1856倒0/1024；双方最终全稳定/fault0。qpos/qvel/raw actor obs/origins逐位匹配，三源全部sc0090_v2/actuator Python hash一致。共同Head5奖励只作评测分数，不冒充fresh原Head2训练回报，也不声称90态/全部私有状态配对。不同谱系/总预算，不是训练因素因果实验；4338没有被偷算成该4274样本结果。
- GPU0 accuracy1920也已nominal/measured均1792/fault0/standing0；头3(1)/111(107)/156(144)/3(3)，静态父micro1856为1(0)/114(112)/171(160)/11(6)，不替换1856、不追加新奖励。其完整formal560.655s/89772.89控制每秒；含物理预检和全部末评pipeline1201.595s，CPU开发/单位另计。
- GPU0新增同1856父、同64的无改动控制`head2-micro-control-allocation-001`已训练到1920，nominal1792/fault0/standing0。头2(2)/113(112)/153(138)/6(4)，1°尾保持27/85/168/10。accuracy1920相对该同预算控制两轴156(144)略高，却单轴/四轴保持更差；因此不能把相对静态父退化全部归因新增奖励。两者都不足以覆盖1856保留点。
- 下一matched对照只有head cap .01→.005；原Head2/micro .25/T24/全部其它PPO/LR/N不变、无accuracy项，同1856父64/50,331,648控制。原拟用GPU1，因root accuracy实际改善，改为GPU0控制末评完成后接实验臂（queue32673），GPU1继续基线有效路线；未启动旧GPU1 cap队列。专职继续管理GPU0。
- 为跨卡计划检查发现并修复实际恢复问题：旧set_rng_state_all未将父训练卡活动流映射到目标卡。真实1856保存的CUDA0/1随机流确实不同。仅新`rise-head-cap005-worktree`增加`rng_resume.py`，按真实parent_run.device映射，CPU/其它流保留；同卡路径与旧恢复全状态逐位。跨0→1后normal/uniform/randperm/CPU随机抽样逐位一致，缺失/模糊设备及provenance冲突在修改前拒绝。helper SHA eb196432cfe67bf2e048f6d5a03c117ccc0ccdf904cdf8379bd5a0c7d5236e53，报告`head2-cap005-rng-oracle.json`/`head2-cap005-rng-change.json`。只证明Torch随机流，不是完整模拟器状态恢复；不热改任何在跑控制源。当前实验仍同卡，预期active_stream_mapped=false。
- GPU1新同配方128：`paired-head-accuracy-allocation-002`，4338→4466，100,663,296控制；资格SHA相同，精确resume，无重复预检/LR切换。当前PID3417279/session14302已实际4341、2,359,296控制，clock104184；resume PASS/source_changed=false，父clock104112、LR3.375e-5，末clock107184。自动4352/4416/4466恢复、4416/4466 head、末common，之后再判是否继续。
- 首续训queue62130遇到报告先落盘、上游COMPLETE稍后写出的竞态，正式0更新、输出目录未创建即退出，原脚本/log/plan与`paired-head-accuracy-allocation-002-startup-repair.json`保留。已改为同时等最终报告和COMPLETE并容忍写入中的临时JSON；确认原句柄exit1后才重启session14302。训练源码/父点/预算不变，不把启动错误当PPO不稳定。

### 2026-10-16 02:43 完整负pitch桶已复核；停止T12，双卡转独立精度附加奖

02:51 GPU0末点追记：accuracy1920正式nominal恢复1792/1792、fault0，standing0/256（前后均0）；head measured PID3382992及后续measured恢复仍自动验收中。专职记录完整formal560.655s/89772.89控制每秒，与稳态131758/s分列；当前还不判定头部奖励成功，micro1856父保留。

02:50准入追记：GPU1单位/成熟64×5/真实32768×16全部PASS；物理oracle分别120/384调用、最大1.19e-7/1.79e-7，smoke69个真实timeout bootstrap。actualN中位131066.23控制/s、max meanKL .014368/maxclip .192030、末5中位 .010433/.111959，未放宽原护栏。queue61527已正式PID3379405，已实际4280/4338、本段4,718,592控制、clock102720；源/准入见`paired-head-accuracy-qualification.json`。GPU0正式已1920/1920、本段50,331,648控制，末评仍由专职顺序完成。

新精度奖口径补充：band按当前目标绝对幅度，匹配零起点step的物理误差带；idle四轴用.002。它不等同任意旧命令到新命令的动态T90，也不等同原return T90按上一step幅度定义的容差。原到位/回中/末保持和附加绝对误差评测都保留，不用新reward值替代验收。

- `paired-negative-pitch-t24-t12/report.json`完成，queue42121正常退出。完整pitch[-10,-5]、roll±10、每seed512×2、8s、真实measured S0/O0、同捕获初态：T24-4274为**2/1024倒**（均首次稳定前），T12-4402为**4/1024倒**（前1后3）；两者最终1024稳定/fault0。两例差异不足以断言显著优劣，但没有支持把T12当承接修复。
- 新样本细分：[-10,-7]605例T24=0、T12=3；[-7,-5]419例T24=2、T12=1。T24残余初态(pitch,roll)分别(-5.256,-9.009)°和(-6.759,-8.573)°，.80/.84s首次倒；见`failure-initial-poses.json`。不能继续把问题只归到-10°边缘，也不能将旧[-10,-7]的4/1024与此次2/1024当同初态改善。当前保持原承接课程，新reward实验不叠加初态改动。
- T12完整共同measured恢复1792/fault0、standing0/256；nominal则1/256（前）。末4402头严格小21(尾12)/单90(86)/两轴136(120)/四轴3(2)，相同100M预算T24-4274为15(8)/110(110)/181(151)/7(5)。附加1°尾保持也从79/66/175/14降至68/57/149/8；最终大范围同样保留，未只挑小幅获益。保留T24-4274作为当前基线根综合参考，停止T12追加；这不等于全头控或实机通过。
- `paired-rollout12-comparison.json`读取真实末checkpoint验证两者父SHA相同、初clock99504/末102576、总候选控制1,142,292,480、各新增100,663,296；其它PPO与去进程地址/源路径的环境配置相同。T24完整formal998.509s/100813.62控制每秒，T12为1011.854s/99484.05。formal开始至全部末评分别1380.557s/1381.429s；T12含新预检全pipeline1681.572s，不能拿这个与未重复预检的T24直接排名。
- GPU0 `head2-accuracy-allocation-001` PID3366906已实际1857（本段786432控制），32768×16准入131757.93控制/s/maxKL .02663/clip .30773，64×5和物理cached reward oracle均PASS。1856→1920/64只加独立accuracy .5，原Head2/micro .25/T24/身体课程不变；没有用root分支通过替代该谱系准入。
- GPU1新真实git源`paired-head-accuracy-worktree`，从保留4274父只加相同独立有界精度项：按被命令轴平均，idle四轴平均，band=max(.1*abs(command),.002)，reward=1/(1+abs(物理误差)/band)，权重.5。原Head5内部乘式精度与active weighting、Catch2s/负pitch/σ/T24/全部其它PPO保留；不重置Adam/LR/N。共享helper SHA746d6b3d7a3e40deaf5a88f9a904604e754db399e6b185d89e324dbb3a0ba29e，未复制Fresh环境。
- GPU1本任务CPU/GPU oracle PASS（原Head5奖励逐位、train/play去唯一新项配置全等、所有PPO相同、命名轴/HOME/正负/idle/激活轴及band边界）。queue61527已实际进入64×5成熟smoke PID3371842，再自动32768×16，原maxKL/clip与>100k准入通过才4274→4338/64，计划50,331,648控制，初clock102576→104112，真实父LR2.25e-5。此时尚未正式学习；`paired-head-accuracy-plan.json`保留边界。两个预检模型丢弃，正式仍从4274。没有新同4274父/同64无奖励控制，不能只凭父子对比宣称严格单因素因果效果。
- 台账、当前阶段、候选与承接页同步；当前工作仍只训练/评测起身、站立、头控和迁移，无其它动作或部署。总体目标ACTIVE未完成。

### 2026-10-16 02:33 T12完成同预算正式学习，补齐整个负pitch桶验收

- GPU1 `paired-rollout12-allocation-001` 已完成4146→4402/256，真实100,663,296控制；common_step=102576，与T24-4274同父/同预算/同物理时钟。完整formal进程1011.854s、**99484.05控制/s**，略低于T24的100813.62；不把准入129063/s或200Hz物理子步冒充本段完整速度。最大update meanKL .03154、clip .34712，256次更新完整保留，正常完成不是头控改善证明。
- 三个nominal节点4224/4352/4402均1792恢复/fault0；standing分别2/256（首次稳定前2）、2/256（前1后1）、1/256（前1）。完整4352/4402 measured头评与最终共同measured仍由queue41425顺序验收，当前head4352 PID3344883。不得将恢复全过写成无倒地。
- 用户要求的整个pitch[-10,-5]桶独立补验已排队：`run-negative-pitch-rollout-comparison.py` / queue42121，待上条训练评测队列完整结束后用GPU1。同4146父和100M预算的T24-4274/T12-4402，各512×2固定新开发种子1603/1604，8s、roll±10、实际measured S0/O0。分列[-10,-7]与[-7,-5]、首次稳定前/后倒地、最终稳定及fault；保持原阈值。捕获初态qpos/qvel/raw actor obs/origins逐位匹配，不声称私有模拟器状态完整回放。只读评测脚本SHA d4a142ebde8f37a3f60101a81d6fd7f309e09ffc62e80807fe0cff80668a8be0；首次跌倒顺序的小型边界oracle通过。
- GPU0仅腿Gaussian冻结对照已PASS：两seed、90捕获态、23项完整奖励、actor/N逐位不变、0fault/0倒；mean→legs当前两轴153(尾145)→210(195)、四轴4(3)→12(10)，小幅仍0/128。两轴尾MAE .689→.578°但step完整return27.7086→27.5598，neck+改善而roll+变差；因此不直接cap腿或宣称解决微幅。完整all-Gaussian补臂由专职继续，不把噪声诊断当正式学习。
- 专职下一有界训练方案：保留micro1856、原宽Head2奖励/微曝光/身体课程/σ/N/Adam/LR，只增加独立有界的命令轴物理精度奖，64更新；完整对照确认后按CPU/GPU oracle、成熟64×5、实际N护栏顺序准入。尚未正式训练，细节/首版零物理启动错误与修复由专职页记录。

### 2026-10-16 02:12 负pitch新种子配对17→4倒；保持平台后切换成熟期T12同预算检验

02:23运行追记：GPU1 T12已实际4290/4402、56,623,104控制，common_step=101232，仍PID3323358/session41425。GPU0 roll-only1920 measured已1792/fault0/standing0；未提升目标头轴，保留1856。新仅腿探索诊断session74282首mean PID3330571，head2-micro1856-fullreward-mean→head2-micro1856-fullreward-legs-gaussian两臂自动，原micro源/full23奖栈/nominal/81×8×2/90捕获初态；源audit_fresh_p04_legs_noise_v3.py SHA1c4c73387e227afaf6c9711b29e6e7575b06a6b8e1307b49ad733a8fd54da081。头外加噪声为0与同一步注入前后头分量逐位，不意味着两条闭环head mean恒同；结果尚未完成，未先改生产探索。

02:18追记：T12实际32768×16已PASS，中位129063.16、aggregate123053.91控制/s，clock99504→99696正确；max meanKL .019199/maxclip .23056、末5中位 .010989/.127155通过未放宽护栏。正式PID3323358/session41425已实际4170/4402、9,437,184控制，clock99792=99504+24×12，exact恢复PASS；`allocation-clock.json`声明最终102576。sample片段缩短不改变2s/8s episode与真实reset规则。GPU0 roll-only1920头3(尾1)/110(110)/171(160)/6(4)，目标roll−MAE .470→.543°、roll+ .729→.909°且7→3到位，停止盲续，保留1856。专职准备只给腿加继承Gaussian噪声的冻结90初态配对诊断，头不额外加噪；不先假定腿运动/静摩擦根因，也不未经验证直接改腿探索。

- 上一目标轮为PROGRESS：完成4146及冻结拼接反证、实际启动第三段；本轮完成第三段、负pitch新种子配对、真实训练账目审计与T12配置/物理smoke，不是单纯等待。总体仍ACTIVE，范围未变。
- GPU1第三段cap4274已4146→4274/128，100,663,296新增控制；完整formal998.509s/100813.62控制/s（Popen→退出），全部continuation pipeline1380.557s另记。所选基线谱系1,142,292,480控制。nominal和共同measured均1792恢复/fault0，standing均1/256，均首次稳定后；不能用最终恢复替代保持。
- 4224头小15(尾10)、当前单110(106)、两轴157(141)、四轴5(3)；4274为15(8)/110(110)/181(151)/7(5)，final大幅单76(73)/两轴224(215)/四轴10(8)。相对4146的10(8)/107(106)/170(150)/7(3)，主要保持指标几乎平台，不自动追加第四段T24。
- 新 `paired-cap-pitch10-frontier-4146-4274` 已完成：实际measured、512×2新固定seeds1601/1602，pitch[-10,-7]、原roll±10、完整8s、共同成熟head5全奖栈；捕获qpos/qvel/raw actor obs/origins逐位配对。4146倒地9+8=17/1024，4274为2+2=4/1024，两者均全1024最终稳定/fault0。每seed无倒99.609%，仍有失败，不称普遍零倒；新种子用于当前开发比较，不冒充最终留出集。两源全部sc0090_v2/actuator Python逐项hash一致，计划/源码/逐项奖励和轨迹保留，queue65451完成。
- [路线与正式预算审计](sc0090-v2-route-comparison.md)新增真实父checkpoint元数据核算：01:51:28快照共60正式目录，基线根含适应2,200,436,736已提交控制，P04随机根含旧执行2,637,103,104；后者分为专职2,379,743,232与早期旧执行257,359,872，和已有专职账目一致。8个验证/无正式status目录明确排除；快照非全GPU成本，未包含预检/评测/快照后更新，不用未选分支冒充所有路线必需成本。
- GPU1新真正git worktree `paired-rollout12-worktree`，任务 `Mjlab-StandUp-Flat-MicroDuck-V2-PairedAdapt-Catch2sPitch10-LateEntropy005-HeadStdCap001-Rollout12-Repaired-S0-O0-R1`。唯一行为配置差异是num_steps_per_env 24→12；train/play、head5全部环境及其它PPO参数逐项一致PASS。采样batch786432→393216、minibatch196608→98304，每更新仍5epoch×4batch；相同100,663,296控制下128→256更新、Adam步2560→5120。
- 新实验从同4146父精确继承actor/critic/N/Adam/实际LR5.0625e−5及common_step99504；不使用默认cfg学习率.001覆盖继承值。最终4402更新时物理clock应102576，恰与旧T24的4274相同；每次更新增加实际12控制，增加只读时钟断言与allocation-clock记录，不能把旧4146×24重算为4146×12。奖励、命令、v5、S0/O0、PPO其它项不变。
- GAE γ=.99/λ=.95保持；更短片段的尾权因子(γλ)^T从.2294到.4790，更依赖critic bootstrap，不能预报更好，也不外推fresh技能。现有官方24配方及控制保留；这只是成熟阶段同控制预算的有界消融。
- 成熟64×5已PASS：真实clock99504→99564、32个timeout bootstrap、精确恢复与冻结N/分布重算/ONNX检查通过。queue41425已进入实际32768×16 PID3319409；同maxKL/clip护栏及稳态>100k通过才正式4146→4402/256。计划 `paired-rollout12-plan.json`，正式最多100,663,296，不将预检训练参数用作正式父。当前未预报正式通过或head改善。
- GPU0 micro1856已nominal与measured均1792/fault0/standing0；nominal头0(0)/114(114)/155(149)/5(4)，measured1(0)/114(112)/171(160)/11(6)，小幅/roll短板在训练条件同样存在，不能归咎只有电压验收变难；非完整初态配对，不作电压效应因果量化。
- GPU0新roll-only隔离源准入全部PASS：非roll奖励/梯度逐位不变，命名轴顺序置换和独立float64 oracle，head评测公式同步；实际32768中位131521/aggregate131837，maxKL .01577/maxclip .2048。正式1856→1920/64已完成50,331,648，nominal1792/fault0/standing0，完整formal552.419s/91111.3控制/s，期间重叠旧1856 nominal头评已记。head measured PID3310839及后续measured由专职继续，未预报roll改善。

### 2026-10-16 01:43 保留4146候选；直接拼接头策略暴露保持退化，不采用

01:46追记：GPU1第三段exact resume/source_changed=false已验证，实际39更新至4185，max update meanKL .01722/maxclip .22437，4160恢复1792/fault0。GPU0微命令1856头评已完成：小1(尾0)、当前单114(112)、两轴171(160)、四轴11(6)，final单114(112)、两轴212(209)、四轴0。当前单/两轴有进展，微幅本身仍未学好；measured全恢复还在自动验收，不以曝光统计代替效果。

- GPU1 `paired-stdcap-allocation-002`全部评测完成：nominal与共同measured均1792/1792、fault0；standing nominal2/256（首次稳定前1、后1），measured1/256（前），仍负pitch。4146头小10(尾8)、当前单107(106)、两轴170(150)、四轴7(3)；4096为8(4)/106(106)/155(137)/7(3)。相对4018的5(3)/110(105)/134(121)/8(5)，小幅与两轴持续改善，单轴/四轴并非同步改善。final大范围单70(69)/两轴232(221)/四轴7(6)，继续保留失败，不称全头控完成。
- 4146所选基线谱系1,041,629,184控制，本段100,663,296。完整formal进程1000.704s/100592.51控制/s，**连同4096/4146 head及共同measured整条continuation pipeline1368.558s**另记（不含此前预检与其它支）；不将全链速度说成100k。全部过程见 `paired-stdcap-allocation-002-timing.json`。
- 只读新 `evaluate_v2_head_frozen_splice.py` 将4146的10个腿均值与健康paired3762的4个头均值组合，两网络各自原normalizer读取同一真实测量61D输入。两者权重/统计冻结，实际输入验证拼接各维逐位来源，self/self组合逐位等于原actor。没有学习、真值补输入或P4/host部署；此两网络组合不是训练候选。源/父文件SHA与执行计划见 `paired-frozen-head-splice-4146-plan.json`，queue33053正常结束。
- 同协议measured81×8×2，捕获初态qpos/qvel/raw actor obs/origins逐位相同（不冒称全部私有传感器状态完整回放）；拼接后头小49(尾1)、单128(105)、两轴334(84)、四轴45(4)，invalid0。到位改善却small/两轴保持显著退化，不采用直接替换，亦未据此添加teacher蒸馏目标。
- `paired-frozen-head-splice-4146-comparison.json`分开尾.5s偏差和抖动：small平均active轴绝对偏差 .928→.668°，trial内中心化RMS .027→.210°；两轴偏差 .748→.760°、RMS .0283→.2775°；四轴偏差 .760→.728°、RMS .0411→.2191°。两轴尾段身体稳定384→382/384，不能只看头到位数。当前单网络有明确的平滑优势，下一步要消除偏差而不是用抖动扫过目标。
- 新0.5°/1°附加指标覆盖4096/4146，见 `head-absolute-precision-4096-4146.json`。4146在1°带尾保持小70/128、当前单64/128、两轴184/384、四轴13/64；与原相对T90不是同一排序，两个口径都保留。用户精度偏好仍待答复，不预设新验收通过线。
- 有持续但有限的单网络行为改善依据，GPU1已精确resume **4146→4274/128**，源/PPO/reward/head5/Catch2s/Pitch10/测量/normalizer均未改，无重复预检、无额外LR重置；`paired-stdcap-allocation-003` / queue75468 / PID3265132已实际4151（3,932,160新控制）。预算100,663,296，4160/4224/4274恢复、4224/4274 head、末共同measured自动；不自动第四段，所有已有候选保留。
- GPU0 `head2-micro-exposure-allocation-001` 1792→1856/64已完成50,331,648：nominal恢复1792/fault0、standing0/256；完整formal553.458s/90940.3控制/s，max update meanKL .02331/clip .2243。真实命令零/单/双/四59.833/20.039/10.071/10.056%，四轴微幅每控制6.047/6.096/5.829/8.867%；采样参数与实际曝光分开核验。from-zero全部formal实耗2,379,743,232、此候选谱系1,459,617,792。head measured PID3267732及共同measured验收仍由专职自动推进，未预报微命令学成。

### 2026-10-16 01:30 cap续段完成且完整训练进程突破10万；头控增加绝对精度审计

- 上一目标轮PROGRESS：真实完成纯cap4018与共同measured、启动相同配方有界续训；本轮实际完成128更新、从零新候选验收与新的精度诊断，仍PROGRESS，不宣布整体目标完成。
- GPU1 `paired-stdcap-allocation-002` 已4018→4146/128、100,663,296新增控制；Popen起点至进程退出**1000.703687s，100592.51控制交互/s**。包含导入/初始化/真实P04测量与物理、PPO、三个内嵌恢复节点，不含既往预检或之后head/common；没有把200Hz物理子步计为控制步。两卡分别已有完整正式进程>100k证据，不等于所有短段或整个试验pipeline都>100k。
- nominal节点4032/4096/4146均恢复1792/1792、fault0；standing依次3(首次稳定前2/后1)、3(2/1)、2(1/1) of256。保留负pitch和首次站稳顺序分析，不能以最终恢复掩盖倒地。末head/common尚在queue83459自动运行，未预报头控变好，也未自动追加第三段。
- GPU0 cap+LR1792已正式及measured均1792恢复/fault0、standing0/256；head小3(3)、当前单105(104)、两轴164(155)、四轴11(7)，final单112(111)/两轴217(213)/四轴0。neck+8/16、MAE2.190°，parent1728为5/16、2.645°；roll+仍4/16/.822°，未全面通过。正式50,331,648控制，完整formal544.334s/92464.59/s，含小smoke/实际N/正式/全部末评的物理pipeline1152.269s另计；候选独立谱系1,409,286,144，全部from-zero正式实耗2,329,411,584含未选支。
- 从零下一因子由真实奖励斜率和曝光决定：fine .04→.02会把当前neck+2.190°处奖励斜率7.304降至1.973，仅27%，故暂不采用收窄。新 `rise-head-micro-exposure-worktree` / HeadStdCap001-MicroExposure025仅加precision_probability .25，zero仍.6，活跃大范围保留75%，轴组合/正负保持；CPU/GPU百万采样验证neck微幅1.43%→6.07%、yaw1.14→5.82%、roll5.12→8.78%。继承1792全部actor/critic/N/Adam/LR，不重复切换LR。
- GPU0新queue22332成熟64×5 PASS，actual32768 PID3245458准入中；smallN首KL4.08，后恢复，不把finite小样本当实际N优化准入。原same护栏通过才能1792父64；无新同起点64控制，因此不宣称严格单因素因果胜利。专职页记录准确进度。
- 新[头部精度和命令格点审计](sc0090-v2-head-precision-audit.md)：现有不可变measured轨迹只读GPU补算0.5°/1°绝对带、.5s/2s到位与尾.5s保持，原T90不改。cap4018在1°带尾保持小60/128、单72/128、两轴141/384、四轴11/64；独立cap+LR1792为29/88/212/14；健康paired3762为113/93/279/35。说明当前头弱不只是原小幅指标严，仍须改善。用户实用精度偏好已异步询问、尚未答复，不自行指定新通过门槛或暂停已授权实验。
- 三层步长复核PASS：native .3222656°/count；实际O0目标 .308938972°/count；v5 motor API .005394751527rad/count。±.03rad最近命令格点落在.003带内，不能归咎单纯量化；0/±.005部分格点超.002，但不代表物理静态平衡不可达。没有修改测量、actor输入、HOME、v5或指令/验收尺度。只读脚本两次小错误/浮点1ULP失败日志保留，修正后PASS，不记入训练进度。

### 2026-10-16 01:14 承接和头控共同验收，GPU1同配方有界续训

- 续接已核验实际GPU/进程/文件；上一目标轮为PROGRESS，本轮也有新完成行为证据和实际下一段，不以复述状态冒充推进。总体目标仍ACTIVE，当前起身/站立/头控/迁移范围不变。
- GPU1纯cap `paired-stdcap-allocation-001` 已完成3954→4018/64、50,331,648控制交互。正式nominal和共同measured恢复均1792/1792、fault0。nominal standing4/256（2首次稳定前、2后），共同measured3/256（均首次稳定前）；全部残余倒地在负pitch桶。比纯低熵的2/256与1/256略差，不把小样本差异抹掉，也不称cap改善了承接。
- 首3968 head小6(尾3)、当前单111(108)、两轴117(100)、四轴4(4)；末4018为5(3)/110(105)/134(121)/8(5)。同父同64低熵控制为3(2)/98(94)/101(82)/4(1)，普通熵控制3(2)/99(98)/86(74)/7(6)。cap有限恢复了当前单轴/两轴能力，远未达到健康paired3762的组合成绩，小幅仍弱。final大范围单72(72)、两轴252(240)、四轴5(5)，不能只挑当前两轴报成功。
- 所有头评仍为真实measured、确定性均值、同时到位持续.1s及末.5s全程保持；保存比较 `paired-head-exploration-4018-comparison.json`。这些报告初态协议一致，但不是完整仿真状态逐位配对；同更新号的三臂必须用目录+SHA区分。
- 本次纯cap正式从/proc进程起点到ALLOCATION_COMPLETE状态592.2188s、84988.26控制/s，见 `completion-status-wall-time.json`。**没有捕获精确process exit**，该口径包含导入/初始化/学习/内嵌恢复和重叠首head，排除状态落盘后的close/退出及前置预检/末head/common；不得写成完整全链突破10万。稳态准入130679/s另列。下一段用Popen开始到wait返回准确记录完整formal进程和整条评测pipeline。
- GPU1继续已有有效方案，而非新增奖励或重复代码优化：`paired-stdcap-allocation-002` 同任务同源码、**--resume 4018→4146/128**，32768×24，计划100,663,296新控制；冻结normalizer/继承actor+critic+Adam+LR及成熟课程，保留Catch2s和75%[-10,-7]+25%全±10初态。没有再次LR切换或无意义重复预检。
- 续训前真实准入已通过：两环境全恢复/0fault、nominal倒地≤5、measured≤6、当前两轴≥134/尾保持≥121、已验证源hash未变。queue83459、PID3219599已实际4023/4146，新增3,932,160控制交互；resume-verification PASS且source_changed=false，`run-paired-stdcap-allocation002.py` / `paired-stdcap-allocation-002-plan.json`保存命令/理由/门槛。正式4032/4096/4146自动恢复，之后4096和4146 head、末共同measured与standing顺序/pitch；无自动第三段。不能把“已启动”当已完成128。
- GPU0 cap+一次性LR切换真实32768×16已PASS：最大更新meanKL .01840/maxclip .1754，末5中位KL .00986/clip .1316，中位132342.0/aggregate132270.6控制/s；未放宽先前护栏。新正式 `head2-stdcap-lr-allocation-001` PID3208754已1728→1792/64完成50,331,648控制，正式nominal恢复1792/1792/fault0、standing0/256已核验；完整formal进程544.334s/92464.59控制/s，短64未达10万全进程。完整head/measured自动验收中。优化稳定不能替代头控行为；此为cap+LR复合改变，非纯cap单因素。专职agent继续负责GPU0。
- 用户负pitch承接目标已有直接证据：普通Pitch10-4018在[-10,-7]同初态1024例中107→35倒地，仍全恢复/fault0；原[-9,-6]3954为0/1024。这里不把不同区间的零倒泛化到[-10,-5]全桶，也不以最终恢复掩盖先倒再起。当前共同目标是继续承接同时找回准确头控，保留所有有取舍候选，无实机验收结论。

### 2026-10-16 01:02 头部探索限制：基线分支已正式学习，独立P04需更小切换步长

- GPU1 `paired-full-head-only-sampling-3954`已完成：相同nominal/全23项head5奖栈/冻结3954、90捕获初态回放，四个head action维度加继承Gaussian噪声、腿外加噪声逐步为0。`paired-head-only-full-reward-comparison.json`中双方预command稳定子集小幅MAE1.130→.671°、两轴.989→.711°、四轴.930→.692°，对应step完整return25.523→25.418、25.494→25.272、24.811→25.121；保持通常变差。局部偏差受头噪声影响，但不存在Gaussian全局更优结论。当前负neck两seedMAE1.66/1.81→.39/.74且回报提高；roll+则误差/保持恶化。完整6份reward配置与formal+runtime polish逐项核验PASS。
- 新共有 `HeadCappedGaussianDistribution` 在同一个update中构造有效Normal：head std取min(raw,.01)、腿不变；采样/log_prob/entropy/params/KL/storage共用该分布。按POLICY_JOINTS名字选头，限制buffer非持久，原参数key及导出Identity不变；仅影响训练探索，不修改测量/反馈或部署均值动作。raw std参数在恢复时继承，之后仍受原Adam更新；不能把“未重置”误写成永久冻结σ。
- 两条谱系CPU/GPU直接分布与实际RolloutStorage测试均PASS：腿同RNG样本逐位、有效σ入库、logprob/entropy对独立Normal一致、直接ratio1/KL0、上下cap梯度、原state_dict严格兼容、确定性导出不变。root `paired-stdcap-unit-tests.json`还核对train/play环境+head5和全部其它agent字段不变。
- 真实小N smoke揭示FP32跨batch重算并非数学严格零：专职同父actor64/384批复现mean最大9.54e−7rad、std逐位、ratio偏差.000164；真实smoke最大ratio约.000359。改为**std必须逐位相同、首KL≤1e−6、首ratio偏差≤.002**，记录mean/std差，未称严格零或放过明显分布错配。还修复timed_log包装丢失原logger签名导致的KeyError 'it'，使用functools.wraps，数值优化运算不改。所有首版失败日志保留，均未产生正式候选进展。
- 共享class SHA e47e8a240c912e45b0f08ce7d5469e2f91f030ce839065fe9a5098cc05612b14；diag v2 SHA ba7b23147263c5b9a54e9475f6996d2e77d189d177c6eb8952ece74667452a8f；共享最小patch-v2 SHA3544aad3404f8cd3dabce9660d4909f6ef07b26228ed70096bfcc45606a0ef45。root仅移植class/diag与必要preflight/恢复元数据，未复制Fresh/Head2环境配置。
- GPU1新真正detached源 `paired-head-stdcap-worktree`，任务 `Mjlab-StandUp-Flat-MicroDuck-V2-PairedAdapt-Catch2sPitch10-LateEntropy005-HeadStdCap001-Repaired-S0-O0-R1`；仍从3954、同64，控制为已完成 `paired-late-entropy-allocation-001` 的4018。唯一臂间行为因素为head有效σ上限.01；entropy .005/head5/[-10,-7]短承接/冻结N/32768×24/5epoch/4minibatch均保持。原parent raw σ/actor/critic/N/Adam准确恢复PASS，不将验证段权重当正式父点。
- root成熟64×5及实际32768×16均PASS；13稳态更新中位130679.11、aggregate125623.41控制交互/s。预先设定optimizer护栏（max update meanKL≤.5、max clip≤.8、末5中位KL≤.1/clip≤.5）也PASS：实测最大KL .060627/clip .345314、末5 .012191/.149214。首step σ重算差0。此为避免明显优化失稳的门槛，不是行为验收。
- **root已正式运行**：queue9163，PID3194594，`paired-stdcap-allocation-001`已实际3956/4018（1,572,864控制交互），实际有效四头σ均.01rad；watch93536记录process start并在3968做head，末4018head/common自动验收。source、plan与资格汇总见 `paired-stdcap-plan.json` / `paired-stdcap-qualification.json`，正式不能仅按4018编号混淆普通/低熵/cap三支。
- GPU0同纯cap在真实32768首更新meanKL5.0805/clip .509，后续恢复正常但预定maxKL护栏FAIL，已阻止正式，formal0；完整16验证与吞吐132325.5/s及失败源已归档，12,582,912验证交互另算，不充作候选正式进度。不得因root通过就把此谱系也放行，亦不放宽护栏。
- GPU0新 `rise-head-stdcap-lr-worktree` / WarmLR1e5任务：明确是**cap+一次性切换学习率**新配方，先准确恢复，再把算法及所有Adam group LR设1e−5，矩/step及其它字段不变，保留原adaptive后续调度；同任务resume拒绝重复切换。CPU配置/Adam状态测试PASS，helper SHA666a3b796a40c3a4a9bd281f021e0c2ea5b50272cc0a463be0d9f22675fb2e02。session25982/小smoke PID3198141已自动进入64→32768→同optimizer护栏→1728父64计划。此分支不再声称只改σ，普通1792仅作综合行为参考，不作单因素因果控制；root无需加入未经自己数据支持的LR切换。
- 当前仍不宣称完整头控、实机或全策略路线完成。保留健康paired3762、from-zero1664/低熵1728及各承接候选，先看真实确定性head、末端保持、完整恢复与standing是否同时改善。

### 2026-10-16 00:39 低熵对照未全面改善，完整头部奖栈审计与头部探索检验

- 上一目标轮为PROGRESS：完成负pitch课程与大样本配对验证、启动同父低熵对照；本轮已实做完对照及完整奖栈诊断，不以状态复述当进展。总体目标仍ACTIVE，当前范围仍仅起身/站立/相关头控/迁移，无其它动作、轮滑或部署。
- GPU1 `paired-late-entropy-allocation-001` 3954→4018/64完成，新增50,331,648控制；nominal和共同measured完整恢复均1792/1792、fault0。nominal standing2/256（1先1后）与普通Pitch10同64一致；共同measured1/256（首次稳定后），普通控制2/256，不以单条差异声明显著胜出。
- 低熵末点head小3(尾2)、当前单98(94)、两轴101(82)、四轴4(1)；普通控制3(2)/99(98)/86(74)/7(6)，仅有限两轴收益，仍弱于3954父的108(93)，不追加同配方。补到的3968小7(4)/中114(114)/两轴103(86)/四轴7(5)，原正式恢复全过；这是冻结早期检查点的延后旁评，不是继续训练。
- 成功formal进程571.51s、88067.11控制/s；run.json口径519.24s、96933.68/s。额外的Git缺失启动失败和之前预检、之后head/common另记，不能混称完整总实验速度；本次首head因监控修复延期，控制组有重叠head，wall比较并非严格同负载。
- 原watch59280因注册Git工作树时目录迁移产生NFS ESTALE后退出，未影响学习。已从仍存活PID3123239的/proc恢复真实process start；新watch60153使用稳定绝对marker，在完整末次head/common后补3968旁评，已完成。原错误与恢复见 `paired-late-entropy-watch-restart.json`。不再将dead watcher记录为活任务。
- GPU0低熵1856当前头pair134(125)/four0，neck+MAE4.019°退步、σ约.038rad已平台；停止同配方盲续，保留1728及1664。专职完整Head2奖栈audit：1728 vs1856在nominal、受控2s idle/2s step/2s return，全23项按真实runtime polish，逐步只调用reward_manager一次，缓存分项×dt和return一致、冻结actor/N、90捕获初态回放逐位。
- GPU0正neck current更差模型step回报26.9896→26.7486，tracking/standing_composite/head_bias/legs姿态项均下降，不支持该关键失败由腿奖励反向鼓励垂头。部分small/组合有身体收益抵消头精度的个例，但不能推广为所有命令同机制。完整Gaussian加噪也不是全局收益更高，且同时加腿噪声不能直接归因头部σ；其具体报告由专职页维护。
- GPU1复用已验证audit，自己的 `paired-frozen-sampling-worktree/scripts/audit_paired_full_head_reward.py` SHA d254e71a4f01904d1e7e542b30d9895e313169245a3788eade7f4c9d7207d14a；显式共同head5奖栈评健康paired3762与适应3954，原trained recipe另存。两臂nominal已完成，source/所有23项func/params/weights与3954 formal+runtime polish的离线逐项核验PASS；90捕获初态显式raw_registers回放逐位，非任意simulator内部状态完整证明。
- `paired-full-head-reward-comparison.json`按命令阶段与末.5s真实分项比较，另列双方t1.5–2.0s已持续稳定子集。该子集small111例active MAE .323→1.134°、2s完整回报25.876→25.516；current两轴346例.371→.981°、25.730→25.489；四轴59例.423→.931°、25.457→24.815。主要组更准确head有更高full return，暂不全局削减腿奖。current单轴跨轴物理角度MAE与相对T90口径不同，必须按各轴目标/容差或逐case判断，不能仅比较聚合MAE选优。
- 精确phase指标、奖励分项、逐case抵消反例、前置稳定子集和全样本均保留；这是观察到的轨迹排序，不是PPO梯度或所有轨迹的普遍因果证明。原head评多只计算stable reward，未把那些报告冒充完整训练回报。
- 下一单因素：专职准备仅头探索σ上限.01rad、腿分布不变，从保留低熵1728做同64对照现有低熵1792。采样/log_prob/entropy/KL/storage必须统一实际有效σ，继承原actor/critic/N/Adam原参数，不改变v5/S0/R1、身体课程、reward或部署确定性输出；CPU/GPU/成熟smoke/实际N准入后才能formal。raw参数σ与effectiveσ分开记录；这是待验证方案，不预报有效。
- GPU1在等待可复用cap实现时，session72323只读 `paired-full-head-only-sampling-3954`：在nominal、共同head5全奖栈、同冻结3954与90初态回放下，仅四个命名head action维度加继承Gaussian噪声，腿外加噪声逐步断言0（腿均值仍可随状态响应），用于区别之前全14维随机动作的影响。源SHA5f7f3ed4e62173807c4123f70aa206bbf8f6698ea2dae7a81e429a81dcad46db，未修改生产分布或部署随机动作。

### 2026-10-16 00:04 低熵准入完成，两条谱系继续有界验证

- 00:09现场确认：GPU1正式PID3123239已实际完成到3960/4018（本段4,718,592控制交互），actor/critic/N/Adam准确恢复PASS，live entropy断言通过，已不是仅启动进程。formal-only queue87293/首点watch59280继续。
- GPU0低熵128已完成1856，恢复1792/fault0、standing先倒/稳后倒0/256，段末head评测中。完整Python进程973.23s（含imports/model/reset/graph/PPO/两次formal eval及重叠一次head）实测103432.5控制交互/s；run.json口径109369.9/s、稳态132062.5/s分列。该128段真实100,663,296控制，达到完整训练进程10万门槛，未新增优化代码；fresh谱系所有正式实验累计2,279,079,936，所选谱系和失败预算详见专职记录。
- 00:07启动修复：第一次formal在Git provenance查询处失败，原因是根新隔离源采用复制且排除了.git，训练器要求git rev-parse HEAD。**正式学习0更新**，原失败log与空准备产物已保留为 `paired-late-entropy-startup-missing-git.log` / `paired-late-entropy-startup-missing-git-artifacts`；不是PPO/物理学习失败。现以原HEAD070d606注册真正detached worktree，并回填相同已验证文件；所有准入源SHA完全不变，不重复smoke/吞吐。修复证据 `paired-late-entropy-startup-repair.json`。新formal-only queue87293替代已退出59109，watch59280继续保存进程start/首head；源与父点/预算完全不变。
- GPU1 `paired-late-entropy-qualification.json` PASS，CPU配置唯一entropy差异、成熟64×5（实际live .005、72条timeout bootstrap、统计逐位冻结）及32768×16全过。13稳态更新中位131178.78、aggregate125977.53控制交互/s，ONNX normalizer parity4.77e−7。queue59109已进入正式3954→4018/64启动，尚待首更新；watch59280自动保存真实进程start与3968头部旁评。对照是已完成的普通Pitch10-4018，不能只用迭代号区分两个模型。
- GPU0同配方128继续；1792节点恢复1792/fault0，head小3(尾2)、当前单103(103)、两轴147(141)、四轴5(4)，最终两轴244(236)、四轴11(11)。相比1728当前两轴158/156有回落、最终范围有进步，是权衡，不能称单调改善；保留1728当前组合保持参考，等待1856完整结果再决定是否延长。
- raw_registers初态差异的代码边界已由专职复核：CausalStream用zeros分配并在reset清零，不是未初始化内存。O0 q/v数值只取line_q3/末LS3 slope，speed/load/voltage历史仍参与整段有限性guard，故不能称完全未使用；未来S0生成使用独立当前speed/load，不从此ring反灌。其余89捕获张量含raw_q/time/status/history/output/ready均配对且全有限。speed/load重建历史为何不同尚未定位，保留覆盖限制，不以诊断回放冒充生产修复，也不停止有效训练。
- 训练候选表新增from-zero低熵1728及其完整measured证据，继续只做训练筛选；实际机器、全范围头控和未来全策略路线尚未验收。

### 2026-10-15 23:59 边缘承接验证完成，针对头控做同父低熵对照

- 本轮不是完成三项目标：GPU1补完Pitch10/64和完整证据，standing改善明确；整体头控与路线选择仍未完成。GPU0低熵同64取得组合保持与neck偏差收益，继续独立从零谱系；无其它动作、尖峰过滤、部署或硬件验收。
- `paired-catch-pitch10-4018-common-measured/pitch104018/report.json`已完成：完整原七桶/原±10°/8s/measured/seeds1591/92恢复1792/1792、fault0，standing2/256均首次稳定后倒，且全部在负pitch桶71条中；其余185条0倒，全部256最终稳定。对照3954同条件6/256（5先1后），不混同原formal的5→2。
- `paired-catch-pitch10-frontier-3954-4018/report.json`：实际measured P04、同完整成熟head5奖栈、[−10,−7]均匀初态512×2、8s。初态qpos/qvel/raw actor obs/origins逐位配对；3954倒61+46=107/1024，4018倒20+15=35/1024，均最终稳定1024/1024、fault0。各seed无倒地96.09%/97.07%，通过事先95%课程门槛，但**不是零倒或完整成功**。不把旧[−9,−6]1024零倒外推到新边缘。
- `paired-frozen-sampling-report.json` PASS_CAPTURED_INITIAL_STATE_REPLAY：健康paired3762与适应3954各mean/Gaussian两臂，81命令×8×2；每对90捕获张量初态逐位，Gaussian仅回放raw_registers历史；私有采样RNG、冻结actor/N与逐步真实奖励oracle通过。跨模型qpos/qvel/rawobs逐位，但raw_registers仍差异，单列而不冒称全状态配对。
- 新诊断mean→Gaussian：3762小86→65（尾8→0）、当前单128→127（尾116→80）、两轴353→325（尾209→15）、四轴51→27（尾13→0）；3954小5→83（尾2→0）、当前单114→128（尾114→84）、两轴106→320（尾96→21）、四轴5→34（尾2→0）。健康parent均值更好，适应后有明显采样依赖，但保持差，不部署随机抖动。
- `paired-frozen-sampling-steady-summary.json`：适应3954小幅正neck MAE1.158°→.615°、signed−1.158°→+.151°；当前负neck MAE1.576°→.582°；当前正neck本来均值已准，Gaussian只增抖，不能把所有轴说成同一问题。完整head诊断1296条包括最终大范围与初始承接，mean身体倒地parent128、adapted18，Gaussian125/31；分range/时段分别记录，不误称头评全程身体无倒地，也不与256 standing正式桶混用。所有feedback invalid为0。
- GPU0低熵1728：小2(尾1)、当前单102(100)、两轴158(156)、四轴5(5)，相比同父普通1728两轴111(103)有保留收益；neck+ MAE2.645° vs原1664 2.918°，neck− .470° vs1.138°。完整measured恢复1792/fault0、standing0/256；final四轴仍0，不能称所有范围改善。专职继续同配方1728→1856/128，session89442/PID3097659，首1792head及段末自动验收，无重复准入。
- GPU1新隔离 `paired-late-entropy-worktree`，任务 `Mjlab-StandUp-Flat-MicroDuck-V2-PairedAdapt-Catch2sPitch10-LateEntropy005-Repaired-S0-O0-R1`。从**3954**而非4018开始、同64预算；现有普通Pitch10 3954→4018作为同父/同预算控制，不重训控制。唯一臂间差异entropy_coef .01→.005；两臂都采用[−10,−7]、2s短站姿、同head5/PPO其它字段/32768×24/5epochs/4minibatches、冻结成熟N。σ、actor/critic、Adam、学习率和时钟准确恢复，不手动缩σ。
- CPU完整训练/play环境与head5 recipe逐项相等、agent只entropy不同，`paired-late-entropy-config.json` PASS。smoke与formal新增恢复后live entropy断言与预检匹配，防止只改配置没作用。主queue59109当前先64×5→32768×16→formal3954→4018，再完整head/common measured；watch59280保存真实进程start并首3968旁评。此时尚为准入，不预写低熵有效。
- 限制：正式训练不是逐rollout逐样本配对，仍是一种子受控对照；配置唯一差异/父状态逐位不等于优化后轨迹相同。GPU0首rollout也有微小非逐位差异，已单列。降低熵是否改善头部确定性与承接仍由本次结果决定。

### 2026-10-15 23:36 短承接课程取得实质改善，推进到 −10°边缘

- 23:49续记：Pitch10正式3954→4018/64已完成，50,331,648控制，恢复1792/1792、fault0，standing进一步降至2/256（1先、1后），均负pitch桶。`remaining-falls.json`显示初始pitch−9.9479°/−9.9248°，首次倒地.76s/.96s；第二条首次稳定.68s，因此是起始1秒内失稳，不是2s截断后才出现的长时保持退化。全部最终稳定。
- Pitch10段末head小3/中99/两轴86/四轴7，尾保持2/98/74/6；承接进步同时头仍下降。定向reset72465、原域24136，实际站姿时间21.6289%、首.5s5.60979%。完整Python进程含imports 587.05s、85736.69控制/s；run.json→完成538.24s、93511.30/s；稳态130937单独计。64段所有开销未过10万，不能用稳态替代。
- GPU1最终head已完成，冻结采样队列31865与完整measured/新前沿队列36683正在执行；两个评测并行、正式训练已停止，避免无人处理的空档。新前沿512×2 seed只将审计区间改[−10,−7]，原完整奖栈/8秒/初态配对要求不变；不能将旧[−9,−6]零倒外推新范围。
- GPU0低熵64完成，恢复1792/fault0/standing0，头评中。实际头动作std下降18–21%，不是手动重置；完整Python进程542.35s、92803.1/s，run.json口径102569.3/s和稳态131730.7/s分别记录。若行为无改善不追加；若改善才以有界长段摊薄启动，不为吞吐再做无止境代码优化。
- 候选记录已更新 `sc0090-v2-evening-candidates.md`：保留paired3762、from-zero1664、Catch2s3954、Pitch10-4018各自优势/短板与原文件路径；不打包部署、不提前宣称综合最优。
- 23:40续记：Pitch10全部准入PASS，64×5自然/强制合计72条terminal bootstrap；32768×16稳态中位130937.37、aggregate125536.20控制交互/s，28586条bootstrap。正式PID3065856已3990/4018，正常loss有限，首3968完整standing3/256倒（2先、1后）、恢复全过；首头小3/当前单109/两轴101/四轴4，未恢复头能力。`paired-catch-pitch10-qualification.json`汇总先于formal的分项检查；`process-start.json`另存/proc真实启动时间以便含imports核算。现场GPU0/1为95%/89%，不是空闲。
- 3954完整measured共同seeds1591/1592补分桶：standing6/256倒（5先、1后），全部在[−10,−5]的71条中，其余185条0倒；256条最终全稳定。这与nominal formal5/256是不同工况，分别保存，不混并。
- 新冻结诊断源 `paired-frozen-sampling-worktree` 复用子代理v3，仅补root head5独立active/inactive权重oracle（`active_count`避免覆盖300帧count）；脚本SHA bc1c43ce6b7e3ca48280c2096f860b83e7277bd31fd94d1fae0797c840139b24。mean/Gaussian四臂比较健康paired3762与承接3954，每对显式回放初始raw_registers、私有动作RNG、捕获初态逐位检查、冻结actor/N、逐步实际reward oracle。队列31865等Pitch10末次head完成后运行，不污染吞吐准入。跨模型初态差异单列，不能预称所有内部状态严格配对。
- GPU1 `paired-short-catch-allocation-001` 已完成3762→3826/64，standing倒地20/256（18首次稳定前、2后），恢复1792/fault0；定向reset72324、原域24190，比8s站姿控制的14705/4972定向次数约4.918倍。全段wall550.181s、91481.98控制交互/s，含旁评，不以稳态冒充全段。
- 同配方 `paired-short-catch-allocation-002` 已完成3826→3954/128，新增100,663,296控制交互；3840/3904/3954恢复均1792/1792、fault0，standing倒地18→10→5/256。段末5条为3条首次稳定前、2条后；完整±10°尚未全过。定向reset145750、原域48508；实际站姿控制时间20.7536%，首.5s为5.23376%。run.json至完成965.436833s，104267.10控制交互/s；含内嵌恢复/重叠承接诊断，不含之前导入和之后head/common验收，见 `paired-short-catch-allocation-002/actual-wall-time.json`。
- `paired-short-catch-frontier-3826-3954/report.json`：实际native-count-v2 measured P04，当前前沿pitch均匀[−9,−6]、原roll/yaw/高度、零命令、完整8秒，512×seeds2026101571/1572；3826倒37+30，3954倒0+0，双方final_stable1024/1024、fault0；初态qpos/qvel/rawobs/origins逐位配对。实际完整奖栈总和、有限性、冻结actor/N检查PASS。事先声明每seed≥95%无倒地、≥99%最终稳定、fault0的当前前沿门槛通过，不等同整个±10°合格。
- `paired-short-catch-3954-common-measured/short3954/report.json`：原完整七桶/完整±10°/8s，measured三电压、development seeds1591/1592，1792/1792、fault0，actor/N不变。与前沿门槛共同允许课程向更负pitch推进。
- 3954头控仍退步：measured小5/128、当前单轴113/128、两轴108/384、四轴6/64，尾.5s保持1/113/93/3。不能称综合最好；保留健康随机化baseline3762与from-zero1664参考。仅将3954用于承接课程后继，不覆盖头更好的候选。
- 新隔离源 `paired-catch-pitch10-worktree`，任务 `Mjlab-StandUp-Flat-MicroDuck-V2-PairedAdapt-Catch2sPitch10-Repaired-S0-O0-R1`，只把训练站姿定向区间(.75,−9,−6)改为(.75,−10,−7)，宽度不变、左移1°。保留25%原±10°、2s站姿/8s其它桶、原reset概率、root head5、冻结成熟N、32768×24/5epochs/4minibatches。CPU配置/部分reset/原eval初态及RNG逐位PASS；GPU与成熟64×5/32768×16队列18074按门槛串行，通过后3954→4018/64；首3968旁评watch61301。此时32768预检中，不提前称正式训练已开始。
- GPU0 Head2零命令.6→.3曝光64未整体胜出（1728小3/中96/两轴116/四轴2），保留1664。冻结mean/Gaussian诊断v3完成 `PASS_CAPTURED_INITIAL_STATE_REPLAY`：每seed90个捕获张量逐位配对，仅Gaussian起始回放raw_registers历史speed/load以消除非行为初始化差异，生产模型不变；未宣称捕获任意simulator内部/私有Generator全状态。1664 mean→Gaussian小2→108但尾保持0→0、两轴144→378但尾137→86；正neck平均偏差−2.897°→−.195°，随机保持仍不合格，不能部署抖动。
- GPU0专职隔离 `rise-late-entropy-worktree` 从保留1664做晚期entropy_coef .01→.005唯一因素64对照，Head2/零命令.6/全部身体课程/σ/actor/critic/Adam/N保持恢复；queue9752先成熟64×5与32768×16、检查live runner实际熵系数，再正式。是否改善均值行为必须看完整恢复/standing/head结果，不预判降低熵必有效。
- 原失败诊断、未配对首版及raw_target变量遮蔽修正均保留；本轮不加尖峰过滤、不动v5/CAD、不做部署或轮滑，路线选择仍未完成。

### 2026-10-15 22:42 完整奖励排除摔倒刷奖，改进承接片段的采样效率

- 22:50续记：适应后head5-3826 nominal小9/中120/两轴130/四轴8（保持8/119/115/7），与其measured13/118/170/13相比也存在严重头短板；parent nominal84/128/354/49。头遗忘在训练名义条件仍发生，不能主要解释为三电压测试差异，亦不把未严格配对的nominal/measured差异说成电压因果效应。
- Catch2s正式3790继续；首3776恢复全过，standing26/256（24首次稳定前、2后），负pitch24/68，其余2/188，尚未宣布突破。新旧同节点8秒验收的实际400帧/无termination/initial_qpos+bucket+origins逐位复核PASS，`full-evaluation-protocol-parity.json`证明没有靠缩短评测隐藏后续跌倒。
- GPU0同1792/64的Head2+balance已完成：恢复1792/fault0/standing0；head小3、中96、两轴125、四轴0，未整体赢过普通控制1856，更未赢过1664最佳两轴156。专职按自身名义/测量证据准备从保留1664只改零命令概率.6→.3的曝光对照（Head2奖励/PPO/噪声不变）；已有普通1728可作同父同64控制，不重复花算力重训控制。
- 22:46续记：Catch2s 32768×16 PASS；13稳态中位130925.49、aggregate125870.93控制交互/s，真实28545次terminal bootstrap，ONNX4.77e−7。12582912控制全归属，站姿3024852=24.04%、初.5s764325=6.07%，有限初始过渡如实记录，不把它说成已精确20%。`short-catch-qualification.json`已固定证据/源SHA。正式PID2974109已RUNNING3770/3826，新增6291456控制，初.5s612550，学习loss有限。
- 22:46电压复核首项：parent3762 nominal小84/128（保持8）、中128（保持117）、两轴354/384（保持203）、四轴49/64（保持13），与既有measured小78/中128/两轴351/四轴53接近；两次并非严格同初态逐轨迹配对。stage5-3826 nominal尚待完成，不能提前排除续训丢失电压泛化。
- 上一目标轮为PROGRESS：完成基线head局部奖励排序审计、实现/验证分轴权重，并实际启动64更新；本轮stage5正式完成3762→3826，新增50,331,648控制交互，恢复1792/fault0，standing28/256全部首次稳定前，尚未改善承接。段末measured head小13/128（保持6）、中118（保持116）、两轴170/384（保持153）、四轴13/64（保持9）；同父点同64的stage4为10/119/154/8，只有有限组合收益，仍弱于健康parent3762。没有宣称奖励已完全修好或迁移已完成。
- stage5整段run.json至完成551.78s，91216.17实际控制交互/s；含重叠head与完整reward诊断，稳态准入131706另计。`paired-head-balance-allocation-001/actual-wall-time.json`。短实验不以稳态冒充整段超过10万。
- `paired-standing-full-reward-audit/report.json`已完成：同native-count-v2 measured P04、同成熟head4全奖栈、同初态qpos/qvel/rawobs/origins逐位配对，512站姿×2新seed1561/1562。冻结paired3762倒地42+49=91/1024；from-zero Head2-1664参考0/1024，双方fault0；参考1条尾段未达连续稳定，不能写所有1024稳定全过。
- 在基线倒地的**相同初态样本**上，完整8秒回报baseline约92.98/93.19，直接catch参考102.75/102.39；gamma.99折扣18.36/18.27 vs24.44/24.37；首秒5.55/4.96 vs11.77/11.75。最大收益来自standing_composite、stable_standing/upright，动作平滑成本也更低。真reward_manager逐步分项和返回总和核验PASS，未额外重算有状态奖励。此证据不支持已观察到的“先倒再起刷到更高总奖励”，不证明所有轨迹奖励都合理，也不将参考actor当基线训练teacher。
- `initial-action-comparison.json`只读显示失败初态上baseline两髋roll均值约+.30/−.26rad，参考约+.03/−.06；parent对应探索std约.10rad。参考动作不是唯一正确动作，暂不改PPO或用教师权重灌入，先验证初段曝光效率。
- 新隔离`paired-short-catch-worktree`任务`PairedAdapt-Catch2s-Repaired-S0-O0-R1`：仍健康paired3762起点、head recipe `v2-head-precision-5`、冻结N、32768×24/5epochs/4minibatches、64更新。站姿episode仅time_out截断到2s，其余保持8s；reset概率(.5,.0625,.0625,.1875,.0625,.0625,.0625)。timeout-only稳态按时长加权仍20%站姿时间、约4倍站姿初段次数，有限初始过渡及异常结束另测；更频繁reset也会重采reset随机量，因此这是一组课程变化，不称轨迹严格配对。
- `standing_catch_timeout`显式time_out=True，继续现有true pre-reset successor critic bootstrap，既不是feedback fault也不把它记作成功/失败。完整正式/head评测仍8秒/原协议、取消所有termination，等桶±10°不缩窄。原定向75%[−9,−6]和25%原范围保留；奖励/物理/观测/动作不再改。
- 新GPU计数在每次动作前记录原episode bucket与standing前.5s样本；只在稀疏status/save读CPU，sum必须覆盖全部控制交互，不能把auto-reset后新bucket误算给上一个动作。smoke也使用同计数器，因此本次吞吐包含这项额外开销。
- CPU/GPU配置/timeout边界/部分bucket/曝光计数PASS，成熟64×5实际PASS、59条terminal bootstrap样本（32强制以外已有自然短timeout），冻结N逐位，ONNX4.25e−7；7680条控制全部归属已核算。32768×16准入进行中，主queue64073，首head旁评59946。预检未通过则自动停止，不把它当正式学习。计划和源差分见`short-catch-plan.json`。
- 电压泛化另查：当前实际P04段训练为11.1V nominal，head评均measured三电压。旧1152的nominal/measured接近不能外推当前模型。根queue31172等本次吞吐报告PASS后，冻结parent3762和stage5-3826做nominal头评，与各自已有measured对照；不污染吞吐准入、不热改当前训练条件。GPU0也独立补1792/1856 nominal检查。
- GPU0 Head2-1792正式恢复仍1792/fault0，但measured两轴114（1664最佳156）、neck+3.677°（1664 2.918°）；1856控制两轴119、neck+5.768°，保留1664而非按最新点选。专职新`rise-head-balance-worktree`同1792父点/64预算只加组均衡、保留Head2 Gaussian及原曝光，无head4 precision_scale。CPU/GPU/64×5已PASS，独立32768准入中。
- **CLI stage5跨隔离源不等价**：根的`v2-head-precision-5`以head4为底；子代理的`v2-head-precision-2-active-balance-v1`以head2为底。必须以recipe ID、源码hash、checkpoint谱系识别，不能只凭数字5复现或混用预检。两条谱系仍独立，未做部署或其它技能。

### 2026-10-15 22:10 定向承接64负结果，定位单轴奖励排序反例并做受控修复

- 22:16现场：stage5正式PID2928029、RUNNING3770/3826，新增6,291,456控制交互，loss有限，源未热改；相同父点/课程/64预算对照正在学习。GPU0 Head2-1792正式恢复再次1792满、fault0，专职排1792完整head并续有界64至1856作后续控制。standing完整reward审计脚本已补逐轨迹/逐项回报、同初态baseline倒地样本对比、总和与有限性检查，watch54995仍等3776head结束，尚无审计结果。
- 22:14续记：stage5的32768×16准入PASS，13稳态中位131706.12控制交互/s、aggregate131551.13/s，ONNX4.77e−7，统计逐位冻结；`head-balance-qualification.json`固定四报告与生产/驱动源SHA。正式已进入启动恢复，first-update须由status实读确认。完整standing奖励审计watch session54995，等首3776 head报告后自动运行，避免同时多项小评测叠卡。
- 上一目标轮为PROGRESS：实现并验证训练专用负pitch课程、启动正式学习；本轮继续实际完成64更新。`paired-catch-allocation-001` 3762→3826，50,331,648控制交互；恢复1792/fault0，但standing27/256倒地（26首次稳定前、1后），负pitch26/68，其余1/188。全段新增standing reset19677，其中14705定向、4972保留原域；首次构造/恢复reset另计。未见承接实质改善，不继续同配方盲加预算。
- 段末measured head小10/128（保持5）、中119（保持118）、两轴154/384（保持133）、四轴8/64（保持3），仍明显弱于健康parent3762。首14 head58/123/283/23和standing28也不构成长期改善。段末独立报告和所有失败保留。
- 本64段run.json至完成533.13s，含内嵌恢复评测及重叠早期head，实际94407.66控制交互/s；与稳态131658/s分别报告。短段验证开销仍使整段不足10万，不把物理子步或稳态当完整wall。
- `paired-head-reward-ordering/report.json`完成只读审计：复核四臂中parent_parent/child_child的初态qpos/qvel/raw actor obs/origins逐位一致，原生产head函数与归档完全相同，原rollout逐步reward边界oracle已PASS。小幅单轴tail四轴平均head奖0.70921→0.71038，active轴0.70053→0.54763、inactive三轴0.71210→0.76463，active MAE .343°→.962°，T90 83→9；中幅单轴均奖.68727→.70051，但active MAE .510°→.889°。这是**head tracking局部奖励**对已观察行为的排序反例，不是完整训练回报或梯度根因证明。
- 新隔离`paired-head-balance-worktree` head stage5保留同parent3762、CatchFrontier、冻结N、head4全部命令/细奖励尺度/PPO和64预算，只改单轴时active/inactive两组各50%（原25%/75%）；闲置、两轴、四轴保持原均值逐位。3轴也采用组均衡，当前课程未采3轴。仍[0,1]、总head权重3，不提高奖励总量。候选在同saved单轴数据上恢复更准确策略得分更高，但不能数学上保证所有补偿消失，也不预报学成。
- CPU/GPU实际生产函数检查PASS：环境递归diff仅balance_active_axes参数；独立逐轴权重oracle、0/2/4轴逐位一致、有界/完美误差/命令误差单调性。成熟64×5/ONNX7.15e−7/统计逐位PASS。32768准入中；主queue59078，通过才正式；首3776head旁评98210。相同随机父点/课程/预算与上一段对照，未热改任何运行源。新head evaluator独立oracle同步声明新聚合，指标定义仍不变。
- 站立完整奖励审计已准备：同实际measured P04、相同成熟head4完整奖栈、独立512站姿×2seed，冻结paired3762与从零Head2-1664做行为参考；逐项记录真实reward_manager一次计算值，并检查总和/初态配对。不得把from-zero不同配方回报直接混为同训练objective。排在首head旁评后运行，避免污染吞吐准入；结果未出前不增加摔倒税。
- GPU0专职head1/head2同1600父点各64对照：head2恢复1792/fault0、measured同满、standing0；小3 vs3、中97 vs94、两轴156 vs99、四轴3 vs0、最终单轴113 vs100。正neck MAE2.918° vs4.098°，负neck1.138° vs2.035°；正roll仍弱。Head2仅fine .08→.04，存在实质组合收益，续128至1792；1728正式仍满，未把head控制叫完成。其离线同公式active/inactive总体都改善，不能借基线局部反例替从零分支定因。

### 2026-10-15 21:44 冻统计对照未修复基线承接，定向负pitch课程开始准入

- 21:54现场：GPU1 PID2896272已实际3767/3826，新增3,932,160控制交互，loss有限；GPU0/1利用率96%/92%。主queue14562、首点旁评24930。parent actor/critic/N/Adam/LR/clock逐位恢复PASS；当前不是仅启动进程。`catch-counter-scope.json`注明恢复时计数包括构造与恢复两次reset，正式段计数在其后清零，不混称首次reset样本。
- 21:52续记：32768×16也PASS，13稳态更新中位131658.36、aggregate131943.10控制交互/s，采样149240.49/s；统计逐更新冻结。`catch-frontier-qualification.json`汇总四份PASS证据和源码SHA。正式64更新自动进入启动阶段（不能以进程启动等同已有学习）；首节点旁评session24930、主queue14562，失败自动停，不使用预检权重。
- 21:49续记：CPU和GPU配置/部分reset/原评测初态及RNG逐位检查均PASS；成熟64×5、强制timeout、ONNX7.15e−7和冻结统计逐位PASS。正式32768×16准入进行中，自动queue session14562；首3776承接+头部旁评亦排好。1°分桶进一步确认[−10,−9)17/17倒、[−9,−8)8/12、[−8,−5)0/39，首段定向区间横跨边界，不替代−10°最终验收。
- GPU1 `paired-frozen20-allocation-001` 3762→3890完成128更新/100,663,296控制交互；三个正式节点恢复1792/1792、fault0。3776/3840/3890近站姿倒地28/25/25 of256，段末24条首次稳定前、1条稳定后，全部25条来自负pitch[−10,−5)桶68条。冻结统计没有解决此处承接短板。
- 段末measured头控小9/128（保持6）、中109（保持107）、两轴148/384（保持124）、四轴4/64（保持2），仍严重低于健康3762，停止该配方追加。共同目标seeds1591/1592动态3890与冻结3890均恢复1792/fault0，站立分别24/25，不支持冻结统计能单独改善该基线分支。冻结统计在从零P04的恢复收益保留为另一条独立证据。
- 完整段run.json创建至完成961.78s，实际104663.93控制交互/s；包含此区间构建/内嵌评测/并行早期头评，不含之前导入及之后head/common验收。`paired-frozen20-allocation-001/actual-wall-time.json`；不混作131820稳态准入。
- 用户要求改善负pitch承接，新增隔离 `paired-catch-frontier-worktree` / `CatchFrontier`：仍从健康paired3762开始、standing20%、head4、冻结成熟N、32768×24/5epochs/4minibatches，仅改变站立训练初态：75%将pitch换为[−9°,−6°]，25%保留原±10°；roll/yaw/高度/关节/速度不改。总训练初态约15%定向边界，并保留原姿态覆盖。第一段64更新，后续是否加深至−10°由完整±10°分桶承接与头保持结果决定，不按迭代数盲目升级。
- 原formal eval仍相同等桶±10°，参数只在非evaluation_buckets路径生效；head evaluator恢复原standing事件。新随机抽样会改变后续训练RNG，不称逐轨迹配对。源递归差分只两环境文件：recovery.py与task注册feedback_repair.py，v5/CAD/测量/奖励/动作/PPO全部保留。方案见`catch-frontier-plan.json`，完整source将随正式manifest归档。
- 配置+CPU/GPU部分reset/roll-yaw保留/eval逐位RNG不变验证、成熟64×5、32768×16准入均排队，通过后才正式；初次CPU检查给multinomial传入全整数测试概率导致Long类型报错，生产任务配置全float不受影响。修正测试输入，失败日志保留。当前重试queue session需现场查询，不把预检当训练。
- GPU0专职独立从零1600恢复1792/fault0、standing0；head小2、中87、两轴91、四轴0，正neck误差3.293°但其它轴退化。正在做同1600父点、各64更新的head1与head2对照；head2唯一fine_std .08→.04，仍须新64×5/32768准入。该分支不改站立课程。

### 2026-10-15 21:19 standing20负结果完成，归一化漂移定位与两卡受控续训

- 21:27续记：现场GPU0 93%/GPU1 92%，actual status分别1570与3820，确有持续学习。GPU1冻结20%成熟32768×16已PASS，稳态131819.58、aggregate131860.89控制交互/s；正式父3762、两N count739639296冻结并逐更新核验。first-minibatch KL均0，首更新pooled mean.0110/max.119。3776恢复1792/1792/fault0，先倒28/256均在首次稳定前，负pitch27/68；head小56/128（保持26）、中125（保持121）、两轴294/384（保持239）、四轴34/64（保持23），只比动态20%同14略好，尚无128行为结论。
- GPU0冻结1472由GPU1完成actual measured head：小4/128（保持4）、中97（保持97）、两轴103/384（保持100）、四轴0/64，fault0；正neck仍0/16，稳态MAE约4.424°（1408约4.98），负neck13/16。新种子1591/1592恢复1789/1792，缺3条均left，其余全过，standing0/256。报告`frozen-head1-1472-head-summary`与`frozen-head1-1472-common-measured`。该head原标签本就是native2，sidecar仅核验`execution-metadata-verification.json`，不算修复。GPU0后续1536正式已1792/1792/fault0，继续到1600；完整头控仍未合格。

- 上一目标轮为实际进展：完成新读数域+standing20正式128更新3762→3890，新增100,663,296控制交互；三个恢复节点3776/3840/3890均1792/1792、fault0。首次reset实际standing6565/32768=20.03%，课程已生效。standing先倒29→25→26/256；3890负pitch[−10°,−5°)桶仍25/68，其余三桶共1/188，故单纯增至20%未解决承接。
- 该128段run.json至结束960.90s，含初始化/内嵌评测/导出及并行头诊断，实际104759.13控制交互/s；更新日志部分121865.45/s。见`paired-p04-standing20-allocation-001/actual-wall-time.json`，不混为稳态136480/s。正式新种子共同目标before3762/after3890均1792/1792、fault0，先倒25→24/256（后者全发生在首次稳定之前）。
- 段末actual P04 measured头控明显退化：parent3762小78/128（末端11）、中128（末端116）、两轴351/384（末端206）、四轴53/64（末端12）；3890小7（末端6）、中107（末端107）、两轴132（末端118）、四轴1（末端1）。不继续追加该动态统计配方，不覆盖较好3762。证据`paired-standing20-final-head-comparison/report.json`。
- 同14更新的只读补对照：旧5%课程3776小58/128（保持32）、中125（保持125）、两轴290/384（保持233）、四轴29/64（保持11）；20%课程3776为52（保持22）、126（保持124）、275（保持217）、31（保持17）。早期相对parent保持变好不应归因于20%本身；两段还含读域差异，不称单因素课程因果结论。
- GPU1完成actor网络/观测normalizer四臂，**每种子初始qpos/qvel/raw actor obs/origins逐位相同**：parentW/parentN、parentW/newN、newW/parentN、newW/newN的小幅分别83/75/7/9 of128，两轴353/342/158/130 of384，四轴52/47/5/2 of64。最终头遗忘主要跟随已学权重，换回旧N不能挽回；不排除训练中N漂移改变梯度。新四臂结果是内部配对数据，不把83强称与旧单次78轨迹完全重放。`paired-standing20-normalizer-four-arms/report.json`，没有把换N诊断模型当正式候选。
- GPU0的精确32768首rollout审计定位另一项真实机制：24步storage raw obs与act前快照全相同；尚无Adam时，用各步当时normalizer重算KL逐位0，用rollout结束N却mean .73093/p95 3.560/max40.975，首minibatch pre-Adam约.7285。N漂移使同权重已非同动作分布；不能只看存储权重一致。证据`native-read-preupdate-audit-001/report.json`。冻结已成熟actor/critic统计采用RSL已有until=count，mean/var/std/count保持逐位，runner.train不会重新开统计。
- 初次actor-only/freshAdam/stage0冻结预检value loss变inf，失败保留；这不匹配成熟续训。新增显式`--mature-resume`后恢复actor/critic/N/Adam/LR/clock/课程并重置，CPU2项、64×5及32768×16 PASS。helper SHA `e75be9b1214c4bd8a0fabd79987284849c866ebb73914786dce61bda48f37a02`，冻结模式写入run/checkpoint并禁止静默丢失，每更新逐位验证。没有改reward/LR/Adam矩。
- GPU0冻结统计同1408父点、同64更新1472恢复1790/1792、fault0：sit/prone各256，left254，其余全满；同新读域动态统计为1717/1792，旧读域动态为1546/1792。站立先倒0/256。冻结使first-minibatch KL严格0，但optimizer内仍有mean7.13首轮及.3448/1.110周期峰，不能称所有PPO问题消失。该64段actual101346控制交互/s。子代理已同配方续128至1600（session88639），1500原COM加难须单列；1472冻结head/新种子measured由GPU1排队session3880承接。
- GPU1新隔离`paired-adapt-frozen20-worktree`保留head4/standing20/新读域/parent3762/128更新预算，仅试验冻结成熟actor+critic统计；helper和诊断源码按qualified SHA复制，train/smoke只补当前head4与证据字段。session56701已启动，成熟64×5实际PASS、normalizer五更新逐位不变、ONNX5.96e−7；32768准入后自动正式`paired-frozen20-allocation-001`至3890，首3776头评session46711。尚未宣称其行为通过，未继承已坏3890权重。
- 评测元数据修正：旧head脚本literal execution_version仍v1，而三个standing20/native测评的归档stream SHA已为新原生域v2。对应目录追加`execution-metadata-erratum.json`，原报告/轨迹不改；新`evaluate_v2_head_normalizer.py`从实际stream读取反馈/命令域并正确标识版本。统计值不受标签错误影响。尚未改板端部署。

### 2026-10-15 20:29 避免继承头控退化，修复读数域后从较好3762推进standing20

- 20:33续记：新域+standing20的32768×16准入已PASS，13稳态中位136480.46控制交互/s、aggregate135934.89/s。队列已启动正式PID2779466，`paired-p04-standing20-allocation-001`，父3762→目标3890，当前处于加载/建环境阶段；以status实际更新推进判断学习，不以PID替代。首段64的KL mean最大.0493、单样本max19.66，与GPU0曾出现mean1–2、max3412并不相同，未据此盲改PPO。

- 首段冻结实际P04 measured头控前后已完整核算：paired3762小幅78/128（末端11）、中幅128/128（末端116）、两轴351/384（末端206）、四轴53/64（末端12）；适应3826分别10（末端4）、117（末端115）、139（末端125）、4（末端1）。有明显头控遗忘，不应把旧3826当优于3762的新候选。证据`paired-adaptation-head-comparison/report.json`。两者最终大幅单轴83→91，不能掩盖中小幅/组合退化。
- 同一真实目标、新种子1591/1592前后复核均1792/1792、fault0，standing25→23 of256，改善有限。首段50,331,648控制交互和所有失败成本保留；`actual-wall-time.json`记录run.json至完成539.24s、93337.94/s（不含先前Python导入和后续头评），更新日志部分122589.69/s。稳态预检136242/s不能冒充整段突破10万；下一段采用128更新减少短段频繁构建/评测占比，不展开无界性能优化。
- 用户要求的近站姿20%课程继续推进，但**父节点改回较好paired3762**，128更新目标3890；不从已知头控退化的3826继续。这是替代上条20:18计划，未重复启动旧正式任务。新的`run_paired_standing20_from_best.sh` session15600/PID2772719，首14更新3776头控旁评session99869/PID2772796；结束后自动分pitch桶、头控、共同目标比较。
- 已实读专职代理CPU8项PASS/2.81s、GPU4项PASS/97.14s证据，覆盖原生0/13/19/20/512/1001/1002/1023有效、真正越界/NaN/坏包/时间拒绝、eager/compiled全状态、部分reset、完整fused/rollout/reset CUDA Graph普通/高动态等。两个生产文件固定：stream.py SHA `2ffcf8729f6482c67f5622cc3a0ca5165f5b6f2e5ae6e992941f36c5a569e5d0`；contract.py SHA `3de49158641ebfdc92f22ab5a7a086fb49f01457d1414ce752ce45f7e609c0bf`。指令保护仍20–1001，原生反馈接受0–1023，不去尖峰、未动v5/CAD、GPU批量结构不变。
- GPU1仅复制上述两文件到standing20隔离源，hash逐一匹配；本卡新域64×5/强制timeout/ONNX也PASS，误差8.94e−8。`feedback-read-domain-qualified.json`已生成并放行32768×16准入（PID2773385），仍须PASS后才正式训练；不是把编译或预检称学习。被替代的等待队列50156/68552已停，原日志保留，当前仅15600负责正式启动。
- GPU0 1472用新域冻结重测坐姿仍仅16/256，fault59→1；读域修复解决大多数误fault，但不等于修复策略坐姿退化。子代理回1408验证并续接自己的从零谱系。残余1条仍需按真实原因诊断，不能写fault彻底归零。

### 2026-10-15 20:18 首段P04适应完成，近站姿20%课程落实

- 用户明确“随机化基线+p04续训练抓紧推进”，并要求一起改善初始负pitch承接。GPU1首段 `paired-p04-adapt-allocation-001` 已完成3762→3826（64更新/50,331,648实际控制交互），三个正式节点3776/3808/3826恢复均1792/1792、fault0；standing分别28/25/27 of256先倒，首段未解决该短板。3826首次站稳前26、站稳后1。保留所有源权重/证据，不以恢复满分称完成。
- 该段64×5 PASS、ONNX误差1.19e−7；32768×16的13稳态中位136242.25控制交互/s，aggregate136273.62/s。仅为独立预检吞吐，不冒充整段wall；正式恢复actor/critic/normalizer/Adam逐位一致、stage5/polish与全部初始课程桶一致检查PASS。
- 明确供电条件：实际S0/O0适应首段为nominal（11.1V、含真实S0误差），从基线近似三电压切换而来；measured三电压另做冻结验收，不宣称首段三电压同时训练。head4/命令曝光保留，近似误差bank替换而非叠加。冻结3826 actual measured head已生成，parent3762 actual measured head→共同目标前后比较队列session27706/PID2742212正在推进。
- 已落实独立 `paired-adapt-standing20-worktree`：新增显式训练桶比例(.20,.10,.10,.30,.10,.10,.10)，相对原(.05,.10,.10,.40,.10,.10,.15)提高standing四倍；不借用discovery开关，避免预检/成熟恢复比例不一致。配置核对已证实唯一环境差异是此比例，reward/命令/物理/观测/PPO不变；64×5已PASS。训练评测保存准确reset qpos，新增`analyze_standing_pitch_bins.py`可独立计算四个pitch桶，不拿首个已运动帧冒充初态；已用基础3506原始数据复算30/68负pitch跌倒。
- 同时发现GPU0从零1472新失败：坐姿仅14/256且59条feedback fault；1408仍保留全姿态通过证据。专职代理自动续训guard已拒绝盲续并用GPU0诊断；不能归因尚未发生的1500 COM课程。其首fault初查为head_yaw原生raw count13–19、raw_valid=1、numeric0，被误按commands.native_count_limits[20,1001]拒绝；固件原生解析域0–1023。原生反馈读数域与目标保护域分离正在隔离验证，尚未称修复完成。
- 因这个已知边界问题，standing20原等待队列session68192/PID2746838在smoke完成后已停止，未运行正式128段。替换队列session50156/PID2755616等待`feedback-read-domain-qualified.json`，校验源hash后重新smoke→32768预检→3826继续128至3954→分桶/头控/共同目标比较。只等具体代码修复证据，不等用户许可；最多30分钟窗口。正式预算100,663,296控制交互尚未花费。禁止热改已冻结的首段源或GPU0源。

### 2026-10-15 20:03 先倒再起拆分、共同目标验证与新的有界适应

- 用户连续追问奖励约束、随机化基线+P04续训、随机扰动是否本身导致跌倒。已核查源奖励与真实轨迹，详细事实/指标/方案见[站立保持诊断](sc0090-v2-standing-retention.md)。standing初态带±10°倾斜和小高度变化，不等于已站稳；两种失稳必须分开。现有持续站稳、锁存倒地税、有符号势差均存在，无证据支持直接归因于重复起身刷奖励。
- `paired-feedback-allocation-001` 3506→3762已完成256更新/50,331,648控制交互。真实P04 nominal/measured均1792/1792、fault0，但standing两条件各28/256先倒；measured为首次有效站稳前26、站稳后2。自身含近似误差的head小幅47/128（末端15）、中幅124/128（末端110）、两轴319/384（末端227）、四轴35/64（末端18）；这些不是实际P04头控成绩。
- 新原环境只读评测 `baseline/native-standing-retention-3506-3762-v2`：基础3506自身无额外推扰且关闭普通观测噪声仍32/256倒地（30在首稳前、2在后），paired自身28（26+2）。所以并非只有迁移或新增随机化造成。基础32条中30来自初始负pitch [−10°,−5°) 桶68条，paired为27/68；提示承接覆盖薄弱，仍非奖励因果证明。原采集器Tensor代理cat递归失败已修正并留档，未影响正式训练。
- 五候选相同真实P04目标、新seeds1591/1592复核已完成：P04-1024、P04-head1-1280、基线DR2-4012、head4DR2-4018、paired3762分别1792、1792、1724、1744、1792 of1792，fault均0；standing倒地分别2、0、33、41、25 of256。代码/权重/normalizer冻结校验通过，报告`p04/common-candidates-verification-1591-1592`。新种子原用于独立复核，此后重复选模须视为开发集。
- GPU0由head1-1280续128至1408已完成；1344/1408 nominal1792/1792，fault0。1280 measured1791/1792、standing0；head中幅107/128、两轴132/384、四轴1/64，小幅3/128。子代理已接1408→1472（PID2725784/session10778），冻结1408 measured/head session97724；不跨1500 COM课程边界。
- GPU1 `paired-adapt-worktree`独立复制当前实际P04源，精确移植已测head4函数/采样器，不修改GPU0源码。session36375已启动64×5→32768×16预检→64正式更新→measured head的有界队列；尚未写正式训练完成。32768×24/5epochs/4minibatches，计划额外50,331,648控制交互。使用显式`--adapt-from paired3762`，保留head4/Adam/normalizer/成熟时钟与stage5，实际S0/O0替换近似bank而非叠加，保留不去尖峰。首段不加新奖励，以判据分项观察是否需要加强近站姿曝光/条件失稳约束。

### 2026-10-15 19:40 反馈误差覆盖实测与新的有界基线随机化

- GPU1 head4+DR2-4018已完成：自身恢复保持1792/1792；目标P04 nominal1782/1792、measured1736/1792、fault0。头控small0/128、中88/128、final56/128，仍未同时解决迁移和精度；未覆盖较好基础3506或旧候选。
- 新只读采集`p04/repaired-measured-onpolicy-errors-1024`：冻结从零1024，在真正修复raw S0/O0、measured三电压条件下2种子×7桶×32，恢复448/448、fault0。策略输入未改，actor/normalizer逐位不变。每控制边界GPU保存q/v观测与对应物理状态，O0查询为now−20ms，因此分别保存相对上一控制边界的误差和相对当前真值的诊断误差；不混用这两种口径。
- 丢前.3s后，各关节q误差绝对值P95约.031–.056rad，lag1约.45–.73；v误差中位约.267rad/s，P95约1.08–1.63rad/s，P99约2.4–4.1rad/s。当前独立白速度噪声仅±.25，存在实际分布覆盖缺口；这还不是唯一根因或完整随机化可行性的证明。
- 采集辅助voltage数组在环境构造时快照，早于独立评测reset，不能用作逐轨迹电压标签。已保存`voltage-metadata-erratum.json`，原文件不改，采集器以后在reset后取值。本次bank builder只用真实samples的query误差前两通道，未读取错误辅助voltage数组；q/v误差与统计不受影响。
- 新隔离`baseline/paired-feedback-worktree`从3506训练源快照恢复，并在checkout完成后逐项核对64个归档文件。构造150528×16×2成对误差bank，SHA `d2eaedd5e76d315765f9ceb8f3cdf93baa628d3944b0660331c693822562a0af`。原先proxy-v1来自修复前前端，未复用其误差数据。
- 新任务`Mjlab-StandUp-Flat-MicroDuck-V2-PairedError-Baseline`：用连续.32s的q/v误差对替代原q/v独立白噪声及encoder-bias，保留两通道20ms延迟；各env共享BAM供电从10.2/11.1/12.4V独立抽样。继续head配方4、同曝光和PPO，从基础3506追加256至3762。此为近似随机化组合，不是单因素因果试验，更不是完整P04；不修改GPU0训练输入。
- 明确保留局限：bank只来自一份冻结策略的零头指令起身/站立，混合关节和电压，无在线条件匹配；块内q/v和时间相关保留，跨块连续性/故障不复现；开发种子用于误差设计，不能宣称held-out验收。物理供电和误差片段独立抽样，因此不声称保留电压条件耦合。
- `paired-feedback-checks.json`实际PASS：成对采样关系、每控制步只推进一次、部分reset不改变其它env噪声/供电；实际env reset后、六个控制步后以及部分reset后的vin_tensor均与共享电压一致。供电进入BAM计算而非仅标签。
- 新工作区最初拿到旧smoke驱动，CLI拒绝force-timeouts/head-precision参数，队列正确停止，未启动正式训练。补拷当前已用驱动并保留失败日志；重跑session33504，64×5强制timeout/ONNX已PASS，误差1.79e−7，8192×16亦PASS：13稳态中位132212.62控制交互/s，aggregate131243.51/s。正式PID2701949实际3560/3762、RUNNING，后接真实P04双条件评测。没有用64环境低吞吐冒充正式吞吐。
- 该近似任务的自身head/recovery评测保留paired reset事件和噪声，不因enable_corruption=false而默默关掉；导出元数据明确empirical-paired-qv/not-full-S0-O0、bank SHA与三电压。与早期理想baseline评测的条件不同，最终路线比较仍用真实raw O0目标。
- GPU0 head1-1152（新增128）nominal/measured恢复均1792/1792、fault0，standing先倒均0。measured head small4/128、current87/128、两轴78/384、四轴0/64；自然同预算1152为21/128、9/128、11/384、0/64。精度阶段中幅明显改善、小幅仍退，且两分支还含polish/课程状态差异，不能将所有差异归因于fine奖励。nominal拆分已完成：small2/128、中88/128、两轴70/384、四轴0/64，与measured接近，不支持把主要头控缺口归于三电压泛化。专职代理继续既定256预算。

### 2026-10-15 19:15 官方发布模型对照完成，下一段训练仍有明确短板

- 用户明确要求验证官方XL330发布策略。已从HF固定revision下载最新velstand v8，使用发布清单指定源码3750776、原XL330 M6/全碰撞/齿隙模型实测；ONNX原图与GPU执行576条头控/320条起身实际观测等价核验通过。完整来源、条件、容差和结果见[官方对照](sc0090-v2-official-xl330-comparison.md)。
- 官方共同严格起身判据1790/1792（99.89%），右侧254/256、其它6桶全过；standing先倒0/256。P04从零1024 measured1792/1792，但不同原本体及2条差异不构成整体优胜证明。
- 同81case头控：官方small4/128、mid36/128、两轴12/384、四轴0/64；V2基础3506为47/128、128/128、265/384、54/64。现在有直接实测依据称基础3506在这组静止头控上更好，不能泛化到实机/步行或P04从零策略。末端.5s保持也分别核算，原始轨迹及单轴oracle均保留。
- P04从零1024 measured头控small12/128（末端2/128）、mid0/128、两轴0/384、四轴0/64，fault0。所有角度/测量恢复通过仅证明恢复，不证明完整头控成功。专职代理用严格覆盖证据单独派生stage10/polish resume，原件不改，随后head1精度配方（fine .08，固定当前范围）256更新；64×5及32768预检已PASS，吞吐135737.5/s；session20041。1024第128更新冻结measured/head复查已自动排队session92562。
- GPU1 foundation4+DR1-3762自身1792/1792；head small0/128、mid126/128、final105/128；实际raw P04 nominal1165/1792、measured1059/1792、fault0。这是未达标结果，未以自身满分称迁移成功。
- 新DR2段session45815：由该DR1-3762继续256至4018，将位置噪声半宽 .02→.04rad，保留20ms延迟、head4、曝光及PPO。新64×5/8192×16均PASS，预检129796.22/s（aggregate129870.47/s），正式PID2671775已实际3840；后接head与P04双条件评测，不加载smoke权重。官方恢复进程结束后再做吞吐，避免共享评测污染独立吞吐口径。

### 2026-10-15 18:54 P04首次名义全过；基线停止无效追加并转有界DR对照

- GPU0 P04从零M4-1024首次名义1792/1792、fault0，standing先倒0/256；当前前沿双方向/双种子全部通过，按原门槛晋级90–120°。正式PID2627938继续到1152，session2695同时只读扫剩余stage4–10，再接final measured R1恢复、与基线3506完全相同81cases×8×2seed的measured头控。不能提前写measured/head已通过，尚未选定全部技能路线。
- GPU1 foundation4-4018：自身1792/1792、fault0；小幅49/128、中幅128/128、最终大幅106/128，但当前两轴254/384、四轴23/64，四轴显著差于3506的54/64。小幅末端.5s保持48/128。停止此配方继续追加，不因单轴恢复而忽略组合退化。3506之后两段512更新/100,663,296实际交互作为已花费分支预算保留，不计入新DR支路继承量，也不抹去。
- GPU1新队列`run_head_foundation4_dr1.sh`，session4662：从较均衡的3506继续256至3762，`head-foundation4-dr1-allocation-001`。保持head配方4/曝光/物理/PPO，仅启用已有位置DR1：uniform半宽 .001→.02rad，位置观测lag 0→1控制步（20ms）。其余观测/encoder偏置/velocity/物理DR保留。配置递归差分精确仅四字段，报告`head-foundation4-dr1-config-check.json` PASS；新64×5真实timeout/ONNX PASS，8192×16吞吐已PASS：稳态132578.49/s、整段130969.33/s。正式PID2647040已到3530，状态RUNNING。前置失败自动停，不拿预检权重正式训练。
- 队列后续自动做延迟匹配的自身head、实际修复raw O0 P04 nominal/measured恢复。目的为验证增强基础头控经有效随机化是否保留并改善迁移；不声称基础微幅已完全理想，也不把增加随机化本身当成功。
- 冻结均值/Gaussian追加配对检查保留了一项失败：新均值重放与旧3506原始轨迹并不逐位相等，不能把旧report强称严格配对。新均值与Gaussian的初始qpos/qvel/env_origins逐位相等；证据`head-foundation4-sampling-pairing.json`同时保留原断言失败及差异。新均值小幅T90 51/128、末端.5s48/128；Gaussian105/128、末端.5s0/128。原均值47/128和44/128仍是原报告结果，不覆盖、不混称。
- `summarize_v2_head_saved.py`已增加所有激活轴末端.5s持续到位计数和带内占比。`head-foundation4-sampling-paired-audit/report.json`逐轴复核原report、不同时间到位反例guard均通过，源快照/hash留存。随机动作使短时到位变多但未改善持续精度，仍不用它替代确定性策略验收。

### 2026-10-15 18:47 P04从零接近全姿态，头控组合/持续精度诊断

- GPU0 M4-896为1391/1792、fault0，左右/俯卧/站立各256/256，坐姿124、仰卧0，120–150°前沿243/256，左右各种子门槛均通过。与同父768、同128更新的M16-896（686/1792）相比有实质改善，但优化步数不同、没有声称逐轨迹配对。
- M4首128段正式新增100,663,296控制交互；更新中位6.08s约129347/s。run.json到完成939.64s（含构建/恢复/评测）约107130/s；若把Python导入也计入，启动wall保守区间939.64–990.50s，吞吐下界仍101628/s。证据`p04/fresh-p04-m4-allocation-001/actual-wall-time.json`，不以稳态代替整段成本。
- 896原配方续256至1152，`fresh-p04-m4-allocation-002` session2775/PID2627938。960节点1789/1792、fault0：坐姿253/256、其余6桶全部256/256，仰卧首次全过。105–135°前沿两方向两种子100%；仍需覆盖剩余前沿窗口、measured反馈、头控与最终验收。
- 使用已保存真实qpos只读核算896/960的standing：各256条在t>.2s后均无tilt>45°且z<.08m的先倒，最终稳定256/256、fault0。对应evidence目录`standing-retention.json`。不把曾经适应支路3648的27/256先倒问题混入新谱系。
- 冻结768成功right与M16-832遗忘right奖励审计：64/64 vs0/64，未折扣回报均约79.3 vs6.8，gamma .99折扣约13.80 vs.61；action-rate累计−.187 vs−.295。当前审计不支持“失败拿更高奖励”或“起身尝试税压过收益”，未据猜测改奖励。未逐条归档初态，因此不标严格配对，也不证明所有奖励场景已合理。
- GPU1新`summarize_v2_head_saved.py`独立核算所有激活轴同时进入误差带并持续.1s；有各轴错时到位反例guard，并逐轴与原report精确核对。保存trace/hash/脚本副本；无训练源热修改。旧3250重新按同81工况评测`head-response-3250-common81`：小幅22/128、中幅128/128，当前两轴232/384、四轴50/64。3506对应47/128、128/128、265/384、54/64；先前231/384与48/64来自含reference额外工况的旧批次，保留历史、不冒充相同批量结果。
- foundation4-3762出现退化：小幅33/128、中幅123/128、大幅92/128，当前两轴240/384、四轴29/64。自身恢复仍需以该正式evaluation文件为准，较好3506权重不覆盖。GPU1从3762继续最后一个256更新观察段至4018，`head-foundation4-allocation-003` session86909/PID2635184；若仍无改善停止同配方追加，不无界堆迭代。
- 冻结3506的Gaussian/均值诊断，Gaussian用独立CUDA generator、不改变环境RNG，actor/normalizer逐位不变。随机动作小幅T90为105/128、均值47/128，但末端.5s持续处于同误差带为0/128 vs44/128。随机neck平均有符号误差减小，关节波动却约.49–.54°，相机角速度约.90–.96rad/s；均值neck波动约.009–.014°、相机约.048–.052rad/s。不能把随机动作T90变高称为精度/稳像全面改善，不能直接用于部署。
- 当前T90仍保留原.1s定义，新增末端.5s保持指标用于揭示短时到位后未保持；未偷偷改变旧验收数值。诊断源`diagnose_v2_head_sampling.py`、结果`head-foundation4-sampling-diagnostic-summary.json`，均值重测同时记录初始qpos/qvel用于核对。
- 历史`sc0090-discovered-precision-entropy-trial.md`已复核：降低熵曾改善相机但削弱头部响应。没有仅根据动作std较大就改熵/PPO；Gaussian诊断不是降熵有效的因果证明。

### 2026-10-15 18:32 基础头控进展、官方比较边界与双卡续接

- “比官方好吗”的准确结论：尚无依据。`head-response-3250-official` 的 official 是 V2 基线任务配方名称，不是官方发布的成品策略；未完成同本体/命令/指标官方成品对照。历史官方 V1/XL330 步行头控报告采用不同模型和协议，不能直接比较当前 V2 站立 T90。
- GPU1 `head-foundation4-allocation-001` 从基础3250继续256更新至3506，没有额外位置随机化，保留原官方轻噪声/物理随机化。配方4把非零指令的一半明确分配至 .005–.04rad，小/组合曝光增加；头部原奖励乘有界精度因子（scale .003rad），最大权重仍3。奖励与曝光一起变动，因此不是单因素归因实验。
- CPU配置/单调有界奖励/采样分布检查、64×5真实timeout/ONNX检查通过；8192×24、5epochs/4minibatches预检稳态中位131280.80控制交互/s，整段127154.29/s。不是GPU物理子步吞吐，也不是并行运行成绩。
- 实测head单轴T90：3250基础小幅22/128、中幅128/128、最终大幅96/128；foundation4的3328为24/128、128/128、96/128；3506为47/128、128/128、96/128。3506两方向小幅neck仍0/16，精度远未完成。每次达到90%误差带须持续 .1s，不能以瞬时穿越冒充到位。自身恢复3328/3456/3506均1792/1792、fault0。
- 旧3250评测额外含reference范围，批量大小和固定种子轨迹不完全等价；以上是相同指标/共有命令的开发评测，不能称逐轨迹配对因果结果。新评测显式固定未使用训练采样器的RNG消费，避免配方4额外抽样改变评测初态。组合需进一步核算各激活轴同时持续到位，不以各轴分别曾到位替代。
- GPU1完成3506头部评测后出现实际空档，已续接同配方256更新至3762：`head-foundation4-allocation-002`，session54569，8192×24/5epochs/4minibatches，128更新独立恢复评测，末尾自动接head评测。未加大随机化，未恢复P4/host部署工作。
- 已完成的随机化诊断不掩盖负结果：head-noise3将头噪声半宽降至 .015rad，4012小幅1/128、中幅111/128、最终大幅96/128，目标P04 nominal1637/1792、measured1471/1792、fault0；不继续追加。axis3-4268小幅1/128、中幅100/128、大幅79/128，迁移1786/1723 of1792、fault0；起身高不等于头控好。
- GPU0专职代理当前从同随机谱系768比较M4与M16。M4的832为908/1792、fault0（M16同832为677）；right193/256、left21、frontier159、prone234、standing254、sitting47、supine0。前沿正向各种子38.5/43.3%，负向79.4/90.2%，仍不得升级。完成同128更新预算896再判断，不能仅靠总分宣称M4胜出。M4稳态预检135916.6控制交互/s。
- 正在只读检查成功/遗忘策略的实际奖励排序；normalizer四臂诊断已表明右侧恢复丢失跟随网络权重变化，不能归因于单独更换normalizer版本。未把猜测直接写入正式PPO。

### 基础基线头控复核：不能只看中幅单轴，更不能声称超过官方

- 用户追问基础基线是否头控很好、是否比官方好。复核真实 `baseline/head-response-3250-official/report.json`（目录名official表示该V2基线配方，不是官方成品策略对照）：当前中幅单轴T90为128/128，但小幅单轴仅22/128，最终大幅单轴96/128，当前组合指令也存在失败。此前“头控很好”只在中幅单轴成立，不能泛化为完整头控达标。
- 3250当前幅度neck/pitch±.39、yaw±.49、roll±.11rad，单轴T90种子中位neck .16–.18s、其它 .08–.10s；稳态MAE neck1.49–1.65°、pitch .48–.91°、yaw .11–.54°、roll .36–.45°。小幅约.03rad，按相同90%到位/持续门槛更能暴露精度不足。
- 未完成官方成品策略在同本体/命令/指标上的公平对照，不得称优于官方。官方V1策略与V2本体不可直接用不匹配的HOME/动力学作优劣结论。
- axis3追加到4268仍未解决：小幅单轴1/128、中幅100/128、最终大幅79/128，旧3250应保留为基础能力参考。停止这条奖励收窄分支的继续追加，已有检查点/实际预算保留。
- GPU1 head-noise3隔离对照已从3756实际训练，保持腿.04、头.015rad，回到原head precision2。CPU同RNG腿样本逐位不变/其它配置不变检查、64×5、8192×16已PASS，稳态131633.62控制交互/s；session51088/PID2585134。当前有限预算及评测完成后，优先补基础基线小幅/组合头控，再检验随机化能否保留这些能力，不把现状称理想效果已完全完成。

### 17:48 两卡重新实际学习，axis3初次结果未解决头控

- 现场GPU0 94%、PID2565942、DirectionGate773；GPU1 67%、PID2571695、axis3续段4110。GPU0只是编译/节点切换短空档；GPU1此前评测TypeError已实际修复，v2完整head报告通过同边界reward oracle，失败原目录保留。
- axis3首段4012：head当前负/正neck T90为13/16、15/16；pitch/yaw两方向各16/16；负roll9/16、正roll仍0/16。MAE neck2.067/2.051°，pitch .942/1.392°，yaw1.481/1.103°，roll .633/.930°。尚未解决精度问题。
- 同一axis3-4012→实际修复O0 raw目标nominal1775/1792、measured1648/1792，fault0。保留更好的3756及旧stage2-4012；不能仅因新奖励更贴近指标就认定改善。额外有界256更新到4268正在实际运行，完成后用同协议决定是否保留。
- 进一步随机化诊断备选（尚未实现/启动）：目前基线所有关节白噪声半宽.04rad，其标准差约4.3count，高于已测静态S0大多数lane的1–2count且缺少时间相关性。头部专门降低/匹配观测随机化是可检验假设，不能先称根因；优先读取当前4268结果后作单因素对照。
- 最新范围仍只训练与训练评测；P4/host部署工作暂停，不继续原生O0或控制环工程。

### 最新用户要求：停止部署工作，专注两卡模型训练

- 用户明确不先做P4和host部署。已停止部署工程，不继续隔离固件接线/测试/打包；目标仍是P04从零、基线随机化迁移和路线选择。已产生的固件raw/O0准备API、portable core编译和旧stream测试记录仅保留为未集成草稿，不标部署完成。固件主工作区未改、未硬件操作。
- GPU1 axis3正式4012已保存；后续head evaluator的独立reward oracle用Tensor除tuple报TypeError，队列退出，造成真实空闲。修复为在闭包外一次构造device四轴尺度，实际失败日志/目录保留；session74532用head-axis3-4012-head-response-v2重测后接nominal/measured P04目标。
- GPU0低利用率现场为新DirectionGate任务32768预检的编译初始化，PID2566574 CPU97%；不是训练队列消失。专职代理session62723自动预检→从已评测768续训，保留旧重叠805尾部37更新/29,097,984实际交互。左右每种子独立门槛，先补左向120–150°，不改变奖励/PPO/测量。
- GPU1同时启动head-axis3-allocation-002，从当前axis3真实4012继续256至4268；与冻结4012的小批量评测并行，按实测头控和迁移结果决定续训/回滚候选。使用已通过的相同配方准入，不重新扩大范围。

### 17:32 Web/P4交付、双卡继续和实测候选

- 用户确认现有Web上传模型、ESP32-P4板端运行。今晚候选清单/交付缺项见[晚间候选台账](sc0090-v2-evening-candidates.md)。主固件未修改/未刷写/未触发硬件；独立runtime worktree准备对齐实际HOME/EMA/反馈差异。
- GPU1原stage2续训已完成3756→4012；3840/3968/4012自身恢复全部1792/1792、fault0。新4012→修复O0 raw目标nominal1780/1792、measured1716/1792、fault0。但head正roll仍0/16，负neck退6/16；没有把“多训”自动当改善。
- GPU1改为有限axis3单因素精度实验，从较好头控父3756继续256：fine std从统一.04变(.03,.03,.03,.012)rad；宽Gaussian、精度分量权重、指令曝光、noise .04/20ms、其它reward/PPO不变。GPU缓存4轴尺度，未每步CPU传参。新64×5/timeout/ONNX PASS，8192×16稳态134535.92/s；session51021自动预检→正式→head→P04 nominal/measured，无需人工逐项再启动。PID2553004实测GPU1 71%、3780；GPU0同时81%、710。两个4012不同实验目录，不得混淆。
- GPU0旧stage2不重叠课程704/768已掌握prone掉至106/34 of256，实际安全停774。专职代理按640父转重叠105–135°窗口，门槛/奖励不变；旧失败尾部实际121,896,960交互保留不抹去。新32768×16预检132626.9/s，session94077/PID2539064。新704为907/1792，prone252/right251/standing254/frontier123/sitting27、left/supine0、fault0，继续训练。
- 已冻结baseline3756和4012规范候选；3756生产C推理核普通/packed路径通过259条及2048真实obs，后者最大误差3.34e−6。CPU MuJoCo+ONNX 14条8s闭环14/14；GPU动作CPU开环3/14，保留物理漂移限制，不夸大对齐。
- P04同步400步episode reset处出现KL峰值（如新段第17更新均值3.19），尚未定位是normalizer/样本分布还是更新问题，不在当前重叠课程对照中混改PPO。若反复出现学习退化，再有限验证首次episode age错开。

### 17:08 最新要求、双卡任务和 count 跳动核查

- 用户最新取消尖峰去除：有效原始尖峰交给策略，随机化覆盖；此前16:25及更早“no-spike仅过渡/最终须拒绝异常”的要求已被覆盖。bounded分支保留历史诊断，不继续趋势拒绝和原生C移植，不合入从零源。
- 用户要求今晚提供几套训练好的起身策略用于实机比较，综合看训练时间及real指标。当前优先随机化基线、P04适应、P04从零三种独立谱系的候选筛选、规范ONNX/契约/运行说明和指标；不扩大其它技能。不把不合格的从零模型标成功。
- GPU0从零640开发恢复778/1792，fault0；前沿189/256、两个种子各过70%，已进入stage2（90–120°），standing255、prone254、sitting55、right_side25、left/supine0。前后阶段前沿难度不同，不能直接比总分称退步。session11795，原随机256再训768，end1024。
- GPU1原始尖峰审计结束后出现空闲，已承认调度衔接不足。现启动baseline `position-robustness2-allocation-002`，session68804，从真实resume3756继续256至4012，8192×24、5epochs/4minibatch、eval128。保持stage2位置随机化和head-precision2；沿已通过64×5和8192×16/132544控制交互每秒的配方，不盲目增加噪声。与父run代码差异仅评测位置延迟口径修复和已经过完整模型等价核验的origin构建优化，训练奖励/PPO/物理配方不变。
- 已完成 `valid-raw-count-jitter-audit/report.json`：GPU1真实S0完成交易采样，128虚拟机器人×两姿态×4s，nominal/measured，丢弃前.3s；每组约33.2万有效交易，invalid0。相邻新交易count差P95=1–2、P99=3–7；lane标准差中位约1.09–1.77count，lag1相关中位约.74–.84。量化每count映射约.30894°。说明已有小幅且时间相关的跳动，停止额外加噪工程。来源冻结经验残差+量化；这是模型输出，不是V2实机新采集。脚本/源码快照/原始NPZ已保存，source不变断言PASS。
- 冻结baseline3756合法count尖峰审计：0/8/20/40°分别442/440/444/434 of448，sensor fault0/0/0/9；actor/normalizer不变。40°仍可能通过后级数值检查触发故障，尚待定位。不要从小样本成功数差推断尖峰有益或已完全鲁棒。
- 历史bounded-v2缺测0/20/40/80ms结果443/440/110/0 of448，fault0/0/336/448。同种子测试在注入前已有轨迹差异，不能再称严格配对；定位未完成。详[历史反馈记录](sc0090-v2-feedback-tolerance.md)。
- reset穿地为独立正确性问题：512策略A/B仅666→668 of1792，未证明它导致平台；GPU最低点候选窄验证已过，但正式训练保持不动。今晚优先训练/验收，不展开无因果证据的奖励改动。

### 16:25 有界预测候选实际缺测失败与修复（不可略过）

- v1左膝valid=false闭环：20ms为224/448恢复、222/448 sensor fault；40/80ms均448/448 sensor fault。不能因为前面的单位/GPU/nominal检查通过而宣布容错完成。已发给子代理：最终前端尚不准入正式训练。
- 20ms首次无效全部在control step101，只涉及left_knee；reason=2，说明当步已经有有效真实新输出。旧25ms插值跨度门槛拒绝跨缺测的左右真实支撑，退回左侧单边预测，σ最大约.1727rad超过.15，产生误失效。每种子分别116/106条；非整机崩溃，也非真值缺失。
- 新 `o0-bounded-prediction-candidate-v2` 保留.15rad不确定度/30ms外推/50ms测量年龄上限；两侧都有按wall时间完成的真实输出时，使用端点已有估计不确定度加 `0.5 * acceleration_uncertainty * 左时间差 * 右时间差` 的桥接界，支持间隔仍≤50ms。同时显式约束速度不确定度≤8rad/s。预测仍不进历史，失联超时仍无效。
- 新真实交易完成延迟/缺测后恢复用例覆盖旧失败：查询时刻两侧有真实输出、跨度40ms，解析恒速预期仍应可用。CPU协议现**10 passed/2.29s**；不能把此单元检查外推为闭环通过。
- GPU1新顺序队列 **session83250**，脚本`feedback-tolerance-worktree/scripts/run_bounded_o0_v2_validation.sh`：device parity-v4→graph parity-v4（源码不变验证）→smoke-v2→**同N32/同种子0、20、40、80ms**配对闭环。任何前置执行失败停止后续；当前未提前写PASS。旧v1评测和代码快照全部保留。
- GPU0从本随机256续到新课程300，actor/critic/normalizer/Adam逐位恢复已核验，无基线输入。首新课程决策320；阶段评测短暂低利用率后自动续，不需要人工重复启动。
- 在容错版正式合入前，还必须用fresh64/256探索大动作作正常轨迹审计。专家baseline3756动作平顺，不能代表从零探索；分别判物理动作失败与反馈拒绝/invalid，不用低起身率冒充传感器错误。


### 16:14 目标对齐后的最新验证与运行

- GPU0从零192恢复762/1792，256为761/1792、fault0；站立/俯卧/当前frontier已较好，坐姿/侧背仍0。颈部偏置已自然改善到约20°，不启用备用L1奖励。专职代理只改课程依赖，让已掌握stand/prone/frontier可继续反向roll；坐姿仍训练/评测，最终polish仍必须三类困难姿态全部通过。新队列session11795在对应32768准入，之后从**本随机256**继续768。
- 原从零实际停277，217,841,664控制交互；选择已评测256为课程对照父=201,326,592。尾部21更新/16,515,072实际消耗保留checkpoint277及日志，不能抹去，也不能计为新课程学习量。
- bounded完整图v3带源码快照/SHA与结束源不变断言已PASS；诊断reason最终版CPU9复核PASS/2.22s。64×5/32真实timeout/ONNX PASS，导出仍candidate状态。
- baseline3756→bounded候选完成：nominal1772/1792=98.88%，measured1709/1792=95.37%，均fault0，actor/normalizer逐位未变。读取拒绝率约nominal0.00327%、measured0.0507%；控制级预测次数0、invalid0，说明正常测量有真实历史支撑，不能拿这些回放替代缺测容错验证。每种子的计数/原始NPZ SHA见`tolerance-summary.json`。
- GPU1 session82847已启动冻结同一baseline3756的20→40→80ms左膝坏包（valid=false）顺序回放，2种子7桶32、实际闭环仍运行；短时是否可用、超时是否失效、恢复速度须读结果。必须补同N32/同种子0ms对照，不能直接把N128正常报告作配对基线。注入只改变原生测量valid，不修改真实物理状态、不注入真值输入。
- 更完整当前工作边界及后续路线：[当前阶段合同](sc0090-v2-current-stage.md)。正式候选仍缺参数/原生C部署一致性、32768吞吐、头部/站立保持及完整native CPU闭环证据，未选择后续全部技能训练路线。


### 16:04 三项目标已落实到运行与验证

- GPU0独立 `rise-zero-worktree` / task `Mjlab-StandUp-Flat-MicroDuck-V2-Fresh-Repaired-S0-O0-R1` / session75364 / PID2423620。`fresh-initialization.json`实际PASS：checkpoint=null、Adam空、normalizer count0、clock/iteration/stage0、discovery=true、polish=false；初始网络seed20261115，SHA `10749e37d0f45adf04a25b8f609a868ea68de3b891b92a293d1419f0eb27554f`。**没有加载基线或适应模型**。旧适应实际停3793，新增165更新/129,761,280交互，保留对照。
- 从零发现阶段standing/sitting/prone=40/30/20%，其它4/2/2/2%；两种子三易项≥70%且≥2400步才转后续roll课程，后续站立保留20%。32768×24、5epoch、16minibatch=49152样本/批，与官方8192×24/4一致；新预检13稳态中位132131.36控制交互/s、整段132056.36/s，不能套旧4minibatch成绩。
- 从零64开发恢复0/1792、fault0，存在颈约90–110°/倾斜23–26°局部姿态。**128未经奖励改变已到726/1792**：standing220、prone251、frontier255（各256），其它0，故障0；颈约71°仍不合格。不能把64早期局部解称为已证实死结。GPU0继续观察192；备用门控L1头部bias仅隔离审计/预检，未贸然正式启用。发现课程因sitting不达标正确保持。
- 父级候选异常拒绝+有限时间预测已实现，9项CPU协议检查及CPU/GPU/编译一致性PASS；完整反馈CUDA图正常/压力/部分reset两组PASS，64×5实际PPO/ONNX PASS。参数与部署尚未最终验证，详见[反馈容错台账](sc0090-v2-feedback-tolerance.md)，不能把这些窄验证外推为最终sim2real完成。
- GPU1冻结baseline3756→候选目标的nominal/measured恢复队列session55968已启动，两种子7桶128；记录拒绝/预测/不确定度诊断，保持actor/normalizer不变。后续基线路线判断必须用真正异常容错目标，不以no-spike代替最终方案。
- native CPU回放7桶×20步基础预检已运行，ONNX与初始原始观测核对通过；后续修正CPU评测线速度为root link而非COM，与训练源码公式一致。完整8秒CPU闭环和运动观测对照尚未完成，脚本保留；当前优先满足最终反馈与两路线对照。


### 15:34 用户重新明确阶段验收与最终反馈要求（最高优先级）

1. **修复后的P04从零训练成功**：不能用基线权重适应代替；从随机actor/critic、全新normalizer/Adam、课程stage0开始，修复可以沿此从零谱系继续。专职GPU0子代理已收到要求，将适应支路在更新边界保存作历史对照，转从零发现课程/64×5/吞吐准入/正式学习。
2. **基线增加有效随机化，成功迁移到修复后的P04**：GPU1负责，冻结源模型后使用目标真实反馈前端评测；恢复、站立不倒、头控和扰动均需评价，不能只凭末端恢复率。已有位置噪声/延迟收益，但旧尖峰异常处理不合理造成大量误故障。
3. **根据两条路线实际表现决定所有策略的训练路线**：比较成功率、实际wall time、样本及优化预算、迁移与响应表现；现在不预选P04或基线，不开展其它技能/轮滑。

用户进一步明确最终异常处理：基于采样间隔/真实运动变化/预测不确定度判断异常；短时缺测保留有效历史并预测；预测值不能伪装测量回写；有效测量恢复后校正；持续失联或不确定度过大才失效；严格预测到约定查询时刻，不能用物理真值补actor。**现有no-spike + O0局部历史版本是诊断/过渡方案，不能标最终反馈完成。**

- 父代理独占新隔离 `runs-v2/feedback-tolerance-worktree`，从已记录的适应run source快照恢复，实施与验证有限时间预测前端；不热改GPU0训练源。最终前端必须通过坏包/真实加速/连续缺测/重连/时间错序/部分reset/CPU–GPU/编译图一致性及吞吐准入。
- GPU0从零与GPU1基线迁移可在过渡前端继续获取证据，最终结论须在已验证最终前端复核。当前99.94%/98.72%恢复来自基线适应，只是已有对照证据，绝不计作P04从零成功。
- native CPU回放脚本 `baseline/scripts/compare_v2_cpu_gpu.py` 已完成7桶×20控制步基础预检（ONNX与初始物理观测核对通过），不是8秒恢复评测；完整闭环/运动观测对齐仍待完成，保留脚本和日志。优先加入最终反馈修复与两路线对照，不把这个短测试标sim2sim成功。


### 15:21 独立实测扰动/头部结果与站立保持缺项

- GPU0现场93%、39.3GB，status3710、新增64,487,424个P04控制交互，仍在训练。GPU1上一轮独立评测已完成，下一任务只限相关迁移/修复，不因利用率而启动无关动作。
- repaired3648 measured R1恢复 **1769/1792=98.72%，fault0**，actor/normalizer逐位不变。`repair-local-o0-3648-measured/report.json`使用同专职修复源保存完整来源与轨迹。baseline3628与3756补上checkpoint真实20ms位置延迟后的名义恢复均1792/1792、fault0，消除了旧评测配置遗漏的不确定性。
- repaired3648名义头部81case×2seed×8重复，无invalid。当前训练范围8个单轴方向全部16/16满足持续T90；颈负/正MAE .732/.564°、pitch .394/.463°、yaw .294/.508°、roll .507/.408°。负roll各种子T90中位.24/.42s，其它约.08–.18s。只对当前单轴范围做此结论，组合、最终幅度、measured头控及相机运动还需审计。
- baseline3756匹配延迟的头部仍有问题：正roll T90到位0/16、MAE .895°；正颈15/16到位、MAE2.024°。3628两个roll方向均0/16，均值误差 .927/.785°。因此不能仅按迁移恢复率选基线候选，头部性能必须独立保留。
- **新增明确失败项：恢复成功不等于站立保持成功。** 子代理复核3648实际保存轨迹与七桶首样本拼图，发现standing初态200ms后曾同时tilt>45°且z<.08m的有10/128和17/128，共 **27/256≈10.5%先倒再起**。现有末端持续.5s稳定指标会把恢复后站住计成功，因此1791/1792应严格称“恢复率”，不能称站立保持全通过。不改写旧报告、不降低旧门槛；专职代理正补站立不倒和到稳时间审计，依据3712对照决定曝光/课程修复。
- 已把origin-marker启动优化移植到baseline与主P04（未更动当前子代理训练源）。两分支各自64/1024/4096/8192 CPU建模对照PASS：最终XML、484项编译模型数组、环境坐标精确相同，非V2 guard通过；8192该构建段约14.8/14.2s降至.0088/.0083s。证据各`origin-marker-port-equivalence/report.json`，不把此单因素数字当全程序启动耗时。
- 实际native CPU MuJoCo/ONNX闭环仍待建立。已核对正确V2/v5 CPU BAM、命名关节、足底适配、51起始头指令槽；通用`infer_policy.py`仍带旧XL330默认值，不可直接作为V2验证。下一步先同初态/同动作CPU–GPU物理回放，再闭环，避免把GPU测量迁移冒充sim2sim完成。


### 15:17 GPU0低利用率核实、反馈修复与双卡推进

- 用户追问GPU0空闲。15:13:58实测GPU0 **1% / 34.6GB**、PID2391499存活；结合日志定位为3648双种子小批量恢复评测及轨迹压缩写盘，不能把此阶段称为高GPU利用率。15:14之后自动恢复学习，父代理连续5秒采样为 **91/92/92/90/96%**，显存38.5GB；status3650且后续继续推进。GPU0专职代理继续监督，未重复启动训练。
- 原失败P04配方在328更新边界停下并保存，累计257,359,872控制交互；320只3/1792成功、585故障。新修复执行 `p04-o0-valid-raw-local-history-v1` 接受有效原始测量包，去除无硬件依据的预测尖峰判死，并让O0只等待所需5个新有效LS采样；保留坏包/count/NaN/时序/PID/数值/限速/query支撑和新鲜度检查，不加入真值或预测替代。O1短历史仍拒绝。详见[专职台账](sc0090-v2-p04-rise-agent.md)。
- 修复版8项CPU语义检查、2项eager/compiled/CUDA图故障及reset对照、64×5真实超时/ONNX均PASS。32768×24×16的13稳态更新中位 **135147.61控制交互/s**，整段稳态134890.45/s；没有乘物理子步。8192环境构建中的origin-marker冗余处理已精确等价修复；实际正式启动落盘至首PPO日志353.24s→89.11s，二者不能作为单因素实验混报。
- 新正式 `repair-local-o0-adapt-001` 位于专职隔离worktree，显式从baseline3628适应，保留actor/critic/normalizer/Adam逐位恢复证据；713,293,824继承交互属于baseline，新增P04交互单独记。3648=新增20更新，不是从零完成3648轮P04。首不可变checkpoint SHA `ee96812da356b5e2b22f160be38b0ddb131356902402da747d0af1c7ecb38ef9`。
- 首3648开发名义恢复 **1791/1792=99.94%，fault0**；standing255/256，其余6桶各256/256。末秒高度约.112m，tilt P95约3.05–3.68°，角速度P95约.22–.25rad/s。仅为训练源实际修复前端的GPU开发评测，尚不代表CPU MuJoCo/ONNX sim2sim或实机通过。
- GPU1已启动同一3648的measured R1恢复→头部响应顺序队列session77775，严格使用专职worktree真实修复源，新增独立评测脚本、不改正在训练的源码。另session14300补baseline3628→3756实际位置延迟回归；两卡本轮仍只做起身、站立及相关头部验证。

### 15:17 基线阶段2结果及评测延迟口径纠正

- `position-robustness2-allocation-001` 已完成3628→3756，位置噪声半宽.04rad、位置延迟1控制步，其余保持。64×5/32真实超时/ONNX通过，8192×16稳态端到端132544.04/s。
- 冻结3756的原完整P04迁移：nominal **1616/1792=90.18%，fault151**；measured **1436/1792=80.13%，fault287**。no-spike：nominal **1764/1792=98.44%**，measured **1696/1792=94.64%**，均fault0。所有actor/normalizer逐位未变。阶段1相应no-spike为1304/1181，证明阶段2带来进一步迁移收益；不能将旧尖峰判死仍存在的完整P04报告与修复版身份混用。
- 已发现并修复baseline独立 `evaluate()` 沿用registry play位置0延迟、未应用checkpoint位置鲁棒配方的问题。**旧3584/3628/3712/3756的名义报告是0位置延迟回归，不能称训练延迟完全匹配**；原报告保留不改。独立主P04迁移报告确实走S0，不受该配置遗漏影响。
- 新baseline `evaluate_v2_recovery.py`、正式trainer evaluate及 `evaluate_v2_head.py` 读取checkpoint position recipe；名义评测有意关闭加性噪声，但保留实际1控制步位置延迟，并保存实际cfg/source/checkpoint SHA。session14300实际运行补测，结果未提前写PASS。


### 14:56 用户要求子代理优先处理故障；位置鲁棒性已产生迁移收益

- 用户进一步明确“子代理也要处理好反馈故障，严重影响训练有效性、浪费算力”。已把反馈失效检测/容错/重预热修复列为子代理首要任务之一；不必把无效旧预算硬跑完，可在更新边界保存后转验证过的修复支路。要保留真实坏包/count/NaN/时序检查，不能默默用真值补输入。子代理最新已确认GPU0实际280更新/利用率78%，独立台账[专职起身记录](sc0090-v2-p04-rise-agent.md)。
- GPU1阶段1候选3628冻结迁移完整完成，actor/normalizer不变。完整P04 nominal **1228/1792=68.53%**、fault115；measured **1023/1792=57.09%**、fault336。no-spike nominal **1304/1792=72.77%**、measured **1181/1792=65.90%**，均fault0。原3500完整P04为0/0通过、no-spike为0/4。仅128更新的显式位置噪声/延迟适应带来实质收益，但没有任何一组达到最终验收。
- 已将3628不可变checkpoint和resume路径、head recipe、stage5/polish、旧position recipe的血统边界交给子代理，供P04反馈修复适应候选。跨任务须显式记录，不把baseline子支路冒充从零P04；若切完整S0，不应再双计baseline额外位置噪声。
- GPU1启动位置鲁棒性阶段2预检（噪声半宽.04rad，仍20ms位置延迟，其他不变）：64×5后串行8192×16，目录`position-robustness2-smoke`/`position-robustness2-throughput-8192`。通过后再从3628给有限追加预算，并检查名义恢复、完整/no-spike迁移及头部响应；这不是把随机化越大越好当定论。

### 14:50 明确委派与GPU1候选完成

- 用户明确要求“让一个subagent专门推进P04起身，训练成功为止”。已创建 `/root/p04_rise_training`，独占GPU0训练监督/阶段评测/平台期诊断与修复，维护独立`docs/sc0090-v2-p04-rise-agent.md`并向主代理同步。当前004进程PID2372973/session55872交接，不重复启动或因初始化低利用率误杀。主代理继续GPU1起身迁移与CPU/观测核验，不开展其它动作。
- 基线位置鲁棒性阶段1实际完成3628，3584/3628名义开发恢复均1792/1792，无退化。只证明原任务保留，尚不代表测量迁移修复。
- GPU1已启动同一3628检查点在完整P04和no-spike下的nominal/measured四组回放，目录`baseline3628-full-p04-transfer`及`baseline3628-no-spike-transfer`。不训练、不改变actor/normalizer；结果出来后对比原3500的0/0与0/4，通过后才决定更强噪声阶段或P04适应。

### 14:46 当前实际作业（接续先核对）

- P04 `standup-allocation-003-32768`实际完成257：256/257均0/1792通过，分别702/692条反馈失效，无晋级。总实际controls201523200（001196608 + 两段各100663296）。不能以更多交互或10万/s声称学成。
- 按用户“P04继续训练、GPU1找修复”的安排，GPU0新`standup-allocation-004-32768`从003/resume_000257.pt追加256到513，每64评测；仍在独立冻结工作树，主参考配方未混入测量消融。当前刚启动、需要约6min初始化，先核对实际PID/恢复报告/新更新再报进度。该参考续训与GPU1修复同步；候选得到可复核改善后应及时调整后续路线，不无限保持无效课程。
- GPU1 `position-robustness1-allocation-001` **session19118/PID2371204**，实际3520+，目标3628。名义课程回归之外，结束后必须在完整P04和no-spike反馈下复测，才能回答随机化是否帮助迁移。未增加其它动作。

### 14:43 起身位置鲁棒性候选开始有限续训

- 位置/速度独立替换完整结果：baseline3500/no-spike在20ms延迟理想q下1508/1792，在理想v下2/1792，两者同时理想1585/1792（88.45%），全部无反馈失效。最后一组仍未100%，保留未解决的延迟/控制敏感性；不把理想输入替换当修复或部署方案。
- GPU0目前实际250更新，192节点0/1792、606失效，256/257评测进行中。原配方训练仍在运行；未因诊断停卡。接下来根据末端结果和GPU1候选证据决定课程/测量调整，不用总reward上涨代替成功。
- GPU1新候选`position-robustness1-allocation-001` session（见实时进程/run目录）从baseline3500/resume追加128到3628，N8192/T24，eval间隔128。保留原3500对照、actor/critic/normalizer/Adam和head-precision-2配方；只改变actor位置覆盖：均匀噪声半宽.02rad + 1控制步延迟，原encoder bias/速度观测/物理/critic不变。它是独立baseline子实验，不是完整S0/P04替代。
- 阶段1先64×5实际PPO/32超时/ONNX PASS，再8192×16剔除3更新，**133013.55端到端控制步/s**、161721.71采样/s，13稳态更新。报告均含position_robustness_recipe，trainer与预检逐字段比对，checkpoint也记录；不能静默续训时丢掉父位置配方。阶段2.04rad只定义未启用，需阶段1证据决定。
- 后续：该候选结束后做名义恢复回归及冻结checkpoint在完整P04/无尖峰测量下迁移复测（同开发种子/桶），并检查头部是否退化；若无改善，继续定位而非宣称随机化必然有效。GPU1没有其它P04动作任务。
- 两分支CPU摩擦修正各2测试PASS；目前还没有完整V2原生CPU闭环结果。不要将观测切换诊断冒充CPU sim2sim验证。

### 14:32 继续P04训练；去尖峰与位置/速度因果对照

- 用户要求P04继续训练，纠正两卡都用于诊断的安排：**GPU0持续原配方P04起身；GPU1用于相关实验找修复**。最新要求优先于14:11“暂不追加”的当时决定，goal始终ACTIVE，其它动作继续暂缓。
- 已从002源码tar建立独立detached工作树`runs-v2/frozen-source-allocation-002`，86文件SHA全部精确匹配。GPU0`standup-allocation-003-32768` session70567/PID2349108，从002/resume_000129.pt追加128至257，每64评测；恢复actor/critic/normalizer/Adam/clock完全一致、`source_changed=false`。目前实际160，不是只有启动壳进程。重新初始化32768环境实际约6min，不能把这段低利用率当成GPU高负载。该工作树运行wrapper需`SC0090_CUDA_COMPAT_DIR=/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat`。
- 完整P04新回放129 nominal首次失效595条，所有首失效lane均为连续尖峰status4；回包valid=1、count范围内、numeric拒绝0、嘴部首失效0。与原正式627不同，分别保留。去尖峰单因素关闭后：P04129 nominal/measured=4/10通过，baseline3500=0/4通过（均1792条），**四组反馈失效全为0**。该门槛制造了高失效率，但删除它没有恢复起身，不能把全部问题归为“仿真故障”。
- 新`spike_rejection=False`只用于明确标记的评测消融，保留raw有效性/count/时序/LS/query；默认True，原冻结`sc0090`源码不改。导出标为simulation-measurement-ablation-only，普通完整P04 trainer拒绝消融预检。合成有效阶跃独立测试：旧状态0/0/0/1/4、新0/0/0/0/0；越界和坏包两臂都拒绝，1 passed/1.61s。新主工作区测量源码SHA已变化，**旧预检只能用于上述精确恢复的冻结工作树**，不得跳过哈希准入。
- GPU1固定baseline3500，关闭尖峰检测后分别替换actor单一输入为20ms延迟物理参考：`baseline3500-ideal-q-diagnostic-v2` **1508/1792=84.15%**，`baseline3500-ideal-v-diagnostic-v2` **2/1792**，均fault0。对照完整测量为0/1792。支持优先定位位置误差，不是仅速度或fault门槛。正在做`baseline3500-ideal-qv-diagnostic` session61804，验证两通道都理想延迟的参考上限。真值输入明确仅用于诊断，不用于正式策略/部署。
- q/v分离首次脚本把env返回的dict误当TensorDict调用clone，两次都在第0步退出，日志`baseline3500-ideal-{q,v}-diagnostic.log`保留；已改为复制字典并只克隆actor，再用新v2目录完整重做，不把失败充作通过。
- 新[起身迁移专题审计](sc0090-v2-rise-transfer-audit.md)区分观测切换与GPU→原生CPU MuJoCo/ONNX两条检查。发现通用CPU回放漏掉历史DOF-friction索引修正；已在两分支移植DofFrictionMujocoController并接入infer_policy.py。冻结V5+原生MuJoCo约束Jacobian独立oracle每分支2项测试通过。V2完整CPU闭环尚未完成，旧infer默认仍为XL330，不能直接宣称可用。
- `scripts/analyze_v2_feedback_faults.py`已对四个完整/无尖峰目录生成fault-diagnostics.json，逐源SHA、首失效关节/原因、物理稳定与故障计分分别保留；`analyze_v2_transfer_inputs.py`用于原始测量路径，不能误用于含真值输入替换的消融报告。

### 14:11 用户收束范围：先解决两分支起身迁移

- 用户明确GPU1暂不做其它动作的P04训练，可用于各种起身迁移实验找修复。已停止正在初始化的`walking-head-contract-smoke`（PID2333842，SIGINT，日志KeyboardInterrupt）；保留代码/CPU测试和失败日志，不标PASS，不继续其它动作预算。此前`walking-successor-critic-smoke`已完成64×5/真实超时/ONNX PASS，但只覆盖更早的PPO修复配置。
- 当前P04正式预算实际结束：累计129更新，继承196608 + 本段100663296 = **100859904控制交互**。64/128/129评测分别0/2/7条通过（各1792），反馈失效25/598/627条；没有课程晋级。P04单卡>10万吞吐不等于学成。**暂不追加大规模正式预算**，先诊断学习、反馈链和CPU回放。
- baseline实际完成3500，恢复1792/1792继续通过。头部3500同幅单轴：颈负/正MAE1.755/1.414°，roll负/正.174/.604°；roll正严格到位降至75%，保留3250作为对照，不把最新checkpoint自动当最好。不继续扩幅或其它动作。
- 已回答反馈故障语义：训练任一15路无效在20ms控制边界终止并重置该环境；其它世界继续。开发评测禁自动重置、跑满8秒，但任一时刻无效整条计失败。无效关节只保留最后有效q/v，不用预测或物理真值代替；恢复有效后才更新。627是“回放曾经失效”的条数，不是627次GPU/物理仿真崩溃，更不是真机故障率。
- 用户质疑故障模拟算力收益，质疑有效：尚无证据证明此高失效率有实机代表性或能改善迁移。下一步必须区分真实测量不确定性、历史frontend重预热/窗口门槛和实现缺陷；保留完整P04对照，允许独立命名的合理缺测/有效测量对照，不能默认越苛刻越真实。
- GPU0：`p04-update129-fault-diagnostic` session70020；GPU1：`baseline3500-fault-diagnostic` session32754。同2种子7桶128，nominal/measured分别回放，不训练。新增只读诊断记录每控制实际actor输入、q/v估计、物理真值、命令/raw/timestamps；每5ms记录source故障与numeric rejection，每条首次控制边界失效保存关节/reason/原始回包/预热状态，保留原失效门槛。
- 64更新输入诊断`p04-update64-observation-diagnostic/input-diagnostics.json`通过actor q/v槽位与反馈缓存逐项相等断言。nominal重放0通过/28失效，原正式GPU0记录25失效，未宣称跨进程/卡轨迹逐位复现。未失效的俯卧/前沿末秒约23.5°倾斜、颈约2.01rad；低成功率不只是失效计分造成。有效测量对20ms延迟物理参考多数关节MAE约.9–1.25°，速度RMSE约.53–.74rad/s，不能仅看平均值判断闭环安全。
- 步行有限审计代码留待以后：V2 nominal/measured改用TerminalCriticPPO，站高参考.095→.112m；明确零/单/双/四轴命令；移植只用于无门控步行的当前命令EMA重基准，避免旧命令误差残留。6项CPU测试PASS；完整腾空3.0/.125–.300保持，未加窄头部精度税。最新版GPU预检按用户要求中断，因此未准入正式步行。

### 13:55 两卡实际进展与头部课程决策

- P04正式`standup-allocation-002-32768`已经进入实际PPO更新（最近status30，不是仍在初始化）；resume-verification确认actor/critic/normalizer/Adam完全相同，恢复LR=.0006666667，原common_step=24；新episode课程stage0/头部小幅度恢复通过。GPU0一次实测利用率85%，不能把瞬时值当全程平均。当前正常更新约7.4–7.6s，等待64/128节点的真正学习评测。
- baseline `head-precision2-allocation-001`实际完成3250；两种子7桶各128全部通过（1792/1792），无fault。`head-response-3250-official`117条件×8×2评测完成：当前幅度单轴正负的T90全部16/16到位，颈约.16–.18s，pitch/yaw/roll约.08–.10s；颈负/正MAE两种子均值1.655/1.486°，roll .449/.361°。相较3000颈2.324/3.177°、roll .770/.650°改善。
- **未扩幅**：组合命令用半幅度、更窄10%带，颈和roll仍有未到位；大幅度最差MAE约16.44°，不能只看单轴通过。保持阶段2同配置/同范围，再追加250到3500后复测，GPU1`head-precision2-allocation-002` session19614。连续两节点无进展再定位曝光/奖励，不无限延长。
- 3250头部评测首次命令漏写任务后缀，`head-response-3250.log`以KeyError退出、没有得到指标；修正为注册的`Official-Baseline`后另存`head-response-3250-official`，失败目录/日志保留。
- 预训练评审已同步N32768实际batch/Adam频率、两分支准入及训练契约优先约定，避免正文仍把P04写成性能未过。运行时审计中的旧固件默认不再当设备依赖。

### 用户最新澄清（13:42后，优先于此前设备配置假设）

- 用户明确：当前还没有实机；电机性能以v5数据集为依据，未来实机关节零位对齐训练模型。因此**训练契约是基准**：冻结v5 M6参数 + V2 CAD质量/惯量/几何 + V2 HOME/轴向；未来物理零位、方向和比例标定对齐它。当前不再查询设备地址或等待config，前面的API问题已由此回答解决。
- 当前固件ee77131的传统LS3、旧HOME坐标、EMA和限速仅作为待适配的部署接口，不反向强制训练继承这些旧默认。需要的V2运行时适配应以训练模型契约准备并独立回放；实际硬件动作/效果必须等实机存在后验收，不能用v5电机数据或仿真替代整机实测。
- 原完整台架P04 S0/O0/R1身份继续保留作为训练/研究基准；Firmware-LS3变体是独立接口诊断，不自动取代P04目标。其嘴部模型读数/健康边界冲突仍作为发现保留，不再以“等实机配置”为当前训练阻塞。

### 13:42 当前两卡正式工作及证据

- `current-bench-p04-throughput-32768`实际exit0，13稳态更新中位**107624.78控制交互/s**、采样119241.33/s，当前测量执行源一致。没有把旧报告SHA套到新代码。
- P04`scaled-resume-validation`实际exit0：从原正式1更新/8192环境改为64环境做1更新诊断，actor/critic/normalizer/Adam逐位恢复PASS；累计198144=196608继承+1536本段，且独立command_exposure恰为1536。此诊断不计入正式P042更新。
- GPU0已启动正式`standup-allocation-002-32768` session71430/PID2312707，从原001/checkpoint_000001.pt续接，32768环境追加128更新、每64更新双种子7桶×128评测；13:42进程存活且仍初始化，尚未声称有新完成更新。原1更新消耗保留，计划新增100663296控制交互。后续先检查resume-verification、status和实际日志再记录进度。
- GPU1正式`head-precision2-allocation-001` session33841/PID2310888，baseline3000→3250；更窄fine=.04的8192预检132305.24控制交互/s。13:40实测3050更新。当前头部开发比较图在baseline `head-response-3000/comparison-to-2000.png/.svg`：同幅度、两种子×8次，真实物理MAE明显下降；不是全范围/相机/实机验收。
- 重用的恢复/头部种子用于课程和奖励调整，属于开发评测。最终验收仍需冻结配方后另取未用于调参的种子/条件，并做运行时和实机检查。
- Error proxy再现脚本已保存，独立重建得到相同bank SHA `d2870d6c60c9ab2771ae97b427e8bb3b39ed54b954411ff6dcaaf0e9a2b97c39`；当前对照否定了把该有限误差库直接等同完整P04的假设。

### 13:31 近似模型不能替代完整P04；准备恢复实际P04预算

- 四强度误差proxy只读评测实际完成，冻结actor/normalizer未变：0/.25/.5/1分别1723/1715/1724/1678条通过（各1792）；最差单桶分别91.4/91.4/91.4/86.7%。满强度仍93.64%，与完整台架P04 0.28%差距巨大。这套有限轨迹误差块没有覆盖导致真实闭环失败的条件，不能标为P04替代或无条件推广所有技能。保留结果，用于后续原始/估计输入、动作依赖与复杂运动条件诊断。
- baseline3000评测完成：idle四轴MAE约.44/.32/.73/.12°；neck负/正2.33/3.18°，pitch/yaw当前幅度均8/8快速到位，roll .81/.64°但严格10%带到位不足。下一受限配方阶段2仅fine_std .08→.04，仍weight3、coarse_std .5、fine_weight .5、同范围/PPO/恢复分布；`head-precision2-smoke`64×5、真实超时/ONNX PASS。GPU1当前`head-precision2-throughput-8192`，通过后再决定250轮预算。
- GPU0`current-bench-p04-smoke`64×5、强制超时、完整融合S0/O0/query图及ONNX PASS，继续`current-bench-p04-throughput-32768` session36987/PID2302311。这是对当前执行源的准入复核；此前106706/s成绩保留，不用旧SHA冒充新版本。
- P04正式trainer现在允许有对应实际吞吐预检的8192/16384/32768，默认仍8192；不再将用户“8192吧？”误写成禁止环境数实验。源预检仍要求同task、同N、10个稳态更新、端到端>100k、图启用及测量执行源码SHA一致。
- 已修正N改变后的消耗记账：继承旧checkpoint实际controls，旧格式从其原run.json的N/T计算；本段仅累加新增更新×本段N×T。checkpoint、status和run.json记录task、N、父路径/SHA、额外交互预算与继承消耗。跨任务必须显式`--adapt-from`，不能悄悄当同任务resume；旧1更新保持196608控制步，不乘成32768环境下的数字。
- 正式P04仍保留起点001/checkpoint_000001.pt。当前吞吐复核通过后，先用32768追加128更新（100663296额外交互），每64更新独立恢复评测；实际per-env课程时钟保持，不以更多环境直接晋级。此有限完整P04预算用于获得真实学习证据，未宣布整体sim2real或当前固件已对齐。实际硬件配置路径问题仍待用户回复，独立工作继续。

### 13:20 延迟/误差分离与GPU轻量扰动已运行

- 20ms位置延迟单因素`baseline2000-delay20-transfer`完成：两种子各桶wins为128/120/121/126/118/123/125与126/121/124/117/121/126/125，合计1721/1792（96.04%）。相比完整台架P04的5/1792，延迟无法解释主要差距。基线训练本身位置白噪声仅±.001rad、速度±.25rad/s、startup位置偏置±.015rad；P04审计的相关误差和尾部更强。
- `head-response-2750`完成：idle MAE约.39/.39/.40/.42°；neck负/正误差2.73/8.61°（仍未通过），pitch正负约.08s、yaw正负约.10s到位，roll .97/.67°但严格到位率仍不足。持续有明显改善，因此同配方仅追加250至3000：baseline `head-precision-allocation-004` session69979。之后须再评测，不以改善代替最终全幅度验收。
- 新GPU `ErrorBlocks`从模型诊断nominal数据构建9600×16×2成对位置/速度误差片段，dt=.02s。原始输入和bank SHA、有限覆盖/无新硬件数据/不复现跨块连续性或fault/仅11.1V等局限保存在`training-assets/fast-proxy-v1/manifest.json`。未改冻结P04训练资产原文件。
- 新任务`Fast-Native-ErrorBlocks`使用原生量化命令、20ms q/v延迟与这组误差片段，移除重复的原位置/速度白噪声和encoder-bias事件；只改变actor，critic和奖励继续物理真值。两观测共用同一误差对，每控制步仅推进一次，部分reset即时不改变其它世界。针对性测试1 passed/1.49s；`error-proxy-smoke`64×5、真实32超时、ONNX PASS。导出仍为simulation-pretraining-only，明确proxy身份。
- GPU0当前`baseline2000-error-proxy-transfer`：同冻结baseline2000父模型，0/.25/.5/1四档强度，每档2种子×7桶×128×8s，不学习；用于决定渐进噪声课程而非预设大噪声最优。
- 只读mDNS直连（绕过不适合.local的HTTP代理）也因3s域名解析超时失败。用户的设备API/config路径问题仍未收到回复；没有写配置或设备运动请求。模型校准/健康边界问题继续保留，当前工作不依赖该回答。

### 13:11 固件LS3接入预检、失效定位与下一对照

- `head-response-2500`完成：idle MAE约.83/.24/.34/.15°；pitch正负t90约.08s、yaw正负约.10s且8/8到位；颈负/正误差3.59/14.79°仍失败，roll误差.75/.97°但严格10%带到位率下降。负yaw较2250的16.04°降至1.88°，颈也仍有进步，恢复未退化，因此同配方追加250到2750（`head-precision-allocation-003` session92435）已完成；当前GPU1`head-response-2750` session31099评测，未自动继续预算。
- 新`FirmwareNativeStream`已接到同一S0/R1原始寄存器生成器，保持独立名称`Firmware-LS3-S0-R1`和`simulation-runtime-alignment-only`导出。虚拟affine写成固件zero/scale形式，不把它冒充整机物理校准。全/部分/补齐reset、快速运动/故障的eager/graph全部状态逐位对照2 passed/9.32s；现有完整台架P04仍保留原身份。
- 首个64×5实际环境smoke **FAIL**，完整日志`firmware-ls3-smoke.log`。诊断`firmware-ls3-warmup-diagnostic.log`在64世界×15路中定位两个嘴部lane：物理闭合0，但查询约−.02636/−.02739rad，raw约−.030/−.025rad；时间、驱动速度和查询均就绪，失败来自按CAD mouth [0,.48]套用固件健康边界后的−.02容差。没有关掉门槛、没有理想状态回退；该健康边界本应来自实际校准配置，不能把CAD机械限位直接当已验证板端配置。
- 已向用户异步询问V2本体只读API地址或保存config JSON路径，缺的是实际零位/比例/限位/限速。首次推测mDNS地址的只读GET经代理返回502；没有写设备、没有扫描地址范围。该信息不阻塞其它独立实验；当前配置未知不是整项goal完成或暂停理由。
- `bench-vs-firmware-measurement-audit`已完成（script session8391）：128虚拟世界、静止/静止.2rad/慢正弦/快正弦，各2s，前.3s不统计；两前端处理**完全相同的S0生成原生回包**。这是模型诊断，不是新实测数据。标称静止位置MAE bench .507°、FW .467°，速度RMSE .307/.304rad/s；快正弦速度RMSE .685/.510。measured快正弦bench逐控制全15路有效94.93%、FW100%；本短测不代表整机故障率。嘴部raw<- .02rad比例标称4.57%、measured3.21%，支持继续核对实际校准/边界，而非随机挑过线种子。
- 为进一步分离主要迁移差异，新增明确的`Fast-Native-Delay20`单因素诊断任务：官方关节速度本来延迟20ms，仅再给关节位置加一控制步延迟；没有声称这是完整测量随机化。GPU0正在用冻结baseline2000做同两种子7桶1792条回放`baseline2000-delay20-transfer` session63051，对照此前Fast-Native 1792/1792和完整台架P04 5/1792。

### 12:57 实机接口审计纠正与可复核代码

- 在固件ee77131及`SERVO_OBSERVERS.md`核实：当前默认LS3使用传统原生点拟合，完整神经特征/220ms重预热只用于神经模式。训练台架P04/O0虽然不执行MLP，仍走CausalStream完整特征/输出链。此前只核对P04身份/冻结来源不够，不能将台架模型等同当前固件。详见[运行时接口审计](sc0090-v2-runtime-alignment.md)，含当前源码SHA。
- 另有待明确的V2坐标映射、实际动作比例、头/腿EMA(.5/.7)和配置限速差异。HOME常量不同本身可由正确校准坐标变换处理；EMA/时序需实际闭环匹配。未修改固件、未写设备，没有声称基线/P04已可实机运行。
- 新GPU `FirmwareLS3`用当前真实C函数作独立oracle，覆盖不规则时间、坏包、NaN、大间隔、速度越限及不允许外推；1 passed/2.93s，状态有效性/跨度一致，输出<=5e-7。只完成历史函数，训练接入和控制健康/坐标/滤波全链路仍待做；不把该测试当完整部署验证。
- Fast-Native 8192×24×16已完成，13稳态中位**132176控制步/s**，采样160270/s，按总wall-time129122/s，数值/真实超时/ONNX通过。台架P0432k的106706/s和这个快版成绩分别报告。
- baseline2500本段已完成，GPU1当前`head-response-2500` session82888做同协议响应评测，未追加新预算。2250的明显精度改善仍不等于四轴全范围达标。

### 12:48 迁移失败已定位到测量路径，精度实验有实效

- 完整P04只读迁移`baseline2000-full-p04-transfer-v2`实际完成，actor/normalizer逐位未变。标称两种子7桶共**5/1792通过，681/1792曾发生sensor fault**；实测R1 **13/1792通过，1082/1792 sensor fault**。其它失败并不都由fault解释，测量变化导致的动作失稳仍需诊断。首个评测脚本因MLP配置含None cnn/rnn字段失败；按mjlab runner原逻辑清理后重跑，旧失败目录和日志保留。
- 增加明确分离的Fast-Native任务：与完整P04共用process_actions（机械限位/count量化/可选slew/实际BAM目标转换），观测恢复官方噪声/延迟/encoder bias；不构造全量P04pipeline。导出`microduck-v2-fast-native-pretraining-v1`、`simulation-pretraining-only`，不继承完整P04身份。CPU边界/半计数量化及配置测试2 passed/21.61s；64×5真实32超时和ONNX PASS。
- 同一个baseline2000 checkpoint在`baseline2000-native-command-transfer`中两种子七桶**1792/1792通过**。因此纯命令路径差异不是此次严重失败的主要来源；完整测量路径仍是未解决迁移差距。该对照不能证明任意随机化可代替P04。
- GPU0当前`fast-native-throughput-8192`，8192×24×16，测量高速主干真实吞吐。下一步是可控输入下完整P04观测的误差/延迟/相关性/故障审计，再设计轻量扰动；不把近似观测误标为完整P04，不在已有实际失败下直接宣称可上机。
- baseline2250头部评测完成：一个种子idle四轴MAE约**.66/.16/.17/.10°**，明显优于2000的12.89/1.56/1.17/3.03°。当前幅度pitch正负t90约.08/.10s，正yaw约.10s，负roll约.04s；颈正负与负yaw仍未到位，正roll到位比例约62.5%，组合/最终幅度仍有不足。独立恢复七桶两种子仍全部128/128。
- 有实测改善且恢复未退化，GPU1按同配方只追加250：`head-precision-allocation-002` session39285，从2250到2500，12:48实际2280。不改PPO，不扩大范围，不凭idle改善认定完整头控成功。上一2250的metadata、全部trace及source快照保留。

### 12:41 P04单卡吞吐突破及实际下一步

- `fault-exact-throughput-32768/report.json`已完成16更新，前3剔除、13稳态：**106705.77端到端控制交互/s**，采样118083.61/s，中位7.37010s/更新，按总稳态wall-time为106444.01/s。完整S0/O0/R1、CUDA图、32k环境真实PPO、强制16384超时、ONNX等均通过。没有乘物理子步；此成绩仅适用于32768，8192仍约6.4万。初始化约数分钟，不能只报稳态隐藏冷启动成本；日志及source/provenance保留。
- 结论从“尚无任何单卡10万证据”更新为“32768能达到，8192尚不能”。正式N暂未改；32768批量786432、每minibatch196608，学习效率及阶段曝光不能视作8192等价。用户提出的高速主干+P04补充路线仍有价值，不因FPS过线取消效果对照。
- `head-precision-throughput-8192`实际PASS，中位131893.69端到端/s。GPU1已正式启动`head-precision-allocation-001` session30441/PID2258047，从baseline003/resume_002000.pt追加250至2250；resume参数/Adam/normalizer精确核验、stage5/polish及固定头范围恢复PASS。12:40状态2120。结束后先查恢复回归与同幅度头部响应，不自动追加预算。
- GPU0新只读评测`baseline2000-full-p04-transfer`：使用冻结baseline2000 actor+normalizer，完整P04标称/实测R1，各2种子×7桶×128×8s，学习更新=0。保存实际评测cfg、source.tar.gz、全qpos/qvel/稳定/invalid轨迹及checkpoint SHA；所有反馈fault均记失败，即使held-last后来恢复。该结果用于决定适应训练需求，不代表真机或完整头部验收。
- 上一goal轮属于实质进展：完成分段图反证、32k吞吐试验启动、baseline2000头部及直接目标诊断、头部新配方smoke和台账路线修订。本轮已取得32k突破证据并启动实际250轮精度预算及迁移评测，Goal仍ACTIVE。

### 12:33 用户提出路线调整：高速学习 + P04补充

- 用户追问是否还能突破10万，若不能是否用基线及其它随机化，或P04作扰动补充。当前证据：8192从零64183/s、学成策略诊断64947/s；19tick分段图`chunk19-throughput-8192`实际exit0、64394/s，收益约0.3%，不继续堆此类代码优化。分段完整逐位/故障对照2 passed/61.98s，但默认仍1tick，未采用无收益候选。
- GPU0仅追加一次32768×24、16更新环境数试验：`fault-exact-throughput-32768` session6521。正式8192未改；32768是诊断，B=786432、4个minibatch时每批196608，优化步/交互降为原1/4，不能只按FPS宣布更高学习效率。用户原句“环境应该是8192吧？”是询问/建议；此前台账将其写成不可变硬性要求过强。当前仍使用8192作正式起点，任何规模演化需依据实验并完整记录，不伪称用户明令禁止优化N。
- 建议调整研究路线：在P04工作区复用基线高速训练骨架学习各双足技能，再用完整S0/O0/R1做短预算适应和回归；官方baseline对照仍仅起身站立，冻结现有检查点、保留原正式P04从零1更新，不混写血统或消耗。这是待对照验证的方案，尚未把快版标成完整P04或实机合格。
- 快版必须保留V2 CAD、M6、HOME、50/200Hz、61/14接口，以及真实命令的限位、整数count量化、仿射方向/比例和端到端时序；P04与baseline动作链并非完全一致，不能直接把baseline ONNX改标签上线。基线当前仍`simulation-baseline-only`。
- 轻量随机化按实测数据标定：每episode固定的偏置/比例、电压/摩擦/负载范围；GPU环形缓冲模拟测量延迟和速度估计；时间相关残差块、条件采样（电压/速度/负载）、读数异常/超时/恢复。不能独立白噪声替代所有因素；固定系统偏差应修正标定，确定性软件bug/奖励冲突不能靠噪声掩盖。也不能只给少数env启用mask但仍分配全量完整pipeline而宣称省算力。
- 最小对照三臂共享同一个已学成动作的父checkpoint：A=快版继续相同新增交互；B=快版+实测标定扰动；C=快版后完整P04微调。先在起身站立验证，统计总交互、Adam步、GPU时、父训练成本和各臂新增成本；公平预算优先等新增交互，同时报告等wall-time效果。用未参与拟合/课程的实测条件和种子评测恢复、头部逐轴响应、健康姿态/相机运动，再做CPU运行时/硬件回放。P04模型通过只是其中一项，不能循环证明自身真实。
- 初始预算建议把大部分交互用于高速技能学习，P04只承担短适应/困难条件回归；比例由这次对照决定，不能预先断言80/20或更大噪声一定最优。若C无实测收益则不强制每技能长跑P04；若故障集中某类条件则只增加对应暴露。
- 方法依据：Peng等[动力学随机化研究](https://arxiv.org/abs/1710.06537)展示其迁移价值；Xie等[消融研究](https://arxiv.org/abs/2011.02404)说明随机化的必要性依赖建模及任务，不能据别的机器人成功直接保证本V2。具体路线选择依据本机对照。
- 基线003已实际exit0、累计2000，1250/1500/1750/2000两种子七桶全部128/128无故障，四份评测checkpoint SHA均与保留文件匹配。`head-response-2000`117案例×8×2完成；actor命令槽逐步核对通过，idle颈误差约12.7–12.9°，head_roll约3.0–3.7°，多数阶跃仍失败；不能称头部达标。
- `head-direct-target-diagnostic-2000`仅诊断覆盖头动作：部分pitch/yaw阶跃达约1°，但部分case失稳，说明还需头腿协同；未用于正式policy或部署。1500/2000评测脚本原字节已按原报告SHA恢复保存在各自source目录，未改报告历史哈希。
- 已准备可选`v2-head-precision-1`：冻结已暴露的±.39/±.39/±.49/±.11rad范围，头跟踪weight .75→3，保留std .5并加入.08rad fine(50%)；PPO、恢复七桶和动力学不变。只允许从polish父checkpoint开始，checkpoint记录配方且禁止静默丢失；评测读取实际配方范围。64×5、真实32超时、ONNX验证PASS。GPU1正做8192×16配方吞吐`head-precision-throughput-8192`；通过后先250更新有界实验，复测头部及恢复，不无限续训。此精度修复独立于用户提出的sim2real扰动路线。

### 12:25 最新吞吐、反证及头部评测

- `fault-exact-throughput-8192/report.json`完成16更新、剔除前3更新：8192×24中位3.06324 s/更新，**64183端到端控制交互/s**，采样70102/s。强制4096真实超时、自举、测量/物理/电机CUDA图及ONNX对照通过；数值PASS不代表10万性能目标通过。最新故障严格对照`fused-registers-fault-division-parity.log`2 passed/72.76s：80步快速运动、R1电压、NaN输入及reset恢复的所有因果张量逐位相同；修复spike斜率除法的1 ULP误差，未放宽断言。
- `learned-policy-throughput-8192/report.json`刚完成：仅导入baseline1500 actor/critic/normalizer，Adam与课程全新，明确为诊断；中位**64947控制步/s**、采样70981/s、3.02722s/更新。相对从零策略仅约1.2%提高，因此“学会站立后吞吐自然突破10万”没有得到支持。该16更新不计入P04正式训练，不用它冒充公平比较。
- 下一步仅对当前融合版做一次带CUDA阶段事件的剖析，再决定是否将重置预热57次图调用批量化；保留57个5ms tick、20ms命令与全部私有RNG消耗，必须同故障对照通过才测吞吐。禁止删掉P04测量或降低物理频率换速度。用户尚未回应单卡/合计吞吐口径，继续分别报告。
- 新头部独立评测`evaluate_v2_head.py`已覆盖81案例×8重复×2独立种子：2s零命令、2s阶跃、2s回中，正负小/当前/最终幅度、双/四轴；物理头关节在训练reward同一边界采样并独立校验，camera四子步200Hz，20ms窗口采用最大两两方向跨度。指标3项测试通过，未到位样本不会从统计中丢弃。仍只覆盖站立，尚非行走/转向或实机稳像验收。
- baseline `head-response-1500/report.json`显示**起身全通过但头部控制未达标**：一个独立种子的零命令颈pitch约−18.37°、head_pitch约+7.00°偏差，多数当前幅度阶跃2s内未到90%。此checkpoint恰在polish开启之前；不能据此说后续500更新无效，也不能说头部已成功。保存全关节/相机/指令/健康轨迹与原checkpoint SHA；下一步评测2000，并核对actor命令槽与物理可达性，再决定精度奖励/课程调整。
- baseline003从1000续至2000：1250、1500、1750恢复节点双种子七桶均通过（以实际evaluation文件为准），1500后stage5/polish；12:23实测1980，随后日志已进入末段独立评测。最终状态须等`status.json`和真实进程退出核验。新checkpoint与晋级后resume分开保存，旧002元数据覆写的证据局限保留。

### 12:03 实测口径与当前任务

- `fused-throughput-16384`实际exit0，16更新/13稳态更新，中位4.49546s/轮、**87470控制步/s**（采样96019/s）；N翻倍后仍未达P04单卡100k，不把采样/物理子步冒充端到端交互。8192已测64634/s。
- 按TensorBoard真实wall_time取共同窗口 **03:59:48.434207–04:00:46.996118 UTC，共58.56191s**，只计完全落入窗口的更新，丢弃前3更新和边界不完整轮次：P04 16384环境13更新=5,111,808步，baseline 8192环境38更新=7,471,104步，合计保守下界 **214865控制步/s**。这是真实并发，不是孤立峰值求和；报告`runs-v2/concurrent-throughput.json`。用户吞吐验收口径澄清仍未回复，正式N没有改动。
- GPU0当前补充融合后端故障/再预热严格对照：测试新增measured R1电压、快速运动、一次非有限物理输入、恢复reset，80步，所有浮点/整数状态逐位比较；`fused-registers-fault-parity.log`，session38061。正常40步既有对照也一并执行。该测试必要，因为前一个对照只覆盖温和平滑运动。
- baseline003继续从1000到2000，最近实测1190/stage4；后续应查1250/1500评测及`resume_*.pt`保存哈希，检查最终frontier与polish再扩大头部/稳像验收。头部独立响应与200Hz相机评测代码仍未完成，不能只据起身全通过宣布策略完成。

### 11:59 当前执行与重要记录

- P04完整寄存器分段融合已通过全部逐位状态对照（`fused-registers-storage-parity.log` 1 passed/55.92s）；8192×16实测`fused-throughput-8192`中位**3.04186s/更新、64634控制步/s**，含强制超时、ONNX实际输入对照、执行源元数据。原始除法修复数值差异，普通/推理张量存储统一修复capture重编译；没有放宽数值门槛。该成绩仍不满足先前采用的P04单卡100k门槛。
- GPU0当前做**16384环境、24步、16更新的短吞吐实验**，`fused-throughput-16384`，session59399。正式环境要求没有悄悄改成16384；这是环境并行数对利用率/吞吐的有界比较。N翻倍会把B/mini-batch翻倍，同交互预算优化次数/课程时钟须重新对齐，不能只报速度。实测及是否允许改变正式N均尚未确认。
- P04正式仍停在001的1更新。其正式脚本现在按preflight实际pipeline/frontend启用同一后端，并核对测量执行源码SHA，防止拿编译版成绩给未编译版准入。shell入口默认禁止Triton FP fusion、缓存放数据盘；改变硬件/模型/observer身份不在本次性能优化内。
- baseline `standup-allocation-002`完成累计1000更新，真实exit0，250/500/750/1000双种子七桶均128/128，stage4。单独64环境、1更新的`stage-resume-validation`验证从1000恢复后所有spawn_stage=4，common_step=24000，头部范围已恢复(-.39,.39)/(-.39,.39)/(-.49,.49)/(-.11,.11)，PASS；这是诊断fork，**不把1001标签作为8192正式消耗**。
- GPU1当前正式`standup-allocation-003`，session74659/PID2227155，从002/checkpoint_001000.pt追加1000到累计2000，每250评测；11:58实际1090更新、stage4。源、NN/normalizer/Adam、LR与新episode课程恢复均可查各自resume报告。首段通过是继续学习的证据，尚无最终头部/健康姿态/实机验收。
- 审计发现旧trainer在评测晋级后覆写同名checkpoint，改变了报告绑定的文件哈希。**已修复后续保存**：`checkpoint_*.pt`保留评测前原文件，晋级元数据另存`resume_*.pt`；003已使用该修复。未重写旧报告/伪造原字节。002的四个节点已与保留的model_249/499/749/999逐位比较，actor/critic/normalizer/Adam全部相同；局限与实际新旧SHA在`checkpoint-provenance-audit.json`。旧原始评测checkpoint字节已被当时逻辑覆盖，不能声称仍可按原SHA取回。
- 下一步查16384实际结果和两卡同窗口吞吐，等待用户对单卡/合计口径的澄清；P04性能工作不替代头部响应/稳像评测、其余技能迁移和真机契约核对。Goal ACTIVE，旧SC0090/HD1910任务继续保持原暂停状态。

### 11:52 后续验证状态

- 原始除法修复后，融合版本已越过最初数值对照，但在正常rollout CUDA capture触发Dynamo重新编译，`fused-registers-division-parity.log`实际失败。定位为eager阶段收到inference tensor，而capture阶段换成普通克隆张量，输入类型guard不一致。
- 已统一`FeedbackGraph`从前两个真实eager tick起就使用同一普通q/v存储，不增加隐形tick或RNG抽样。`fused-registers-storage-parity.log`已PASS：1 passed/55.92s，全部浮点及整数状态逐位一致，完整/部分/补齐/重复reset通过。开始8192×16完整吞吐`fused-throughput-8192`（含强制超时、导出及新执行元数据），源码tar/SHA已快照。完整寄存器融合仍默认关闭，需等实测吞吐准入。
- 基线11:51实际完成930更新、stage3，进程继续至1000与最终本段评测；Goal保持ACTIVE。吞吐口径澄清暂未收到回复，仍按原先严格单卡目标推进。

### 11:49 续接状态（先读取实际进程）

- 最新已完成吞吐：`compiled-throughput-8192`前3更新剔除，中位3.76549s/更新、**52213控制步/s**，完整16更新及强制超时/导出PASS，实际exit0。该版本为编译read/tick/query、保留原LS规约、预热图和关节索引缓存；仍未到P04单卡10万。源码快照及SHA在该运行目录，不能把后续未验证代码混称这一成绩。
- `compiled-profile-8192`实际exit0，测量reset约1.354s/更新、正常测量0.625s、物理0.685s（剖析开销单列，不当吞吐）；单独donor延迟15–1920活跃lane约1.25–1.38ms，122880lane/2400活跃约1.93ms，**没有证据支持继续把donor检索认作主要瓶颈**。未实施推测性的32路donor改写。
- 正在试验寄存器生成剩余算子的分段融合（默认关闭的`fuse_registers`）。整段融合触发Triton编译错误，日志`fused-registers-parity.log` exit1；改为在read/tick处明确编译边界后能执行，但严格对照发现physical_context插值1 ULP差异，日志`fused-registers-segmented-parity.log`和`fused-registers-first-difference.log`。没有放宽断言，也没有启动该版本正式训练。
- 差异定位在插值浮点真除法；新增不被编译器改为倒数近似的原始GPU除法算子，保留原BLAS规约、LS规约、私有Torch RNG和Warp donor算子。当前GPU0针对性验证：`fused-registers-division-parity.log`，**session15739**。先查结果，再决定是否做8192吞吐。实验类位于`measurement/compiled_registers.py`，当前普通任务仍默认未启用任何编译后端。
- 新导出元数据增加独立`microduck_measurement_execution`，记录donor sampler版本、真实pipeline/frontend类、源码哈希和图/紧凑reset开关；不改冻结硬件contract身份，不改原sc0090/source-identity.json。该元数据变更待下一次实际导出覆盖。
- GPU1正式baseline进程PID2200765/session43689持续运行；第750更新两种子七桶各128/128通过、stage2→3；目标仍累计1000更新，后续完整头部/姿态/扰动验收未做，不可判最终技能完成。
- 已向用户异步澄清“10万步/s”按P04单卡还是两卡真实并发合计验收；尚待回答。此前按更严格单卡口径推进，不能擅自把单独测得峰值相加当合计通过。该问题不暂停工程工作和已准入基线训练。

### 11:36 算子融合验证及首个实际课程节点

- 新剖析`prehistory-profile-8192`显示预热GPU流约1.979s/更新、正常测量0.846s、物理0.671s。另有反复按正则解析14关节的CPU开销，已按不可变Entity缓存命名映射（两分支源码；已运行基线进程仍使用其原源码快照，未热改进程）。
- 直接torch.compile的read/tick/query在严格对照中失败，保留`compiled-stream-parity.log`。禁用Triton浮点融合后，误差仅剩LS浮点消去尾差，40步及部分reset最大3.9e-15，全部整数/布尔状态一致，仍未把此诊断冒充逐位通过；`compiled-stream-no-fma-differences.json`明确EXPERIMENTAL_NOT_ADMITTED。
- 为保持原语义，把LS3/LS5原FP64规约作为不被编译器重排的GPU自定义算子，其余read/tick/query融合；统一eager/capture推理上下文，消除capture期间Dynamo重编译。最新`compiled-stream-ls-exact-parity.log` **1 passed in 41.63s**：40步、完整/部分/重复/补齐reset的所有张量逐位一致，未放宽断言。编译后端仍为显式opt-in，要求TRITON_DEFAULT_FP_FUSION=0。
- GPU0启动`compiled-throughput-8192` session51731，8192×16、强制超时、实际PPO/导出，明确`--compiled-measurement`。首次编译耗时单独看，不能充作稳态成绩；待实际报告后决定是否准入。
- baseline第250更新已完成两独立种子×7桶×128的8s回放，所有桶每种子128/128稳定通过、无无效状态；首次连续0.5s稳定均值站立约.67s、坐姿约1.7s、其余约1.05–1.48s。记录`evaluation_000250.json`，课程按既定规则stage0→1；这是起身稳定阶段门槛，不是头部响应/稳像/实机验收。当前训练继续至1000预算。

### 11:29 性能修复及基线恢复

- P04重置预热现在按容量桶复用工作区和CUDA图，保留57次5ms真实预热tick/命令/私有RNG/重试；完整+部分+补齐容量+重复重置的全部状态逐位测试PASS（6.45s）。首版capture因Python bool转GPU标量失败，失败日志保留；改为已存在GPU confirmed张量后通过。
- P04 `prehistory-throughput-8192` 16更新完成，中位4.75125s/更新、41380控制步/s，比前版16.009s明显改善，但仍未到10万；正常测量图capture1/replay1534，导出对照1.2e-7。当前GPU0追加单更新剖析`prehistory-profile-8192` session60495，确认剩余瓶颈再做有限修复。
- 独立基线已通过其正确性/吞吐/曝光/评测准入，不继续等待P04专有测量优化。GPU1正式恢复`standup-allocation-002`，session43689，从001/checkpoint_000012.pt追加988更新，到累计1000；为及时晋级，独立双种子评测间隔由500改为250（250/500/750/1000）。物理从新episode开始，原12更新、旧源和消费保留。
- 发现并修复PPO续训LR缺陷：Adam.param_groups已恢复，PPO.learning_rate却仍是构造初值，下一minibatch会覆盖历史LR。修复从加载后的同一Adam组恢复调度变量；旧checkpoint同batch下一次参数与Adam矩逐位测试PASS。两分支同修复。
- baseline实际恢复核验PASS：actor/critic/normalizer/Adam所有张量及状态逐位一致，completed=12、common_step_counter=288、LR=.002962962962962963；旧bootstrap contract=null，新契约为无reset后继critic。证据`standup-allocation-002/resume-verification.json`。最新日志已进入正常更新（约1.95s/轮，含诊断），不是只启动了壳进程。
- 正式脚本新增机器准入：8192、至少10个稳态测量更新且端到端中位>100k，避免把smoke PASS误当性能PASS。P04保持正式未恢复。

### 11:25 身份核实与剖析结论

- 用户追问P04是O0还是O1。只读`git show 3bf729e:physical-validation/sc0090-testbench/common3000/selected-policy.json`核实：2026-10-11 explicit_user_instruction选定P04/seed42，S0/O0/R1，observer.id=O0、neural_observer_enabled=false。当前V2分支172c5ed/070d606与该身份一致，未擅自把O1降为O0。原JSON字节及commit/SHA证据已存`docs/evidence/p04-selected-policy-3bf729e.json`和provenance；原dataset工作区未改动。权重文件叫O1并不等于执行O1，O0共享仿射mapping，forward只在mode=O1时执行MLP。
- 单更新剖析已完成，工具退出0。cProfile总38.052s（存在明显剖析开销，不能当真实吞吐）；24次measurement.reset累计CPU墙钟30.468s，占约80%；96次正常measurement.advance的GPU流耗时0.843s、physics.step0.672s。每次reset在Python驱动57次预热advance，合计1368次。主要瓶颈是重置预热的Python/小核调度，而非O1残差网络；下一处修复应复用预热CUDA图，保留280ms时序/残差/有效性门槛和不重置lane。
- baseline最新曝光/诊断/评测4更新验证完成、实际退出0。已补齐最终status与checkpoint保存本段exposure统计及segment_start_update，避免最后状态覆盖后计数丢失。

### 11:21 实测与下一步（覆盖下方旧快照）

- 用户现场指出GPU利用率低于50%。实时核验GPU0=1%、GPU1=22%；GPU0上一吞吐试验已结束，GPU1在跑P04部分reset候选。没有把空闲卡写成满载并发训练。
- 三个8192×24、16更新试验实际退出码均0，前3更新剔除：baseline `graph-throughput-8192` 中位1.51465 s/更新、129804控制步/s；P04 `graph-throughput-8192` 17.6361 s/更新、11148步/s；P04 `compact-throughput-8192` 16.00946 s/更新、12281步/s。报告PASS仅表示数值/导出测试通过，**P04吞吐未通过**。融合donor版本比先前masked版本更慢，保留失败候选证据，不隐瞒回退。
- P04新增部分reset紧凑工作区，仅对选中lane做预热；同私有RNG抽样消耗，图/eager/部分reset全部因果状态逐位对照PASS。实测约9%改善，远不足目标，停止盲目叠加优化。
- 用户问“残差网络测量是否很慢”：当前身份是S0/O0；O0没有执行O1残差MLP，开销应拆分为经验残差donor检索、因果寄存器/历史状态处理、reset预热，不将整条慢归因于神经网络。
- GPU0运行真实PPO单更新剖析 `sampling-profile-8192`，session53646；先3次warmup，再记录cProfile与CUDA事件的measurement.reset/advance/command、physics.step、ppo.update，不把带剖析开销的结果当性能成绩。
- GPU1运行baseline最新曝光统计+PPO诊断+独立评测完整循环 `exposure-allocation-validation`，session40607，64环境4更新/2评测。两任务目的明确，不用无关负载刷GPU利用率。
- 下一步以剖析证据决定一处针对性修复；头部完整响应/稳像、其它技能、硬件回放仍未完成。正式恢复前按新来源记录、保留此前1/12更新和检查点。

### 11:10 最新状态（覆盖下方10:41历史快照）

- 用户进一步要求：正式训练前把奖励—效果指标、PPO超参数及演化、课程升级/平台期、训练评测计算/来源一致性、GPU流水线讲清楚；P04测量GPU化、减少CPU↔GPU通信；交互吞吐超过10万步/s；两卡并行。用户确认不含轮滑。24步已核实为官方行走/起身配置（基点b02fdea），并非V2最优值已证实。
- 评审文档：[sc0090-v2-pretraining-review.md](sc0090-v2-pretraining-review.md)。已列出具体运动指标、20余项设置/作用/演化原则、来源和当前准入缺口。目标仍是全部P04策略+一个基线起身策略，不能缩小为本轮代码预检。
- 首次8192预检两分支PASS，已启动正式`standup-allocation-001`，后在用户追加“正式训练前评审”要求时按更新边界停止。P04真实完成**1**更新，基线**12**更新；实际进程退出均0。checkpoint_000001.pt / checkpoint_000012.pt、原source.tar.gz、run.json、日志和消耗全部保留；不是0更新，不删除、不改标签。停止记录`p04/pretraining-review-stop.json`，最新状态各自`standup-allocation-001/status.json`。Goal不暂停，继续工程工作。
- 进一步审计发现官方/新分支尚未带入历史PPO修复：坏return/advantage在归一化后清零；真终止+超时同时发生仍自举；纯超时使用V(s_t)。已分别移植finite目标拒绝、终止优先及独立后继critic采集/TerminalCriticPPO。两分支各17项CPU/GPU针对性测试PASS（3个分布式测试未运行，本轮两卡是独立任务不是DDP）。
- P04 `ppo-correctness-smoke` 和baseline `ppo-correctness-smoke-v2` 均64×5 PASS，强制触发32个真实纯超时并自举32次；不是没有触发超时的假覆盖。
- 新`tests/test_v2_terminal_observer.py`在两分支各3项PASS：普通、命令重采样、推扰，独立auto_reset=False实际后继critic对照；除3个CoM线速度浮点尾差≤5e-7外其余71维逐位一致；活环境qpos/qvel/命令/计时器/CUDA RNG不变。首版失败是测试未展开TorchArray，修复测试后重新实际执行，失败日志保留。
- P04测量执行实现现在位于`sc0090_v2/measurement/`，**原`sc0090`冻结源码及source-identity.json保持原字节**，没有伪造旧证据。优化包括固定形状GPU residual donor kernel、持久状态、整段5 ms S0/O0/query CUDA graph、热路径异步GPU断言。仍1024候选、同上下文目标、同残差库/时序；新每lane RNG协议版本`v2-device-donor-1024-v1`，不宣称与旧采样顺序逐位相同。
- 测量graph与eager40步、一次部分reset、所有因果状态逐位对照PASS，graph只capture1次/replay38次；连同冻结源/反馈测试5项PASS。独立候选打分oracle及旧masked scorer共3项PASS，验证相同候选选择一致且未选中lane不变。
- 已加入`custom_v2/exposure.py`，GPU累积实际使用的头部命令（零/单/双/四轴、正负、幅度分位），每10更新才拷小计数到CPU日志。正式脚本加入历史已配对验证的PPO诊断器（实际KL/clip fraction/梯度/EV/学习率），禁止只看reward。
- baseline最新完整训练循环验证`ppo-allocation-validation` 4更新/2次评测已完成，status ALLOCATION_COMPLETE、goal_complete=false；这是小批验证。两卡当前跑最新8192×16吞吐：P04 session**43960**，baseline session**95421**，目录各自`graph-throughput-8192`；GPU采样工具session**42993** (`timeout120s nvidia-smi`)，日志p04/graph-throughput-gpu.csv。
- 下一步：实际查询上述句柄和report，稳态丢弃前3更新；检查端到端是否>100k/卡（同时报告真实并发合计）。若P04仍慢，按计时定位测量/部分reset而非删模型或物理步。随后完成更新后的GPU/评测/曝光/恢复回归，更新准入判定；从保留检查点续接，累计预算对齐到1000（P04剩999，baseline剩988），评测对齐500/1000，不重置原消耗。

### 10:41 历史快照

- 当时按用户的8192环境建议确定初始规模；此前将其转述为固定硬要求过强，后续已按吞吐与学习证据演化至完整P0432768。用户要求高GPU利用率、消除已知软件问题，但不得无止境优化阻碍实验；借鉴历史起身/站立/步行课程与奖励，头部响应单独验收。
- 依赖已固定：BAM `62bd8ce12154340be97e06f7f41a0ca8f116d967`（桥接 API），MuJoCo 3.13.0 / Warp 3.8.1 / mjlab 1.3.0 / torch 2.9.1；冻结电机系数仍是 v5 M6。隔离 venv 已安装到数据盘 `venv-p04`，两工作区用各自 PYTHONPATH 运行，共用同一只读依赖环境。
- 入口统一 `scripts/v2_python.sh`。R535 主机必须加载备份中已有 CUDA 12.9 用户态兼容库；不能直接运行 Python 后假定 graph 已启用。初次 smoke 的 graph 实际关闭，记录保留；后续验证确认物理 graph=true，两个电机图各 replay=599，和 eager 的扭矩/摩擦误差=0。
- 两分支 `runtime-validation-graphs-v2.json` 均 PASS：53 个真实 V2 世界，3 s 姿态保持、HOME 精确重置、摩擦/阻尼编译默认值复原、部分 reset 其它世界 qpos/qvel/电机 duty/摩擦/阻尼逐位不变。
- V2 坐姿经 GPU 扫描选定：两侧 hip_pitch=-.10、knee=-1.10、ankle=-.85，左右 hip_roll=+.0873/-.0873；两分支最大倾斜约 2.3°、最终高度 .06337 m。旧坐姿迁移候选倾斜约 89°，已否决。站立实际高度约 .112 m；奖励使用这个实测值，出生 HOME 仍用 CAD .11638049 m。
- CPU 足底适配初次 V2 对照失败（接触点相差约5 cm）；原因是原适配 C++ 禁用浮点融合，而 Warp 默认启用。改用 FMA 后，100 个 V2 状态×两足的独立 GPU 几何对照以及模型隔离/恢复测试均通过。没有修改 CAD 碰撞网格或 GPU 接触算法。
- P04 姿态 helper 已恢复物理真值；V2 轴向和坐姿限位测试已建立。最新版 P04 配置/反馈/动力学测试 21 passed；基线前一版 18 passed；masked donor 独立 CPU/GPU 等候选逐位对照 2 passed。新课程还需同配置最终 smoke 与训练循环验证。
- 历史 recovery V3 课程/奖励已移植到 `custom_v2/recovery.py` 和 `training.py`：7 起始桶（站、坐、俯卧、滚转前沿、左右侧、仰卧），前沿150→120→90→60→30→0°；正负势能差塑形、连续 .5 s 双足稳定、轻平滑税、颈关节物理限位成本；只有独立双种子评测晋级。新模型从零训练，不照搬旧专家的成熟头部/CoM 随机化。
- 头部 GPU 分桶采样已实现：精确零命令60%，单轴20%，双轴10%，四轴10%，含范围端点；逐步扩大幅度，保持组合暴露。独立头部阶跃/回中/组合响应评测仍待完成，不能把起身门槛代替头部验收。
- 4096 短测（旧全批 donor、10 更新）：P04 147.12 s，基线12.22 s，均 graph=true、无 contact overflow、ONNX 对照通过。P04 实测瓶颈为部分重置时仍对所有舵机做1024候选搜索。只做一次针对性修复：仅给需要新残差块的 lane 搜索，采样和打分仍在 GPU；相同候选对照逐位一致，RNG 消耗变化已明确，正式训练从新采样版本零步开始。
- 正在验证 `train_v2_standup.py` 的完整 2 更新→两种子7桶回放→保存/晋级循环：P04 session37493/PID2161764，基线 session13308/PID2161779，目录各自 `allocation-validation`。启动于新隔离环境，首次 import 较慢；进程10:40仍活跃，不能因观察超时重启。
- 下一步：完成上述验证；当前课程 8192×5 preflight；固定源码/配方后双卡启动各1000更新首个正式预算，500/1000独立评测。P04/基线每臂每1000更新=196,608,000环境控制步；基线只做起身站立，完成后腾卡给 P04其它动作。
- 关键失败、测试实际退出码已保留到 `runs-v2/preflight-process-exits.json` 与各日志。台账下文初始记录为历史，不作为当前状态。

这是本任务唯一主台账。每次续接先检查实际 Git、进程、GPU、日志和产物，再更新本文件；台账中的“运行中”不能替代进程存活证据。失败结果保留，不将预检、训练 reward 或短跑成功写成最终技能达标。

## 1. 用户目标与明确范围

- 使用两张 NVIDIA L40，完成 SC0090 第二版机械结构的 P04 全部双足策略训练。
- 官方基线分支仅训练一个完整起身站立策略，与 P04 同协议比较实际效果。
- 参考 `backup/local-work-20261015` 的 GPU 优化、HOME 重置、编码器偏置修正、CPU/GPU 足底对齐，以及近一个月课程、奖励、训练和回放经验。
- 目标包括动作完成、自然步态、健康姿态、头部稳定、指令跟随准确及时。训练完成必须有逐技能行为证据；不能用吞吐或总 reward 替代。
- P04 与基线都要检查并修复训练和参数设置，核对真实机器人部署接口和 sim2real 条件。
- 用户明确：本轮不包括轮滑。轮滑、轮滑蹲起、轮滑坡道、轮滑起身、Swizzle、轮滑 Spin 不列入本轮。双足原地转向由行走策略覆盖。
- 用户明确要求建立稳定 Markdown 台账，持续记录目标、方案、实际工作。
- 用户询问/建议过8192环境；这不是不可变的硬性限制。初始两分支以8192起步，完整P04后来按实测吞吐采用32768，当前基线适应P04也用32768；实际N/T与继承预算逐段记录。性能优化有边界，以实际瓶颈和学习效果为依据。
- 头部四轴响应需覆盖正负、回中、微调、大阶跃、组合和站/走/转/启停上下文；报告响应延迟、90%到位时间、超调、稳态误差和相机/姿态代价。
- 旧任务此前由用户暂停；本次授权是新的 V2 训练。旧 SC0090/HD1910 attempt27 保持暂停，保留全部旧产物。

## 2. 权威位置与分支

| 用途 | 位置 / 身份 |
|---|---|
| 工作区根 | `/home/l40-vm01/Documents/ai-research/microduck`（自身不是 Git 仓库） |
| 历史只读来源 | `microduck_rl`，`backup/local-work-20261015`，`2ca93c9` |
| P04 工作区 | `microduck-v2-p04`，`train/v2-p04-20261015`，起点 `070d606` |
| P04 上游 | `awesome/codex/v2-sc0090-s0-o0-r1` |
| 基线工作区 | `microduck-v2-baseline`，`train/v2-baseline-20261015`，起点 `bc8a4d3` |
| 基线上游 | `awesome/codex/v2-sc0090-official-baseline` |
| 本台账 | `microduck-v2-p04/docs/sc0090-v2-training-ledger.md` |
| 新实验持久目录 | `/home/l40-vm01/data2/experiments/microduck-v2-20261015/{p04,baseline}` |
| 工作区实验入口 | 两工作区各自的 `runs-v2` 符号链接 |

主磁盘约 11 GiB 可用，数据盘约 1.6 TiB 可用（10:12 初查）；训练日志、检查点和回放放数据盘。不得擅自删除旧实验腾空间。新工作区隔离源代码；不得修改旧暂停实验的冻结运行目录。

## 3. 策略清单与依赖顺序

| 阶段 | P04 内容 | 基线内容 | 状态 |
|---|---|---|---|
| 0 | 依赖、模型、HOME、观测/动作、GPU、回放预检 | 同样预检 | 训练准入通过；完整CPU/运行时回放待验 |
| 1 | 起身并稳定站立：站、坐、俯卧、仰卧、侧卧、随机跌倒、扰动 | 同一任务和评价协议 | 已有独立及基线根完整恢复保留候选；本轮头控对照已停止追加，保留其取舍与未通过项。后续默认完整P04，非实机优胜结论；具体候选见顶部 |
| 2 | Velocity：前后左右、曲线、原地转向、启停反转；头/身体命令 | 不训练 | fresh256+两段YawFine至512累计402,653,184，阶段已关闭；449/512两条件均336/336，转向/零速/步态/头稳仍有局限，保留不同优势候选 |
| 3 | VelStand：运动、静止和跌倒恢复的连续切换 | 不训练 | 阶段256已完成150,994,944；两条件命令336/336，静止头稳改善，运动/停车头稳与恢复保持仍有缺口；阶段关闭保留128/256 |
| 4 | SitStand：受命坐下/起立/保持 | 不训练 | fresh128+ClearSeat128+Lesson4s128累计150,994,944，阶段已关闭未合格；修正实际自干涉后两条件仍无完整保持，不再自动续训 |
| 5 | GroundPick：俯身拾取动作、回正与负载扰动 | 不训练 | 完整P04首128已75,497,472；两条件低位/完整目标均0，阶段关闭未合格，不使用剩42盲续 |
| 6 | BallKick：支撑、踢球、复位 | 不训练 | 物理/接触/多实体successor与64×5通过；actor61/critic82，实际N16384准入中；左右共享总上限100,663,296，正式0 |
| 7 | Roulade：翻滚、起身与稳定落地 | 不训练 | 修复后首128已50,331,648，两条件完整/身体落稳0；剩余128只做有依据的近完成曝光，先物理/跨源准入 |
| 8 | 适用任务的 rough/抗扰动扩展、任务切换、ONNX数值与CPU回放（无固件部署） | 起身站立最终比较 | 待前置技能 |

P04主分支与各冻结独立源分开记录；V2起身、walking、SitStand、VelStand、GroundPick已有本体迁移源。Roulade首段已训练、余段准入中，BallKick实际批量准入中；不能直接启动未迁移的XL330任务充当V2策略。

预算以顶部实际分配为准：基础基线原为8192×24，当前独立P04及基线适应P04均为32768×24；历史不同N不按更新号直接比较。成熟T12同预算对照已完成且未获综合收益，当前仍采用T24。每项新配置须小规模正确性与对应N吞吐预检。当前两卡推进P04 Roulade/BallKick；起身头控平台后停止专项追加。其余动作按独立节点分配，不能因reward上涨无限续训，也不能因预算结束认定完成。

## 4. 模型与 sim2real 不变量

| 项目 | 已知契约 | 待验证事项 |
|---|---|---|
| V2 机械 | CAD 质量 0.600941825 kg；15 个驱动，策略 14 关节、嘴巴独立 | 每个新任务实际编译模型和惯量不漂移 |
| HOME | V2 CAD 保存姿态为关节零，根高度 0.11638049189959615 m | 全部/部分重置、地面高度、控制历史、摩擦/阻尼和电机状态 |
| 策略 | 61D actor、14D action，50 Hz | 顺序、尺度、归一化、上一动作、切换命令语义 |
| 时序 | 物理 5 ms、策略 20 ms；P04 反馈 10 ms、查询延迟 20 ms | CPU/GPU 与部署固件一致，重置无额外隐形电机 tick |
| 动力学 | 冻结 v5 M6，SHA256 `7c35a732575bfad135d5bf7c7eeb44f268ffd96862064f04e9e7a5105bbf6fb4` | 公式与桥接依赖版本，不能混用旧 SC0090 系数 |
| 工况 | P15/D30/I0、goal profile=0、11.1 V 名义供电 | P04 R1 共享 10.2/11.1/12.4 V；不重复叠加 XL330 电机 DR |
| P04 反馈 | S0/O0，从重置预热开始，故障显式终止 | 不偷用真值作 actor 输入；不把传感器偏置当物理奖励目标 |
| P04 目标 | O0 仿射逆变换、native count 四舍五入，count 20–1001 | 关节物理范围/方向/零点、命令限幅与运行时元数据 |
| 基线 | 官方带噪/延迟/encoder-bias 观测，连续物理目标 | 保留对照身份；当前基线导出标为 simulation-baseline-only，不能伪称可直接给 S0/O0 固件 |
| 接触 | GPU Warp 足底为训练标准 | 从备份移植 CPU 足底算法适配，V2 足底上重新验证 |
| 导出 | normalizer 必须烘焙进 ONNX | 实际观测样本 PyTorch/ONNX 数值对照和元数据/固件检查 |
| 实机证据 | 目前没有本轮实机运行数据 | 仿真合格与硬件合格分开；需要实际标定与运行日志才能证明实机效果 |

## 5. 历史经验采用记录

| 来源 | 本轮处理原则 | 状态 |
|---|---|---|
| `AGENTS.md` 官方工作流 | 先稳定姿态、配置测试、64×5 PPO smoke，再长训 | 已阅读，待执行 |
| `docs/sc0090-training.md` 2026-10-09 HOME 修复 | 重置真实物理状态及编译模型摩擦/阻尼；部分 reset 不扰动其它世界 | 已定位，待移植验证 |
| 同文档 physical-truth 奖励修复 | actor 传感器噪声与物理任务奖励分开；头部/姿态目标使用物理真值 | 已定位；P04 `_servo_joint_pos` 观测替换需修复 |
| `sim/recompute.py` | reset 常量重计算 CUDA graph；派生统计按 world 独立，避免跨环境写竞争 | 已阅读，待移植 |
| `actuator/compute_graph.py` | 电机张量计算 CUDA graph、稳定缓存和状态所有权 | 已阅读；V5 类接口不同，需适配，不能整段套旧类 |
| `physics/cpu_warp_contacts.py` / C++ | CPU 足底对齐 Warp；限定模型和版本 | 已阅读；现实现要求 MuJoCo 3.13.0 / Warp 3.8.1 |
| 2026-09-15–18 工程复盘 | 起步保持探索，学会技能后再加平滑；步态看实速/步频/步幅；相机稳定不以站住或跌倒取巧 | 已阅读，待体现在任务配置与评测 |
| 2026-10 长期头部训练 | 提早覆盖组合命令；单轴/组合/回中分别评测；局部训练误差不代表全范围控制 | 已阅读旧任务状态，待采用 |
| 官方奖励经验 | 负罚符号、避免提前到达 jackpot、反向课程、身体目标按新模型实测 | 已阅读，待逐任务检查 |

## 6. 已完成工作与实测问题

### 2026-10-15 首轮

1. 核实当前无训练进程，两张 L40 可用；NoMachine/显示进程保留。
2. 创建两条独立训练分支和 worktree；旧分支、旧模型和旧暂停 checkpoint 未修改。
3. 找到数据盘，建立新实验目录与工作区链接。
4. 阅读 V2 模型接入、S0/O0/R1 反馈、动作/导出契约、官方任务配置与备份关键修复。
5. 用旧共享 venv 对两分支运行初始测试：**均在收集阶段失败**，并非测试通过。
   - P04：`runs-v2/initial-tests.log`，exec session 3268，退出码 2。
   - 基线：`runs-v2/initial-tests.log`，exec session 66867，退出码 2。
   - 原因：旧 venv BAM `3505fba...` 使用 `vin_drop_resistance_range`；新分支官方代码/锁文件 BAM `62bd8ce...` 使用 `vin_drop_gain_range`。直接复用旧环境不兼容。
6. P04 `uv sync --frozen` 正在建立隔离环境；日志 `runs-v2/dependency-sync.log`，工具 session 10059。续接必须查询真实句柄/进程结果。
7. 发现新分支锁文件 MuJoCo 3.10.0，备份对齐代码要求 3.13.0；将固定统一依赖后重新验证，不混用历史 CPU 报告。
8. 发现 P04 `_servo_joint_pos` 被改成反馈估计，对使用该 helper 的姿态奖励构成真值污染风险；实际 head tracking 单独读取物理 q，须逐项审计，不能概括所有奖励都错。
9. 发现 P04 feedback 路径含 `any/all` 的 CUDA→CPU 同步、每次全批 donor 搜索（最多 1024 候选/舵机）及重置预热循环；需要剖析和 GPU 优化，同时保留原始采样/时序/有效性语义。

## 7. 当前运行记录

| 工作 | 句柄/日志 | 已验证状态 |
|---|---|---|
| P04 初始测试 | 3268 / p04/initial-tests.log | 已退出 2，依赖不匹配 |
| 基线初始测试 | 66867 / baseline/initial-tests.log | 已退出 2，依赖不匹配 |
| P04 隔离依赖同步 | 10059 / p04/dependency-sync.log | 启动后有下载日志；需再次核实 |
| GPU0 正式训练 | 无 | 未启动 |
| GPU1 正式训练 | 无 | 未启动 |

## 8. 下一步与完成证据

下一步：完成可复现的共同依赖；修复 P04 物理奖励读数；移植并验证重置、reset CUDA graph 与 CPU 足底对齐；适配 V2 起身任务并测量 HOME/坐姿；双卡各自 64×5 smoke 和吞吐实测；通过后开始有节点预算的起身对照。随后逐项推进清单中所有 P04 双足策略。

最终交付必须逐项具备：真实训练 checkpoint 与配置/来源、完整技能评测（含失败）、步态/姿态/头部/命令响应指标、CPU/GPU 回放、带归一化 ONNX、部署契约及可运行调用示例、基线比较、预算与训练谱系。硬件结果缺失时明确记录，不以仿真代替硬件。

当前总体目标：**ACTIVE / 未完成**。本轮分类：PROGRESS（新工作区、实际测试失败证据、隔离依赖安装、配置问题定位及本台账）。

## 2026-10-16 11:06 — 双卡camera固定对照正式启动

- 目标：按已核验历史经验补运动相机稳定目标，同时检查速度、步态、头控和恢复退化；不把静止或低折叠姿态当头稳成功。
- 完成：实际N24576×16两臂准入通过，各1已定位native越界/未知0/非有限0；核对当前80余源文件和全部资产SHA；资格`velstand-camera-qualification-001.json`、公共评测协议`velstand-camera-eval-protocol.json`已冻结。旧小批smoke复用限数值实现未变；新增只读审计通过真实及负例夹具验证。
- 正式：原256父模型，mild GPU0 PID93712/control GPU1 PID93799，各64更新，合计75,497,472控制交互上限；不继承准入272。训练与四项末评单进程接续，正常结束不再自动加训。
- 评测：两臂同nominal/measured各336命令条目和128初态survey；survey唯一变更为设备参数，原公式/随机化/种子不变。保存原始200Hz相机姿态及控制数据，可独立复算转向残差。
- 吞吐：准入mild99,730/control98,893控制步/s，不能称突破10万。正常速度/有效完成率优先于盲目增加负载。
- BallKick：最后170累计100,270,080完成且阶段关闭未合格；横速和身体减速有收益，但完整/身体保持/头回中为0，相机没有改善。子代理只做CPU归档。

### 11:11 — 双卡负载与相机独立复算

`velstand-camera-paired-utilization-001.json`保存30个1Hz实测点：GPU0平均86.07%、GPU1平均88.57%，不能将一次PPO采样间隙的低占用解释为未运行。此时两臂已分别完成至少8,257,536正式控制步、异常0。

新增只读`audit-camera-world-residual.py`，对原256的nominal/measured四份保存trace哈希核验后，用float64世界坐标SO(3)有限旋转独立复算；三世界轴×三角速（含倾斜初态/四元数符号翻转）夹具通过，原总RMS最大差<6e−7rad/s。另报减去原requested yaw的残差，并标注主动头控；有限区间旋转不是训练瞬时Jacobian角速度，残差也不自动等于无意抖动。原始总RMS/P95/20ms曝光指标仍保留，健康分母沿用原步行边界。

## 2026-10-16T11:21:44+08:00 — camera两臂完成并关闭，转失败动作诊断

原VelStand256同一父模型，两臂各64更新、N24576/T24，新增各37,748,736、合计75,497,472正式控制交互。对照camera权重0；处理−.01。200Hz每物理子步平方后求均值，减去原请求yaw；没有改PPO、教师、actor输入、动力学或原评测。四份nominal/measured×seed初始qpos/qvel/raw actor/origins均逐位相同。独立世界SO(3)复算原RMS误差<6e−7rad/s；有限区间角速与训练瞬时Jacobian角速的区别已注明。


结论：mild转向/头控相机约−3%至−4%，直行无改善、两条件各新增1跌倒；不作为默认，原父和对照保留。对照改善恢复但neck DC偏差更大，窄带T90仍近0。完整证据及取舍见[相机对照复盘](sc0090-v2-camera-pair-review.md)。

GPU0已启动`probe-roulade-rise-handoff-001.py` PID108022/session29903，只有256×5s诊断无PPO；GPU1子代理执行相位一致GroundPick低位初态probe。未经新证据不扩大已关闭配方预算。

## 2026-10-16T11:30:43+08:00 — 翻滚承接可行性与GroundPick旧端点勘误

- Roulade名义/实测随机化均256/256身体站稳+头回中，故障0，完整翻滚0。初态与原评测exact，切换前轨迹不是exact，原因未归因；详见[承接复盘](sc0090-v2-roulade-handoff-review.md)。这是128,000诊断controls，不是PPO训练，不计旧正式预算，不是部署方案。
- GroundPick低位probe预保持health为0，不能说策略从健康目标离开。子代理已定位唯一未过项：head_contact found=1、持续.663–.781N；嘴高12.39–12.90mm、down>.996、双足64/64、self0、故障0，其他门槛均过。旧物理候选只验证>1N冲击为0，没有保存contact found，故“完整健康端点可行”的先前表述过宽。保持原no-contact门槛，先做必要几何修正与一次复验，尚不新训。
- 总预算重审`training-budget-audit-20261016-camera-complete.json`已生成：按父元数据计增量、验证/诊断排除，不把teacher推理或重复父谱系计PPO。

## 2026-10-16T11:37:34+08:00 — GroundPick复验失败，保留预算；翻滚教师独立源准备

GroundPick 002在CPU全部64几何初态满足嘴高15.91–16.38mm、down≥.999843、头壳净距≥6.863mm后，真实P04预保持0.10s又64/64头接触；尾头载荷.612–.814N，mouth11.79–12.20mm、tilt29.29–29.86°。几何净距不等于无头撑的动态平衡。001/002均不构成“从健康低位离开”的因果证据。旧head impact只罚>1N与评测no-contact存在明确口径缺口，但尚未证实全奖励反奖/任务物理不可达。

因此条件授权的相位一致reset正式42更新**未启动**，GroundPick仍75,497,472正式步。子代理只做保存状态的CPU COM/足底支持多边形/头触点静力核对；不扩大GPU候选扫描，不改变门槛。只有存在具体健康候选与动态证据后才考虑剩余24,772,608正式步，总额仍不超过原100,663,296。

Roulade名义/实测teacher承接512/512站稳头回中，完整滚转仍0；旧阶段100,663,296不改。root已建立真实worktree`roulade-landing-teacher-worktree`，106数值源从冻结endpoint源核对一致；目前只复制源，尚未实现/验证教师训练。`roulade-landing-teacher-proposal-001.json`明确候选新教师学习最多64更新/25,165,824新增，若采用则总125,829,120，这是**新增预算**而非冒用旧余额。要先验证末段路由不向actor泄露phase真值、采样和BC对齐、保护起滚行为、原完整滚转指标、PPO/BC/正常化恢复；现在没有新PPO资格或运行。

本轮已完成camera正式成对训练/原末评/独立复算及两项失败原因诊断，分类PROGRESS；整体目标ACTIVE，多个技能仍未合格，无外部阻塞。

## 2026-10-16T11:46:41+08:00 — Roulade末段教师实现并启动准入

冻结原256作为起滚/未到末段的动作参照，冻结起身1856作为末段教师，使用两者各自原归一化。保留原奖励、reset课程、动力学、测量、actor61/critic74、PPO全部超参数。末段训练门控为采样动作之前的head latch有效、supported frontier≥300°、current accumulated≥260°；后一个条件防止已回转的状态仅靠旧最大角度继续取末段标签。门控不进入actor或critic，实时推理仍只有原61维测量/命令。

BC沿用已验证的双教师模式：独立Adam lr3e−5、5epochs、最多4个均衡minibatch、每批最少512（不足整批时保留一个批）、末段不足64帧时不使用该teacher；教师/原策略anchor系数均1，不做权重搜索。原PPO仍T24/E5/M4、native adaptive LR/.99/.95/.2clip/.01entropy，原Adam/N/LR/clock6144精确恢复，BC Adam首次新建；后续同源resume必须恢复两个Adam及BC计数。

PPO存储按time-major排列；每次act前将native frontier/accum/latch复制到独立GPU缓存，记录同control clock，BC只消费完整24步缓存。原actor输入/现场状态不改，partial reset后的下一帧使用reset后的native上下文。单独保存最后验证段原始context和PPO observations供独立门控/维度核对；老师权重和归一化末尾逐位不变。


准入PID138792/session95340，正式0；完整预算/恢复/保护说明见[末段教师计划](sc0090-v2-roulade-teacher-plan.md)。GPU1唯一静力候选3s真实保持PID138802/session55261，仍只诊断，未放宽任何动态健康门槛。

## 2026-10-16T11:59:25+08:00 — teacher实际N突破10万，固定原PPO对照准入中

教师资格002已通过：成熟64×5/ONNX4.77e−7；实际16384×16/6,291,456验证controls/24,576timeout、反馈非有限0、BCmaxKL.34699、稳态105,057控制步/s。001配置比较时机错误已修正，原记录保留，0正式损失。

两臂预算正式修订为各64/25,165,824，共50,331,648新增，Roulade全历史上限150,994,944。理由和差分见[教师计划](sc0090-v2-roulade-teacher-plan.md)：新增原PPO对照验证自然续训收益，不能把增训共同收益归于教师。GPU0已合格teacher由worker002/session11397开始，GPU1control PID160841/session31755准入后接续。旧GPU0等待进程明确0正式后取消，receipt保留，不是重启训练。

GroundPick003闭环固定目标试验：空载无头接触但髋—足自接触32/32；30g同样self32/32且12/32轻头接触；尾完整low_goal0。CPU复现双侧hip_collision与foot_collision穿深.067–.473mm，不能称false found。本轮41,600诊断controls、0新增PPO，端点探查关闭；75,497,472原正式保留。

## 2026-10-16T12:02:41+08:00 — 双卡Roulade固定对照进入正式更新

Control CPU/64/实际N全部通过：64 ONNX5.96e−7/32timeout；N16384×16=6,291,456验证controls、24,576timeout、PPO KLmax.013815/clip.185558、反馈非有限0，稳态aggregate107,389/median107,699控制步/s。root再次逐SHA核对双资格、源和138资产，`roulade-pair-ready-001.json`绑定两资格及原256父后，control worker自动恢复原256开始64。

当前teacher PID164898/session11397 已290，新增13,369,344；control PID160841/session31755 已280，新增9,437,184，两者保护无触发。最近GPU快照89%/94%；正式阶段完整吞吐等结束据实际timing报告，不用准入速度代替。原评测均已排队，之后不自动追加。

预算审计器schema6增加两个实际进程wrapper识别，按/proc argv+实际allocation status+pipeline phase共同验证，不凭状态文件宣称活跃。`training-budget-audit-20261016-roulade-pair-live.json`保留实时下界；正式总额按每目录新增计数，验证/物理诊断不并入。


## 2026-10-16T12:31:52+08:00 — 翻滚固定对照关闭，坐站时序修复与头稳诊断接续

- 目标：保持两卡推进、参考历史运动头稳经验，以动作健康和真实指标决定下一有限训练，避免没有证据地继续堆 PPO。
- 完成：Roulade teacher/control各64，共50,331,648新增，全部累计150,994,944；完整翻滚两条件均0。teacher站立初态末段身体63/128、65/128，但未完成330°且头HOME0，不记技能成功。双方初态精确一致，稳态105,095/108,235控制/s，故障/非有限0。`roulade-teacher-control-0320-review.json`与`roulade-retained-candidates-0320.json`保留原父及两320。
- 修复：SitStand评测旧命令缓存使切换第一动作迟20ms。新调度保持原生观测/历史和当前指令一致，两条件600步断言及五项oracle通过。原384健康保持、头响应仍全0；旧报告保留，不能说时序修复已解决训练。重测307,200诊断控制、正式0，证据`sitstand-clock-reeval-review-001.json`。
- 测量：坐姿头标签72世界×4s=14,400诊断控制；端点身体/双足/无头地/无自碰72/72，但小头指令存在约2.7–2.9°静差，只资格端点安全。外部教师probe001中断12,800+002完成38,400诊断控制：坐→站教师16/16先翻倒超过90°；固定SIT站→坐16/16失败，不能全程做健康教师标签。
- 正在接续：GPU0直接头目标物理对照上限268,800诊断控制；GPU1正确ClearSeat缓慢过渡上限64,000。均不更新模型、不自动开新PPO；原课程/奖励/物理冻结。GPU0探查001在环境创建前发现tuple/list断言错误，0仿真停止，修正002再启动，失败记录保留。
- 计账：baseline根39段2,629,828,608；P04随机根含legacy50段4,096,327,680；未解析0。完成后审计`training-budget-audit-20261016-roulade-pair-complete.json`。当前阶段文件之前同时残留BallKick/camera旧“进行中”表格，已按实际终态整体刷新，旧版本存`current-stage-before-roulade-closeout-001.md`，不改历史台账。


## 2026-10-16T12:43:42+08:00 — 两项有新证据的有限修正进入资格准备

- GPU0运动头稳：直接头目标诊断完成268,800控制；nominal/measured生存330/336、329/336，反馈0，+.30相机RMS6.774→2.137、6.689→1.795rad/s，但转向/生存受损，拒绝推理覆盖。实际N24576×5的原生PPO/BC方向审计完成2,949,120验证，五次walking-head方向余弦全负，反馈/非有限0；不能将局部方向直接解释为相机专属因果。
- 新唯一候选：同原256，camera−.01，walking BC仅去掉4头维；腿梯度分母14、站立全14、两Adam精确恢复保持。新正式预算37,748,736/64更新，VelStand全历史上限264,241,152，资格前正式0，既有mild320作为同父原配方对照。CPU→64×5→实际24576×16通过后自动正式及原两条件末评；不继承验证模型。计划见[head-free BC](sc0090-v2-head-free-bc-plan.md)，worker`run-velstand-head-free-bc-001.py` PID220635已启动。
- GPU1正确ClearSeat缓慢往返64,000诊断完成，2s/4s均最终不合格、头着地64/64、self/反馈/非有限0。CPU发现SIT可有官方允许的shin支撑，因此不能把足域COM外移直接判为完整支撑失败。奖励审计发现pose Gaussian、pose L1和composite pose因子仍全程追踪线性alpha腿模板；endpointstillness已正确gate，不误指它。
- 已给子代理唯一最小修正预算：只在过渡期释放这3项腿模板，保留高度slew/upright/温和升降/head/终态。原384父，单64×16384×24=25,165,824新增正式上限；SitStand全部上限176,160,768。独立源与必要CPU/64×5/实际N16资格通过后自动正式及clock-fixed末评。不自动第二段；与旧384对比不能排除自然续训贡献。


## 2026-10-16T13:06:28+08:00 — 运动头稳取得有取舍的新候选，坐站坏初始化另作一次从零验证

- 头稳单64正式已完成37,748,736，VelStand累计264,241,152；原mild同父对照下相机RMS在平移/转向/主动头行走/停步下降约32%–38%，生存335/336和336/336，反馈0。颈部误差由约5–6°变为11–13°，步行池整体速度误差无普遍改善、回零仍有足步；保留候选而非全头控合格。详见[复盘](sc0090-v2-head-free-bc-review.md)。正式稳态98,606控制/s、反馈/非有限0。GPU0 PID241104接28s长保持诊断，上限1,254,400控制，不再PPO。
- 坐站过渡释放正式64已结束，新25,165,824/累计176,160,768，双条件所有完整身体/头保持仍0，初态与原384精确一致。稳态96,428控制/s、反馈/非有限0；保留384/448，原阶段关闭未合格。吞吐100k硬断言曾单独停止已正确完成的actual_N；续接入口复用原准入、从384正式，不重复测试/不继承验证权重，低吞吐照实计账。
- 新证据：SitStand最初128用了后来修正的错误坐姿，之后256/384/448全部继承它，没有正确ClearSeat配方从随机初始化的谱系。授权唯一fresh固定配方验证：数值源沿用release不变、全新actor/critic/N/Adam/clock0；新128×16384×24=50,331,648正式封顶，SitStand全历史上限226,492,416。不是把旧成本归零，不扫种子/权重、不自动第二段；与旧448对比包含初始化/clock差异。计划`sitstand-fresh-release-plan-001.json`已冻结，GPU1 PID248228单进程CPU→fresh64×5→actual16→fresh128→原双条件评测已启动；验证模型不继承。
- 不变量继续：无轮滑，无P4/host部署，有效尖峰不去除，实际测量与200Hz相机记录保留。头稳阶段的几次小测试错误/immutable plan状态误写已保留痕迹并恢复资格绑定的原SHA，未改变正式数值源、模型或优化。


## 2026-10-16T13:13:02+08:00 — 长运动已完成，双卡继续验证/训练

- 头稳长保持1254400诊断结束：nominal旧mild/新候选均224/224，measured旧224/224、新223/224，反馈0。相机收益在10s运动保持仍存在，颈部偏置及一次新增转向失败保留，不能宣称全面优胜。GPU0已接首条固定轨迹捕获和运动学视频检查，PID256706，16800诊断上限、正式0。
- GPU1 fresh正确坐站CPU/64×5/实际16全部通过；实际median100961、aggregate99211控制/s，反馈/非有限0。PID248228正式至少50/128、19660800新增交互，clock0/全新N/Adam已记录，未继承验证权重。继续128封顶及原双条件评测。
- 关闭阶段预算快照`training-budget-audit-20261016-head-bc-complete.json`：baseline39段2629828608、P04随机根52段4159242240，未解析0；fresh正在进行，未提前计满。审计器已识别fresh worker真实argv/PID/phase，下一快照将按已提交数计入。


## 2026-10-16T13:50:30+08:00 — 纠正短阶段效果判断，沿历史成功配方持续学习

- 用户新要求已纳入：5/16只检查数值；64作为中期观察，不以早期未改善否定成功历史配方。实际重读官方VelStand v8、SitStand两段谱系及历史相机/头指令/连续遥控报告，形成[历史与学习预算](sc0090-v2-history-and-learning-budget.md)。
- GPU0新头偏差配方在独立worktree仅3个数值文件改变：恢复门控的当前命令EMA、权重1、精确源迁移。CPU独立历史oracle1.25e−16、ONNX9.54e−7、实际N16约99,174控制/s、12,288timeout通过，原PPO/两Adam/N/命令/动作不变。PID298058首64正式进行；总观察预算按新计划002明确为256/150,994,944，PID305339预加载同源接续三段。前64不作最终疗效结论。
- 坐站fresh128结束，新50,331,648、历史累计226,492,416；当前body/head末保持仍0。M16与M4各16相同交互和初始状态，Adam320→1280，learning时间增加、吞吐median100,961→92,078。四项冻结评测614,400诊断，初态精确一致；只作为初始曲线快照，不据此否定M16。专职继续核对成功历史奖励/课程和足量预算。
- 头稳28s诊断及视频采集已完成；只实际检查两模型t2s/t8s四张关键帧，未声称完整观看。可见头姿差异和路径偏移，数值跺脚/偏颈问题仍保留。
- 球踢两次诊断各128,000、正式0：先踢2s后站立教师可使身体/头末保持111/104，但完整动作0；原170所有256/256早期self contact。CPU同几何复现16见证为电池—左胫穿透约1–2.3mm，全部512初态无自碰，故不是初态穿模。支持腿/头改用站立教师后clean22/14、有效触球159/158，完整仍0；仅诊断组合，不当单策略成功。原代码bad_contact已经清除valid，不误报“坏接触后继续奖励”；真正不一致是self没有进入valid历史，而评测要求全程无self。尚未改变球踢奖励或重开正式。
- 预算快照schema8：baseline39段2,629,828,608；P04随机根53段4,209,573,888；未解析0。此值包含坐站fresh，不含随后头偏差新增；验证与诊断未混入。原源、失败记录和检查点均保留。


### 2026-10-16T13:54:36+08:00 — 坐站恢复历史基础配方，按完整学习量执行

核对c832147成功1250的实质差异：历史12s episode、原全程slewed腿Gaussian/L1/composite、UniformPose头采样；现Lesson4s只有4s、释放过渡3项腿模板、BalancedHead零命令.5，还提前带入77ffadf才加入的neck_strain−.5。短128不足以否定历史配方，停止继续改造奖励，转为正确V2几何下的历史基础配方。

已明确单条从零320预算：N16384/T24/E5/M16，125,829,120新增controls、25,600 PPO Adam，接近历史1250的122,880,000/25,000；rollout刷新频率仍不同，不称完全等价。step课程边界÷4，2s slew、3.5–6.5s姿态命令、2–5s头命令、12s episode秒数不变，历史1500后坐姿硬化不提前加。SitStand含过去全部成本上限352,321,536。CPU差分/课程检查→64×5→actual16通过才正式，连续320/每64保存，不因首64无收益就换方。子代理负责实施与精确源/预算记录。


### 2026-10-16T13:56:24+08:00 — 头稳首64作为中期观察，固定配方已继续

首64新增37,748,736、VelStand关闭累计301,989,888；稳态99,836控制/s、反馈/非有限0。两条件生存336/336与335/336；相机平移RMS约3.286/3.331→2.905/3.030rad/s，颈部误差仍11–14°。这是早期快照，既不判配方失效也不判全面成功，不改变奖励；PID305339按256总计划接续第二64，源、N/两个Adam/LR/课程恢复保持。证据`velstand-head-bias-interim64-001.json`。

旧320的28s长运动健康子窗口复核完成，`velstand-head-bc-longhold-health-001.json`保留全样本/健康分母。唯一measured失败实际发生在负转向命令结束后的停车阶段12.90s，tilt45.82°、height.1213m、反馈和数值正常，不能写成反馈故障或直接断言转向中跌倒。


### 2026-10-16T14:01:20+08:00 — 历史函数语义已核实，修复外置统计回调后接续

坐站历史CPU配置契约已通过；18项启用函数/命令/课程AST中17项去docstring后与c832147完全相同。head_pose_tracking仅新增可选fine_std/backlash视图，本任务std=.5、fine_std未启用、模型无passive_backlash，沿本轮已有独立奖励oracle验证默认路径。5阶段×4头轴绝对目标均在V2实际model limits内，HOME0；证据`sitstand-history-mdp-semantic-audit.json`、`sitstand-history-model-limits-cpu.json`。

worker001首64环境已完成1次更新（1536验证controls、80Adam、0正式）后，外置counted_log未保留原签名，原诊断器inspect.signature绑定拿不到it。只给外置回调加functools.wraps，不改生产PPO/诊断保护。失败与计数保存，worker002/PID319428/session27653已启动，复用源SHA一致的CPU契约，必要smoke002→actual002→连续320；资格所得权重不继承。当前两条训练流水线都已启动，GPU导入/准入与正式负载分别记录。


## 2026-10-16T14:22:19+08:00 — 固定学习量继续，直行偏航按历史证据复查

- 前一目标轮为实质进展：头偏差首64及评测完成、后三段真实接续；坐站历史基础配置/语义/限位核验并启动连续320。当前两条进程已由/proc与allocation记录核实，目标仍ACTIVE。
- 坐站actual002已16更新/6,291,456验证/1,280Adam/8192timeout、5oracle全0、feedback/nonfinite0；正式已实际推进。原生12s reset-only课程与严格比较保留，所以阈值÷4不是每次生效墙钟严格÷4；实测head和action_rate/DR按已记录reset边界升级。末评源明确清空curriculum且只加载actor，不恢复训练clock，五节点使用同DR/初态协议，结束时逐位核对初态。
- 用户新增强调直行问题。已读取历史heading-feedback、heading-sampling、heading-training及遥控视觉复核；真实航向与50Hz机身wz积分不得混用，横偏必须在逐世界的目标参考系计算。现有GPU轨迹离线复算`v2-straight-paths-audit-001.json`：+0.30命令两秒move横偏约0.10–0.17m，旧长10s样本约0.96–1.39m；不能仅以平均body速度称走直。这里只测初始heading参考路径，不把缺失的实际躯干姿态补成真航向。
- 找到已有Walking512 nominal短程标定176,000controls，439/440生存、反馈0；保留并复用，不重复。当前VelStand walking anchor_yaw_remap=None，官方v8的实测教师逆映射尚未在V2验成。新唯一补充标定用同冻结teacher、完整接触VelStand场景、1s idle+11s保持、nominal/measured及真实qpos角度，共上限940,800诊断控制/正式0。队列`run-velstand-teacher-yaw-001.py`已预加载，待头偏差256结束接GPU0；未改学生请求、奖励或运行端，没有直接抄XL330表。
- Roulade已创建隔离历史课程准备源，保留原奖励/已验V2几何，仅准备足量PPO与课程时钟映射；尚未运行或批准新的正式交互，不占当前训练GPU。


## 2026-10-16T15:00:39+08:00 — 双卡正式训练、完整曲线收尾及GPU实测

- GPU0历史Roulade完整课程已实际450/1536，新增176,947,200控制；CPU/64×5/actual16资格通过，连续训练不因中期表现换方。预算603,979,776新增，全部翻滚上限754,974,720；课程step阈值÷4、M16对齐历史minibatch，物理秒数/奖励/初态不变。依据完整源课程安排学习量，不声称已知官方ONNX原训练长度。
- GPU1历史SitStand320已结束，SIT五节点nominal 0/273/30/82/324、measured 0/233/25/83/330（每节点384）；STAND全0，原初态/评测配置精确一致。继续320→640，仅冻结真实live320难度，预算125,829,120；资格通过，已实际450，不继承验证336权重。
- 头偏差固定256结束，新增150,994,944、VelStand累计415,236,096，5120 PPO Adam已核验。保留320/384/448/512/576与SHA；相机/回零有收益、运动颈部偏差未解，不自动加训。训练第三64有1次非法count保护，其余0；所有末评反馈0。详见[固定256复盘](sc0090-v2-head-bias-fixed256-review.md)。
- 完整接触教师yaw标定940,800与固定表闭环537,600诊断完成，正式0。改善稳态偏航未普遍降低单轨迹横偏；同初态前缀有微小数值差放大，不能称逐轨迹因果严格配对。旧176,000标定保留。[直行复盘](sc0090-v2-straight-walking-review.md)。
- 用户反馈GPU0低：42秒连续采样平均81.4%、中位88%、2次低于50%；近50轮吞吐中位106,818控制/s，真实正式进度在增长。没有为占卡而制造任务，也未中断当前训练改PPO配方。原始采样已归档。
- 总审计schema9已纳入两个真实wrapper：baseline根2,629,828,608；P04随机根含legacy4,679,073,792（动态已提交下界），未解析0。当前合同已整体刷新，旧文件归档`current-stage-before-history-closeout-1500.md`。


## 2026-10-16T15:14:13+08:00 — 继续正式预算，准备两项有证据的后续

- 上一目标轮为实质进展：完成GPU42秒采样、头偏差256曲线/保留manifest、直行闭环复盘与schema9总预算审计；目标继续ACTIVE。当前Roulade/SitStand均由真实/proc进程和新增updates核验，保持完整预算。
- BallKick已明确下一有限从零配方：只将真实200Hz self found加入坏接触永久锁存，与原全过程健康评测一致。原finish方向−4/quiet−2、其他奖励/碰撞/61actor/82critic/14动作保持；N24576/T24/E5/M24×384=226492416新增正式上限，全部Ball成本上限326762496。课程step÷6，5s物理回合不缩，覆盖原最大1500×24阶段并留巩固时间；目前仅独立CPU准备，正式0，资格通过后才连续训练。未找到官方Ball成功run原日志，预算依据源码课程，不冒称历史逐位复现。
- GroundPick004直接平衡补偿CPU限位失败并停止排队，0新GPU/正式；生产会夹住的踝目标不送去仿真。单次005约束求解把实际执行限位与局部稳态偏差纳入，29迭代取得嘴23mm、髋足1.5mm、命令/物理q均有余量的静态候选。仅安排64×3s真实固定目标验证9600诊断，等SitStand640及末评全部结束接卡，然后给Ball。详见[可执行端点](sc0090-v2-groundpick-executable-endpoint.md)。
- 直行原零命令前缀误差不是命令提前曝光：已核对前50控制两臂命令均0、actor前向默认确定性。局部Warp源存在约束/接触浮点atomic累加，但尚未定位这次轨迹差异的根因，不能将候选原因写成已证实修复。原数据与限制保留。


## 2026-10-16T15:25:56+08:00 — 坐站固定预算完整结束，GPU1转入健康踢球准入

- SitStand Hold320到640完整结束，新增125829120、全历史478150656；累计Adam51200、actor/critic normalizer251658240，时钟未重启。576/640两条件SIT末保持均384/384，STAND0/384、严格head0/672，故保留坐稳进展而不称双向技能成功。站命令高度约92–106mm但倾角15–20°与头偏差仍超门槛。四节点初态精确一致，评测cfg按已验资产路径归一后相同。正式1431.033s=87928.88控制/s，稳态88536；fault/nonfinite0、196608真timeout。无自动第三段。
- GroundPick005完整9600诊断exit0，正式0。空载self32/32、head0/32；30g self32/32且均曾head，low_goal尾1s两组0。CPU保存末态确认双髋足接触2–325µm；局部常量稳态补偿未提供健康端点，本轮停止候选，不降低原门槛。
- Ball新健康配方CPU256组合/25子步脉冲/partial-reset/cfg/283资产检查PASS。驻留worker PID397198/session30107实际启动，当前RUNNING/actual24576；真实GPU子步/terminal、64×5、actualN仍是硬准入，formal0，不继承验证模型。末评额外区分raw正确足触球与历史健康valid，避免判据修复改变计数后误称触球能力下降。
- GPU0 Roulade已实际830/1536、326,369,280新增controls，继续完整课程，未触guard。schema10快照baseline根2629828608、P04随机根含legacy4930732032，未解析0；Ball验证不混计正式。


### 2026-10-16T15:30:00+08:00 — 踢球真实准入完成并正式从零启动

真实32世界200Hz self短脉冲/历史valid撤销、partial reset及fast/full terminal通过，原6终止场景保留；64×5有32真timeout、ONNX2.38e−7。actual24576×16通过，稳态median77014/aggregate76991控制/s，低于100k目标如实保留；近期每轮采样6.53s/学习1.07s，不能未经profiling把瓶颈归为Adam。资格`ballkick-health-history-qualification-001.json`，验证权重未继承。正式PID397198已提交10/384、5,898,240controls；GPU0 Roulade已890/1536。两项完整预算持续，无用户授权范围外部署。


## 2026-10-16T16:05:26+08:00 — 低位拾取从几何候选推进到真实健康端点

- 前一目标轮完成SitStand640及全曲线、005探针和Ball从零正式启动，为实质进展。本轮两正式进程持续，由/proc与真实updates核验，不因等待观察重启。
- 006用实际CPU Warp足地算法、原CAD/Jacobian和冻结v5静态电机方程求力矩；5162组生产计算与独立公式最大4.58e−16Nm。005所分析支撑下膝需.16–.18Nm、旧目标仅.003Nm，踝命令余量不足；不以几何COM通过冒充力矩平衡。
- 007/008两次固定候选动态诊断共28800控制，PPO0；005本轮前9600另保留。007无head/self但空载嘴偏高；008将静态高度留余量后nominal64/64、measured61/64原low_goal持续0.5–3s通过。剩3条measured仅嘴高超25mm，原最大25.895mm；全部128在原50Hz边界无head/self/fault/nonfinite，camera独立200Hz。接触短子步未记录，不称全200Hz接触证明。
- 007/008小批与GPU0 Roulade并行，完整formal wall包含竞争，GPU1 Ball未打断；不称此区间独占吞吐。没有CAD/限位/摩擦/执行器增益改动，固定目标亦不当策略。
- 停止端点搜索，下一独立Ground完整剂量fresh512×24576×24=301989888、全部377487360，M24/step÷6、原4s周期/20s回合。仅准备25%相位一致健康低位reset与接触时两个approach正部门控，orientation负部保留；其余历史配方保持。CPU/新oracle/partial reset/terminal/64×5/actualN资格通过后才能正式，当前新增正式0。见[端点及完整学习](sc0090-v2-groundpick-torque-balanced-endpoint.md)。


## 2026-10-16T16:24:44+08:00 — Roulade与Ball完整预算收尾，GPU0正式接续Ground

- 本轮为实质进展，整体目标继续ACTIVE。Roulade1536、14份原评与CPU配对/分项审查完成；新增603,979,776，全部754,974,720，实际122,880 Adam与normalizer计数一致。正式共享phase100,119控制/s，稳态100,278；不含imports/资格/末评。完整动作仍0，不把吞吐当效果。实际late折叠奖励盆地和头目标缺项见[复盘](sc0090-v2-roulade-history-full-review.md)，原源未改。
- Ball384及8份评测完成，新增226,492,416、全历史326,762,496。256/384两条件raw/healthyvalid均256/256；完整动作51/44→44/33，保留256及全部候选。短期64/128全0不应被误判配方无效。后续先核课程/站稳/头偏差再定有限方案，未原样无界加训。
- Ground新健康reset/reward资格全部通过，CPU283资产/实际源、真实GPU12successor及partialreset、64×5 ONNX2.38e−7、actual24576×16。资格aggregate86,237控制/s低于100k，反馈/非有限0。fresh512已实际50更新，验证权重未继承；新增上限301,989,888。
- 用户指出GPU0低利用率：切换窗口均45.3%真实保留；正式45秒平均79.3%、中位86%、37–94%，不是只引用94%高点。此时GPU1评测/释放，不能称双卡均正式训练。
- schema11增加Ground真实wrapper识别，快照baseline根2,629,828,608、P04随机根5,454,495,744、唯一active正式Ground20。审计/验证/正式量分开。当前合同已更新，旧版归档current-stage-before-roulade-ball-closeout-1623.md。


### 2026-10-16T16:29:39+08:00 — 用完整曲线决定踢球精度阶段，纠正课程归因

实际TensorBoard与live记录证实Ball256/320/384 action−1、COM.015/headCOM.01/push.3均相同，原14奖权一致；冻结256不构成降低难度，明确撤回该解释。256→384完整成功丢失nominal26/measured28，其中球速方向gate23/23，主要横向越界17/15、超速6/9，身体失败5/5有重叠；不能泛称反馈或自碰故障。头中性也下降，保留各分项。

GPU1 PID467044补评192/320四场，诊断上限256,000/正式0，以确认中间候选。下一唯一拟正式：成熟候选上替换健康valid后的正球奖为按原speed/lateral容差构造的联合精度高斯，原其它项保持；额外256更新/150,994,944，Ball全部预算上限477,757,440。数值/成熟恢复/terminal/ONNX/actualN准入前不正式，不从验证权重续练。该新奖励未证明有效，按完整固定曲线判断，不声称课程修复。机器决定 `ballkick-precision-stage-root-decision-001.json`。

SitStand640 CPU实际actor复算发现站命令主动把neck/headroll打向限位；不是缺站指令，腿对posture flag仍响应。子代理保留具体诊断，未直接在全域使用已有反证的起身教师标签，也未占用正在推进的Ground。


## 2026-10-16T16:55:52+08:00 — 两卡正式接续，坐站头撑诊断有实证

上一goal轮为实质进展：完整翻滚/踢球收尾、低位健康训练正式启动、精度方案由全曲线确定。当前Ground PID444216与Ball精度PID475752真实持续更新；精度新四文件patch经root静态复核和独立CPU/GPU资格，通过后从原256恢复，额外256固定预算不变。actualN16 aggregate88,036，未达100k；训练不因瞬时利用率或早期reward改方。

SitStand冻结640两臂两条件153,600诊断结束：原HOME站0/16、颈115°/身体20°，头壳2–3秒持续接地力均2.419/2.473N；四头目标回指令后仍站0/16，身体78°/高度45mm，均曾倾倒。只修头输出方案被否定，应全身联合改学；不把大头误差本身当唯一因果。初态exact但未干预SIT前缀数值差放大，原始限制保留。图表/200Hzcontact/camera与健康分母见[诊断复盘](sc0090-v2-sitstand-head-support-review.md)。

唯一下一局部承接诊断51,200控制由PID494468执行：640先运行2秒，仅实际近HOME站姿切冻结1856全身输出，原反馈/history/上一动作不重置，头HOME，不推广已知会先倒的低SIT过渡。第一次CPU因archive路径不同停止0GPU；6文件SHA一致后只归一路径，失败源/log保留，未改生产参数。

schema12总审计快照baseline根2,629,828,608、P04随机根5,598,412,800，两正式进程核验成功；原生更新减真实父256，只计新增精度20,054,016，Ground135,659,520。正式与诊断/资格分开。自动CPUcloseout队列PID490305严格待两worker各自真实终态且退出后复算。


### 2026-10-16T17:11:56+08:00 — 近站全身教师32/32实证与下一固定坐站预算

局部4s×64×4共51,200诊断完整完成，0PPO。初始nearSTAND头撑32/32由1856恢复身体+接触+头HOME，posthandoff最大倾角24.84/26.87°，最低高97.46/96.85mm，末头力0；严格全轴1°仅4/16、6/16，不扩大精控结论。初态exact但未干预prefix不逐位相同，原差值和健康相机分母保留。此前153,600只替头对照证明那条捷径不成立，当前两次诊断合计204,800。

准备唯一Sit近端teacher训练320更新/125,829,120，全部Sit预算上限603,979,776。保留原奖励/物理/PPO/冻结难度，训练期STAND+ramp_done+实际近HOME域用冻结1856全身标签，SIT用冻结640保持anchor，其他STAND过渡不强制教师。独立BC沿历史lr3e−5/E5/最多4均衡块/每有效帧coef1，非分支50/50重采样；准入前新正式0。子代理独立源准备，GPU1等Ball全末评退出；目标仍单61D策略，非运行端切换。详见[头撑与教师复盘](sc0090-v2-sitstand-head-support-review.md)。

外置探针发生配置浅拷贝构造错误：首control12800保留，第二case0steps；仅deepcopy修复补剩38400，未重跑成功case。两个正式worker未受影响。


## 2026-10-16T17:44:11+08:00 — 两段完整收尾、双卡接续与真实空档记录

Ground512、Ball精度256均结束并CPU复核。Ground累计377,487,360正式，low全0但空载低嘴已从91–93mm改善到43mm、回站进步；Ball累计477,757,440，精度中期有收益后退，旧健康256严格头中性48/41仍保留。正式平均分别87,358和80,518控制/s，未达100k；原模型/全曲线与失败保留。schema12关闭审计P04根5,895,684,096，baseline根2,629,828,608，无未解析。

GPU0四动态端点诊断153,600控制/0PPO已结束：常量008在原低阶段接管未达到原100ms低位门槛，回站略降，因此不把静态健康端点当过渡教师。CPU独立复算原low/stand/action路由通过，初态exact但前缀数值不逐位相同。原clean_cycle实现要求整个周期双足接地，文字仅称无头/自碰/故障；初始无足接触必使第一周期clean0，后续也有短离地。保留原计数，额外列明200Hz坏接触与文字所述clean，低位全0不由该口径差异造成。

当前用户反馈低利用率属实：现场GPU0 1%、GPU1 0%处于旧任务结束空档。已按具体证据接新有限方案，未伪装成仍在正式训练。Ground唯一新配方将成熟512嘴高reward从std.10/target0改到std.025/target.020，当前43mm原奖约83%、新约43%，保留其它全部配方与原评门槛。N24576/T24/E5/M24固定256，新增150,994,944上限；准入通过前正式0。首worker导入模块名写错，0GPU/PPO退出并保留；外置修正worker002/PID556944，源冻结。

Sit近端教师3e−5/1e−5资格触BC KL.5护栏，均为数值资格失败；只降BC LR到3e−6、原PPO自适应LR不改。64×5和actual16384×16已PASS，actual最大meanKL.0622、5oracle全0、8192timeout、f0；formal从原640精确恢复，PID548509已实际650并增长。固定320学习量不因早期reward换方，未继承资格模型；head-only与局部fullteacher的204,800诊断另计。


### 2026-10-16T17:50:34+08:00 — 双卡正式运行已确认，GPU0突破10万的口径明确

GPU0 Ground精度actual24576×16完整PASS：9,437,184验证/12,288timeout，7oracle最大5.96e−8，feedback/nonfinite0，contact194403/1572864；64×5 ONNX4.77e−7。原512 actor/critic/N/Adam/LR/clock精确恢复，正式PID556944现提交540/768、16,515,072新增。子代理独立CPU审查无阻断项。GPU1 Sit PID548509现提交750/960、43,253,760新增，原BC护栏正常，按固定320继续。

45秒连续双卡正式采样：GPU0平均80.24/median90/min40/max93，9个样本低于50；GPU1平均81.11/median89/min43/max94，5个低于50，完整低点保留。最后20轮采样+学习aggregate GPU0 108,351.20、GPU1 85,557.92控制/s；这是更新稳态而非导入/构建/保存/末评均计的全worker率。GPU0已达到10万此口径目标，未宣称所有任务达标或每秒持续满载。

schema13真实worker识别通过：快照baseline2,629,828,608、P04根5,935,792,128，两正式活动且未解析0。新Ground末评seed2026101633与旧2607不同，已队列原512同新seed双条件参考307,200诊断、0PPO，仅待主worker全部完成并退出后接GPU0；随后CPU全曲线复核。原门槛与初态配对要求保持，不跨seed比较归因。阶段详情见[精度段](sc0090-v2-groundpick-target-precision.md)。


### 2026-10-16T18:04:13+08:00 — 正式预算持续，排好翻滚局部诊断并排除直行坐标猜测

上一goal轮为实质进展：Ground成熟精度新阶段准入/正式启动、Sit小BC步幅准入后正式、双卡实测80%均利用率与Ground10.8万更新吞吐、预算schema13核验。当前原PID556944/548509实际仍在跑，没有因观察超时重启。新增Roulade1536局部落稳诊断CPU契约和独立静态review通过，64,000诊断/0PPO排GPU1，严格等Sit全阶段与六末评成功退出；原fullroll门槛、midroll预信用和200Hz新筛查分开。

直行已有轨迹CPU审计0GPU：+.30健康轨迹实际偏航约.255rad/s，body-wz与真航向差仅.004–.006，姿态转换项约.0034，不能把主要问题归为统计坐标。纯walking512真实末日志head偏差权重0、standing.05，尚未推进600–2000的后续历史课程；VelStand已有完整头偏差阶段仍保留，不将不同阶段混淆。下一步行方案需有父难度连续性、有限预算与充分课程曝光，尚未启动新PPO。


### 2026-10-16T18:08:01+08:00 — Sit正式预算完整关闭，保留轨迹修复评测类型错误

Sit640→960的320正式更新完整完成：125,829,120新控制/25,600 PPO Adam/6,400 BC，clock23040，反馈0；完整正式1517.177s=82,936.36控制/s，稳态83,498.10。总Sit603,979,776，禁止重训已完成段。

首768 nominal评测已600步仿真并保存traces，外置新增健康统计用float stable直接位与bool，导致报告阶段bitwise_and_cuda报错。原子代理获授权只对统计转bool、CPU恢复已存首报告、补其余5条件；不继承失败资格模型、不修改生产源码/hash、不重跑完整首轨迹。原worker548509已退出，Root Rou前置等待正确以失败停止，0诊断；旧等待/source/status全部保留，将按恢复末评新marker和新PID退出重新排一次队。GPU0 Ground原正式继续。


### 2026-10-16T18:14:18+08:00 — 用户纠正路线选择，恢复基线预训练比较优先级

用户明确指出：P04可训练/吞吐达标不足以证明全技能学习效率，基线预训练不应提前搁置。采纳并撤回“所有技能默认P04从零”的推广结论。下一候选为B0动作基线→B1有效随机化/量化适应→B2完整P04，按同一指标的全路线耗时比较，保留所有P04已有模型/失败/成本。除起身外的基线预训练现在用于最终P04路线，最新用户指示优先于此前仅基线起身范围。

仅取消尚未运行的Rou等待worker583739及CPUwatcher，两者状态已验证等待且0诊断/0PPO；保留源码/独立审查/CPU域证据，没有停止Ground正式或Sit末评。原子代理继续Sit恢复末评，并梳理历史基线DR→P04起身迁移入口；根代理准备非起身代表技能的公平比较和基线定义。具体合同见[路线比较](sc0090-v2-training-route-comparison.md)。
