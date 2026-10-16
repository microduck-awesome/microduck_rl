当前：HealthHistory512完整学习与原12s双条件四节点评测结束，Ground正式全历史377,487,360，预算关闭；完整能力/头稳结论见[固定阶段最终曲线](sc0090-v2-groundpick-health-history.md)。

## 历史阶段

# GroundPick 当前阶段已关闭：未合格

2026-10-16 后续有限端点探查也已关闭：001/002头撑，003髋足自接触；三批41,600诊断步，新增正式0，余42更新未执行。GPU1已释放。最终证据 `groundpick-endpoint-probe-closeout.json`；原CAD/训练源/128检查点未改。

2026-10-16：父代理确认关闭，实际正式预算 **75,497,472 controls / 128更新 / N24576 / T24**；未执行剩余42，不追加配方。两条件3×256重复周期的低位/完整拾取全0，保留候选，不宣称实机或技能成功。完整台账与历史数据继续如下。

# V2 P04 GroundPick 专职台账

## 历史状态（首段正式启动时记录）

GPU0独立GroundPick，源`groundpick-repaired-worktree`，任务`Mjlab-GroundPick-Flat-MicroDuck-V2-Repaired-S0-O0-R1`。fresh actor/critic/N/Adam，未继承旧技能或验证权重。CPU/物理候选/phase+payload terminal10案例、fresh64×5/32真timeout/ONNX、两条件完整evaluator软件烟测通过；指标selector错误已修复并保留0更新失败。

N16384稳态median84,381.53/aggregate83,047.16controls/s；唯一追加N24576通过大批正确性和12,288真实forced-timeouts，median97,831.51/aggregate96,290.69。选择实测更快的24576，明确未达100k目标：`groundpick-qualification-002.json`中throughput_target_met=false/性能例外有父级授权，全部正确性护栏保留，不再性能改造。

自动正式queue82066/PID3972041已启动，输出`runs-v2/groundpick-allocation-001`；首128=75,497,472controls，episode年龄原生0、不传force-timeouts。当前启动时仅初始化，实际控制数以status.json为准。之后自动两条件256env×12s完整三cycle、四载荷×有/无push以及200Hz相机。阶段总上限100,663,296不变，首段后最多42额外更新24,772,608（总100,270,080），最多2配方、不自动续训。

低位/站稳单列头稳，下降/回正视为主动运动；头接触found与超1N冲击分开。动作和头稳都必须有健康姿态数据，不以倒地静止取巧。SitStand150,994,944及walking402,653,184阶段由各自台账关闭；不部署/轮滑。

## 目标、来源和有限预算

- 实际历史源码是 `tasks/microduck_ground_pick_env_cfg.py` 的task-space嘴尖高度/朝向、分段phase、腿/头回HOME和10–40g载荷；7月24日固定DOWN姿态设计稿不是当前实现，不能混用。原4s phase分为下降0–1.5s、低位1.5–1.7s、回正1.7–3.2s、站稳3.2–4.0s；reset phase历史为随机，独立评测须补phase0从站姿开始的完整动作。
- 61D actor/14 policy动作/HOME0，head/body输入槽保持零padding（头由该任务主动控制）；使用完整修复S0/O0、无去尖峰、fused/compiled/graph、冻结v5/CAD和mouth独立通道。物理mouth/feet/head指标只用于奖励/critic/评测，不替代actor测量。
- 发现阶段拟最多256正式更新、最多2行为配方；首128后独立nominal/measured评估，不自动第二段；实际N≤32768/T24准入后锁定精确controls，阶段硬上限201,326,592。若无明确行为收益或可定位的必要修复，不重命名追加。
- 头稳按阶段：下降/回正是主动运动，低位/站稳分别报告实际200Hz相机角速度与20ms曝光跨度、任务身体健康子集及数量。不得把目标需要的转头全部惩罚，也不得把倒地静止计作稳像。

## 必要检查与已发现问题

1. V2原XML已有mouth_tip/jaw_soft/head_camera，仍需验证CAD口端目标可达、朝向/头撞地/自碰撞与负载力矩；不添加假接口或修改碰撞源模型。有限一批物理可行性，不无限候选搜索。
2. 确定bug：旧`mouth_payload_force`挂weight=0 reward，而锁定RewardManager.compute在weight==0直接continue，载荷不会施加。新独立cfg去掉该零奖hook，将同一F=−mg、tau=(mouth−COM)×F作为post-command step事件；须验证真实force/torque、phase ramp、清零/reset及terminal shadow不修改live。尚未资格通过。
3. GroundPickPhaseCommand尚未被TerminalCriticObserver准入；必须验证reset/random phase、wrap、超时successor与外力状态，不能只扩白名单。
4. CPU配置/几何→64×5含真timeout/实际reward oracle/ONNX→实际N吞吐、有限/feedback/容量护栏→首128；全部使用新技能从零actor/critic/N/Adam，独立预算。

## 保留失败

- CPU audit001在仅编译模型/读取site元数据时失败：`m.site(name).bodyid`是1D数组，int转换拒绝；0物理积分/0训练。修为`model.site_bodyid[site.id]`，旧日志与failure JSON保留。无理由重跑任何已完成PPO（本技能本来也尚未PPO）。

## 2026-10-16T07:39:18.476320+08:00 CPU检查及首物理批次

- CPU audit002完成（成功JSON为groundpick-cpu-audit-001.json，001失败日志另保留）：V2原mouth_tip/jaw_soft/head_camera存在；实际episode20s、period4s、reset phase随机，solver10/20原模板保持、capacity64。旧7月设计稿未使用。
- 固定6次CPU几何求解（3口端高度15/30/45mm×2初值），每深度两初值收敛同一邻域，均无原CAD自穿入、足端近地、COM在支撑区；仅几何候选，非动态通过。嘴尖朝下cos>.998，最低目标需要neck/ankle接近现有限位，须真实v5静稳/力矩验证，不以IK当可运行。
- 单批真实GPU物理PID3895998/session86130启动：HOME+三个几何目标，各空载/30g×16worlds，共128×3s，记录actual q相对target、contact/head impact、mouth height/orientation、head torque与200Hz camera。仅等待SitStand末评结束后用GPU0，实际无其它技能PPO。

## 2026-10-16T07:52:24.849281+08:00 物理/终止通过，fresh PPO准入

- 物理002完整：15mm候选真实空载/30g各16worlds末1s嘴高12.57/12.49mm、down cos最小.99711/.99732、tilt28.15/28.30°、q目标MAE约.0075rad、线速max.00123m/s。settled .5–3s自接触及超1N头冲击均0、全3s反馈fault0。其余候选结果完整保留，30mm候选+30g有4/16超1N头冲击，不能归为全部通过。此传感器当时仅force，没有证明完全无head接触。
- 记录器001首step后误用未实现joint_torques接口，改明确qfrc_actuator；旧failure、实际1control留档，非PPO/物理崩溃。
- GroundPickPhaseCommand/载荷事件真实terminal10案例（5phase×fast/full）全部PASS：critic逐位0差，20个live张量含xfrc/RNG不变、世界F/tau误差≤1.49e-7、部分reset phase/payload PASS；actor61/critic74。显式reset后初始payload需初始化一次，自动reset由native post-command step事件更新；新trainer已落实。
- 同一head impact传感器增found只读字段，原force/奖励/碰撞不改，供严格头接触指标；奖励与数据源独立physical oracle共7项，所有reward只调用一次（oracle仅纯函数）。
- fresh64×5/强制32timeouts PID3922580/session89448已启动，0旧技能actor/N/Adam；全自动后续queue24994：N16384×48（覆盖20s自然timeout）→原100k中位与全部软件护栏→首fresh128→两条件12s三完整cycle/四载荷/相机末评。实际N固定16384，阶段最多256更新/100,663,296controls、最多2正式配方；首128新增50,331,648，末评后才决定是否有依据使用余预算，不自动加段。正式run/output/qualification全绝对路径。

## 2026-10-16T08:03:06.794935+08:00 指标选择器修复及全入口软件通过

- smoke001在0更新被指标入口拒绝：env.cfg中的body_ids尚为slice；指标须读event_manager持有的resolved term。局部修复后保留failure；queue24994正常阻断，没有actualN/正式控制。
- 新queue80682/fresh64进程3934826完成smoke002：5更新/7680验证controls、32真实timeout、7项physical reward/F oracle最大5.96e-8、ONNX64实际obs最大3.58e-7；初始env.reset后载荷初始化已实际核对，actor/N/Adam均fresh。
- 同进程两条件32env evaluator烟测全部PASS：8组（4虚拟载荷0/10/25/40g×有/无训练3–6s间隔push），每组3原生4s周期；实际2401个200Hz camera四元数、初态和物理trace保存，actor/N逐位不变，oracle≤5.96e-8。随机5更新表现不是技能验收。负载仅重力wrench/力矩扰动，没有实体抓取或附加物体惯量，不声称真实抓取/硬件已通过。
- 自动已转actualN16384×48 PID3942954，输出groundpick-actual16384-001；48覆盖20s/1000控制自然timeout。之后全部原资格PASS才fresh128，当前正式预算0。完整阶段N固定16384、最多256更新100,663,296controls/2配方，首12850,331,648，不自动续训。

## 2026-10-16T08:10:39.559555+08:00 实际N16384未过吞吐，有限N24576检查

- actual16384×48全部执行但未签准：median84381.53/aggregate83047.16controls/s，collection中位4.205s vs learning.384s，末平均episode27.50controls，70°fell_over使全部提前reset，没有自然20s timeout；这不是bootstrap损坏（64已32强制真实timeout PASS）。contact107964/1048576、oracle≤5.96e-8、feedback/nonfinite0、maxKL.01705/maxclip.25766。原queue因终止覆盖断言阻断，正式0。
- 父级确认仅再一档N24576×16，显式force-timeouts验证大批shadow内存/终止，同时测吞吐；正式绝不伪造episode年龄，不改solver/PPO/奖励/命令。已启动session84938。若仍不达100k，如实按有效学习速度决策，不无穷性能改造。
- 阶段控制总上限100,663,296不放大：若选N24576，首128=75,497,472，余预算最多42更新=24,772,608（总100,270,080）；若留N16384则原首12850,331,648/阶段256。N选择后记录精确预算。

- 输入契约已独立记录`runs-v2/groundpick-training-input-contract.json`：61维保留布局，但48:51是4s phase cos/sin而非速度，head/body槽零padding，物理S0/O0不变，ONNX含normalizer。仅训练契约，runtime_qualified/hardware_tested=false；没有改实机接口或做部署。

## 2026-10-16T08:17:41.014634+08:00 N24576准入与正式128

- 16更新验证9,437,184controls、12288强制真实timeout、contact154205/1572864、maxKL.01508/maxclip.23158、7oracle≤5.96e-8、feedback/nonfinite0。median97831.51/aggregate96290.69仍低于100k；未继续调物理或软件性能。相对不同N/fresh更新过程只是实测吞吐选择，不称逐轨迹/同权重N因果对照。
- 资格002显式记录性能未达与允许有界推进；所有数值源SHA逐一复核后，正式fresh128 PID3972041/session82066启动。预计75,497,472controls，首次正式GroundPick从0累计；父root已同步阶段剩余42更新上限。完整wall/实际行为待末评，不能把预检吞吐或静态候选当模型学成。


## 2026-10-16 正式首段现场续接

PID3972041/queue82066已确认实际60次更新、35,389,440正式controls，无guard；GPU0约89%。中段日志稳态约104–105k/s，不能据此替代含初始化/完整正式段吞吐。累计训练low_goal尚0，仍须等128节点完整nominal/measured三周期独立评测。正式预算与自动末评路径不变；Roulade CPU准备在独立真实worktree进行，不改本段源。


## 首128完成，自动独立末评中

正式75,497,472 controls/clock3072；完整进程820.587s＝92,004.20控制/s，训练循环aggregate100,939.42、median102,275.52。预检曾不足10万、正式循环实际越过，但不得称完整进程也越过。自然timeout successor12,183条；fault/nonfinite0，contact162283/1572864，无guard。训练低位goal累计仍0，不代表已学会。自动名义末评PID3992321于正式退出后启动，随后measured，末评完成才决定剩余42更新。


## 首段最终复盘与关闭依据

- nominal 三周期 return500ms=63/177/63 of256；measured=73/142/76；low/complete均0。相同256worlds重复周期，不是768独立试验。
- 两条件反馈fault/头接触均0；nominal3个cycle-trials自接触、measured0。嘴低位91–93mm，对比物理已证可达约12.5mm，仍远高于25mm门槛。
- 实际低位q均值：head_pitch1.126接近1.13082上限、head_roll.404、左hip_roll.455，neck−.387；不是已证15mm可达pose的neck+1.99/head_pitch−.861。仅根据物理q不能推断执行目标/静摩擦根因。
- low阶段已曝光3,153,796训练samples；std=.10的Gaussian在92mm不饱和。未发现可由简单数值修复解决的缺口；不把42剩余预算本身当续训理由，不推断该动作不可学。
- 报告文字clean与实现差异已用保存轨迹sidecar并列：原clean还要求每控制步bothfeet，声明clean仅排除head/self/invalid。原件不改，两种complete仍0；low/return原门槛和保存指标逐一parity。相机新增最后500ms健康身体子集，避免整个站立phase早段尚在回正影响解释。
- nominal低位健康样本136/211/134，实际相机角速RMS均值.75–.83rad/s、20ms曝光跨度p95均值.0195–.0200rad；站尾健康63/177/63，RMS.12–.13、跨度.0051–.0055rad。两条件完整数值见各`saved-trace-audit.json`，不把嘴未达标但身体健康子集称拾取成功。
- 计时精确定义：100,939.42/s是剔除首3更新后的125更新稳态aggregate；820.587s是内部建环境起计时（92,004.20/s）；外置真实完整进程含imports/配置/打包904.344s（83,483.18/s）。最终使用外置口径讨论端到端，不能把内部字段名full_process字面当含全部过程。
- 保留清单：数据盘`p04-groundpick-retained-candidates-20261016.json`；后续GPU0转Roulade，GroundPick不自动重开。

## 低位物理证据口径纠正（冻结128诊断001）

此前3秒固定目标物理证据的“head impact=0”只表示超过1N部分为0，不能称无头地接触。新64世界预保持显示全部原低位门槛均通过，唯独head_contact found=1：实际0.663–0.781N；CPU确认为头壳接地。旧轨迹未保存found，原件保留；“健康低位已验证”的推断撤回。原正式阶段仍75,497,472步且未合格。唯一后续诊断002只将neck目标−.06rad增加净距，保持严格no-contact gate，不改训练源或开始PPO；实际结果另记。

证据：`groundpick-physical-qualification-erratum.json`、`groundpick-lowstart-contact-audit.json`。

## 低位诊断最终收尾：不进入新PPO

001/002各64世界，1s固定目标预保持+4s冻结128策略原生周期，共新增32,000诊断controls；**正式仍75,497,472 / 128更新，余42更新未花**。两进程均exit0，GPU1已交还，无active PID。源码SHA和actor/normalizer逐位不变，真实actor相位观测/执行目标已保存，未建新训练源。

002仅将neck固定目标减.06rad；同64保存qpos的CPU净距最小6.863mm、嘴15.912–16.376mm/down≥.999843都过几何门槛。但真实预保持至.10s时64/64重新头触地；尾200ms嘴11.79–12.20mm、头力.612–.814N。其它低位门槛全64通过，仅no-head-contact全0。因此两次起点均不合格，不能用“first exit=0”声称策略从健康低位立刻离开。

002冻结策略随后在站立phase持续500ms通过48/64，完整clean cycle0，fault0；这只是恢复诊断，不补判拾取成功。200Hz相机/20ms曝光按真正站稳末500ms子集另列在 `groundpick-lowstart-final-review.json`，低位健康分母为0。分配的30g负载在初始φ=.375按原任务尚不施力，φ=.425后才递增，不应称带载低位保持。

确认的奖励/评测谓词缺口：head impact只惩罚>1N部分，而评测拒绝任何head_contact found；约.7N头撑在该罚项成本为0。未重建全奖栈，不能因此声称全局奖励偏爱头撑，也未授权另一奖励配方。唯一端点reset候选因为动态健康准入失败而关闭；不继续姿态搜索或花剩余预算。

完整机器结果：`groundpick-lowstart-final-review.json`；当前状态：`groundpick-agent-current.json`。原report/traces/checkpoint不改。

## CPU静力追加审查结束

原001/002姿态在宽松整足投影内，不能归为全体COM越足底；实际CPU接触多边形外分别26/64、8/64，所需最小头载上界约.75/.79N，与实测量级相符，但其余例不能判静力必需。单次120迭代对称腿/根约束修正未过锁定足底平面和嘴高边界，标NO_QUALIFIED_STATIC_CANDIDATE；无GPU/训练，余42仍未执行。详细见 [静力审查](sc0090-v2-groundpick-static-support.md)。

## 根代理新增授权：仅一次静力最近候选动态核验003

CPU静态筛选仍NO_QUALIFIED；根代理指出微小筛选超差不是动态不可行证明，授权仅此候选GPU1固定目标3s，不重新求解/搜索。已启动PID138802/session55261，64worlds，32空载/32实际30g（固定原生phase.6使载荷满幅），9600 diagnostic controls、PPO0；原严格low/no-contact门槛保留。保存actual随机质量/COM和完整qpos/qvel，结果出来前不称动态通过。

## 003动态结果：端点探查关闭，未合格

唯一静力最近候选3s实际批次已exit0。空载32/32无头地接触，尾1s嘴8.51–12.62mm、down≥.99837、双足和速度/姿态门槛均过；但是32/32自接触。30g组嘴7.94–10.41mm、down≥.99961，同样32/32自接触，另12/32有≤.190N头地接触。两组尾1s完整low_goal都是0/32，fault和非有限均0；健康低位相机分母0，不作稳像通过声明。

保存末态CPU原CAD接触复现：全部64世界为左右各自 `ground_hip_collision ↔ foot_collision`，穿深约0.067–0.473mm；不改碰撞掩码、不忽略该接触。实际固定目标的关节与物理平衡位置有偏差，这个CPU几何候选因此没有转成健康动力学端点。实际随机质量.5888–.6128kg、body_ipos/inertia、每步COM/qpos/qvel/真实动作链都已保存。

按约定结束，不再候选扫描或追加GPU。三批共41,600诊断controls；正式仍 **75,497,472 / 128更新**，剩余42未花。GPU1释放，无active PID。`groundpick-endpoint-probe-closeout.json`是此次后续诊断最终状态，`groundpick-agent-current.json`已同步；原正式128策略保留且未合格。
