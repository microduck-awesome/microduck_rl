# 当前状态

Sit近教师固定320已正式960完成，全部历史603,979,776控制封顶；外置末评bool统计已修，PID581916/session89539恢复首saved trace报告并补其余5项，训练不重跑。详见近教师专职台账。

# 当前阶段（更新）

GPU1近端教师原3e-5在小批/实际N均触发BC KL护栏；BC LR1e-5小批通过但actualN超原.5护栏；BC3e-6全部资格PASS，PID548509/session12268已实际正式660/960（7,864,320新增控制），从原640至960；最新更新以allocation/status.json为准；[专项台账](sc0090-v2-sitstand-near-teacher.md)。全部旧640结果和失败历史保留。

当前：SitStand历史基础320 + Hold320到640及所有固定末评均已结束，原PID356135 exit0。576/640两条件坐姿末保持384/384，但站姿0/384、严格头响应0/672，仍未双向合格。全部SitStand正式478,150,656封段，无自动第三段。GPU1经root GroundPick005后已转Ball健康锁存准入PID397198/session30107。

## 历史阶段记录（下文旧当前行仅代表当时状态）

当前真实阶段：Hold320 正式续段运行，PID356135/session49554，已到610/640（最新status），本段新增114,032,640控制；每更新实际课程冻结断言通过。原HistoryBase320及其十项末评已完成，原GPU窗口等待也已结束。正式到640后自动448/576/640双条件评测，无自动第三段。随后GPU1先交root GroundPick005，Ball GPU资格等待该窗口结束。详见 [Hold320续段](sc0090-v2-sitstand-history-hold320.md)。

## 历史续接记录（以下当前表述均为当时状态，非现状）

当前：已获批准HistoryBase320后同配方课程冻结续320，见 [Hold320续段](sc0090-v2-sitstand-history-hold320.md)。原worker356135导入暂停，等待root临时GPU1闭环结束后自动恢复，尚0新增正式；全阶段上限478150656。

当前（历史基础320完成）：GPU1已释放；SIT末保持nom324/meas330 of384重复trial-phases，STAND0/384、严格head0/672，fault0。五节点初态逐位配对，部分SIT候选保留，双向/头控未合格。全部正式352321536，不自动续训。详见 [历史基础固定方案及最终曲线](sc0090-v2-sitstand-history-base.md)。

当前续接（2026-10-16）：历史基础固定320已获批准，worker002 PID319428/session27653准备资格；001日志包装错误已留1536验证步并修复；见 [历史基础固定方案](sc0090-v2-sitstand-history-base.md)。此刻尚未正式更新。上一M4/M16四冻结评测完成，16更新仅早期快照。禁止自动加入head L1/EMA或短训换方。

# V2 P04 SitStand 专职台账

## 当前状态（fresh128已关闭，M16仅验证亦完成）

正确TransitionRelease配方从零128新增50,331,648控制，全部SitStand历史226,492,416正式控制封顶；原448和所有失败保留。fresh两条件原12s评测body tail/head到位仍全0，健康相机分母0，未合格。

随后唯一M16实际N16验证已完成（6,291,456诊断、正式0）：实际optimizer.step1280 vs原M4 nativeAdam320，数值/反馈/容量正常，稳态wall约+10.8%；尚无实质技能改善证据，未自动新正式。GPU1已释放。当前机器记录 `sitstand-agent-current.json`。

[从零谱系](sc0090-v2-sitstand-fresh-release.md)、[更新密度与奖励审计](sc0090-v2-sitstand-update-reward-audit.md)、[上一释放模板阶段](sc0090-v2-sitstand-transition-release.md)分别保留。未来如继续需新的明确阶段预算，不将旧226,492,416重置。无实机、部署或轮滑。

- 目标：61D actor、14动作、HOME0的V2原生受命坐下/起立/保持，能中途反向命令；坐/站均有相关头控与抗后仰恢复。实际物理角度/高度/冲击用于奖励和评测，不作为替代测量输入。
- 独立真实worktree：`/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04/sitstand-repaired-worktree`，detached070d606；已从本人冻结起身源逐SHA复制，18个测量/电机/runtime核心文件与父级最新walking源逐字一致。未复制正在验证的walking评测作为资格。计划task `Mjlab-SitStand-Flat-MicroDuck-V2-Repaired-S0-O0-R1`。
- 合同：冻结v5/V2 CAD、S0/O0无趋势去尖峰、O0 local history、compiled/fused/CUDA graph、200Hz物理/50Hz控制；新actor/critic/normalizer/Adam从零，不继承起身零twist分布。critic宽度按真实配置获取，不硬编码74。
- 已读本仓AGENTS.md、历史`docs/sitstand_policy.md`、原SitStand配置/命令与主技能清单。旧XL330成功经验：posture flag在twist vx槽0站1坐、独立50/50 reset×command覆盖四组合、3.5–6.5s驻留、2s内部alpha渐变，速度上限只作后备；学习后才增加坐姿head jerk/后仰kick，不提前加扰动税；峰值/终点jackpot不可替代渐变目标。
- 已有V2候选：左右hip_pitch=-.10、knee=-1.10、ankle=-.85，hip_roll左+.0873右-.0873；sit z约.06337、tilt约2.3°，stand z=.112。必须在本任务全碰撞与真实P04动力学下重新验证静稳、命令过渡和碰撞，不能照搬XL330 ±1.35膝或.115站高。

## 本轮方案与准入

1. 迁移旧SitStand配置的语义与成功课程到V2；所有SIT override/高度/选择器以V2命名映射核对，原已有起身和步行配置不改。
2. 至少3秒扰动初态hold/过渡物理试验，记录高度、tilt、接触、速度、冲击与限位；先确认可达稳态再PPO。
3. TerminalCriticObserver当前未支持SitStand/zero padding/坐姿interval事件：必须验证shadow复制alpha/flag/timer/command与pre-reset successor时序，以及live状态不变，之后才能限定支持；不得只扩白名单。
4. CPU配置/符号/命令与目标奖励检查→64环境5真实PPO更新（含真实timeout与ONNX契约）→实际N吞吐/稳定性；新技能PPO先保留T24、5epochs、4minibatches，N选用实际准入证明，不把旧起身bench冒用。
5. 独立评测四初态/目标组合、双向与反转、坐站持续保持、过渡峰值速度/冲击及头控；按结果安排有界正式段。`run.json`明确allocation_kind=formal/validation、task、num_envs/allocation_updates/resume、agent.num_steps_per_env；status保存completed_updates/allocation_controls/inherited_controls/controls。

## 记录规则

每个假设、修改、源码/配置/检查点SHA、准入、失败和真实wall/controls都留档。暂未证明的风险不冒充已修复，预检不记正式；真实硬件尚未可用，不宣称sim2real最终成功。

## 2026-10-16T05:32:31+08:00 物理端点与终止观测

- CPU cfg PASS：61 actor/14 action、原T24/E5/M4；新技能从随机初始化，未继承起身模型或归一化。
- `sitstand-physical-feasibility-001`：128环境四组10s实际P04；前3s坐/站×精确/扰动均稳定，坐高0.06344–0.06348m、最大tilt3.79°；站高0.11217–0.11223m、最大tilt2.53°。0反馈故障。
- 2s直线关节插值过渡双向共128/128倒；这是否定开环插值模板的证据，不否定可学习的重心转移。奖励仍为宽松平滑内部目标；正式验收看物理到位/保持/速度而不强制动作照抄插值。
- 首terminal检查拒绝旧模板未迁移的critic foot readers，失败保留。已替换成现有V2同名安全reader。
- `sitstand-terminal-validation.json` PASS：真实GPU五情景（坐姿激活jerk/tip、ramp边界、命令到期、fresh height、反向），critic与native无reset successor逐位相同，21个live状态tensor不变；部分reset alpha匹配实际传送高度，其他env不变。critic实际74，仅本任务。
- 新原生trainer＋SitStand物理metrics＋独立坐/站/反向/头控电池已写；64×5强制timeout软件验证已启动，尚未formal，当前预算0。

## 2026-10-16T05:41:48+08:00 软件准入→实际N

- smoke-001完成5更新/7680验证controls后，ONNX验证器误用batch64喂官方固定batch1输入而失败；failure.json与原件保留。改逐条验证；未修改网络导出或数值训练源。smoke-002完整PASS，32个真实timeout、五项物理reward oracle max0、ONNX实际64 obs最大2.38e-7、actor/N不变。其余pure PPO数据保留，max pooledKL .09734、maxclip .61862，不能以小N推定大N。
- 独立nominal/measured各32环境×600控制电池软件PASS（原始随机策略仅训练5更新，实际保持全0，不能称技能通过）。两份采用各自smoke checkpoint，非电压因果配对。五oracle仍全0，actor/N逐位不变、无fault。
- 实际N：`sitstand-actual32768-001`，32768×T24×32=25,165,824验证controls，覆盖12s自然timeout；独占GPU0，不混评测吞吐。保留原nconmax200/solver30/50以先验证真实配置，不为100k擅改物理。护栏finite、contact、PPO和中位≥100k。
- 预排首正式128（100,663,296 controls，fresh无resume）仅等该源资格PASS，queue71844；准入失败则不启动。正式完整进程wall另记含Python imports，与trainer内部env+learning时间分开。正式后自动nominal/measured四初态×反向×返回×head检查，非自动续训。

## 2026-10-16T05:48:41+08:00 碰撞容量必要修复

- 原200/world实际32768在初始化0控制OOM：Warp convex EPA单次临时申请20,132,659,200 bytes；失败记录`actual32768-001/failure.json`，未损失正式预算。
- 锁定Warp支持naccdmax独立配置，但锁定mjlab SimulationCfg/Simulation构造器只暴露nconmax/njmax，不传CCD参数；本轮不扩分配框架。唯一cfg改动`nconmax=64`，求解器30/50、dt、CCD迭代、V2碰撞、v5/PPO不变。
- 新64×5/32timeout/五reward oracle/官方ONNX通过；同seed同动作128×10s对照两端全部稳定/0fault，采样contact峰1778/8192。1–3s旧/新最大高度差.265mm、tilt差.770°；全段跌倒后的差更大。旧原始fullstate未存，故不称严格初态配对或逐步动力学等价，只是端点可行与容量检查。开环双向仍128/128倒。
- 新actual32768×32运行中，每物理step nacon容量硬检查，外置每2s扫描CCD/contact/constraint overflow并安全停止。formal128仍仅等待新源资格，queue44394，不自动放宽100k/正确性护栏。

## 2026-10-16T05:56:53+08:00 实际16384准入与首正式段

- cap64/N32768首step graph launch仍OOM，0更新；未削solver、碰撞或改PPO，原件`actual32768-002/failure.json`。
- 同源N16384×32完整PASS：稳态中位100936.72 controls/s，稳态聚合99037.67/s；内部env+learning wall198.49s（不是包含imports的全进程）。16384真实timeout，5 oracle max0，feedback/NaN0，max pooledKL.06155、maxclip.32860，contact峰129479/1048576。8个容量告警模式无命中。
- 资格`runs-v2/sitstand-training-qualification-003.json`；达到已约定稳态中位准入，不再探索24576。正式源未修改：`scripts/train_v2_sitstand.py`；task仍`Mjlab-SitStand-Flat-MicroDuck-V2-Repaired-S0-O0-R1`；N16384/T24/E5/M4，fresh actor/critic/normalizer/Adam，原LR1e-3 adaptive，未继承smoke/起身checkpoint。
- formal目录`runs-v2/sitstand-allocation-001`，PID3697414，外置队列`run-sitstand-formal-003.py`/session19086；128次更新=50,331,648 controls。它替代从未执行的N32768×128计划100,663,296；不合并预算。
- 独立8模式外置watch `watch-sitstand-capacity-003.py`/session19537覆盖当前formal与旧预检。正式后自动nominal/measured256×600相同命令电池，先读姿态保持/双向与反向、速度冲击、头部/身体保持，再决定下一段；无自动续训。

- 2026-10-16T06:00:48+08:00 formal状态快照：20更新，7,864,320 controls、GPU0约90%。反馈/非有限0，训练sample稳定比例仍近0，尚未学习成功。最新单update5.49s，不能将预检稳态中位100937直接写成正式整段速度；保留末段真实wall与时序再汇总。

## 2026-10-16T06:35:45+08:00 128失败模式与有限诊断

- 正式128全部完成：50,331,648 controls；trainer env+learn wall561.879s，稳态median103400/aggregate101994 controls/s；外置含imports进程641.603s/78446.7controls/s。未称全程10万。
- nominal/measured原定256×600均自动完成，各四组×三阶段tail_hold全0，fault全0；站→站并未>45°倒地，却尾tilt27°、neck误差约115°；站→坐停高.0956–.0971m；坐→站/坐→坐尾tilt67–69°，整段>45°约89–95%。
- 段后自动评测完成而未及时衔接，造成GPU0空档；已承认，恢复64环境真实full reward诊断并等待单因素修复，不原样盲续。
- `sitstand-0128-fullreward-comparison.json`：每步只读真实reward_manager一次计算的21分项，权重/演员/N未变；第一阶段2–3s正确站目标15.669>策略高颈站11.513，正确坐10.533>侧倒坐6.143；固定端点两组均16/16hold。说明不是全局反奖，长期倒地占样本问题明显。
- 坐姿正确端点self_collisions约−3.50/步（站0），须先识别实际GPU接触pair/dist/constraint rows；`audit_v2_sitstand_contacts.py`/session36154正在跑。不给无证据“假碰撞”结论、不删碰撞/惩罚。接触结果后收敛到一个最小修复；不改整个奖励栈，不以tilt立即终止而切断头/膝支撑过渡。

## 2026-10-16T06:54:55+08:00 已确认目标自干涉→ClearSeat最小修复

- 实际GPU settled碰撞：128/128双侧hip凸包与foot持续约8.297–8.299mm穿入，最大单约束行约22（未当作合力），与CPU静态几何8.30mm一致。旧目标仅高度/tilt稳定，不是健康姿态；CAD/碰撞/v5未改。`sitstand-sit-contact-audit-v2/{report.json,raw-contact-state.pt}`保留。首contact导出工具误用旧efc_force字段，修正为锁定Warp data.efc.force，失败原件保留。
- 有限CPU几何筛选→唯一一批4候选×32环境×3s；批001统计器错误调用microduck mdp别名，第1步失败，已改为复用生产self_collision term并重跑批002（未扩大GPU候选）。只有hip−.7/knee−.8/ankle−.15、左右roll±.0873 qualified；其余3候选实际倒地，不采用。
- 资格窗口准确：settled .5–3s无self_contact且max tilt6.44765°；末1s高.0759992m（范围.075631–.076391）、max线速度.000703m/s、角速度.00975rad/s；fault检查全3s为0。不是全程无自接触的声明，也不是过渡技能通过。
- 新真实worktree `sitstand-clear-pose-worktree`；task `Mjlab-SitStand-Flat-MicroDuck-V2-ClearSeat-Repaired-S0-O0-R1`；SIT高度取.075999。唯一行为变更是健康SIT目标和对应高度，原12s/初态×命令50/50/2s内部ramp/正常head奖/PPO/碰撞/冻v5保持。指标读取实际task目标，避免旧常量污染。
- CPU递归cfg差分PASS：只sit_overrides/sit_z/reset坐高字段；actor obs/actions与所有奖励weight逐位配置一致，agent仅run_name变。source-transition.json硬限定4个变动源并锁定128父SHA。
- mature64×5从128→133为验证：exact actor/critic/N/Adam/LR/clock恢复通过，32真实timeout、5physical oracle全0、ONNX64obs max1.19e-6、actor/N不变。正式预算仍50,331,648，不计smoke inherited或新增7680。
- actualN16384×32 PID3793156，自动队列`run-sitstand-clear-pipeline-001.py`/session83141；准入后从原128父正式128→256，新增50,331,648，不从验证160节点续。后续完整12s名义/measured256×600自动评测，保留错误/PPO/8容量告警护栏。
- 用户新指令：头控长期平台可保留局限继续其它策略；不再起身专项head。本SitStand聚焦真实双向/健康姿态，正常head奖励与读数保留，不让微幅精度阻塞；原颈114°/身体27°属于健康姿态错误，不能作为可接受头精度平台。

## ClearSeat 128更新正式与相机评测增量

- 当前source：`sitstand-clear-pose-worktree`；任务`Mjlab-SitStand-Flat-MicroDuck-V2-ClearSeat-Repaired-S0-O0-R1`；资格`sitstand-clear-qualification-001.json`已PASS。mature父仍原失败128，validation160不继承；原actor/critic/N/Adam/LR/clock/RNG精确恢复，唯一行为变化为已物理验证的坐姿和耦合高度。
- 实际N16384×32稳态中位103320.99、聚合101757.95controls/s，自动正式PID3799173，新增50,331,648controls。启动时尚导入/初始化，不以PID替代实际更新。
- 外置evaluator添加只读camera记录，旧脚本与变更SHA保留`sitstand-clear-evaluator-before-camera.py`及`sitstand-clear-camera-eval-change.json`。数字训练source hash不含该独立评测脚本，训练源不变。待独立测试与真正末评验证；不能把关节精度或身体omega当相机稳像。

## 2026-10-16T07:08:45.864998+08:00 ClearSeat128完成，自动独立评测

- 正式128→256全部完成，新增50,331,648 controls；SitStand累计100,663,296，原失败128未删除。trainer env+learn wall560.173s/89850.17controls/s；稳态median103523.63/aggregate102485.96。外置含imports完整进程640.037s/78638.60controls/s，不能把稳态10万写成完整wall。
- contact峰113022/1048576，真实timeout 81920，feedback/nonfinite均0、无optimizer/capacity告警。累计训练健康保持比例0.00000270，不能称技能学成。
- 自动nominal完整256×600电池PID3817478已启动，之后measured；相机外置脚本SHA3c02e2ab91f838ad59c03dfec44ce8e15a12fce0325f66144ea47c7c831aea06，CPU7检查PASS、98数值源SHA未变。等待实测行为后决定，不自动同配方加段。

## 2026-10-16T07:13:57.041437+08:00 4秒初级课程有界阶段

- ClearSeat256名义四组合×三phase保持全0；站姿起点不超45°但头平均误差33–34°，坐姿起点64/64超45°，feedback0。相机采集2401×256四元数/finite/unit检查通过，但健康保持子集0，倒地后低角速不能算稳像成功。实测末评仍在自动队列，无新反馈“放宽”。
- 父级已按完整行为证据确认一个新阶段：同256父，只把训练episode从12s改4s；50/50 reset×command、2s内部目标slew、奖励/头控/全部PPO/动力学/测量不变。原SitStandCommand与shadow successor机制不变；保留双向4s尝试，不依据tilt提前终止。独立play/末评仍完整12s反向/返回及head/camera。
- 新真实worktree`sitstand-short-lesson-worktree`，任务`Mjlab-SitStand-Flat-MicroDuck-V2-ClearSeat-Lesson4s-Repaired-S0-O0-R1`，source-transition显式限定3文件差分与256父SHA，精确继承actor/critic/N/Adam/LR/RNG/clock6144。
- 本阶段最多128正式更新/50,331,648新增controls，累计将150,994,944；这是本轮第3个行为配方。末评若仍无健康保持实质改善，停止本方向转下一技能准备，不再追加reset/奖励/熵试验；已结束起身head方向不重开。
- 自动queue98901当前`cpu-and-mature-64-smoke`/PID3833146：CPU精确cfg diff→同进程64×5 mature/32timeout/ONNX→N16384×32验证→通过同原护栏再formal256→384→nominal/measured各256×12s相机电池；validation父始终256，验证所得检查点不用于正式起点。

- 2026-10-16T07:18:25.687085+08:00 Lesson4成熟64×5已PASS：32真实timeout，五物理oracle全0，ONNX64真实obs CPU误差1.43e-6，actor/N导出不变。CPU train仅episode_length_s变化，play完全相同、PPO仅run_name。自动进入actual16384×32 PID3838150；formal尚未启动，预检控数不混入正式。

## 2026-10-16T07:21:27.874822+08:00 Lesson4实际N准入与正式启动

- `sitstand-lesson4-actual16384-001`32验证更新通过：稳态median101142.21/aggregate98734.12controls/s，49152自然timeouts，contact112501/1048576，5oracle全0、无feedback/nonfinite/optimizer/capacity告警。中位达到原100k准入；聚合稍低如实保留，不改N/solver/T/PPO。
- 自动正式PID3848934/session98901，输出`sitstand-lesson4-allocation-001`，256→384；不继承validation288。启动后会核首更新及精确时钟6144→9216，累计目标150,994,944正式controls。

## 2026-10-16T07:39:18.475433+08:00 Lesson4最终收尾

- 256→384已完成50,331,648新增、累计150,994,944；稳态median100966.09/aggregate99787.57controls/s，外置完整660.002s/76259.84controls/s。245760自然timeout，contact121231/1048576，无feedback/nonfinite/optimizer/capacity异常。
- 两条件末评全12组健康保持仍0，200Hz camera已保存2401样本/世界并按过渡、零头保持、主动头/idle分组，健康子集为0，不能用原始低角速替代稳像合格。
- source与所有失败原件保留；已启动下一GroundPick单批128worlds×3s物理可行性PID3895998/session86130，未开始PPO。


## 2026-10-16 正确ClearSeat慢路径与唯一过渡模板释放

慢路径64×20s已完成，正式0；正确端点63/64健康但2s/4s往返均0/64，头接触64/64、self/fault/nonfinite0。足域COM不足须结合双侧shin支持，不能判端点或任务不可行。新单64预算已明确获准，原384父/16384/T24，累计正式封顶176,160,768；完整证据与真实worker见[专职增量记录](sc0090-v2-sitstand-transition-release.md)。
