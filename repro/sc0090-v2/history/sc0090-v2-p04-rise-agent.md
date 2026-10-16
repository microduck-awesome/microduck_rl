# P04 起身专职代理台账

2026-10-15 14:50 开始。主台账：[训练台账](sc0090-v2-training-ledger.md)；父代理负责 GPU1 起身迁移诊断，本代理独占 GPU0 起身站立训练。本轮不训练其它动作或轮滑。

## 当前状态（2026-10-16 05:15）：本轮起身实验已收尾，GPU0转SitStand

父代理已明确停止追加本轮起身head试验，并将GPU0分配给独立V2 SitStand（受命坐下/起立/保持）。新工作记录于[坐站专职台账](sc0090-v2-sitstand-agent.md)。本节没有声称头部全指令或实机已经合格；起身全姿态/测量故障的成功与严格头控缺口分开保留。

| 保留候选 | 小单到位/保持 | 当前单 | 当前双 | 当前四 | 大双 | 用途 |
|---|---:|---:|---:|---:|---:|---|
| micro1856 | 1/0 | 114/112 | 171/160 | 11/6 | 212/209 | 当前组合保持较好；共同完整负pitch 0/1024倒 |
| cap0051920 | 3/3 | 116/114 | 165/150 | 8/3 | 229/221 | 当前单轴及大双轴取舍 |
| 同步cap0051984 | 2/2 | 114/113 | 162/148 | 4/3 | 222/218 | .5°单轴保持59/128；曾有优化峰值并安全停 |

表中分母依次128/128/384/64/384，均为真实P04 measured、81case×8×2协议；三者nominal/measured恢复1792/1792、fault0、standing0/256。完整负pitch0/1024只属于micro1856，不外推到其它候选。完整checkpoint/source SHA和报告路径在`runs-v2/p04-rise-retained-candidates-20261016.json`；没有重新导出或部署。

- 最后phase+accuracy1984已完成同1920父64，nominal/measured各1792/fault0/standing0。头3(保持2)/115(114)/129(120)/8(6)，比phase-only单/四轴好、两轴明显差，绝对1°30/97/170/7、.5°9/55/34/1；不继续该方向。正式50,331,648/672.090653s=74,888.18 controls/s，52预检稳态100806.76/s独立记录。
- 一次离线聚合审计完成：current双轴mean bonus随保持140→120同步下降.57612→.56747，oldHead2+bonus3.14205→3.14033，整体并未出现奖励反向。存在局部同case持续失败高于成功的反例（accuracy臂55/571对，worst仅修正头部子栈31/55），不足以证明全奖根因或支持无限再改奖励。`head2-active-axis-aggregation-audit.json`明确不含body/action/有状态bias；未新开min聚合试验。
- 随机起身谱系正式实际累计 **2,732,064,768 controls**；所有失败、未选段、同步64与两个phase64均包含，预检/评测另列。最后候选谱系1,560,281,088；micro1856谱系1,459,617,792。所有源/模型/时长与未执行预算保留；SitStand从新预算账起，不混算、不继承起身零twist normalizer。
- v5/CAD/S0/O0、HOME0、200Hz/50Hz不变，合法测量不趋势去尖峰，无baseline权重/N/Adam进入本随机谱系。起身训练进程和头评均已结束，旧PID不再是当前运行；GPU0由同一专职代理继续下一任务，不做部署或轮滑。

## 最近完成的修复与证据索引

- 续1920父时，旧RNG资格检查误要求新父活动流等于1856预检流。仅修资格逻辑：同父严格hash；同任务新父实际流必须等于新父，helper/设备仍匹配。11边界PASS；原src计算源未变，旧preflight未改。`head2-cap005-continuation-guard.json`及patch；实际1920父活动流1f06fefe...正确恢复，未复用e407c603...旧流。
- `head2-cap005-allocation-002/optimizer-event-1971.json`：同步rollout17/34/51跨8s超时，前两KL .0428/.0384，第51 2.603；逐样advantage与完整奖励未存，不能断言根因。首minibatch分布/ratio严格一致，N冻结，反馈/NaN0。后来phase53前后无同峰值只是该因素实证，未宣称解决所有优化问题。
- `head2-timeout-phase-implementation.json`、`head2-timeout-phase-qualification.json`、`head2-timeout-phase1984-comparison.json`记录独立超时deadline、52长预检及同64行为负结果；CPU/GPU首单位曾因str对象地址比较失败，规范化地址后PASS，生产未因此改动，失败原件保留。
- phase measured旧allowlist漏新task，在模型载入/物理步前拒绝；原目录/log改名`attempt1-blocked`保留。仅补明确phase/accuracy task ID，`head2-timeout-phase-eval-allowlist-fix.json`含前后SHA；session21787重跑exit0，measured1792/fault0/standing0。没有将0步入口失败算策略失败或成功。
- `head2-phase-accuracy-implementation.json`及最小patch：原Head2源码/输出逐位不变、train/play/RL去掉唯一新增reward后相同，正负/active/idle/阈值/命名HOME/独立float64 oracle PASS。其指标是按当前目标幅度归一化的稳态精度，不等于任意previous→new动态T90或return；原head评测和.5°/1°诊断持续独立保留。
- 真实执行探针`head2-micro1856-execution-analysis.json`已证明small指令时target相对idle改变1.78–2.55°而physical仅约.05–.64°；有实际命令响应，不能断言网络没读命令或单独归因静摩擦。probe17工况与旧81工况初态不逐位配对；v1仅在checkpoint自身source/task语义下使用。

## 历史快照（2026-10-16 03:54，运行状态已由上方覆盖）

- **GPU0 已启动同配方 cap005 1920→2048 /128**：`rise-head-cap005-worktree`，任务 `Mjlab-StandUp-Flat-MicroDuck-V2-Fresh-M4-HeadStdCap005-MicroExposure025-Repaired-S0-O0-R1`；队列 session67085、正式 PID3472734，输出 `head2-cap005-allocation-002`。03:56实际已1930更新、新增7,864,320 controls，GPU0约90%；首minibatch分布/ratio逐位一致。新1920父活动流1f06fefe...恢复完全相同，LR1e-5/clock46080精确继承。父为 cap005001 的 `resume_001920.pt`，明确同任务 `--resume`；Head2/micro.25/zero.6/head cap.005/腿探索/T24/N32768/冻结N/Adam及实际LR均继承，没有新奖励或再次LR切换。1984/2048 formal恢复、两个节点measured头评及2048 measured恢复/standing已自动排队。
- **修复实际阻断续训的 RNG 资格检查**：旧入口无条件要求本次父活动流hash等于旧1856预检父hash，会拒绝正确的1920恢复。仅新增资格函数：同父仍严格hash比对；不同父仅允许同任务显式resume，且实际恢复流必须等于新父流；helper/source/target设备匹配仍严格，跨任务或未准入卡变更拒绝。11个边界测试PASS；所有既有`src`计算源SHA未变，预检原JSON未改、不重复物理基准。trainer新SHA `044c88a0d11ff334bde9ee4e5f9cfceef9c9b998b41c5789de851fa18c80f29b`，最小patch SHA `3c7d61c404e65837ed1683d0c4697b29ab2f8fc5ca2d49ba333c63b9b4811328`；`head2-cap005-continuation-guard.json`记录原/新SHA及11边界，真实恢复另写`resume-rng-qualification.json`。这不是新的物理/PPO训练因素。
- **cap005首64已完整验收**：nominal/measured各1792/1792、fault0、standing0/256。head小3(保持3)、current116(114)、两轴165(150)、四轴8(3)，final两轴229(221)/四轴2(2)。相同micro1856父/64控制为2(2)/113(112)/153(138)/6(4)、final204(196)/5(5)；1°尾保持control27/85/168/10→cap33/100/202/11；.5°control11/37/38/0→cap6/48/45/0。保留有限单/双轴收益，不称微幅已学成、cap全面更好；静态micro1856当前两轴171(160)/四轴11(6)、1°尾217/13仍是重要参考。cap正式50,331,648控制/553.052891s=91,006.93/s，预检稳态132642.08/s单列。
- **执行探针已完成，不继续扩诊断**：`head2-micro1856-execution-probe` session52063为历史完成，17case×8×2、每seed300真实MLP/动作调用和每actuator1200真实compute，0重复计算/0动作替换，actor/N逐位不变、fault0。保存实际raw actor观测、MLP输入、mean/processed action、原生目标count、O0逆目标、真实motor输入target/q/v、PWM/torque及物理输出。±1.719°小幅命令时，实际target相对idle已改变约1.78–2.55°，物理仅改变约.05–.64°；不能称网络没读命令，也不能仅据此归因静摩擦。neck+实际target4.286°/physical.471°/PWM.155，yaw−target−1.171°/physical−.157°/PWM−.023。分析`head2-micro1856-execution-analysis.json`；通用v1脚本SHA `449406aa54dfb971c24507e8322adf4fad5741484e6a0597d50f9225e5b5a91d`，只在checkpoint自身实际source/task语义下使用，rootstage5不可隐式套freshstage解释。17case与旧81case不是逐位初态配对。
- **保留从零micro1856与cap1920**。父级共同物理源、捕获qpos/qvel/rawobs/origins逐位配对的pitch[-10,-5]、roll±10、1024例：micro1856倒0，baseline T24-4274倒2（首次稳前）；两者最终全稳定/fault0。不是所有倾斜域、90态全配或硬件证明。头部原严格T90、末保持/return和绝对.5°/1°诊断均保留。
- 已完成随机谱系正式 **2,581,069,824 controls**，cap1920候选谱系 **1,509,949,440**，micro1856谱系1,459,617,792；其余对照/未选尾部全部留账。本128计划100,663,296，实际新增只按新status计，未预记已完成。v5/CAD/S0/O0、200Hz/50Hz不变；合法反馈不趋势去尖峰；没有加载baseline权重/N/Adam；仅训练评测、不做部署或轮滑，目标仍未完成。

## 历史快照（2026-10-16 03:09，以下运行状态已被上方覆盖）

- **已完成的同父/同预算无改动控制实验**：`rise-head-micro-exposure-worktree`原source SHA全部匹配既有准入；`head2-micro-control-allocation-001`，PID3391839/session97401，原micro1856→1920/64，已完成1920/64，本段50,331,648 controls，nominal1792/1792、fault0、standing0/256；head/measured已全部完成：measured1792/1792、fault0、standing0；head小2(保持2)、current113(112)、两轴153(138)、四轴6(4)，final两轴204(196)/四轴5(5)。原父1856仍优先保留；相同64控制表明accuracy1920的两轴156(144)只较控制小幅保留，不能把其相对静态父退化全归给新奖励。只resume原微曝光配方，旧Head2、micro.25、head cap.01、腿σ、T24、PPO/N/Adam/实际LR全部沿用；没有加载accuracy1920。成熟恢复与stage10/polish/clock44544 PASS。tracker94102，段末head/measured/standing自动接。
- **GPU0当前为匹配cap.005正式实验臂**：真实隔离`rise-head-cap005-worktree`，task `Mjlab-StandUp-Flat-MicroDuck-V2-Fresh-M4-HeadStdCap005-MicroExposure025-Repaired-S0-O0-R1`，同micro1856父/64预算，唯一训练变量head有效σ上限.01→.005；没有精度附加奖，不改LR。CPU单位PASS，原distribution class/MDP/Head2源码逐字不变。formal护栏改为识别真实distribution class，防止新标签绕过旧HeadStdCap001字符串检查。父级已完成**跨卡活动CUDA RNG恢复映射**且同/跨卡draw oracle PASS；helper SHA eb196432cfe67bf2e048f6d5a03c117ccc0ccdf904cdf8379bd5a0c7d5236e53，修复源已交还本代理。为保留GPU1 baseline新奖励的实际收益，本代理改为control→cap005同卡串接；`run_head2_cap005_gpu0_queue.py` session32673已自动接CPU/GPU unit及64×5 PASS；smoke实际mapped=false，活动hash e407c603147c0cfd68d279e548f71bc43c22cd9b5f7820b3e2fd79ec808394a3与父一致，CPU/其它流也逐位。小N首pooledKL .457/clip .589，后续下降，未以小N代实际N；32768×16已PASS：median132642.08/s、aggregate132238.88/s；首meanKL .06519，后.008–.017，maxclip .23721，原护栏均过。同卡mapped=false、CPU/活动CUDA hash与父一致。正式PID3431316/session32673已实际1861/1920（+3,932,160controls，clock44664），GPU0约93%；tracker16822。resume-verification及resume-rng实际PASS，初始LR3.375e-5/Adam/N精确，同卡mapped=false/e407c603...父流一致；没有手动LR切换。首/末评自动排队，matched结果与1856静态保留同时比较。硬核RNG资格及入口/helper/test SHA；同卡mapped必须false且父活动流hash相同。原GPU1 queue不启动，GPU1继续baseline4338→4466。匹配父/配置/预算不自动等于完整物理初态逐位配对。
- **accuracy1920未选，不继续新奖**：nominal/measured均1792/1792、fault0、standing0/256；measured head small3(保持1)、current111(107)、两轴156(144)、四轴3(3)，parent1856为1(0)/114(112)/171(160)/11(6)。1°尾保持small28→34、current90→93，但两轴217→165、四轴13→5；neck+MAE1.555→1.605°、roll+.729→.907°。单项微弱改善不能覆盖组合退化，保留1856。正式50,331,648controls/560.655s=89,772.89/s，实际32768准入稳态131757.93/s；完整物理流水线1201.595s，CPU实现/单位另计。报告、错误分项和全部失败/未选尾保留。
- **当前保留micro1856**：measured头小1(保持0)、current114(112)、两轴171(160)、四轴11(6)；nominal0(0)/114(114)/155(149)/5(4)。nominal/measured均全恢复/0fault/standing0，但仅开发256条standing，不能外推连续完整倾斜域；父级共同真实P04环境同[-10,-5]pitch/roll±10的1024条比较已完成：micro1856跌倒0、baseline T24-4274跌倒2（均首次稳前），两者最终全稳定/0fault。报告`runs-v2/paired-negative-pitch-routes-4274-1856/report.json`；物理源相同，捕获qpos/qvel/rawobs/origins逐位配对，未扩为90态全覆盖或硬件证明。头部尚未通过；原严格T90、return及末保持不改，.5°/1°额外统计是诊断，不冒充用户新验收。
- 已结算完成正式段 **2,530,738,176** controls；control64已全部计入，当前仅评测，cap005已启动正式进程，实际新更新以status/诊断落盘为准。保留micro1856谱系1,459,617,792；control1920待验收，谱系1,509,949,440。cap完成后再据真实更新合并，未把预检/诊断算正式。
- 从随机初始化独立谱系持续推进，未加载baseline actor/N/Adam；v5/CAD/S0/O0资产、200Hz/50Hz不变，合法反馈不趋势去尖峰。只训练评测，不部署或轮滑，不宣称硬件/完整sim2real成功。

## 历史快照（2026-10-16 02:35，优先于下面历史记录）

- GPU0 已完成 roll-only 1856→1920 的64更新及全部评测，PID3292643/session10607、tracker7481均为历史。nominal/measured 恢复各1792/1792、fault0、standing0/256。头控small3(保持1)、current110(110)、两轴171(160)、四轴6(4)；目标roll+从micro1856的7/16降到3/16，MAE .729→.909°，roll−MAE .470→.543°。故不延长此因素，保留micro1856参考；正neck改善1.555→1.095°不能掩盖目标轴退化。
- roll-only正式完整进程552.419s，50,331,648控制交互，91,111.32/s；启动/早段并行冻结1856 nominal头评，真实wall包含竞争。32768准入稳态131521/s与全进程口径分开。
- 有限冻结诊断已全部完成：同micro1856、nominal、完整23项奖栈；mean→仅腿Gaussian：小0(保持0)→0(0)，current118(117)→117(115)，两轴153(145)→210(195)，四轴4(3)→12(10)；mean→当前实际全Gaussian：小4(3)、current122(119)、两轴236(228)、四轴20(16)。90项捕获初态经raw_registers显式回放全配，0fault/0倒，权重/N不变；直接头噪声为0只适用于legs臂同一步同观测，后续闭环mean会随状态变。所有条件微幅仍未学成。共同命令前站稳子集，两轴仅腿尾MAE .689→.578°、完整step回报27.7086→27.5598；head项提高被身体/动作成本抵消。不由此认定静摩擦因果或直接cap腿。
- **当前唯一新训练因素：独立命令轴精度附加奖**。源`rise-head-accuracy-worktree`，task `Mjlab-StandUp-Flat-MicroDuck-V2-Fresh-M4-HeadStdCap001-MicroExposure025-AccuracyBonus05-Repaired-S0-O0-R1`，从保留micro1856完整恢复，预算64至1920；T24、micro.25、head cap.01、腿σ、旧Head2整项、冻结N、Adam和实际LR不变。新项weight .5，score=mean_active(1/(1+abs(physical_error)/max(.1*abs(command),.002)))；idle四轴平均，避免静止轴补偿被命令轴。没有替换原Gaussian或削弱其远端梯度。CPU/GPU单位session66194已PASS（旧head源码与输出逐位、正负/idle/阈值/不活跃轴/命名物理轴和float64 oracle），64×5成熟PPO/timeout/ONNX已PASS，physical oracle120调用最大差1.79e-7，LR3.375e-5/clock44544精确继承、无新LR切换。32768×16及optimizer护栏已PASS：中位131757.93/s，最大meanKL .026628/clip .307726，physical oracle384步最大差1.79e-7。正式PID3366906已完成1920；nominal1792/1792、fault0、standing0/256，完整formal560.655s/50,331,648controls=89,772.89/s（稳态另计）。session2264已自动接measured头评PID3382992，再measured恢复/standing；tracker57722保留完整wall/谱系。头结果待出，不以新reward或起身替代头控完成。真实smoke按一次RewardManager缓存核对独立物理oracle，完整错误和分项保留。
- 保留micro1856：measured头小1(保持0)、current114(112)、两轴171(160)、四轴11(6)；nominal小0(0)、current114(114)、两轴155(149)、四轴5(4)。两条件均有微幅/roll缺口，不能归于只有measured电压才出现。nominal/measured全恢复各1792、standing0，head尚未合格。
- 正式随机根实际累计 **2,480,406,528** 控制交互；当前保留micro1856谱系 **1,459,617,792**，其它对照/尚未选择尾部 **1,020,788,736**。新accuracy1920候选谱系1,509,949,440，头/实测验收未完暂不选；所有失败与未选段保留。最新凭据`head2-accuracy-allocation-001/completed-zero-lineage.json`。预检/冻结诊断不算正式训练。
- 本谱系始终从随机初始化独立续训，不加载baseline权重/normalizer/Adam；v5/CAD/S0冻结、O0合法测量不去尖峰，200Hz/50Hz不变。目标仍全姿态恢复+倾斜承接+头部响应/末端保持，未宣称实机或完整头控成功；只训练评测，不做部署、轮滑。

## 历史快照（2026-10-16 01:58，进程状态已被下方新结果覆盖）

- 低熵128续段 `head2-late-entropy-allocation-002` 已完整结束1856，PID3097659/session89442为历史进程。1856起身1792/1792、fault0、standing0/256；头控small3(保持3)/current99(98)/两轴134(125)/四轴0，低于保留1728的2(1)/102(100)/158(156)/5(5)。正neck误差2.645→4.019°，实际σ已在.038–.040rad平台，**不再盲续低熵**。
- 当前选择低熵1728为下一因素父点，1664仍保留组合参考。纯headσ≤.01候选已完成64×5和32768×16，但**优化稳定性准入FAIL、正式更新0**：实际N首pooled KL5.0805/clip.5094，后续才恢复.008–.016；132325.5/s吞吐通过不能覆盖优化失败。旧源`rise-head-stdcap-worktree`和失败source tar全部保留。
- cap+LR1792完整measured头评为small3/128(保持3)、current105/128(104)、两轴164/384(155)、四轴11/64(7)；parent1728为2(1)/102(100)/158(156)/5(5)，微幅与组合保持仍未解决，不能宣布头控成功。neck+8/16、MAE2.190°（parent5/16/2.645°），roll+4/16/.822°仍偏差。measured恢复也1792/1792、fault0、standing先倒/稳后倒0/256。保留1792作下一阶段父，1728/1664仍保留参考。
- cap+LR正式进程544.334s、50,331,648交互=92464.6/s；本次完整物理流水线（64smoke→32768bench→正式→formal恢复→head→measured恢复）1152.269s，另含旧纯cap失败研发链，未宣称全链10万。实际32768稳态132342/s，与此清楚分开。
- 微命令.25首64已完成1856：measured head小1/128(保持0)、current114/128(112)、两轴171/384(160)、四轴11/64(6)，final两轴212/384(209)、final四轴0。当前neck/pitch/yaw各方向到位均16/16，neck+保持14/16、其余保持16；roll−11/16、roll+7/16。正neck MAE2.190→1.555°，roll+.822→.729°，但小幅各轴仍欠响应约1.2–1.7°，**微幅问题未解决**。nominal/measured恢复各1792/1792、fault0、standing先倒/稳后倒均0/256。
- 冻结micro1856的nominal同81case补测已完成：small0(保持0)、current114(114)、两轴155(149)、四轴5(4)，measured相应1(0)/114(112)/171(160)/11(6)。nominal neck+15/16/MAE1.562°，roll−13/.419°、roll+6/.680°；说明当前微幅与横滚短板也存在于训练测量条件，不是只三电压验收才出现。两条件未做全部初态tensor回放配对，不作电压差额因果结论。
- **当前推进roll-only精度候选**：新源`rise-head-roll-fine-worktree`、task `Mjlab-StandUp-Flat-MicroDuck-V2-Fresh-M4-HeadStdCap001-MicroExposure025-RollFine002-Repaired-S0-O0-R1`，从1856完整状态继承，64预算至1920。只把head_roll fine_std .04→.02；neck/pitch/yaw仍.04，micro .25/cap .01/entropy.005/冻结N/当前LR不变，无新LR切换，也未同时加入四轴近零精度项。CPU/GPU非roll分项奖励和梯度逐位相同、旧无override公式逐位、命名轴置换、独立float64 oracle均PASS。64×5与实际32768×16/optimizer护栏全部PASS：中位131521/s、整体131837/s，max meanKL.01577/clip.2048；**正式PID3292643/session10607已启动1856→1920/64**，无重复LR切换。session7481计wall/交互/standing，段末head/measured自动接。另session40881在吞吐准入后补冻结1856 nominal同头评，使用原micro源并行，后续formal wall如实包括竞争。准入不代表横滚行为已改善。

- **下一最小因素为微命令曝光**：独立`rise-head-micro-exposure-worktree`，task `Mjlab-StandUp-Flat-MicroDuck-V2-Fresh-M4-HeadStdCap001-MicroExposure025-Repaired-S0-O0-R1`。只引入precision_probability=.25（幅度.005–.04rad）；zero仍.6，活跃指令75%保留原全范围，轴组合/正负抽样规则不改；Head2 fine.04、cap.01、entropy.005、冻结N、actor/critic/Adam及当前LR不变，**不再次施加切换LR**。父为cap+LR1792，预算64至1856，无新同父64控制，仅作阶段进展比较，不声称纯因果胜利。
- 源复用父级已验precision sampler，新增实际每轴微幅正负及全微幅组合曝光计数（GPU累计）。CPU/GPU配置/百万采样、64×5和32768×16/optimizer护栏已全部PASS；32768中位132562.3/s、整体132761.1/s，最大meanKL.03030/clip.2091。**该段已完成1792→1856/64，原PID3252183/session22332为历史**，输出`head2-micro-exposure-allocation-001`，随后head/measured自动评测；session84185跟踪全wall/交互/站立先倒顺序。继承LR7.59375e-5，明确未再次切换LR。CPU配置测试首次把float0.0设成int0导致JSON规范化比较失败，已改测试恢复相同float类型，失败日志保留；生产采样行为没有因此修改。准入通过不等于微幅头控已改善。
- 不收窄fine .04→.02：真公式exp(-(e/std)^2)下，当前neck+2.190°附近加权误差梯度7.304→1.973（只剩27%），会削弱未掌握方向；roll+.822°则5.958→16.123。离线报告`head2-cap1792-fine-scale-sensitivity.json`，未改奖励。旧baseline2000 direct-target为开环头目标且部分失稳，不能作为当前P04可达精度下界，也不重复冒用。

- 分布实现仅在自有Gaussian子类update构造实际Normal前统一限制；采样/log_prob/entropy/KL/storage同分布，原std_param及Adam精确继承，limit非持久buffer不污染checkpoint key，确定性导出仍Identity。rawσ和effectiveσ分字段记录；原冻结Gaussian诊断直接拿rawσ的方法不适用于cap新任务。
- 完整Head2奖栈只读审计1728/1856已完成：23项与真实formal+polish权重精确一致，逐步读取RewardManager一次计算缓存，未重复有状态head_bias EMA；每seed90捕获初态tensor回放配对（raw_registers显式复制），脚本与报告保留。neck+变差对应完整step return26.9896→26.7486，head/standing/腿项均下降，不支持删腿奖励。两轴总体return几乎持平，有一明确组合退化被身体收益小幅抵消，未外推全部奖励反向。
- 同1728全动作Gaussian诊断：neck+fullreturn26.9896→27.0335略高，但small/current-pair/idle回报均低于mean；不是全局采样更优，也不能把腿噪声影响归为头σ。只支持有限head-only cap试验，父代理GPU1另做head-only冻结诊断。
- 低熵128段真实100,663,296交互，**完整进程973.227s=103,432.50控制交互/s**（包括导入、初始化、两次formal eval），中位更新132,062.47/s；短64首段完整进程92,803.09/s，明确口径，不混报。
- 正式随机根实际累计 **2,379,743,232** 交互；当前下一阶段父micro1856谱系 **1,459,617,792**，其它对照/尾部 **920,125,440**；保留1728和1664独立参考。本段真实新增50,331,648。完整formal进程544.334s=92464.6控制/s，短64段全进程未达10万；未把预检/诊断算正式训练。所有失败候选保留；仅训练评测，不做部署/轮滑，不声称实机成功。

### 2026-10-16 00:48 cap准入的FP32重算诊断

- CPU与GPU直接分布/实际RolloutStorage的同均值重算严格ratio1/KL0均PASS。成熟64环境PPO的首minibatch KL mean1.55e-10、max5.96e-8，ratio最大偏差.0003337（0.03337%）；原严格零assert正确留下失败文件，未签准/正式训练。
- 新监控补首mean/std重算最大差，实际effective std仍要求逐位相同；KL上限1e-6、ratio相对偏差≤.002明确作为FP32异batch重算容差，报告名称改为distribution_consistent，不伪称零。同真实父actor的64/384 batch重算诊断并行；新的64×5与32768准入均须真实过检查。
- 首64小批量更新内部pooled KL20.35/clip.818是真实优化结果，也完整保留；不能把它等同首更新前分布错配。正式32768是否仍过强待大batch准入/实际行为证据判断。

- v2第一个真实更新分布容差检查已通过，随后新增diagnostics包装timed_log时丢失原logger签名而KeyError `it`。修复为`@functools.wraps(original_log)`，只恢复日志签名；原smoke-v2日志/源码保留，没有正式交互。v3 smoke/v3 bench以同生产class重跑，队列session60903。
- 实际32768×16额外optimizer准入护栏：单更新meanKL≤.5/clip≤.8、末5更新中位KL≤.1/clip≤.5；越界存FAIL并阻止正式。此为避免明显优化破坏的前置筛查，非头控验收。若实际N也过强，必须另列新配方而不能暗调LR后仍称单因素。
- 异batch直接复现：真实1728 actor、相同按既有normalizer标准化的合成输入，64/384 batch的mean最大差9.54e-7rad、std逐位相同、KL0、ratio差.000164。该输入不是物理轨迹，不夸大诊断覆盖。

### 20:48 native2 首64结果与有界PPO归因

- 1472同64更新：standing256/sitting202/prone235/frontier256/left256/right256/supine256，总1717/1792、fault0；对比旧域sitting14和fault59，修复有效但行为未恢复到1408全过。
- 原每桶每seed70%guard通过，只代表不触发最低故障停机，不冒充完整恢复验收。基于反复极端KL及已学坐姿/俯卧退化，另行在1480安全保存停止（未跨1500 COM加难）。保留checkpoint_001480及全部72更新56,623,104交互；随机根实际总1,623,982,080，未抹除失败尾。
- 立即接session38206 `native-read-preupdate-audit-001`，同1408、同32768环境/24步/配方，外置审计脚本不改运行源。逐步比较动作采样时obs快照与storage缓存、当时/当前normalizer更新前KL，记录20次minibatch的KL和LR；只做一次临时更新、不保存成候选模型，算诊断消耗。尚未归因或改变PPO。

### 20:55 已定位rollout内统计漂移，冻结成熟normalizer单因素准入

- 同1408精确32768×24的临时诊断：每步act前原始obs快照和storage缓存逐位相同；同一实际样本用采样时normalizer重算KL严格0，用rollout结束normalizer则在Adam前KL mean0.73093/p95 3.56017/max40.97507。首minibatch尚未更新即mean0.7285；随后20 minibatch扩大到约2.8。normalizer count1,107,296,256→1,108,082,688，最大std变化.014。证据 `native-read-preupdate-audit-001/report.json`。诊断全程内部对照严格同样本，但与先前formal独立物理rollout未宣称逐轨迹配对（pooled KL分别2.323和1.527）。
- 新隔离源 `rise-frozen-normalizer-worktree`，唯一训练因素是显式 `--freeze-normalizers`：actor/critic既有mean/var/std/count原值全部保留，RSL已有until=count禁止继续统计；每次runner.train()仍受until保护。每个正式更新检查冻结张量逐位不变，checkpoint/run保留recipe，已有冻结父若漏CLI拒绝续训；不改actor/Adam/LR/reward/曝光/测量。PPO telemetry新增首minibatch KL和更新后LR，不改优化运算。
- CPU生命周期两项PASS，覆盖train()、梯度更新后统计不变、重载重新应用冻结、拒绝未学习count0。session2609自动64×5真实timeout/ONNX→32768×16准入→1408同64预算学习，未签GPU PASS前不开始正式。准入源hash写 `frozen-normalizer-qualification.json`。
- 此为针对本谱系已证实的rollout统计漂移对照，不外推父级GPU1头遗忘由同机制导致。先比较恢复保持、首minibatch/全更新KL及head，不以loss下降代替行为。

### 20:08 前已完成的头部精度阶段（历史状态）

### 先前Head1阶段记录（历史）

- 用户当前范围：只专注模型训练/评测，不做P4/host部署或导出工程。P04必须从随机初始化学成，本谱系从未加载baseline actor/critic/normalizer/Adam。合法测量不去尖峰，冻结v5/S0/O0、CAD及时钟不改；无其它动作或轮滑。
- **当前GPU0正式续段PID2703439/session36888**，head1同配方1280→1408已实际1300，GPU0稳定计算，输出fresh-p04-head1-allocation-002；工作树rise-zero-m4-worktree、任务DirectionGate-M4-Repaired-S0-O0-R1。原PID2662275/001已完整256更新结束1280；该点nominal恢复1792/1792、fault0。head-precision1、32768×24/5epoch/4minibatch、stage10/polish均不变，预检135,737.5/s有效。
- 冻结1280实际measured恢复1791/1792（sitting255，其余满）、fault0、standing先倒0/256。头控measured current107/128（保持107）、两轴132/384、四轴1/64、final单轴59/128；small仍3/128。neck负向已16/16，正向仍0但误差13.47→8.12°，pitch/yaw双向满、roll27/32。整体仍在进步，继续到1408再定精度/微幅曝光；不是头控完成。
- 实际恢复PASS：派生resume SHA `b8762408528ee183b714838dbd8377507831db204c80e679ad0aec7f970600e1`，actor/critic/normalizer/Adam逐位恢复，source_changed=false，LR=.00038443359375、common_step24576。32768 reset全部stage10/polish，head中幅及五项polish权重已实查。每64正式恢复检查；128更新节点1152的measured恢复/头控/末端保持已由session92562自动排队。
- 只读补充对照session37480：原M4自然续段1152同measured头控/保持。1024刚跨1000扩幅，不能把它的current失败当充分曝光后的平台；将与head1-1152同新增128预算比较，但两者还存在polish/课程状态/episode初始化差异，不做纯fine奖励因果结论。
- head1首64更新1088：nominal最终0–30°及全七桶1792/1792，feedback_fault0；standing同口径先倒0/256、最终稳定256/256。1152和1216也继续1792/1792；1152实际measured也1792/1792、standing先倒0/256。
- head1-1152头控：measured small4/128（尾保持2）、current87/128（保持86）、两轴78/384、四轴0；nominal对应2/88/70/0，fault均0。缺口在nominal同样存在，不能归因三电压。current neck±各0/16（正误差13.47°、负2.84°）、roll+8/16，pitch/yaw各方向16/16；继续256预算，依据1280评测再定精度/曝光。
- 自动续接已排：session36888在1280全姿态门槛保持后，同head1续128到1408（不跨1500 COM加难）；session81535同步冻结1280做measured恢复/头控/保持。无新增源修改，不空等评测。原自然续段1152头控measured small21/128、current9/128、两轴11/384、四轴0，作为相同新增128预算对照；非严格单因素。
- **头控明确尚未通过**：1024 measured 81case×8×2，全部无fault；small单轴T90 12/128、末端.5s保持2/128；current单轴0/128、两轴0/384、四轴0/64；final单轴11/128且末端保持0。同baseline物理reward时刻和到位/保持判据，原指标与独立同时到位oracle一致。证据fresh-p04-1024-head-measured及-summary。
- **基于1024真实全窗与measured覆盖转polish**：fresh-p04-1024-covered-polish/resume_001024.pt是独立派生文件，原checkpoint不改。严格核对同SHA、源、两种子/每桶128、方向至少32、各≥70%及无fault（实际全100%）；把剩余窗口机械2400步等待替换为已经完成的全覆盖证据，单独curriculum-transition.json记录，没有伪造普通晋级历史。除recovery/eval-history及coverage_transition元数据外，actor/critic/normalizer/Adam/LR/RNG/clock/controls逐位相同。
- 下一段进入既有polish：action_rate -.2→-.5、arrival_damping0→-.025、torque_rate0→-.0005、head_bias0→.5、body_tracking0→.5，并启用历史headprecision1。属于能力门控阶段转换加头精度配方，**不声称单因素因果实验**。保留1152自然扩幅对照；该点1781/1792(sitting245)，fault0、standing先倒0/256。
- 新head1前真实随机谱系累计1,215,037,440训练交互；选定1024谱系805,306,368；其它对照/未选尾部409,731,072。002实际896→1152共201,326,592，未选1024→1152的100,663,296单列；证据fresh-p04-head1-lineage.json。未把预检或只读评测伪算成有效训练。
### 19:01 前已完成的恢复阶段（以下为历史状态）

- **1024冻结覆盖评测完成**：stage4–10七个剩余前沿窗口每窗1792/1792，总12544/12544，逐方向/逐种子均100%，fault0；实际measured R1最终stage10=0–30°也1792/1792、fault0。actor/normalizer逐位未变，源码结束核验一致；证据fresh-p04-1024-frontier-sweep和fresh-p04-1024-measured。1088正式再次1792/1792、standing先倒0/256、fault0。尚未把1024覆盖证据冒用为1152验收，也尚未修改训练阶段/开启polish。
- **1024正式七桶1792/1792、fault0**，standing先倒0/256，stage3=105–135°两方向两种子全过；达到2400步后按原门槛晋级stage4=90–120°。1000后的reset已实际扩大head到±.39/.39/.49/.11，未改课程。1024 checkpoint SHA `10cbb3995f9b32c7643623d74e82793ad3c6e44b01cacf7784bd404a18467c69`。
- 只读并行queue session2695：冻结1024原formal evaluate扫剩余stage4–10、每窗2×7×128；随后final stage10 measured R1恢复，再同baseline3506的81case×8×2头控measured评测。真实S0 compiled、sensor_error_reset保留、actor/normalizer逐位不变检查；主线PID2627938继续1152，不阻塞训练。若所有窗口已通过，将依据覆盖证据处理机械阶段等待，当前尚未改变训练状态。
- **960首次七桶接近全过：1789/1792、fault0**，sitting253/256，其余standing/prone/frontier/left/right/supine均256/256；105–135°前沿逐种子逐方向全部100%。stage3尚未达到最小2400步，正确暂不晋级。仍需后续持续性、剩余前沿覆盖、头控和measured R1评测，不能将此称最终sim2real完成。
- 站立保持同口径只读审计：896、960各256条，t>.2s且physical tilt>45°且root height<.08m的先倒均0/256，最终稳定均256/256、fault0。结果在各evaluation_evidence/standing-retention.json，未重跑模拟或改变策略；对比早期baseline适应3648的27/256缺口已有本批实测改善。
- **同128更新M4对照已有实质收益**：896 M4为1391/1792、fault0，standing/prone/left/right首次均256/256，sitting124，supine仍0。120–150°frontier243/256；正向90.8/89.6%、负向两种子100%，按独立方向门槛晋级stage3=105–135°。不是完整起身训练成功，背姿与坐姿未完成，头控/实测R1尚待后续。
- M16同父768、同128更新只686/1792，左右侧均0；额外到960仍平台，原M16实际停969并完整保留。M4只改num_mini_batches16→4：32768×24、5epochs不变；每batch49152→196608，80→20次Adam/LR判断，都是每rollout3,932,160 sample-visits。未把任一称为天然官方等价；首rollout不是逐条物理轨迹完全配对，单训练种子不能推出普遍结论。
- M4首128正式实际新增100,663,296控制步。更新中位6.08s≈129,347/s；PPO更新段含首图准备累计817.73s≈123,101/s。run.json→完成939.64s（含模型/恢复/评测）≈107,130/s；包含Python导入的实际启动wall保守939.64–990.50s，对应仍≥101,628/s。详001/actual-wall-time.json；只读奖励审计曾并行，不伪称孤立基准。
- 方向门槛继续严格：每种子×正负各≥70%，窗口须2400步；当前stage4时钟24576。最终坐姿/两侧/背条件不放宽，不由已学右向抵消左向失败。阶段表(150,135,120,105,90,75,60,45,30,15,0)。头命令已按原1000课程在reset扩大至±.39/.39/.49/.11rad，继续监控保持与响应。
- normalizer四臂完成：right63/64(W768/N768)、64/64(W768/N832)、0/64(W832/N832)、0/64(W832/N768)，fault全0。大退化随权重而非本次统计版本，未改normalizer。reward排序审计已完成归档并停止：成功right两种子79.83/78.75，失败6.58/7.04；gamma=.99均值13.80/.61。站立复合/直立/稳定项排序正确，action-rate成功−.187、失败−.295，不支持尝试税压过收益。没有归档逐条初态，不能称严格轨迹配对或所有奖励均已验证。
- 002前随机根实际1,013,710,848交互，选896谱系704,643,072，其它实验/未选尾部309,067,776全部留账；002/zero-lineage.json含来源/哈希/累计消耗。没有抹去M16/先前课程实验。
- GPUreset几何候选、异常拒绝方案、部署草稿均不进入主线。GPU1由父代理推进baseline头控/随机化/迁移。现在保持M4连续训练，不新增泛化诊断或奖励修改。

### 从零第一段及已完成准入（历史）
- 独立工作树 `runs-v2/rise-zero-worktree`，task `Mjlab-StandUp-Flat-MicroDuck-V2-Fresh-Repaired-S0-O0-R1`。历史队列session75364 / PID2423620：64×5和32768×16均PASS，`fresh-p04-allocation-001`计划0→1024，实际在277保存后因课程平台转新配置。15:43曾实际5更新/3,932,160交互、inherited_controls=0、GPU0约96%。起点无resume/adapt-from/head-precision。
- `fresh-initialization.json` PASS：checkpoint=null、网络seed20261115、Adam state entries0、actor normalizer count0、clock/iteration/recovery stage0、discovery=true、polish=false、head_precision=null。初始状态序列化SHA `10749e37d0f45adf04a25b8f609a868ea68de3b891b92a293d1419f0eb27554f`。这条谱系没有加载基线策略状态；v5/S0/O0冻结测量资产照常保留。
- 32768×24、5epoch×16minibatch实际13稳态端到端中位**132,131.36控制交互/s**、采样155,082.98/s、整体132,056.36/s；正式初始更新约5.93s。课程CPU检查2项PASS/30.60s；首轮测试漏任务入口导致KeyError的失败日志保留，修正入口后配置与阶段门槛检查通过。
- 从零发现课程：standing/sitting/prone=40/30/20%，其它姿态4/2/2/2%，没有完全未曝光的困难动作；双种子三易项均≥70%且≥2400控制步才退出发现阶段；随后standing保留20%、sitting/prone各10%、frontier30%、侧/背各10%。后续roll升级仍用原独立门槛与最小阶段时长，未放宽恢复成功判据。
- 从零PPO保留T24与5epochs，N32768用16 minibatches，使每minibatch49152与官方N8192×24/4一致；每更新80次Adam，必须通过该实际配置的新吞吐准入，不能套旧4-minibatch报告。
- **已被最新用户指令覆盖的历史要求**：用户此前要求最终前端采用**异常拒绝+有限时长预测**。父代理独占`feedback-tolerance-worktree`实现/验证采样间隔、运动变化、不确定度、有效历史保留、预测不回灌、恢复校正、超限invalid、指定查询时刻等。当前valid-raw-local-history是过渡对照，不代表最终反馈修复完成；验证后只从本随机谱系续训，绝不加载baseline谱系。
- 当前从零已140更新/110,100,480交互，GPU0约94%，每更新约6s，fault0。64开发评测0/1792（4条曾稳定）、无故障；128未经奖励修改已**726/1792**：standing220/256、prone251/256、frontier255/256，其余桶0。坐姿未过，discovery正确保持，未自动升级。
- 128近站姿态相比64明显改善：prone/frontier高度≈.1144m、倾角≈11°（原23–26°）；颈偏置仍约71°（原≈110°），不满足头部效果。站立扰动后先倒约35/256。继续到192再按真实趋势决策，不能把早期0成功直接断言为无法逃出的奖励死结。
- 备用单因素候选位于`rise-zero-bias-worktree`：只把历史已有的、近站立门控L1头部bias项提前到发现阶段，weight=.5，1s EMA；不改Gaussian、动作税或地面回转。实际64轨迹反事实剂量为frontier约−.21/s、standing−.16~.18/s，而未起身sitting/侧背为0。历史门控/落地清空测试4项PASS；64×5/超时/ONNX smoke PASS。但因128自然进步明显，**尚未正式采用**，未换奖励或中断主线。
- 原适应支路实际停在3793，保存`repair-local-o0-adapt-001/checkpoint_003793.pt`，新增165更新/129,761,280 P04适应交互；保留为基线→修复P04对照。站立曝光候选64×5 PASS，但未正式训练，不能把阶段5/head-precision配方套到从零阶段。

### 先前适应对照（已停止，不是当前从零主线）

- 旧GPU0正式进程 PID2391499/session39607已停止，工作树 `runs-v2/rise-agent-worktree`，输出 `repair-local-o0-adapt-001`；3712/3776开发恢复均1792/1792、fault0。
- 3648在GPU1 measured R1恢复1769/1792（98.72%）、fault0；当前幅度8个单轴头部两种子全部到位，组合/大幅度尚未通过完整审计。不能将恢复率外推为站立保持、头部全指令或实机通过。
- 新发现：3648和3712两节点，站立桶都有27/256先倒后起（pitch/roll初态各±10°、高度高于接触面、初始速度0）。最后站稳标准不足以评价“保持站立”；单列抗初态扰动保持与已稳定HOME保持，旧恢复门槛不改。
- 下一单因素候选在独立 `runs-v2/rise-retention-worktree`：已掌握阶段站立曝光5%→25%；其它6桶始终保留，frontier40%→25%、supine15%→10%；初态难度、reward、head、检测器、S0/v5均不变。64×5预检session65669与主训练并行在GPU0小批量运行；尚未启动该配方正式训练，等3776节点及预检证据后决定。父代理的GPU1继续CPU/GPU真实sim2sim工作。

## 目标与约束

- 推进真实 P04 S0/O0/R1 起身学习，按七类初态、双种子物理稳定性和反馈健康评测；不能用总 reward、更新数、删故障或放宽门槛冒充成功。
- V2 CAD、冻结 v5 电机、HOME/轴向/观测动作契约不变；没有实机，未来实机对齐训练契约。
- 每个训练预算段有检查点、源码快照、预检和真实消耗；根据评测及时修复课程，避免只延长失败配方。GPU0 连续推进，GPU1 由父代理做相关修复实验。
- 不修改正在运行的冻结源，也不覆盖父代理正在编辑的诊断与测量文件。

## 接管时的证据

- 原完整 P04 已完成 257 更新，累计 201,523,200 控制交互；192/256/257 开发评测均 0/1792，257 反馈故障 692/1792，stage 0 未晋级。
- 257 TensorBoard：stable_standing 始终 0；平均 episode 约 5.8–6.2s；策略动作 std≈0.816；action_rate 项约 −2.46/s。多数物理正奖励已经较高，不能仅把失败归于故障计分。
- 父代理位置/速度分离：固定基线3500，在无尖峰P04测量中仅替换20ms延迟物理q可得1508/1792，仅替换v为2/1792，两者1585/1792；这是真值输入诊断，不是可部署方案。

## 当前执行

- 历史进程 PID2372973已停止，`standup-allocation-004-32768`，冻结工作树 `runs-v2/frozen-source-allocation-002`，原计划从257追加256至513，实际在328保存后转修复版本。
- 14:52 已实际完成258更新；启动前约6分钟CPU初始化，不把这段算GPU利用率或吞吐。正式采样保持32768×24、5epoch×4minibatch、200Hz物理/50Hz控制。
- 下一节点320检查真实成功率/故障与训练指标；若仍陷入局部解，先做实际末态诊断，再在隔离版本修复，不盲目续同配方。

### 15:04 修复分支已进入连续GPU准入队列

- 用户追加明确要求：本代理也要解决反馈故障，不能机械续训浪费有效episode。320开发评测实际仅3/1792成功、585/1792有故障、76条曾短暂稳定；已在更新边界停止，最终328 `checkpoint_000328.pt`保留，累计257,359,872交互，stage0。未消耗到原计划513。
- 父代理的基线3628位置鲁棒性候选在原完整P04名义1228/1792、measured1023/1792；禁旧尖峰后1304/1181且fault0。明确选作**跨任务适应**起点，保留原从零对照，不假称它是P04血统。
- 口径修正：3628自身的1792/1792沿用了旧baseline play配置、位置0步延迟，因此只是旧名义回归，不能称为完全匹配新训练的1步位置延迟。父代理正在补齐；上述独立完整P04迁移结果确实走S0反馈链，不受此问题影响。本代理新任务的play与train均走其注册的实际S0/O0修复版本，不叠加baseline位置噪声/延迟。
- 新任务 `Mjlab-StandUp-Flat-MicroDuck-V2-Repaired-S0-O0-R1`，执行版本 `p04-o0-valid-raw-local-history-v1`：接受有效原生测量包，移除无硬件证据的8°预测尖峰判死规则；O0保留5个当前有效采样准备LS，取消其不使用的19帧/O1 220ms重预热。S0上下文、1024候选donor、冻结O0 affine、真实timestamps与v5/CAD均不变。原始valid/count/NaN、单调时间、PID/命令、输出限速/数值、query支撑/新鲜度门槛全部保留；不注入物理真值/预测回退。
- 新CPU测试8项PASS/2.70s：成熟O0输出与旧值逐位一致；O0准备点由第22个tick改第4个tick；坏包、越界、NaN、乱序仍使query无效且必须5个新有效采样才能重新产出；有效快速阶跃保持真实原始值；O1禁止短历史。
- CUDA检查2项PASS/69.08s：普通/高速/坏值、完整/部分/补齐reset，eager与compiled+fused+graph全部因果状态逐位一致，故障和恢复路径实际被覆盖。
- 已启动连续顺序队列session39607：`repair-local-o0-smoke` 64×5/真实超时/ONNX → `repair-local-o0-throughput-32768` 32768×16 → `repair-local-o0-adapt-001` 256额外更新、每64双种子7桶评测。任一检查失败自动中止正式阶段。显式 `--adapt-from baseline/.../resume_003628.pt`，保留head-precision2、Adam/normalizer和stage5；基线位置噪声配方明确由完整S0测量替换，不双计噪声。此时正式新更新尚未声称完成。
- 15:10准入队列全部通过并自动进入正式适应，PID2391499/session39607。64×5/真实超时/ONNX PASS，CPU归一化导出最大差1.79e−7；32768×16有13个稳态更新，端到端中位**135,147.61控制交互/s**、采样154,401.35/s、按稳态总wall 134,890.45/s，中位5.819s/更新。此为新修复执行版本的实测，不覆盖原P04旧报告。正式首个节点3648（新增20更新），后续3712/3776…；新正式输出 `repair-local-o0-adapt-001`。
- 15:12已实际完成3632，新增3,145,728 P04交互；713,293,824继承交互属于基线。actor/critic/normalizer/Adam恢复逐位PASS，LR=.000170859375，common_step=87072；重置后的恢复stage5、head-precision2范围保持。现场GPU0为93%，训练约5.93s/更新、fault0。
- 15:14首开发评测3648：**1791/1792成功，fault0**；standing255/256，其余6桶各256/256。仅增加20个P04更新（15,728,640交互）。不可变checkpoint SHA `ee96812da356b5e2b22f160be38b0ddb131356902402da747d0af1c7ecb38ef9`。一次开发评测接近全过不等于完成，继续3712等节点，GPU1并行measured R1与头部复测。
- 3648真实物理轨迹：各桶末秒平均高度≈.112m，tilt P95 3.05–3.68°，角速度P95 .22–.25rad/s；零命令头部四轴均值绝对值≤.28°。唯一失败条目之前曾稳定，末端0.5s连续稳定未达标；保持原判据。诊断保存在 `evaluation_003648_evidence/physical-diagnostics.json`。
- 15:13:58用户看到的1%利用率对应双种子小规模评测及约120MB轨迹压缩写盘，未卡死；15:14:27已3650且94%，完整episode400步，feedback_fault0。评测仍必要，不将瞬间低利用率或PID存活冒充吞吐证明。

## 启动空档修复实验

- 新隔离工作树 `runs-v2/rise-agent-worktree` 基于070d606，恢复allocation002的source.tar.gz；其与正在训练的冻结工作树独立。
- 小改动仅针对 V2 terrain：在 mjlab 建模前跳过原本最终会被删除的viewer origin sites，避免32768个site创建/附加/逐个删除。非V2配置保留原行为，不改共享依赖文件。
- CPU独立验证正在运行：64/1024/4096/8192世界旧/新构建时间；要求最终XML、所有编译模型数组、环境坐标逐项完全相等，另验证非V2仍创建sites。证据 `runs-v2/rise-agent-origin-marker-check`；尚未写PASS或用于正式训练。
- 结果更新：首次因standup重新构造官方terrain导致断言失败，失败证据保留；补上standup末尾配置后，`rise-agent-origin-marker-check-v2/report.json` 全部PASS。8192构建13.986s→0.01179s；64/1024/4096/8192每组最终XML、484个编译模型数组、origins精确相等；非V2继续生成原site。已用于上述新队列，旧正在运行进程未热改。
- 实际启动比较：run.json落盘到首PPO日志，旧004为353.24s，新修复适应为89.11s；文件 `actual-startup-comparison.json`。该差值包含建模/图准备/恢复/首更新，不能当作纯origin单因素测量。

## 待做

1. 查320评测和真实末态，定位稳定性缺项/头颈局部解/课程曝光。
2. 对确有依据的课程或奖励改动做64×5 smoke与对应吞吐准入，保留旧模型对照。
3. 接收父代理GPU1鲁棒性修复结果，选择有证据的迁移候选；完整P04与测量消融始终区分身份。
4. 学成后仍需冻结配方的新种子、CPU MuJoCo/ONNX闭环和最终实机验收；开发评测不是sim2real完成证明。

## 16:55 同前沿失败与reset初态几何诊断

- 512轨迹20ms后的前沿姿态：120–130°几乎全失败、140–150°全部成功，失败后进入supine盆地。20ms是实际首步状态，不冒充精确reset角。
- 使用注册play cfg的Scene.compile、实际启用碰撞mask及全0 HOME独立CPU静态核对，120/135/150°、root z=.05的地面穿透约2.5/3.1/3.3cm（头壳），z=.07仍约.5–1.3cm。原始XML粗审计与真实训练mask复核均留存；物理模型未改。证据fresh-p04-reset-clearance-audit/actual-training-model-report.json。
- 尚未证明穿透是平台原因。只读512策略A/B已启动session53876：同初态随机数，只将实际穿地初态抬到1mm间隙，CPU几何只在reset计算一次；后续真实S0/GPU闭环、策略和归一化不变。该CPU诊断不是拟采用的正式GPUreset实现，结果fresh-p04-512-reset-clearance-diagnostic。当前训练不受源码热改。

- 512只读穿透修正A/B已完成：总666→668，frontier145→153、prone256→253、sitting9→6、standing仍256、侧背0，fault均0；actor/normalizer逐位不变。808/1792初态穿地，最大33.36mm，包含94/256站立和200/256前沿。结论：几何缺陷真实，但仅修初态没有明显解决当时学习瓶颈；不夸大微小差异。
- 独立rise-zero-clearance-worktree准备按实际floor碰撞mesh最低点抬升的GPU实现，只在reset作用，不是测量替代。第一次几何测试因编译模型terrain占body1、自由根实际body2而显式断言失败，尚未进入环境/正式训练；修复为按freejoint真实body id解析，保留失败日志。当前正式源不动。

## 17:06 今晚候选交付优先级

- 用户要求今晚优先获得多套起身候选比较训练时间与实机指标。GPU0持续本随机初始化谱系，先做全姿态恢复、站立保持与头部响应，不能用baseline适应3648代替从零，也不能把尚未掌握侧背的当前节点说成已合格。
- 每次runner.save会覆盖同目录同名ONNX。合格候选需冻结确切pt+ONNX、源/配置/契约及零谱系，CPU对照PyTorch确认61→14和归一化烘焙，并区分开发评测、CPU sim2sim与真实机器验收。
- GPU reset最低点候选已通过256姿态的独立MuJoCo oracle，最大误差7.88e−8m；CUDA graph结果逐位一致，非穿地初态和除z外状态完全不变。证据fresh-p04-reset-clearance-geometry-v2/report.json。8676个实际floor碰撞mesh顶点，没有改模型。尚未完成完整环境smoke/吞吐准入，**未进入正式源**；为今晚候选目标先停在此验证状态，不阻塞改善中的训练。

## 17:17 重叠前沿与连续执行

- 704退化后先等768验证，768俯卧进一步降至13.3%，才停止旧配置。保留640成功蒙太奇及704物理诊断；没有因为单一早期低分就断言永久死锁。
- 重叠课程不改变最终目标：stage2从90–120°改为105–135°，后续15°逐级推进，最终仍0–30°和独立侧/背/坐姿通过才能polish。课程判断显式记录evaluated_frontier_low_deg与next_frontier_low_deg，避免不同任务的stage编号被混淆。
- CPU5项PASS/41.04s；64×5真实训练/强制超时/ONNX PASS。串行自动队列已进入32768实际配置准入，任一失败不会启动长训，预检后自动续选定随机谱系，不等待额外许可。

## 17:30 重叠课程首评及PPO诊断

- 704重叠配方prone252/256、right251/256，旧配方同704为106/0；新105–135°前沿48%，当前不再引入其它训练因素，继续768。
- PPO观测到同步400步timeout附近KL异常：重叠新段更新17的KL mean3.19/max minibatch3.78，平常约.007–.016；更新34 critic MSE10.86。旧非重叠656/657也曾KL20.28/12.46，baseline位置随机化两段没有>0.1峰值。仅相关性，不声称timeout为已证实原因。RSL normalizer count为int64且until=None，尚无冻结/计数溢出证据；未来若反复退化需区分normalizer漂移、return/advantage分布、更新前与更新后KL，再做独立episode-age错峰对照。当前PPO保持原样。

## 17:42 分方向门槛修复

- 640真实trace拆分右120/124、左69/132；重叠704右123/124、左0/132，768右115/124、左2/132。依据足够指向聚合门槛过早升级，而不是奖励/PPO不足；保留每种子原始轨迹与统计，不把方向失败混为随机总失败。
- 独立direction工作树只增加reset方向记录、eval方向统计、方向独立晋级门槛和改变窗口的显式计时语义。用已学好prone/right的768继续，在120–150°补未掌握方向。源数据、v5/CAD、奖励和PPO不变。
- 8项CPU门槛回归PASS/45.34s；64×5/真实强制timeout/ONNX PASS。自动连续队列先32768准入，随后从768显式adapt，不等确认。旧重叠实际停805，额外37更新29,097,984控制步保留；不是只计到768。

## 17:44 最新范围收敛

- 用户明确不要先做P4或host部署，当前只做训练与评测。停止把候选导出/固件当本轮下一项；没有正在进行的本代理部署或导出进程。父代理也已停部署工程。
- 17:43:49 GPU0实际72%、预检稳态5.90s/更新；正在训练而非卡在CPU构建。预检→正式仍为同session62723自动连续队列，后续模型/图重建短暂空档须按真实状态报告。

## 18:05 方向课程平台与normalizer因果拆分

- 832：standing256/prone253，right_side从父768的254降到0，前沿120–150°128/256；两方向逐种子40–59%，未升级。896：standing256/prone254、sitting57，left/right/supine均0、前沿119/256（正49.2/44.8%，负41.3/50.8%），fault均0。门槛修复正确阻止假晋级，但未解决权重遗忘。
- 固定网络W768/W832与normalizer N768/N832的4臂只读同种子评测，每臂2×7×32：right依次63、64、0、0（分母64），fault全0；W832换回N768也不能恢复。standing各64，prone63/63/63/61，frontier41/41/26/25。所有参数/统计逐位不变。证据fresh-p04-normalizer-768-832/report.json。
- 结论仅为右侧退化主要随网络权重，非本次normalizer版本变化；不外推所有历史遗忘或PPO的具体原因。当前高KL峰并非退化必要条件（本段没有>1峰），不继续凭猜测改normalizer。
- 下一有限单因素候选为32768×24仍5epochs，只把16 minibatch/80次Adam改回官方4 minibatch/20次Adam；同768父点、同方向课程/奖励/反馈与128更新预算对照。16保持官方单batch样本数的选择并不自动保证学习效果等价，adaptive LR也按minibatch调整。尚在准备候选，未称其更优或启动正式。

## 18:13 M4单因素进入实际准入

- 原DirectionGate M16到960仍standing256/prone255、sitting72、frontier125、侧背0，fault0。安全停在969（非960），实际201新更新158,072,832交互全部保留。
- M4恢复同768，128更新预算与M16的768→896作主要对照，M16额外到969只作补充趋势证据；不把多花预算的结果混成等预算比较。
- CPU唯一差异检查第一次因为独立Entity callable对象repr含不同内存地址误报；保留失败日志，canonical只屏蔽地址而保留名称/所有参数后PASS/42.04s。实际训练配置没有因此修改。64×5/真实timeout/ONNX PASS，32768真实准入自动连接正式。

## 18:19 M4正式启动与比较口径

- run.json实际18:14:49.856，resume核验18:15:43.606、环境/RNG/新窗复位18:15:44.962；之后进入真实PPO。queue的bash通过exec变为PID2606253，ps显示的PID存活时长包含前面预检，不能误当全部为正式初始化。尝试py-spy当前栈被ptrace权限拒绝，未声称已取得18:15:33低利用率时的调用栈。
- 同parent768 actor/critic/normalizer/Adam逐位相同，CPU配置校验只有minibatch差异；但首rollout回报聚合量并非逐位一致（M16 return variance71.20862/M4 71.11602，MSE3.40599/3.39006）。记录first-rollout-comparison.json，不能将本实验称逐条轨迹完全配对或由一个训练种子推出普遍结论。主要比较仍为同检查点/同配置种子、同交互预算的学习效果。

## 18:39 M4同预算结果与原配方续训

- 896：1391/1792、fault0，四个standing/prone/left/right桶首次全256；sitting124、supine0。两个方向独立通过120–150°门槛，晋级105–135°，保留精确checkpoint和另存的resume元数据。
- 同源继续896→1152，PID2627938/session2775，父checkpoint/resume SHA和真实零谱系保留；未因M4结果再改reward/PPO/DR，亦未声称实机或全头控通过。
- 已停止normalizer和reward只读审计。奖励各项只读reward_manager每步原计算结果，逐项累计×dt与returned reward总和核对通过；未重采样有状态reward。按桶/项/首秒/折扣分别存档，不能用未保存的逐条初态声称严格配对。

## 19:07 全恢复覆盖后进入头部精度阶段

1024从零谱系首次完成全部剩余角窗和实际measured恢复，训练继续到1152的原检查点完整保留。头部中幅T90仍0，因此没有宣称起身/头控整体完成。覆盖派生只改课程元数据，自动预检后从同1024状态进入既有polish和headprecision1；无baseline权重、无新增测量噪声、无固件部署。

## 19:31 派生状态描述字段澄清

覆盖派生resume严格验证1024原件与所有窗口/方向/种子完整通过，模型、Adam、RNG、clock未改。但新eval_history描述性counts/rates沿用了父checkpoint的旧960 history（sitting253/256），同时stage已写10；这部分描述字段不参与覆盖判定。保留已消费父文件不可变，在fresh-p04-1024-covered-polish/history-field-clarification.json给出真实stage10全256规范计数，修正生成器未来输出；head1实际1088/1152正式评测已覆盖为当前正确统计。无模型/训练门槛影响，不重启训练。

## 19:54 区分近站姿接住失败与已稳定后再倒

只读standing-order.py按原fall(t>.2s、tilt>45°且z<.08m)与保存的严格连续.5s stable标志做事件顺序审计。旧适应3648的27/256跌倒全部在首次有效稳定之前，已站稳后再倒0；P041024与head1-1280 measured两类均0。初态是HOME关节、root高度.11638–.12638m、pitch/roll±10°、根速度0，不能称已静置平衡。现有remaining_fallen(-.5)已用_fallen_tax_armed锁存，tilt>40°触发，tilt<20°且height>STAND_HEIGHT-.010释放；没有曾稳定后再跌专属一次性惩罚。当前奖励保持不变，父代理独立核查基线source/target和新种子。脚本scripts/analyze_v2_standing_order.py可复用，结果各evidence/standing-order.json。

### 20:58 冻结统计预检初始化口径失败保留

- 第一64×5用了旧diagnostic initial-checkpoint模式：只load actor/critic/normalizer，Adam/LR和stage/clock仍fresh。第1更新value1.82，第2约5.82e20、第3inf，队列正确中止，未进入正式。`frozen-norm-head1-smoke.log`保留，不能以CPU功能测试代替GPU数值稳定性。
- 新增显式`--mature-resume`验证模式，先逐位核对完整actor/critic/normalizer/Adam，再恢复LR、common_step、recovery stage/polish及reset；不偷换原initial-checkpoint含义。session42657重跑64×5真实timeout→32768×16→同父1408的64更新。若精确恢复仍失败，再按实际机制收窄控制变量，当前未声称已通过。

### 21:00 成熟恢复预检通过

- `frozen-norm-head1-mature-smoke/report.json`64×5 PASS：完整父态/Adam恢复逐位、LR .000170859375、clock33792、stage10/polish；32实际timeout。五次更新所有冻结actor/critic统计逐位不变，critic value loss .542/.683/.335/.049/.009有限；ONNX CPU最大差7.15e−7。
- 新队列session42657自动进入32768×16准入，尚未签长批次PASS。保留不匹配初始化的失败日志，只能说明正确恢复后本路径通过，不能将旧失败未经消融地归到单个Adam或课程因素。

### 21:04 冻结统计长批次准入通过，正式已自动衔接

- `frozen-norm-head1-mature-throughput-32768/report.json`16更新PASS；13稳态中位 **130737.43控制交互/s**、整段130878.09/s、采样148892.21/s，中位6.015s。使用真实成熟1408网络/Adam/课程，冻结actor/critic所有统计16更新均逐位不变，ONNX CPU最大8.34e−7。
- qualified marker `frozen-normalizer-qualification.json`已PASS，带最小文件SHA及64/32768报告；父级可复用机制但各自实际配方仍需独立验证。session42657自动开始正式1408→1472，当前CPU导入/构建，未虚报首更新。仍同head1/PPO/曝光；先看首minibatch KL与同64恢复，头评另行。

### 21:08 正式冻结统计对照运行与首轮反证

- 正式PID2821942/session42657，`frozen-norm-head1-allocation-001`。完整父1408恢复逐位PASS，冻结counts均1,107,296,256，source_changed=true；新学习controls照常增长，不能把冻结normalizer.count当训练交互数。
- 首更新first-minibatch KL严格0，证明已消除采样内统计漂移；但20次Adam内部pooled KL mean7.133/max460.30、actor梯度pre_max63845.9（实际clip后1），比动态统计首更新更大。故不能称冻结统计已完成优化稳定性修复，数值finite仍继续约定64行为对照。
- 至1430，除首更新外其余mean KL最大.0511；同步episode节点1424=.0441，所有first-minibatch KL0。是否保持起身/头控仍待1472，下一调整若必要才针对转换首轮optimizer步幅，不混改reward或直接判定GPU1同根因。

### 21:18 同配方续段已1482，保持原预算

- `frozen-norm-head1-allocation-002`完整恢复PASS、source_changed=false，parent resume SHA cb357a6c2d7e2b9428bee8e383649a12f75f5f82c23ac569b3761b2f31609822。继承LR .0002562890625，首1472更新first-minibatch KL0、pooled1.8586、actor pre_max347（clip后1），不是已消除所有optimizer大步。
- 现场GPU0 92%，实际1482，继续约定128更新至1600，1536检查坐姿/侧姿保持及故障；不因单个高KL反复中断已出现行为收益的配方。父级GPU1在自家吞吐准入后自动接1472 head measured+common heldout1591/1592。
- 正式冻结64段实际run.json→结束496.63s/50,331,648交互=101346.41/s（含初始化与最终评测，不含更早imports）；更新中位6.04s=130203.97/s。native2动态72同口径102601.91/s；其SIGINT末更新先保存再退出未正常logger，logged71与actual72已区分，不能把全部72除以71次日志耗时。

### 21:26 1536恢复保持与1472实际反馈/头部结果

- 1536正式1792/1792、fault0；standing256均稳定，先倒/首次稳定后再倒均0。已跨1500原COM随机化加难，继续128预算到1600，head1和PPO不变。
- 父级实际同修复源1472 common measured新seeds1591/1592：1789/1792、fault0，left126+127，其余满；standing先倒0/256。这是新种子实测条件泛化，仍不是硬件验收。
- 1472 measured头控：small4/128(末端4)、current97/128(97)、两轴103/384(100)、四轴0/64，fault0。neck正仍0/16，稳态MAE两seed4.492/4.355°；负13/16，MAE约1.916°。相比1408 current101/两轴134/四轴3，头部总体未证明改善，不能拿恢复1790当头控完成。
- 暂不把fine_std .08直接收紧.04：在当前正neck误差约4.424°，单关节窄Gaussian奖励对物理误差的导数，.04仅为.08约24.5%；交叉点约3.12°。这只是奖励函数解析梯度，不是已测policy gradient或通用课程晋级门槛；更大误差当前仍需要可见梯度，1600再根据真实head误差决定。解析记录frozen-head1-1472-reward-scale-note.json。

### 21:42 1600头部短板与Head1/Head2同预算单因素

- 1600实际measured head：small2/128(末端1)、current87/128(86)、两轴91/384(87)、四轴0/64，fault0。正neck首次1/16、MAE3.293°（较1472约4.424°继续降低）；负neck4/16/2.384°、roll+5/16/.772°，pitch/yaw四方向各16/16。恢复全过不能外推头控完成。报告frozen-head1-1600-head-measured和-head-summary。
- 当前head1控制臂1600→1664已完成更新，正在正式恢复评测；session67764将在评测完成后安全停，保存实际可能多1更新尾部，再顺序新Head2 32768×16准入→从同1600父点重新训练64至1664。只改变fine_std .08→.04，其余环境全字段、stats freeze、Adam/LR/PPO、head范围/零命令/组合曝光、S0误差均不变；不继承控制臂1664权重。控制臂1664头评自动与Head2正式启动并行，完整保留控制算力。
- 全env递归配置diff PASS，唯一差异environment.rewards.head_pose_tracking.params.fine_std；首次diff工具拼接int字典key失败，改成str(key)后复核，生产代码未动，旧log保留。当前转阶段理由记录frozen-head2-config-diff.json；helper沿用旧“2000-update”reason文本只属历史模板，不作为本次决策证据。
- Head2成熟64×5实际超时/ONNX已PASS，统计全程逐位不变；正式仍等待新32768准入，未套Head1预检。当前从1600到1664的Head1控制和后续Head2将按同64预算比较恢复、每轴误差、组合到位/保持，不只看reward。

### 21:48 Head2准入已通过并启动正式同64对照

- Head1控制1664正式1792/1792、fault0；实际安全停1665，65更新51,118,080交互全部保留，末尾1更新不混作匹配预算。累计实际随机根1,826,095,104，Head2选用同1600谱系1,258,291,200，未选对照/尾部567,803,904。
- Head2成熟32768×16 PASS；13稳态中位131087.26控制交互/s、整段131133.42/s、采样149316.98/s，normalizer统计逐位不变，ONNX最大5.96e−7。session67764自动开始`frozen-head2-allocation-001`1600→1664，fine_std .04；Head1控制1664 measured head并行评测。无源码热改、无body/standing课程变化。
- 完整父来源/配置单因素/各预算均已写`frozen-head2-controlled-comparison.json`及新run `zero-lineage.json`；正式首更新和终端表现尚待验证，不能把准入或更尖奖励当成功。

### 21:56 Head2同64恢复全过，头部验收并行

- Head2 1664正式1792/1792、fault0，stage10七姿态保持。64更新first-minibatch KL全部严格0，pooled mean KL最大.06505；Head1同64控制最大2.27892。此为训练稳定性趋势，不能代替头部动作指标，也未凭此称Head2赢家。
- Head1控制1664 measured head：small3/128(末端3)、current94/128(93)、两轴99/384(94)、四轴0；正neck0/16/4.098°，负12/16/2.035°，roll+4/16/.811°。相对1600未出现整体突破，控制证据完整留存。
- session23473已启动`frozen-head2-allocation-002`1664→1792有界128，PID2903092；同GPU并行冻结Head2-1664 measured head→独立同时到位summary→measured全7桶恢复/standing-order。若头明显退化，1728节点重新决策，不把继续训练冒充验收。
- Head2首64新增50,331,648正式交互，实际随机根总1,876,426,752；新选1664谱系1,308,622,848，未选控制/尾部567,803,904，所有budget/源/父checkpoint在zero-lineage。

## 21:56 状态快照（历史）

- **GPU0 Head2同64已恢复1792/1792、fault0，正接续1664→1792有界128**，session23473/PID2903092，输出`frozen-head2-allocation-002`，实际源`rise-frozen-normalizer-worktree`；Head2-1664 measured head/恢复正在同GPU并行只读评测。头部指标尚未据此宣布胜出，1728依据实际结果复核。Head1同1600起点控制1664完整保留，详21:56记录。
- 1536/1600 nominal恢复均1792/1792、fault0，standing先倒和稳后再倒均0。1600 measured head small2/current87/两轴91/四轴0，正neck MAE3.293°且首次1/16，但负neck/roll精度下降，头控未完成；因此开展单因素精度对照，不能无限延长Head1。
- **同64更新结果**：native2动态统计1717/1792（sitting202/prone235/其余满）；冻结统计1790/1792（standing/sitting/prone/frontier/right/supine各256，left254），fault均0。旧读域动态为1546/1792、59fault，不能把这两因素混作一个比较。1472新checkpoint SHA `a0eeac0ef22a2d58517789688aeb75ae7cf04a7ec703d640a08e05f69469b556`；完整合格恢复1408仍保留，头控未完成。
- 已证实rollout内统计漂移：缓存obs逐位正确；采样时N重算KL0，rollout结束N的Adam前KL mean.73093/max40.975。显式`--freeze-normalizers`用已有until=count固定actor/critic原mean/var/std/count，runner.train()不改变冻结；每更新逐位检查、recipe入run/checkpoint、防静默丢模式。权重/Adam/LR/head1/reward/曝光未改。
- **不能说PPO全部稳定**：冻结首minibatch KL均0，但首更新内部mean7.133/max460.30；1441/1458内部仍有.3448/1.1100峰，更新前0。统计漂移和optimizer大步并存；现行为改善有实证，保持有界预算，不为了每个高KL立即改多个因素。续段继承末LR .0002562890625，首轮要继续监控。
- 新源CPU生命周期2项、成熟64×5/32真实timeout、32768×16全部PASS；13稳态中位130737.43控制交互/s，actor/critic所有统计16更新逐位不变，ONNX最大8.34e−7。第一旧actor-only+freshAdam/stage0预检inf失败日志保留，后明确完整mature-resume验证通过，不覆盖失败或硬作单因素归因。
- native2原生read域0–1023/目标域20–1001修复继续保留，冻结v5/S0/O0、CAD、坏包/NaN/真越界/时间/LS检查不变；合法尖峰不做趋势门控。raw=-4来源是连续S0统计生成器越域，实际编码/饱和语义另待审，不冒称硬件坏包，不凭猜测钳位。
- 1472 standing严格顺序审计：256条均完成连续.5s稳定，.2s后先倒0、首次稳定后再倒0；只读真实轨迹，同既有门槛。
- 1472→1600将跨原有1500 COM随机化.01→.015，明确记账，不将后续加难效果伪装成纯冻结统计因果。父级GPU1按新源接1472 measured/head，等待其自家预检结束，当前未声称1472头控或measured通过。
- 正式随机根已花 **1,674,313,728** 交互；当前选1472谱系 **1,157,627,904**；未选对照/尾部 **516,685,824** 全保留。新002继续另计；验证/临时诊断不冒充正式学习。只训练评测，不做部署/轮滑/其它动作，无baseline学习状态混入。


### 22:54 Head2 平衡对照结束与下一曝光因素

- 平衡1856实际三电压 head 结果没有全面好于同预算plain1856，更未超出保留1664组合参考；正neck MAE3.832°（plain1856 5.768°、1664 2.918°），仍0/16。不延长这条平衡奖励支路。证据head2-balance-1856-head-measured、-summary，源码与actor/N不变。
- 普通1792 nominal small2/current101/两轴114/四轴2，measured4/99/114/2；plain1856 nominal1/99/125/2，measured3/98/119/2，均fault0。主要短板在训练名义条件同样存在，不能主要归于三电压评测差异；初态未逐位配对，数个百分点差别不作电压因果推断。
- 新候选隔离源rise-head-exposure-worktree从普通Head2源复制。已有实际1664曝光零约60.02%、单19.97%、双10.01%、四9.99%；相应幅度/方向已记录，不能说其完全未曝光。现有限假设是降低易idle占比可增加运动任务优化权重，先采用保守.6→.3（初稿.2尚未验证/训练，改.3后才准入），保持其余幅度、20%端点、非零1/2/4比例、head奖励、body/standing/PPO/N/测量全部不变。
- CPU/GPU配置diff和采样验证要求只有zero_command_prob变化；同RNG原有非零幅度逐位保留，新增非零而非改噪声/幅度。成熟64×5及32768准入后方可正式同1664→1728共64更新，对照原allocation002的1728及1664静态参考，idle与完整恢复单列。

### 23:00 曝光候选完成准入并自动接正式

- 成熟64×5 PASS，32真实timeout，actor/critic统计逐位冻结、ONNX CPU最大7.15e−7。32768×16 PASS：13稳态中位131383.66控制/s、整段131816.61/s、采样149193.52/s，中位5.986s；同1664成熟状态与新.3命令曝光验证，不冒用旧预检。
- queue73612自动恢复1664并执行64更新到1728，随后同源measured头评/summary，GPU0无其它吞吐污染。`head2-exposure-qualification.json`含源SHA及三份验证报告；`head2-exposure-lineage.json`含完整真实消耗。
- 当前选1664谱系1,308,622,848，已花正式2,077,753,344，未继承对照/尾769,130,496（包括原1664→1792/1856及balance全部预算）。本段新增最多50,331,648另计，不抹除未选权重。
- plain1856控制正式run.json→结束495.53s=101571.5/s；balance64为497.29s=101212.8/s；均含构建/最终恢复、不含更早imports。balance pooled KL max.0656较plain .337更稳，但动作指标未整体改善，未将其作成功证据。

### 23:03 曝光正式恢复与真实计数

完整1664 actor/critic/N/Adam逐位PASS，resume SHA33beff2ce6b43e39e95910fb29d981e8881023949e1bae47e903cb9213dff184，继承LR.0002562890625000001、clock39936。首更新pooled KL.03203，前8更新first-minibatch KL全部0；实际曝光零30.42%/单34.97%/双17.21%/四17.40%，没有只改配置而未生效。PID3009430、session73612，预算仍64至1728，待真实行为验收，不把利用率或曝光达标当头控成功。

## 23:00 状态快照（历史）

- GPU0 新 Head2 活跃/静止轴各50%平衡候选已完成同1792→1856的64更新：`head2-balance-allocation-001`，PID2961926已正常结束，实际新源 `rise-head-balance-worktree`。正式恢复七桶1792/1792、fault0，**头控仍未通过，平衡奖励未证明整体胜出**。新候选measured small3(末端1)、current96(96)、两轴125(117)、四轴0；恢复standing先倒/稳后倒0/256。接下来的有限候选从较好1664出发，同Head2只把零命令.6→.3，stage6显式记录，独立源rise-head-exposure-worktree，64至1728；CPU/GPU、成熟64×5、32768×16准入均PASS，queue73612/PID3009430已完整恢复并正式RUNNING，已实际1670、GPU0约93%；原1728同预算控制measured small2/current101/两轴111/四轴1，fault0。
- 新stage5明确是`v2-head-precision-2-active-balance-v1`，基于本谱系Head2 Gaussian(.5/.04各半)，不是父级Head4/Cauchy的stage5。只改单活跃轴时活跃组与其余组各50%，idle/两轴/四轴逐位保持；命令零概率.6、1/2/4轴曝光、范围、PPO、冻结统计、测量/电机/standing课程不变。
- 语义CPU/GPU、完整env单因素diff、成熟64×5真实timeout/ONNX、32768×16均PASS；13稳态中位130363.59控制交互/s、整段130281.50/s，qualified marker `head2-balance-qualification.json`已PASS。前一64控制使用普通Head2源，未混用新奖励oracle。
- 保留最好组合参考Head2-1664 measured：small3/128(末端2)、current97/128(96)、两轴156/384(148)、四轴3/64(2)，正neck MAE2.918°；nominal/measured恢复均1792/1792、fault0，standing0。不能只因新模型iteration更高就选它。
- 普通Head2-1792 measured为small4(3)、current99(98)、两轴114(106)、四轴2(2)；再64到1856对照为3(2)、98(98)、119(115)、2(2)。1856正neck仍0/16、MAE5.768°，恢复仍1792/1792。延长原配方未获整体进步，故作有界新奖励对照。新候选必须同时比较同预算1856和保留1664。
- 本谱系归因审计并未复现父级的聚合平均奖改善但活跃轴下降的整体趋势：1664→1792 active和overall多数同时下降；仅部分case有补偿排序。平衡奖励是有依据的有限假设，不能预先称已找到全部根因，也不能声称50/50从数学上杜绝所有补偿。
- 冻结成熟normalizer继续显式入checkpoint/CLI，所有统计逐位不变、first-minibatch KL0；这消除了已证实的rollout内统计漂移，未宣称消除所有optimizer大步。native2反馈read0–1023与command20–1001分离保持，合法尖峰无趋势门控；原始S0生成器raw=-4来源审计保留，不偷换为真实硬件故障。
- 正式随机根累计 **2,077,753,344** 交互；本新1856候选谱系 **1,459,617,792**，其它对照和未选尾部 **618,135,552** 全留账。各段新增`completed-zero-lineage.json`，预检/诊断/评测另计，无baseline学得状态混入。只训练评测，不做部署、轮滑或其它动作；无硬件验收结论。


## 23:14 状态快照（历史）

- GPU0曝光.3同1664+64已完整1728并自动评测结束：恢复1792/1792、fault0、standing先倒与稳后倒均0/256；measured head small3/128(末端2)、current96/128(95)、两轴116/384(106)、四轴2/64(1)。普通同1728为2(2)/101(101)/111(103)/1(1)，没有整体突破；保留1664组合156/384仍较好。没有以单项改善或最新iteration冒称成功。
- 当前GPU0只读冻结诊断自动队列session4907/57391：保留1664与新曝光1728，各做mean与Gaussian两臂，同81case×8×2、实际measured S0，独立采样RNG，严格保存并核对初态qpos/qvel/raw obs/CPU+CUDA RNG和可达S0 tensor state。脚本`runs-v2/diagnose_fresh_p04_head_sampling.py`在运行工作树外，不改训练源/策略；配对与行为结果尚待完成。随机穿过目标不能代替末端持续保持，不部署随机动作。
- 已核实GaussianDistribution std_type=scalar，actor的std_param直接为σ，action scale1。1664头部σ=.04998/.05260/.05532/.05434rad；新1728=.05059/.05440/.05566/.05428。后续降低熵/探索尚未授权成确定方案或实施，先依当前诊断而非外推旧baseline结果。
- .3曝光实际零/单/双/四约30/35/17.5/17.5，除零命令.6→.3外env配置全字段不变，CPU/GPU/成熟64×5/32768×16均PASS。稳态131383.66/s；正式64的run.json→结束492.56s=102183.1/s，更新中位5.99s=131290.8/s，含构建/最终恢复、不含更早imports。
- 先前Head2 balance同1792+64也未整体胜出：small3/current96/两轴125/四轴0；plain1856为3/98/119/2。普通1792和1856 nominal与measured缺口相近，主因不能直接归三电压。两条新试验均完整保留，不叠加失败因素盲续。
- 下一精度乘项只做离线分析，未实施：precision_scale=.003会令e=.003/.01rad的物理reward敏感度提高22.89×/2.07×，但当前neck2.918/3.832/5.768°只剩.575/.578/.649×；不是通用升级，也不是实际policy gradient结论。文件head2-precision-sensitivity-note.json。
- native2读数0–1023/目标20–1001、冻结v5/S0/O0/CAD、合法尖峰无趋势门控、成熟normalizer显式冻结继续保留。随机根正式已花 **2,128,084,992** 交互；曝光1728候选谱系1,358,954,496，保留最好1664谱系1,308,622,848，其余对照/尾部均有completed-zero-lineage账；没有baseline学得状态混入。只训练评测，不做部署或其它技能，无硬件合格结论。


### 23:39 低熵准入通过且正式运行

- 成熟64×5 PASS，live entropy=.005、32真实timeout、N统计逐位不变；32768×16 PASS：13稳态中位132503.24控制/s、整段132239.06/s，采样150890.56/s，中位5.935s。`head2-late-entropy-qualification.json`保留源码与各报告SHA。
- 正式PID3072045/session9752，首1665更新pooled KL.030797、first-minibatch0；父1664 actor/critic/N/Adam逐位PASS，LR.0002562890625。仅算法entropy变更，未手工重置σ；run.json完整agent配置写明.005，新任务ID显式区分。
- `/proc`实时记录正式Python process start1792078635.1，段末可计算包含imports的完整进程墙钟；此前run.json墙钟明确不含更早imports，两个口径不得混称。已将process-start和zero-lineage写入正式output。

## 23:32 状态快照（历史）

- GPU0下一唯一因素为**成熟阶段entropy_coef .01→.005**，不是手工改σ；新源`rise-late-entropy-worktree`、任务`Mjlab-StandUp-Flat-MicroDuck-V2-Fresh-M4-LateEntropy005-Repaired-S0-O0-R1`。同保留1664精确actor/critic/normalizer/Adam/LR恢复，Head2奖励、零命令.6及全部环境/课程不变，64至1728对照原普通1728。全env/agent配置CPU检查PASS；queue9752前置验证均PASS，正式PID3072045已实际1665、GPU0约93%，64更新后自动measured头评。完整parent actor/critic/N/Adam逐位恢复，live entropy .005；不把诊断当降低熵一定有效。
- 当前Gaussian/mean诊断已完成：明确初始raw_registers历史环显式恢复后，每seed90捕获tensor逐位相同，涵盖qpos/qvel/raw actor obs/全局CPU+CUDA RNG/可达S0 tensor；不声称所有私有Generator对象或任意模拟器内部状态全捕获。报告`head2-frozen-sampling-replayed-v3-report.json`，脚本`diagnose_fresh_p04_head_sampling_v3.py` SHA761d6272abe0fe053312a8d6d356288af1a1358ee6c583d6d157a6028b9ddffe。只读诊断state replay，不是生产反馈修改。
- 1664严格捕获态配对mean→Gaussian：small T90 2→108/128但末端保持0→0；中单98→128(保持96→98)；两轴144→378/384(保持137→86)；四轴0→53/64(保持0→0)。正neck稳态signed bias−2.897°→−.195°、MAE2.897°→.541°。因此均值偏差也确实改变，不只是随机穿过阈值，但随机持续保持不合格，不能用于部署。
- 新曝光1728 mean→Gaussian：small3→105(保持2→0)、中单92→128(90→95)、两轴127→372(115→65)、四轴2→54(1→0)；neck signed−2.938°→−.205°。全部fault0、身体倒地0/1296。诊断mean与早前formal头评有少量差别，沿用各自保存初态，不覆盖原报告或假称跨报告初态完全一致。
- 前一.3曝光64结果恢复1792/1792/fault0/standing0；原正式measured head3/96/116/2，对照普通同1728为2/101/111/1，未整体胜出。Head2 balance同1792+64也未整体改善。保留最好组合1664（原协议两轴156/384）和所有对照，不继承失败新因素叠加训练。
- 诊断失败证据完整保留：第一版PAIRING_FAILED唯一不一致为raw speed/load历史环（q/status/time/history/output相同），未据“无影响”冒称PASS；v2初态复制后变量target覆盖导致形状错误，中止评测；v3改raw_target并检查evaluate_seed头target只有一处赋值后重跑完成。另v3首次启动早于源写入的失败log已另存launch-before-source，未算训练或覆盖证据。没有因此修改训练模型/生产源。
- 正式随机根累计仍 **2,128,084,992** 交互；本次新父1664谱系1,308,622,848，未选对照/尾部819,462,144全保留。曝光1728候选谱系1,358,954,496只作保留候选。diag/preflight算力另记，不混作正式学习；当前不做部署/轮滑/其它技能，未声称实机或完整头控验收通过。


### 2026-10-16 00:02 中间节点与诊断边界

- 同低熵续段1792正式恢复1792/1792、fault0；头部sigma=.03742/.03979/.03987/.03822rad，实际source_changed=false、actor/critic/N/Adam精确恢复。冻结1792 measured头评与主训练并行，主线已1800、GPU0接近100%，未改训练参数。1728完整measured恢复与standing顺序审计均全过且先倒0。
- raw_registers分配和reset代码明确zeros，因此并非未初始化张量；历史speed/load差异的生成机制仍未定位。O0 q/v数值由line_q3及LS3 slope得到，不数值消费raw speed/load，但这些历史字段仍进入isfinite检查，不能称完全未使用。其它89捕获张量及有限性匹配；该初始ring显式回放只用于诊断，不是生产修复。
- 同父低熵/普通64正式首rollout return variance14.348248/14.362422、MSE.630075/.629504有微小不同，正式rollout没有逐样本配对证据；配置唯一因素与精确父态不等于每条物理轨迹逐位一致。对照为单训练种子的有界实证，不能作普遍因果或实机保证。

<!-- HEAD_COMPARISON_BEGIN -->
## 当前候选同协议头部比较（2026-10-16）

均为实际P04 measured、81case×8×2、物理reward边界评测。每格是“到位条数/末端.5s连续保持条数”，分母由列头给出，全部fault0；不是百分数或部署资格。1664是静态保留参考；1728对照同1664+64，balance1856对照同1792+64，不能把所有行混成同预算实验。

| 候选 | 小单128 | 中单128 | 中双384 | 中四64 | 大双384 | 大四64 |
|---|---:|---:|---:|---:|---:|---:|
| Head2保留1664 | 3/2 | 97/96 | 156/148 | 3/2 | 213/204 | 3/3 |
| 普通Head2控制1728 | 2/2 | 101/101 | 111/103 | 1/1 | 193/184 | 9/9 |
| 普通Head2 1792 | 4/3 | 99/98 | 114/106 | 2/2 | 211/204 | 14/14 |
| 普通Head2控制1856 | 3/2 | 98/98 | 119/115 | 2/2 | 200/194 | 12/11 |
| 平衡奖励1856 | 3/1 | 96/96 | 125/117 | 0/0 | 180/177 | 13/12 |
| 曝光.3 1728 | 3/2 | 96/95 | 116/106 | 2/1 | 176/174 | 13/13 |
| 熵.005 1728 | 2/1 | 102/100 | 158/156 | 5/5 | 230/223 | 0/0 |
| 熵.005 1792 | 3/2 | 103/103 | 147/141 | 5/4 | 244/236 | 11/11 |
| 熵.005 1856 | 3/3 | 99/98 | 134/125 | 0/0 | 212/205 | 14/14 |
| 头σcap.01+切换LR1792 | 3/3 | 105/104 | 164/155 | 11/7 | 217/213 | 0/0 |
| 微命令.25 1856 | 1/0 | 114/112 | 171/160 | 11/6 | 212/209 | 0/0 |
| 仅roll细尺度1920（未选） | 3/1 | 110/110 | 171/160 | 6/4 | 211/202 | 3/2 |
| 精度附加.5 1920（未选） | 3/1 | 111/107 | 156/144 | 3/3 | 199/190 | 0/0 |
| 同父微曝光无改动控制1920 | 2/2 | 113/112 | 153/138 | 6/4 | 204/196 | 5/5 |
| 同父head cap.005 1920 | 3/3 | 116/114 | 165/150 | 8/3 | 229/221 | 2/2 |
| cap.005继续64 1984（安全停） | 2/2 | 114/113 | 162/148 | 4/3 | 222/218 | 2/2 |
| 首次timeout相位1984 | 2/2 | 109/108 | 152/140 | 4/1 | 213/205 | 2/2 |

证据索引：`runs-v2/p04-head-comparison-20261016.json`。完整原报告与另做捕获初态回放配对的Gaussian诊断分开保留；第一次配对失败也另存。
<!-- HEAD_COMPARISON_END -->

### 02:28 诊断输出目录与元数据修复

- 全Gaussian补臂首次在0物理步时FileExistsError退出：为纠正v3通用元数据“零头噪声检查”作用域，我提前把sidecar写进了脚本要求新建的输出目录。失败日志`head2-micro1856-fullreward-all-gaussian-directory-failure.log`保留。sidecar已移到外部，另存v4把标记限定为sampled且legs；未修改已通过legs原件。重启session80774完整PASS，v4 SHA `2e2d9ebb7c8204df02e20a6b781faeeaf3209de4631a65c085d5ba4a9c87bfa1`。行为公式和实际σ未改。

### 精度附加项的指标边界

- 新项band使用**当前目标绝对幅值**，与本头评0→step的max(.1*|target|,.002)一致；不是跟随命令增量重建动态T90。idle时四轴均用.002rad。原return T90仍按原commanded_step幅值报告，新项idle定价与return相对到位带不同，不改原验收、不把两者宣称同一指标。所有误差均物理输出q−HOME−当前命令，保存未裁剪error/单轴score；奖励不引用带噪actor测量代替物理输出。
