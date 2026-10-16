# 当前：历史完整课程1536与七节点双条件评测已结束，完整技能未合格

全部Roulade正式754,974,720；GPU0已接Ground512，Roulade没有活动训练。完整曲线、吞吐/奖励分项与下一步见[历史预算复盘](sc0090-v2-roulade-history-full-review.md)。以下旧阶段按原样保留。

# 历史：Roulade预算100,663,296已用完；两配方均未合格，阶段关闭

首128为50,331,648正式controls；完整进程617.694s/81,483.14控制每秒，排除首3更新稳态median112,059.47/aggregate110,438.62。nominal/measured各256例原始评测：standing完整0/128、body落稳0/128；四个midroll桶均body/头中性0/32。所有健康身体相机分母0，不能把坏姿态静止当头稳；两条件fault/nonfinite0。原128检查点和失败轨迹保留。

最后128方案已获父代理授权：新真实worktree `p04/roulade-endpoint-worktree`；25%原standing、50%原50–340°midroll、25%355–360°近完成桶(tuck0–.15，初始角速0)。其它reward/PPO/观测/DR不改，精确继承128 actor/critic/normalizer/Adam/LR/clock。原reverse curriculum的accum/max/paid=spawn及latch预置明确为教学先验，paid预置避免空奖；不加actor/critic进度输入、不将端点稳定计作完整翻滚。正式完整评测仍原站立与90/180/270/330初态。

先进行128world×3s近完成HOME物理和身体/头/200Hz相机检查，PID4160315/session72728；随后64×5成熟恢复/终止验证、N16384×16容量与速度准入，通过才自动最后128，再两条件原评测。阶段总上限100,663,296；这是第2且最后行为配方，不追加第3种或扩预算。新计划/来源见 `roulade-endpoint-plan.json`、`roulade-endpoint-source-transition.json`。下面全部历史按原件保留。

# 历史：首128正式段放行

新源已签 `roulade-qualification-001.json`；fresh N16384/T24/128＝50,331,648正式controls，阶段总≤100,663,296、最多2行为配方。正式队列session57877，实时 `p04/roulade-formal-status.json`。首段结束自动256world nominal/measured各5s评测，不自动续第二段。下面保留全部前期证据与失败。

# 历史：最初物理准入，正式更新0

父代理已授权 GroundPick 关闭后转入 Roulade。任务 `Mjlab-Roulade-Flat-MicroDuck-V2-Repaired-S0-O0-R1`；阶段正式上限 **100,663,296 controls**，最多两种行为配方，优先N16384首128（50,331,648），若N24576更合适首128（75,497,472）后最多42，不增加总额。所有物理/软件准入独立计账，不计正式。

# V2 P04 Roulade 专职台账

## 历史：最初CPU准备

- 主责：子代理 `/root/p04_rise_training`，GPU0 当前仍属于 GroundPick 首128正式段及其自动末评。父代理负责 GPU1/VelStand/BallKick。
- 范围：V2 原地前滚翻→双足落地→站稳；无轮滑、Spin、部署。大幅滚转本身为主动动作，头/相机稳定只在相应保持阶段评价，动作阶段另报冲击/角速度。
- 独立真实工作树：`/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04/roulade-repaired-prep-worktree`，detached `070d6069e1aacbba9a22a905eaa8f4fe0dd6729a`。
- 从冻结 `groundpick-repaired-worktree` 复制 src/scripts/tests；188个数值/入口文件 SHA 逐一核对，来源 `roulade-source-parent.json`。没有修改正在训练的 GroundPick 源码。
- 冻结约束：v5 电机与 V2 CAD 不改；最终使用完整 S0/O0、合法读数不去尖峰、O0局部历史、GPU compiled/fused/graph，61 actor/14动作/HOME0。当前还未注册/准入 Roulade V2 任务。
- 本技能正式预算尚未开始计数；GroundPick/SitStand/起身交互不得继承或混计。首段/阶段总上限须在物理与协议审计完成后明确锁定。

## 真实历史源码与经验

依据 `tasks/microduck_roulade_env_cfg.py` 和 `tasks/mdp.py` 的实际实现，而非文件顶部残留的旧注释：

- episodic、5秒，无相位时钟/动作轨迹模板；从站立或 midroll 开始，学习前滚和落地。
- supported-only 的 +body-y 角速度积分，sagittal flatness 门控、max frontier 增量奖；不奖励空翻、来回摇摆或停在半程。
- 当前 progress paid-rate cap **5 rad/s**，overspeed阈值 **7 rad/s**（顶部早期说明仍是3/4，不能照抄）。
- head-top 实际接触锁存，landing 在260–330°渐开；midroll重置显式赋予已经过头的锁存，不能把 midroll 完成当 standing-origin 全动作通过。
- 真实 spawn mix 起始50/50，历史将晋级延后到3000/6000更新，避免未掌握就减少后半程曝光。动作率/落地税按实际源阶段，不能无证据提前加税。
- 原始 symmetry=True、XL330 tuck及jaw轴均有 V2 迁移缺口（如下）；不会直接照搬。

## CPU 几何审计 001（已完成）

入口：`scripts/audit_v2_roulade_cpu.py`；报告：数据盘 `p04/roulade-cpu-geometry-001.json`。显式 `CUDA_VISIBLE_DEVICES=''`，零物理积分，仅 mj_forward 和一个自由姿态微分检查，不是 GPU/动力学准入。

1. **镜像符号**：按轴矢量在矢状面反射 `-R` 推导为 `[-1,-1,+1,+1,+1,+1,+1,-1,-1,-1,-1,+1,+1,+1]`。置换/符号 involution 通过；33姿态（HOME+32组腿±.15rad、头0）足端几何镜像新表最大 `2.46e-8 m`，旧表 `0.0487394 m`。髋roll范围镜像约2e-6rad差异；这只证明上述关节运动学，不证明质量/摩擦/反馈随机化全动力学完全对称，不能自动打开旧mirror loss。
2. **tuck**：老绝对角经已验证 old→V2 HOME/轴向映射后，两膝为−1.2549rad，越V2下限−1.19276；满幅 self penetration 最大6.455mm。仅投到限位内保留20mrad余量仍自穿6.039mm，不能以“限位合法”当姿态健康。旧映射和投影候选的0.3/0.6比例均在本静态检查中无自穿；尚未选择最终 reset，未证明动态可滚。
3. **head-top**：V2真实主壳碰撞 `ground_head_shell_collision` 在 `bottom_head_shell`；旧sensor仅body `jaw_soft`会漏主壳。主壳主要近水平顶面法向约 `(-.078459,0,.996917)`（HOME shell局部系），旧 `(.882,0,.471)` 不适配。HOME身体pitch50°时旧轴world-z=−.3729（错误判top-down），真实壳top-z=+.7009；因此锁存与pivot奖励必须同步使用正确刚体/几何定义。
4. **前滚符号**：root body+Y速度1rad/s积分1ms后forward+X的z变负，确认V2仍+Y为鼻端向下，不沿用历史错误−ω_y。
5. 模型XML、mesh、限位、电机未修改；全部失败姿态/几何记录保留。

## 下一项有限检查

- 一次有限CPU检查收敛无自穿的tuck幅度与实际头顶区域；不得盲裁切关节后直接进入训练。
- 独立V2配置：正确head-shell接触/顶面定义、V2 reset名字索引与地面高度、真实S0/O0、旧reward/pacing其余保留；镜像损失须明确选择并验证，不拿运动学结果冒称完整对称。
- GroundPick末评/剩余42更新决策之后才安排GPU物理样本；随后64×5真实PPO、状态性roll accumulator一次更新/奖励独立oracle、true timeout successor、ONNX真实obs和实际N容量/吞吐护栏。
- 任何Roulade奖励都不得为oracle重复调用有状态progress函数；记录原reward_manager一次计算的分项。分 standing-origin 与 midroll-origin 验收，前者必须实际获得head-top latch并全滚完成。


## CPU reset survey 001（单次有限补查）

报告 `p04/roulade-reset-cpu-001.json`：固定7个tuck比例，.60/.65/.70/.75/.80未检测静态自穿，.85为2.433mm，.90为5.637mm。不继续扩展候选搜索。另固定32个midroll（8 pitch ×2 tuck比例×2高度），暂以旧映射0.6幅作为几何探针，旧z=.05/.10m让24例碰撞几何低于地面，最深106.379mm。最低几何来自实际地面碰撞头壳/脚/电池，纯根高度平移至1mm净距需要z=.0529–.1574m。不能让正式训练依赖解算器从深穿中弹出；后续只做最小姿态相关ground-clearance reset，不改CAD接触或把CPU几何探针当动态动作模板。0.6幅尚非选定训练tuck，未验随机关节噪声与动态承接。

此轮CPU准备至此收敛；当前等待 GroundPick 首128正式段及独立末评，不占额外GPU，不提前启动Roulade正式训练。


## 实施版本与 CPU 准入

- `custom_v2/roulade.py` 独立工厂/注册；原奖励权重逐一相同，站高只从XL330 .115替换已测V2 .112；solver10/20、5ms×4控制保持，contact容量64并待实际N护栏。
- 真实壳geom `ground_head_shell_collision`，局部顶轴(-.0784586,0,.9969174)由startup显式赋值；mdp旧任务仍默认原轴，不改其它worktree。
- `roulade_symmetry.py` 新V2轴表，actor mirror loss保留原0.5但不做critic/data augmentation，61/14 involution通过；这是几何对称先验，不宣称S0统计噪声/质量完全对称。
- tuck=旧HOME/轴转换delta的0.6（双hip−.41526/knee−.75294/ankle+.3582，neck+.80946/headpitch+.39054）；静态锚无自穿，256带原σ=.08噪声样本0越限、最大初始自穿.405mm，不能写“带噪全零自穿”。实际GPU还需检查。
- `roulade_geometry.py` 16 hinge FK+实际floor-bit1 collision OBB角点，在重置q/根姿态下给1mm保守净距，mouth_reset已移到本reset前保证最终mouth状态一致。CPU64个含任意3D根姿态对照FK最大1.12e−16、FP32 height差3.17e−8；实际mesh净距3.29–34.04mm，保守额外落差与冲击必须GPU测，不能称已支持地面。无额外sim.forward、无CAD修改、无物理真值喂actor。
- CPU配置/256噪声tuck/镜像/原奖励/PPO检查：`roulade-cfg-cpu-001.json` PASS，仅CPU资格。
- 第一次物理入口在建环境前发现诊断contact API误写sim.data.ncon，主动终止，0physics/0PPO；保存`roulade-physical-001-abort.json`和原脚本，改为锁定Warp实际wp_data.nacon。第二次session16659已启动，输出`roulade-physical-002`，每次stateful进度reward只调用原函数一次并核对独立state oracle。


## 真实重置反例与有界噪声修复（正式仍0）

物理002/003均在第1物理步前被初态1mm自穿护栏拦截，未PPO。003保存完整128初态；CPU逐位复现唯一world100/pitch180°：`ground_battery_collision`↔`self_right_shin_structure_00_collision`最大3.441mm，右hip_yaw−.19655、hip_roll+.14225。关节命名/索引正常。严格同其它q/根姿态，仅yaw限制−.04就无自穿；只roll限制+.04仍1.211mm。见`roulade-actual-reset-failure-001.json`和`roulade-noise-bound-cpu-001.json`。

唯一修复是原高斯σ=.08的**噪声值**截断±.04rad（原draw数量和顺序不变），无动作裁剪、无CAD/碰撞改动；原全yaw、midroll50–340°、tuck blend.3–1、初始omega0–3、原DR全部保留。对001相同256个CPU目标/RNGdraw，截断后自穿最大0；配置001原报告保留，002记录增量及新SHA。新增spawn_angle/spawn_mid仅日志元数据，记录实际reset谱系，不参与actor/critic/reward。

新的物理004：session9459/PID4045441，仍128world×3s和原初态门槛，增加保存native self/head sensor-found以与CPU几何并列。尚在初始化，未宣布物理通过。之后接64×5真实PPO+终止前successor状态验证、actualN16384准入，未获准入不启动正式。


## 物理004完成，smoke自动衔接

actual128初态self penetration0、mesh净距1.210–21.691mm；128×3s反馈fault/nonfinite均0。原生progress/accum/max/paid/latch五项150次独立oracle最大误差全0；真实actor61/critic74，控制边界观测contact峰958。HOME16/16最后1s健康站稳，height.111895m、tilt2.114°；健康相机RMS均值.02029rad/s、20ms跨度p95均值.000392rad。七个midroll固定tuck组都有真实头壳接触、没有动态self-contact，但未回正，不能作为全滚翻技能成功或强制动作模板。自动64×5已接PID4053468，session99615队列后续actualN16384×16，正式仍0。


## PPO诊断兼容性：保留mirror配方，先严格对照

smoke001的6组successor（accum0/3/6×fast/full）全部critic差0、23live状态/CPU-CUDA RNG不变、partial reset其它世界不变，但旧`ppo_diagnostics.py`拒绝任何symmetry，故0次PPO退出；不是反馈或物理故障。保留失败日志/终止验证结果。

最小扩展只放行V2 Roulade actor mirror loss，仍拒绝data augmentation/RND/多GPU/未知aux。按锁定原生RSL原样插入mirror loss的位置/均值动作/反传顺序。`roulade-mirror-diagnostics-parity-001.json`：CPU、cuda:0，各mirror off/on×3更新×20minibatch；相同rollout下actor/critic/N/Adam/LR/RNG/loss全部逐位相同。无symmetry路径同样逐位，原PPO超参与mirror coeff.5未变。新diagnostics SHA `1e1e31f40484a800e26362050803fc65b4111c10d85b47cd6fa0d5a4c3eda1cb`。

队列002/session20306已接smoke002 PID4061333 → actual16384×16，不重做已过的物理检查。actualN至少一条真实timeout可触发完整N的shadow分配并验证时序，数量完整报告，不要求所有world恰好同时超时。正式仍0。

独立终评入口 `scripts/evaluate_v2_roulade.py` 已写，尚待真实冻结checkpoint软件验证：5s/256world，128 standing-origin与4组midroll各32；前者必须实际获得head-top latch，后者显式precredit不当完整前滚成功。同时报告支持进度、真实初始朝向平面的orientation展开、侧向偏离、落地500ms身体/中性头保持、200Hz相机与20ms曝光运动；主动滚转不算稳像失败。最终stage仅通过有界预算决策，不能把开发指标代替硬件。


## 实际N准入完成：N16384选定，正式尚未启动

- smoke002：64×5实际PPO/32timeout、6组successor与partialreset PASS、5项原生progress/state最大4.77e−7、ONNX真实64obs CPU2.38e−7、fault/nonfinite0、contact459/4096。
- actual16384-001：16更新=6,291,456 validation controls，16,384自然timeout、完整N shadow已实际使用；fault/nonfinite0，contact100814/1048576，无8类capacity日志，无optimizer guard；max pooledKL.016853、clip.229557。
- 排除首3更新的稳态median112,275.20控制/s、aggregate108,336.73；内部构建起128.209s，外置含imports总224.614s。本段是短validation，不将稳态当完整端到端。
- 选择N16384，不再跑N24576或其它档。首128正式=50,331,648，整个阶段最多256同N更新=100,663,296、最多2行为配方，所有5/16更新验证模型不得作为正式父模型。
- tail/session58532已转冻结5更新checkpoint的32world nominal/measured独立评测入口软件验证；nominal PID4075718。该模型没有技能声明，入口通过后才签全source资格。
- 父代理BallKick获GPU0小批窗口，完成marker：数据盘`p04/ballkick-physical-001-process.json`；允许与两项32world入口并行，禁止与大N benchmark/正式并行。我必须等其marker/进程结束后启动正式。
- 正式外置watch将保留8类capacity日志扫描；若status累计feedback/nonfinite新增则请求安全更新边界停、保存并做明确raw原因诊断，不猜测故障语义。当前数值源不为未知稀有事件继续扩框架。


## 独立入口抓到 episodic命令继承缺口，新源重签

旧Roulade从velocity继承`rel_forward_envs=.2`，锁定UniformVelocityCommand会将这些world的vx强制`abs().clamp(min=.3)`；即使eval ranges全0也不是实际0。首冻结入口的全零command断言在第一物理步前拦下，保留原执行脚本/log；CUDA assert是该语义护栏触发，非反馈故障或GPU硬件问题。

最小V2修复：显式`cfg.commands['twist'].rel_forward_envs=0`，训练仍保留原tiny范围（vx/vy±.01、wz±.05），独立eval才全0。其`init_velocity_prob`一直0，因此无物理速度注入；覆盖分支只是命令值、无额外RNG。已有物理证据不受影响，但actor输入分布确实改变，所以旧smoke/N16384 source资格不冒用。

新队列003/session8311：smoke003 PID4085462 → 等父BallKick窗口完成 → actual16384-002×16 → 新冻结5更新checkpoint nominal/measured各32world入口。父BallKick已确认完成marker落盘，可继续大N。阶段正式仍0，预算/配方数不扩。旧16384吞吐是修复前结果，最终吞吐以002为准；本次重跑是输入配置修复，不是性能档位搜索。


## 最终新源准入与正式放行

修复rel_forward_envs后的smoke003：32真实timeouts、6successor/partialreset PASS、ONNX64actualobs2.09e−7、fault/nonfinite0。actual16384-002：6,291,456 validation controls，16,384自然timeouts，contact100631/1048576，maxKL.016267/clip.241036，5项oracle≤3.81e−6，fault/nonfinite0。剔除首3更新median112,716.56、aggregate108,585.52控制/s；外置短预检完整220.508s，内部125.856s，未宣称完整进程10万。

冻结5更新checkpoint的两个32world×5s独立入口均完成、actor/N逐位不变、fault/nonfinite0；只有软件资格，模型core_completion仍0。measured实际取10.2/11.1/12.4V，重置后BAM vin逐位等于共享反馈电压，nominal11.1V。106个数值source与138个CAD/测量/v5资产SHA核对；v5 JSON固定7c35a732…，没改CAD或系数。

正式只使用新随机actor/critic/N/Adam（无resume）；5s episode、T24/E5/M4、LR.001 adaptive KL.01、entropy.01、mirror.5、原奖励权重。初态保留yaw/roll/pitch、midroll50–340°/omega0–3和tuck blend.3–1；仅V2tuck0.6、noise σ.08截断±.04、实际CAD净距与壳顶轴、纠正mirror、去掉错误forward-command分支。14动作/61 actor/74 critic、HOME0、S0/O0 validraw/local/GPU compiled+fused/graph维持。

父BallKick窗口已退出marker就绪。正式状态以实时PID/已提交更新为准，不把签准当已经训练成功；外置watch保留8capacity模式和新反馈/非有限安全停止，PPO自身有KL/clip/finite护栏。末评区分standing-origin完整滚与midroll后段恢复，报告尾500ms身体/头中性保持、200Hz相机/20ms曝光跨度；不以动作过程必需转动当稳像失败。


## 首段实际进度

PID4120956已40/128提交，15,728,640正式controls；fault/nonfinite0，无optimizer/capacity guard。累计lateral-axis绝对均值.093（早10更新.176），但standing-origin-after-roll stable仍0，不能以平面改善称会翻滚。维持同配方到既定128并自动末评。实时状态以`roulade-allocation-001/status.json`为准。


## 最后端点课程物理与准入（2026-10-16）

`roulade-endpoint-physical-001` 128×3s：HOME16/16；355/356/357/358/359/359.5/360°末1s身体健康与head max<.1rad依次16/14/12/15/16/16/16，共105/112。7个失败保留，未扩大候选搜索。全程fault/nonfinite/self-contact0，初始mesh净距1.25–7.14mm、自穿0；progress/accum/max/paid/latch150次oracle全0。健康子集200Hz camera角速RMS各组均值.0162–.0206rad/s，20ms exposure span p95均值.000373–.000432rad；分母随组12–16，不能将其当策略或完整翻滚通过。

CPU配置比较入口第一次因dict整数joint索引拼路径的诊断TypeError失败，0PPO/0新物理，原入口/log保存failed001。仅改test中`str(k)`，生产数值源不变，正在重跑一次。自动队列51251在其通过后接成熟64×5、actual16384×16、精确从原128启动最后128和原两条件评测；任何容量/反馈/非有限/optimizer异常停止保留。

CPU复核PASS：cfg仅reset函数与旧spawn-mix删除两处，原play逐字段相同、actor/critic/PPO/观测原样。测试16384端点4148/普通12236，旧桶条件全部相同；三种错误transition拒绝。跨源manifest严格锁父checkpoint SHA、完整from/to源SHA、唯一4文件白名单；正式仍先逐位确认actor/critic/normalizer/Adam，再继承原LR/RNG/3072时钟，不使用validation尾模型。数值源已经冻结，成熟64×5 PID4180631自动队列session51251。

成熟64×5 PASS：32真实timeout，6组successor最大7.06e−10/24live张量与RNG保持、partial reset包括endpoint标志；ONNX64实obs CPU最大7.15e−7；fault/nonfinite0，contact421/4096。5更新max pooledKL.04977/clip.49505，未达安全停止阈值。精确恢复原128的actor/critic/N/Adam、LR.0008779149519890257和clock3072，validation尾133不作为正式父。actual16384×16 PID4188926运行中，GPU0实测91%，未谎称初始化PID即已学习。

实际16384×16准入PASS：16,384真实timeouts，feedback/nonfinite0，contact95,433/1,048,576，原生progress/state最大9.54e−7；排除首3更新稳态median111,084.98/aggregate107,472.10控制/s。见roulade-endpoint-actual16384-001与roulade-endpoint-qualification.json；末端教学样本占24.966%，这16验证更新没有身体稳态，不能把软件通过当行为通过。正式PID4758已启动（系统PID回卷），session51251，源固定，从原128继承，最后128/50,331,648，阶段最终100,663,296；所有原eval在自动队列内。

最后128正式完成，累计256更新/100,663,296正式controls（达到阶段上限，无第3配方）。新段50,331,648，完整625.436s/80,474.53控制/s；内部构建起530.717s，排除首3稳态median109,529.65/aggregate108,349.75。196,608真实timeouts，feedback/nonfinite0，maxKL.01622/clip.21779，无安全guard；endpoint实际曝光24.9722%，训练身体/头稳定仍0，standing-body仅一个控制样本，不能作保持通过。原nominal256world末评PID21630已自动接续，再measured；结束后GPU0交还根代理有历史依据的有限camera试验，无新增Rou训练。

## 阶段关闭与GPU交接

最后256两条件完整原验收均standing完整/body/head0/128、四midroll各body/head0/32，fault/nonfinite0、健康相机分母全部0。晚段330°从旧tilt68.6°变为10.33°，但高.05971m、双足0/32、自碰31/32，属低位折叠回正，非站稳。保留两份不可变checkpoint与报告，清单`p04-roulade-retained-candidates-20261016.json`；阶段正式100,663,296，2配方上限已满，关闭未合格。GPU0 marker STAGE_BUDGET_COMPLETE_NO_FURTHER_TRAINING已触发，交由根代理camera准入，本子代理后续不占GPU0。

一次有界CPU奖励审计`roulade-landing-reward-audit.json`：五个落地相关生产AST与独立公式最大1.19e−7，末330低位加权/dt前upright1.473+height.1816+composite.02093+sharp约.00001−tax.26144=1.41416(每控制.02828)。门槛只用frontier/latch，没有双足/无自碰/头中性要求。105个真实健康端点HOME参考高.11161m，与.112目标吻合；实际腿q未保存，composite只给[0,上界]，五项均值下界4.455、上界8.416。故无证据称低位优于健康HOME，能证实的只是坏低位仍收正部分奖、sharp在该处约零。policy数据为reward边界，HOME物理probe为post-step/forward边界，报告补此metadata说明，原版并留。完整return因实际action及前action、raw自碰count、angmom/softlimit等缺失未重建。原奖栈也无直接head HOME/camera项，landing pose明确仅legs。只读无GPU、未提出或启动第3配方。
