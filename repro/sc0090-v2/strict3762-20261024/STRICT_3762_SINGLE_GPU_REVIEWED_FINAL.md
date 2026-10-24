# 原 3762 完整历史对齐：完成后的审阅版

状态：**十段正式训练、启动前资格检查、恢复边界及最终产物审计已完成。行为比较存在明确限制，见 FINAL_RESULTS.md；不表示实机或全角度性能验收通过。**

## 0. 用户目标与当前结果归档

用户要求：先推送已有结果；再核实原版从初始化到 3762 的完整链路，锁定环境、奖励、课程、噪声和阶段切换条件，唯一训练差异为后背/左右凸起碰撞；只用一个 GPU；文档 review 修正后，按文档完成训练。

已有双 GPU 6804 结果已推送 `microduck-awesome/microduck_rl` 的 `repro/sc0090-v2-sim2real` 分支，远程提交 `768a43321d6d0ded736f8ad73bb02c5d6fbd4020`。186 个清单文件的提交后 Git blob 字节均与本地交付 SHA256 一致，包含 ONNX 和两个 checkpoint。见 `publication-receipt.json`。这份归档明确标为适配结果，不是严格历史复现；不得作为本轮权重初始化。

## 1. 证据优先级与“相同”的边界

本方案依据原始运行记录，而非上轮总结或最终快照猜测历史：

1. `historical-launch-evidence.json`：从原会话中提取的十条实际启动命令及时间。
2. `evidence/<历史目录>/run.json`：每段有效环境、agent、seed、参数和源码 SHA。
3. 同目录 `source.tar.gz`：每段实际源码；已逐文件与该段 `source_sha256` 核对，十段全部一致。
4. 同目录 `status.json`、`evaluation_*.json`：实际完成计数、评测时点和原课程决策。
5. `history-inventory.json`：完整归集索引、所有 checkpoint SHA、源代码变更列表、全量配置摘要。

本方案唯一目标是 `paired-feedback-allocation-001/checkpoint_003762.pt`，SHA256 `e753856776bb43e44d9465b4067a0b9cd2d10fe03fc8cb88ee4177ec786f93e1`。同样编号3762的 Official 头部旁支不是这个模型；`head-precision2-allocation-002` 等旁支不在该祖先链内。

对齐对象是**逐阶段代码、有效数值配置、初始 seed、算法、执行顺序和切换条件**。碰撞发生变化后，接触、轨迹、随机重置次数、评测结果及最终权重可能变化，不能承诺这些也逐位等于旧模型。不得为了得到历史通过率而写入历史评测结果或强制晋级。

历史 checkpoint 仅作核查参照；本轮必须从随机初始化开始，中间恢复只读取本轮前一段生成的同名 checkpoint。不得载入旧3762、上轮6804或其他实验的权重/归一化器/优化器/RNG。

## 2. 唯一允许的训练模型差异

十段原始 preflight 记录的起身 XML 均为：

`d52e06595e737ebe0b96529bfa6410b26b5f0e2d221e03ee2c28c4c9fcf0ed69`

它与 `rise_baseline_paired3762` 冻结模型完全一致。本轮仅在这个模型上补入后背/左右外壳的 32 个同 CAD 来源凸分块。必须逐元素证明：原有 XML 的刚体、关节、执行器、限位、质量、质心、惯量、既有碰撞和接触参数不变；新增部分密度为零且不重复计质量。原名义质量 0.600941825 kg 不替换成其他版本。

不引入后来的站立奖励、头部探索改造、动作滤波、物理限位修复、其他随机化、当前双 GPU 归一化同步封装或修改后的训练器。所有历史 Python/C++ 文件按各段归档原样恢复；只在外部编排器管理路径、证据和原本存在的停止/恢复事件。

## 3. 单 GPU 和基础数值契约

- 使用原历史设备参数 `cuda:1`，一个 Python 训练进程；不用 torchrun/DDP，不把 8192 环境拆到两张卡。保留历史设备可见性：十个父checkpoint都存有两张可见卡的CUDA RNG，但仅cuda:1承担训练；不可通过隐藏设备把cuda:1改成cuda:0。
- `num_envs=8192`；每次 PPO 24 个控制步、5 个 epoch、4 个 minibatch，全局采样批量 196608。
- 环境与 agent seed 均为 `20261015`；构建 runner 前原代码再执行 `torch.manual_seed(seed + 100)`。
- 物理步长 0.005 s，decimation=4，控制 50 Hz；episode 8 s；非有限 horizon；无新的动作滤波。
- actor 61D、critic 74D、policy action 14D，嘴部独立执行器保留；动作 scale=1、HOME offset 和关节顺序不变。
- actor/critic 均为 ELU MLP `[512,256,128]`，观测归一化开启；初始 Gaussian std=1。
- Adam；初始 LR=0.001、adaptive、desired_KL=0.01、gamma=0.99、lambda=0.95、entropy_coef=0.01、clip=0.2、max_grad_norm=1、value_loss_coef=1。恢复时按该段原代码恢复优化器和实际自适应 LR，不重新初始化。历史checkpoint恢复Torch CPU及CUDA RNG；没有另存Python/NumPy的跨进程连续RNG或完整物理状态，本轮不能擅自加上这些改变原恢复语义。
- 固定记录的依赖：torch 2.9.1、mjlab 1.3.0、MuJoCo 3.13.0、mujoco-warp 3.8.1、warp-lang 1.12.0、better-actuator-models 1.0.1、rsl-rl-lib 5.0.1。实际环境与源码 API 一致性还须通过启动前检查。
- 不能把所有阶段都套用最终编译设置。前三段归档 `v2_python.sh` 不设置 `TRITON_DEFAULT_FP_FUSION`；从第 2000 次后的 head-precision-allocation-001 才加入默认 `TRITON_DEFAULT_FP_FUSION=0` 和 no-fma cache。遵循原启动命令和各段原 wrapper；不得从当前运行壳层意外继承新变量。

以上是便于阅读的摘要，**全量 `environment`/`agent` 配置均受锁定**；未在此列出的字段也不能随意修改。

## 4. 完整十段执行表

`Official` 表示 `Mjlab-StandUp-Flat-MicroDuck-V2-Official-Baseline`；`Paired` 表示 `Mjlab-StandUp-Flat-MicroDuck-V2-PairedError-Baseline`。

| 段 | 原运行目录 | 本轮累计区间 | 原 iterations 参数 | task / 头部设置 | 本轮读取的父文件 | 评测间隔 |
|---|---|---:|---:|---|---|---:|
| 0 | standup-allocation-001 | 0→12 | 1000，原实际在12停止 | Official / 无精度开关 | 无 | 500；实际未评测 |
| 1 | standup-allocation-002 | 12→1000 | 988 | Official / 无精度开关 | 段0/checkpoint_000012.pt | 250 |
| 2 | standup-allocation-003 | 1000→2000 | 1000 | Official / 无精度开关 | 段1/checkpoint_001000.pt | 250 |
| 3 | head-precision-allocation-001 | 2000→2250 | 250 | Official / `--head-precision` | 段2/resume_002000.pt | 250 |
| 4 | head-precision-allocation-002 | 2250→2500 | 250 | Official / `--head-precision` | 段3/resume_002250.pt | 250 |
| 5 | head-precision-allocation-003 | 2500→2750 | 250 | Official / `--head-precision` | 段4/resume_002500.pt | 250 |
| 6 | head-precision-allocation-004 | 2750→3000 | 250 | Official / `--head-precision` | 段5/resume_002750.pt | 250 |
| 7 | head-precision2-allocation-001 | 3000→3250 | 250 | Official / precision stage 2 | 段6/resume_003000.pt | 250 |
| 8 | head-foundation4-allocation-001 | 3250→3506 | 256 | Official / precision stage 4 | 段7/resume_003250.pt | 128 |
| 9 | paired-feedback-allocation-001 | 3506→3762 | 256 | Paired / precision stage 4 | 段8/resume_003506.pt | 128 |

每段都是原来那一段的源码，不是十段共用最终源码。精度阶段1的四次250更新不能合并成一次1000：原历史在每段之间有新进程构造、加载、RNG恢复和新 episode 初始化。

段0不得简单改成 `--iterations 12`：这会在第12次额外执行原本没有的评测。需要保留原参数1000，由外部边界停止装置在第12次更新的原 SIGTERM 检查点触发原停止路径；原调用顺序已通过CPU测试、真实64环境停止验证及正式8192环境运行核验，均准确在12停止且未额外评测。

评测的实际全局时点：250/500/750/1000；1250/1500/1750/2000；2250/2500/2750/3000/3250；3328/3456/3506；3584/3712/3762。128 间隔不是“从段起点再加128”，原训练器对累计更新取模。

## 5. 必须按历史时间发生的实现变化

- 0→12：原 PPO 与当时的 reward/advantage 防护代码；不能提前使用后来的 TerminalCriticPPO。
- 12→1000：原 source archive 加入 terminal critic/bootstrap、诊断、曝光统计及新的评测读数位置，并调整 interval 对齐逻辑。逐文件变化见 inventory。
- 1000→2000：原恢复逻辑开始在恢复时重置新 episode 并恢复 polish reward；checkpoint 与评测后的 resume 开始分开保留。前段 checkpoint_001000.pt 是原训练器评测后再次保存的版本，不得擅自用另一种文件替代。
- 2000→3000：原头部 stage 1 与当时 wrapper（包括 no-fma 设置）；四段分别恢复。
- 3000→3250：原 stage 2。
- 3250→3506：原 stage 4，同时按这一段 source archive 使用当时 actuator/runtime/commands/mdp 实现变化。
- 3506→3762：才切入 PairedError，包括 error_proxy、paired_randomization、任务注册和当时观察函数变更。

历史记录中的 git HEAD 不足以代表这些未提交过的逐段实现；源码归档 SHA 才是实际锁定依据。

## 6. 奖励、噪声、课程与晋级条件

完整数值和函数身份以每段原 run.json 为准，构建后的配置必须逐字段比较，不可仅看 task 名称。

早期 Official 的 actor：角速度加性噪声 ±0.03、gravity ±0.01，IMU安装误差范围6°且观测延迟0–1控制步；joint_pos 加性噪声 ±0.001、0步延迟、原 encoder bias；joint_vel 加性噪声 ±0.25、1步延迟。后续 PairedError 的替换、bias事件移除与配对误差重置严格以最后一段记录为准，不提前启用。其余质量/惯量、CoM、摩擦、推力、电压等事件的范围和采样时序全部锁定，不用本摘要替代完整配置。

头部时间课程的控制步节点为 0/12000/24000/36000/48000（对应0/500/1000/1500/2000更新）。stage1/2/4 的精度范围均冻结在原1000更新范围：颈±0.39、头俯仰±0.39、yaw±0.49、roll±0.11 rad；不是把 final_ranges 直接当训练范围。stage1 fine_std=0.08；stage2 fine_std=0.04；stage4 fine_std=0.04、precision_scale=0.003，零指令比例0.4、precision_probability=0.5及原组合轴采样。阶段1/2的零指令比例为0.6。各段完整头部 reward weight/函数参数和采样器由对应源代码与 run.json 双重核对。

起身稳定判据按原代码：高度不低于 `STAND_HEIGHT−0.010`，倾角不超过20°，身体线速度范数<0.15 m/s、角速度范数<1.5 rad/s、双足接触，连续至少0.5 s（25个控制步）。晋级要求累计frontier评测样本至少256、距上次晋级至少2400控制步，并且两个种子的standing/sitting/prone/frontier各自成功率≥0.7；到最后stage才额外要求left_side/right_side/supine各自≥0.7后启用polish。不得误写成每次晋级都检查七桶或要求合并总成功率即可。

`run.json` 是构建时配置，不涵盖随后所有运行时权重变化。polish触发后及后续按原逻辑恢复时，五项权重为 action_rate_l2=-0.5、arrival_damping=-0.025、joint_torque_rate_l2=-0.0005、head_pose_bias=0.5、body_pose_tracking=0.5。必须同时核对这些动态变化，不能只比较启动时JSON。

起身课程前沿 `(150,120,90,60,30,0)` 度；原七桶、两个固定开发种子、每桶每种子128、8秒，使用原判断函数。原历史在250/500/750/1000/1250晋级，1500进入polish；这是历史结果，**不是本轮可以强制写入的日程**。保持原评测阈值、最小阶段步数和逻辑；后续段按原代码检查准入条件，不降低门槛或无记录延长训练。需要明确：原polish标志首次通过后单向保持，后续复评失败不会清零；头部段只检查该历史标志。因此2500次行为失败并未触发原程序的停止条件，必须单独报告，不能把标志当作当前行为通过。详见POLISH_STATE_REVIEW_ADDENDUM.md。

## 7. 重建与启动前门槛（完成记录）

- [x] 已推送当前结果，远程提交与模型字节核对完成。
- [x] 已追溯十条真实启动命令、十段 source archive，并与每段源码 SHA 一致。
- [x] 已确认十段原模型 XML SHA 相同，且等于原3762冻结资产。
- [x] 已读取十个父 checkpoint 的实际课程、计数、优化器和RNG元数据；计数与24控制步/更新相符，设备RNG均为2份，见 checkpoint-history.json。模型/normalizer的本轮恢复仍须实际资格验证。
- [x] 在独立目录重建十段源码；每段数值源码文件集合与历史严格一致，不能混入最终版额外模块。
- [x] 非碰撞静态资源与原冻结资源核对；碰撞变化逐元素审计与质量/接触验证。
- [x] 原环境依赖、编译设置和每段有效配置重新实例化核对，任何数值差异都必须先解决。
- [x] 段0第12次停止边界验证；禁止意外第12次评测或多训练一次。
- [x] 文档条目与原启动命令/源码/状态独立反查，修正所有发现的错误并记录 review 结果。
- [x] 在通过文档审阅后，做64×5和原实际批量的新碰撞资格检查；资格权重不得进入正式初始化。
- [x] 原训练器的吞吐准入（适用阶段 ≥10个实测更新、>100000 steps/s）保持原值，不伪造报告或绕过。

以上门槛未关闭时，不启动正式训练。阶段执行将保留实际命令、source/config比较、checkpoint哈希和各次评测，不按“跑完3762”代替验证。

## 8. Review 纠错记录

目前已从上轮认识中纠正：完整链路有10个进程分段；初始第12次中断不能省略；初始PPO实现不是最终PPO；配对误差只在3506后切入；FMA设置不是从头一致；四次头部250不可合并；128更新间隔按累计步对齐；checkpoint与resume的保存语义存在历史变化。后续 review 发现的问题继续记录，未验证项目不标通过。


### 首轮文档反查修正

- R01：明确唯一PairedError目标SHA，排除相同更新编号的旁支。
- R02：纠正笼统“恢复所有RNG”的表述，按历史只恢复其实际保存的状态，并保留两卡可见、单卡训练的设备语义。
- R03：补充运行时polish奖励，避免只锁启动JSON却漏掉训练中权重改变。
- R04：补齐每种子、分桶和最小阶段时长的晋级条件，区分晋级与polish条件。
- R05：确认第1000次父文件是原代码评测后覆盖保存的checkpoint；它的当前SHA可能不同于当次评测请求中的保存前SHA，这是历史保存语义，不能“修复”为后来的文件流程。

以下保留首轮及第二轮审阅记录。真实GPU资格验证和第12次停止边界均已通过，准入证据见formal-admission.json。


### 第二轮反查与执行约束

- 十段数值源码逐文件匹配各自历史归档；源码库提交 `782d7c362765c225580feaed53b70040abdd74de`。281个静态文件逐项核对，XML原有元素不变；新增32块带来的派生配置项为96个（contype、conaffinity、condim各32），并非仅64个掩码项。不得放宽为忽略全部碰撞配置。
- `config-review.json`及十份config-checks均通过。Python为3.12.14，依赖版本匹配；通过实际Triton语言选项核对FP fusion：前三段True，之后False。没有修改归档训练源码以适配检查工具。
- `collision-identity-review.json`验证原刚体/关节/执行器/geom字段完全一致，总质量0.600941825 kg；3秒被动重力测试验证背部及左右新增外壳接触。该测试不证明训练成功或硬件性能。
- `stop-boundary-unit-check.json`只证明CPU调用顺序与RNG一致，不能代替真实环境停止验证。外部historical_entry在原训练器注册SIGTERM处理器之后安装计数包装，在第12次原update执行前发信号，由原update完成更新并走原停止路径。不更改原iterations=1000，不增加第12次评测。
- `execution-plan.json`逐段反向还原原argv；正式训练仅替换output/preflight/resume路径，首段另包原停止事件。所有正式父文件必须来自本轮formal目录，资格检查权重不得进入正式链路。
- 资格验证使用各段原wrapper、原源码、cuda:1；64环境5更新后再测8192环境（首段5更新、其余16更新）。历史bootstrap资格报告计数4096的阶段保留force-timeouts验证；这仅作用于独立资格进程，绝不传入正式训练。旧stage1函数没有stage参数，检查工具只允许stage=1，拒绝静默忽略错误stage。
- 正式链路之前另以64环境、validation-only、iterations=1000运行原首段停止验证，要求STOPPED12、common_step_counter=288、无评测文件、无第13次更新。验证后丢弃这些权重，正式首段重新随机初始化8192环境。
- 资格进程和正式进程串行，保留两卡可见，但仅cuda:1执行计算。清除execution-plan环境契约所列分布式、warm-start和编译覆盖变量后调用该阶段原wrapper。
- 每段启动前重新检查冻结文件哈希。进程异常退出即保留产物并停止，不自动续训部分段落、不修改奖励或课程、不绕过原吞吐门槛。额外进程恢复会改变历史重置次数；若需重试，只能重新审查后从该段指定父文件开始整段重跑。
- 文档审核PASS仅准入资格验证。正式准入要求当前碰撞模型的新smoke/batch报告、真实停止边界检查均通过；适用阶段保持原>=10实测更新及>100000 steps/s门槛。阶段父文件的课程前提仍由原训练器检查，不强制晋级。
- 文档与执行计划冻结后，运行结果另写qualification/formal记录；不通过改写合同把失败改成通过。

资格执行入口为 `tools/run_qualification.py --phase <0..9> --kind smoke|batch`，仅运行资格检查、验证审核文件哈希、拒绝覆盖已有尝试并使用排他进程锁；不能启动正式训练。启动时的文档审核证书保留为PASS_FOR_QUALIFICATION_ONLY；随后20次资格检查与真实停止边界通过，正式准入见formal-admission.json，最终完成核验见training-completion-audit.json。


## 完成审阅与原始合同的关系

实际训练使用的冻结文档仍为STRICT_3762_SINGLE_GPU.md，SHA256 `23fad64fab5b85963e06edf84f7b38d31e201e21c911e90882cc3d609e8f9385`；本完成版更新已闭合的门槛状态并补齐polish单向保持说明，不改写训练期间的文件或数值配置。最终核验见training-completion-audit.json：10段、19次原定评测、3762更新、每环境90288控制步、单GPU cuda:1。所有正式进程正常退出，无正式训练重试或回退。

2500次右侧卧退化、随后恢复、资格进程一次退出段错误及不改变数值配置的资格复验，均保留证据。原始polish条件和调用顺序保持不变。行为结果、全14关节诊断及尚未达到的性能见FINAL_RESULTS.md，不把完成训练写成性能全面通过。
