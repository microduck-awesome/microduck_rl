# 当前阶段已完成

BC3e-6固定320正式和六项原评测已完成，禁止自动追加。CPU完整复算见 `sitstand-near-teacher-lr3e6-final-review.json`；是否技能合格按原指标，不能用预算结束代替。

# 当前：正式完整结束，末评恢复

原320正式已到960，新增125,829,120控制、25,600 PPO Adam/6,400 BC，正式含建环境1517.177s=82,936.36控制/s，反馈0。首768 nominal600步已完整；新增contactscreen聚合将float stable直接做按位AND导致外置report失败。只修stable.bool()，PID581916/session89539正在CPU恢复首report并补其余5项末评，训练源码不变、不重训、首仿真不重跑。正式预算已封顶603,979,776。

# 当前真实阶段

BC3e-6全部资格PASS，PID548509/session12268已在GPU1真实正式学习，已核660/960、7,864,320新增控制；从原640精确恢复，连续320至960。最新值以allocation/status.json为准。原3e-5与1e-5失败源保留，不再换配方。

# SitStand 近端教师固定阶段

当前：64×5小批在第4更新因post-BC mean KL1.20659触发原0.5护栏停止，正式0。原源、原640父的actualN也在第2更新触发原护栏（meanKL1.64036），已停，正式0；唯一BC步长修复已获root授权：v2小批全PASS、actualN第3更新meanKL .56403超原.5门槛；root已批准仅BC LR3e-6，PID548509/session12268已启动新资格，正式0。原640坐姿能力及站姿头支撑失败保留。原Ball精度阶段512已结束，GPU1已交接。

目标：一个61维观测、14动作的单策略，保留受命坐下能力，并在接近站姿时学会无头支撑站立。不将教师外置切换称为技能成功。

方案：从历史Hold640精确恢复actor、critic、normalizer、PPO Adam、LR、CUDA RNG及原时钟15360；保持原冻结课程、奖励、物理、测量和命令。仅增加独立BC Adam（3e-5、5 epochs、最多4个均衡大小批次），每有效帧权重1。STAND且ramp_done、z>0.1m、tilt<30°、腿HOME MAE<0.2rad、双足、finite-ready时使用1856完整起身教师；不排除头触地。SIT用640完整动作anchor，其余STAND仅PPO。STAND不足64帧则跳过该分支；不做50/50分支重采样。教师只接真实raw actor观测，起身教师twist/body置0、head命令保留，独立冻结normalizer。门控真值仅选择训练标签，不进入actor/critic。

正式上限：N16384/T24/E5/M16连续320更新，新增125,829,120控制、25,600原生PPO Adam；本技能全历史603,979,776。只此一段，无自动下一段。验证权重不继承。实际BC次数、各分支样本、ignored STAND逐更新记录。

资格：CPU cfg精确差分/边界真值表→成熟64×5（ONNX、真实超时、当次24步缓存oracle）→actual16384×16（数值/容量/PPO和BC KL守门）。吞吐照实记录，不用10万硬挡。合格后才自动从原640正式学习。末评768/896/960两条件原12s命令/反向/保持电池，原阈值不变；另加200Hz头壳/自碰/双足与相机数据及健康分母。

源：`/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04/sitstand-near-teacher-worktree`。计划/补丁/worker/状态见同目录上一级 `sitstand-near-teacher-plan-001.json`、`sitstand-near-teacher-production-and-eval.patch`、`run-sitstand-near-teacher-001.py`、`sitstand-near-teacher-pipeline-status.json`。

证据限制：近端教师probe两条件各16条身体+健康+粗头HOME全过，但全四轴持续1°仅4/16、6/16；不能推广到低SIT→STAND。初态逐位相同，前2s轨迹已有浮点差，不称严格逐轨迹因果配对。此前head-only替换仍倾倒，不使用该方案。

## 小批001失败保留

实际4更新、6144 validation controls、320 PPO Adam、25 BC Adam。BC mean KL .01414/.06563/.24807/1.20659；STAND有效帧0/108/241/447，SIT621/624/624/624，最后selected1071触发原balanced批数1→2，BC步5→10。此为已观察相关时序，不据此单独归因。32真实timeout、96步preaction cache独立逐位oracle、反馈/非有限0、原640恢复和教师/N冻结、ONNX7.15e-7均过。原source/失败checkpoint/report留在smoke-001，不继承其权重。

actual诊断002使用原640、相同源/BC/PPO/.5 KL守门；只调查实际N，不能跳过未完成小批资格进入正式。

## actualN诊断002失败保留

2更新786432验证控制、160PPO Adam/40BC，8192真实timeout。postBC meanKL .44683/1.64036，max6.937/10.373，各20BC；48步preaction缓存独立逐位、reward oracle0误差、feedback/nonfinite0，teacher/N不变。实际N也不合格，不以小N噪声解释放行。候选只将BC独立LR3e-5降1e-5，保留全部其它量和护栏，原640父重验；当前没有正式新增。

## 唯一BC步长修复v2

只将独立BC Adam学习率3e-5→1e-5。PPO adaptive LR/旧Adam矩/原640父/coef/门控/分组/epochs/奖励/预算和meanKL≤.5不变。新worktree `sitstand-near-teacher-lr1e5-worktree`，worker `run-sitstand-near-teacher-lr1e5-001.py`，status `sitstand-near-teacher-lr1e5-pipeline-status.json`。`sitstand-near-teacher-lr-only-source-diff.json`严格确认所有源码/资产相同，仅cfg一行不同；外置eval仅改源码路径。旧两次失败源不可覆盖。资格通过后允许同一原定320预算，无新增正式额度。

## LR1e-5结果

64×5全PASS：7680控制/400PPO Adam/35BC、32timeout、ONNX9.54e-7、BC meanKL最高.15404。actual16384在第3更新停：1179648控制/240PPO Adam/60BC、8192timeout、BC meanKL .06101/.23709/.56403；reward oracle0、72步preaction逐位、feedback/nonfinite0、teacher/N不变。实际Adam次数由保存状态再次验证，并非只按epoch乘法推算。累计此方案验证1,979,904控制，正式仍0。提出只继续同一BC步长数值因素3e-6，不更改其余或护栏，等待root明确选择。

## 固定BC步长3e-6准入（root明确授权）

新真实worktree `sitstand-near-teacher-lr3e6-worktree`，worker `run-sitstand-near-teacher-lr3e6-001.py`，PID548509/session12268。只BC独立Adam LR1e-5→3e-6；原0.5 guard、PPO、门控、coef、5epochs/最多4批和原320完整学习预算保持。失败停止，不自动再换配方。每轮BC mean/maxKL、有效STAND/SIT/ignored曝光、真实优化次数和耗时均已有落盘。原640的全部模型/数值资产按SHA逐项确认相同，见physical-assets-exact.json；验证权重不可进入formal。

## 实际N最终资格PASS

16384×16：6,291,456 validation controls、1280实际PPO Adam/320BC、8192真实timeout；feedback/nonfinite0、5项reward oracle误差0、contact155970/1048576。BC max meanKL .062218；PPO maxmeanKL .015424/clip .209764；稳态中位81301.87、整段聚合81437.65控制/s，未到10万，不改物理/PPO追指标。资格SHA `a791460ac2c14681f2ccbfbbad2015fccb57ae8fa178ad9d55144fab20f883e4`。所有验证结果不继承，正式只原640。末评/CPU复算均已串联；若正式数值护栏停，保留并停止，不自动改方。

## 末评外置修复

恢复入口 `resume_sitstand_near_teacher_evaluation_002.py`，同pipeline-status终态及新PID581916退出才交GPU1。首trace已含全部600×256真实actor输入/动作/qpos/qvel和2401相机样本/2400接触样本；actor/N冻结及原reward oracle600次断言均在保存trace前完成。首oracle最大误差张量未持久化，恢复report记null及证据边界，不伪造0；其余门槛/原始数据不改。其余5项768000诊断控制自动接续。失败源码SHA/状态和修复SHA保留。
