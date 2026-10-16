# BallKick 健康接触锁存与完整历史课程

当前：384连续正式及原nominal/measured四节点评测全部完成，预算关闭，无自动续训。新增226,492,416、BallKick全历史326,762,496；最终曲线和头稳/健康分母见本文末及 `p04/ballkick-health-history-final-review.json`。

批准一条fresh右足谱系：N24576/T24/E5/M24连续384，新增226,492,416正式controls、46,080 Adam；原Ball100,270,080保留，全历史上限326,762,496。每64存节点，原协议末评64/128/256/384。中期观察不换方，无teacher，不因原85/170短段失败否定原奖励，不自动下一段。

唯一健康语义修复：原200Hz bad锁存仅other-ball/head-ground，漏self；新独立startup读同substep self_collision.found(any)，与其它bad并集、优先于同帧正确触球，永久清valid至该env原生reset。旧ball_kick_state.py和旧任务行为不改，新HealthHistory注册替换startup hook。保留方向−4/回稳−2、原奖栈、CAD碰撞、S0/O0无尖峰去除、61actor/82critic/14action。

按官方模板N4096/T24/E5/M4的24576单minibatch，当前M24保持同批量；每交互Adam量相同，总剂量相当参考2304更新。旧课程max1500×24=36000，step÷6后6000；原严格>加5s=250control自然reset，末级预计6250才生效，384末nativeclock9216覆盖完整课程后的训练。所有物理秒数/5s episode不缩。未找到独立Ball成功run原始日志，此为实际模板课程与参数证据，非声称精确复现某已证实机checkpoint。

CPU已PASS：256状态组合独立真值、25个四substep脉冲先后、实际partial-reset函数选中隔离与critic副本、cfg仅新hook/课程step、agent仅M24、283非代码资产相同。旧kick_state_transition只作self=False等价，不作新self真值。GPU仍必须实际保存碰撞pose的单substep脉冲/部分reset/真实timeout successor及原六情境，随后64×5/ONNX与actual24576×16；正确性通过才从零384，验证权重不继承，吞吐如实报告不独立阻挡。

新末评额外200Hz保存raw正确足/其它球/头地/self及valid/bad。分别报告raw触球与健康valid，不能把健康拒绝导致valid减少写成触球能力下降。身体/方向/末保持原阈值和200Hz相机健康分母保持，动作阶段与稳像分开。原170和失败证据不覆盖。

## 实际准入与正式启动

真实32世界独立200Hz验证通过：16世界单物理子步self接触撤销valid并锁存bad，4世界部分重置仅清选中状态；12世界真实timeout successor保存reset前[valid=0,bad=1]、reset后[0,0]。六种普通/命令重采样/interval push的fast/full terminal观察逐位或最大差0，actor61/critic82。该接触脉冲通过保存实测姿态和诊断恢复制造，不冒称自然策略轨迹。

64×5 PASS：32timeout，ONNX最大2.38e−7，600次实际Adam。actual24576×16 PASS：9,437,184验证controls、1,920次实际Adam、12,288timeout；contact173,670/1,572,864，feedback/nonfinite0。稳态median77,014.49、aggregate76,990.61 controls/s，未达10万，按已授权规则保留结果、不追加性能搜索。

small、actual和formal原生初始化快照SHA一致，clock0/空Adam/normalizer count0；正式禁止load且不等于actual16训练后权重。正式每更新回调核对120次Adam；新预算226,492,416、全历史326,762,496不变。资格及外置fixture/evaluator/worker SHA和只读评测差分见 `p04/ballkick-health-history-admission-execution-receipt.json`。原准备plan及production.patch保留；评测raw-contact增量单列，不偷换数值源。

## 384 完整课程最终曲线（同原门槛）

| 节点 | 条件 | raw足触球 | 健康valid | clean | 滚速方向 | 身体保持 | 身体+头中性 | 完整 | fault |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | nominal | 143 | 127 | 164 | 0 | 0 | 0 | 0 | 0 |
| 64 | measured | 154 | 138 | 157 | 1 | 0 | 0 | 0 | 0 |
| 128 | nominal | 126 | 11 | 21 | 0 | 65 | 60 | 0 | 0 |
| 128 | measured | 137 | 12 | 25 | 0 | 63 | 53 | 0 | 0 |
| 256 | nominal | 256 | 256 | 255 | 127 | 107 | 101 | 51 | 0 |
| 256 | measured | 256 | 256 | 256 | 107 | 103 | 96 | 44 | 0 |
| 384 | nominal | 256 | 256 | 253 | 105 | 105 | 46 | 44 | 0 |
| 384 | measured | 256 | 256 | 253 | 84 | 101 | 51 | 33 | 0 |

新增正式226,492,416，累计326,762,496；实际46,080次Adam、nativeclock9216。完整formal共享进程阶段3025.505s/74861.04controls/s；稳态median75068.76、aggregate75248.88。完整worker含资格和8次末评另列3401.599s，不混为学习吞吐。

各条件64/128/256/384的原始qpos/qvel/actor_obs/origins配对核验、200Hz首次触球/自碰时刻和分窗口相机RMS/20ms曝光及健康分母见 `p04/ballkick-health-history-final-review.json`。完整资格不由某一分项代替；未完成原全门槛者仍未合格，runtime_qualified/hardware_tested=false。预算到此关闭，不自动续训。

## 补评192/320及成熟父选择

新增仅256,000诊断controls、正式0。192两条件健康valid86/89、完整1/0且头中性1/0；320健康valid256/256、完整34/27、完整且头中性24/17。各条件与64原始qpos/qvel/rawactor/origins逐位相同，200Hz锁存独立核验通过。原四节点报告不覆盖；新机器结果 `ballkick-health-history-extra-nodes-review.json`。

按两条件严格完整且头中性成功数的较小值，256的48/41仍最好，选其作新成熟精度父。实际manager日志256/320/384均action−1、COM.015、headCOM.01、push.3及全reward权重相同，不将退化归因后续加难。paired256→384 lost中球速/方向gate失败名义/实测各23，身体5/5（可重叠）；详见`ballkick256-384-course-failure-cpu-audit.json`。新Gaussian精度阶段是另列明确预算，未覆盖本384阶段消耗和全部候选。
