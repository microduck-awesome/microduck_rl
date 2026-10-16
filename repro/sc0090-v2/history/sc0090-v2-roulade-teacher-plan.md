# Roulade末段教师训练准入

2026-10-16T11:46:41+08:00。目标仍是单一61D策略完成有支撑的完整前滚并健康站稳；512/512外置教师恢复只证明落稳能力可用，不能替代完整技能。旧正式100,663,296已关闭；本次拟最多64×16384×24=25,165,824新增，累计125,829,120，明确是新增有限试验，未通过准入前正式0。

## 唯一行为方案

冻结原256作为起滚/未到末段的动作参照，冻结起身1856作为末段教师，使用两者各自原归一化。保留原奖励、reset课程、动力学、测量、actor61/critic74、PPO全部超参数。末段训练门控为采样动作之前的head latch有效、supported frontier≥300°、current accumulated≥260°；后一个条件防止已回转的状态仅靠旧最大角度继续取末段标签。门控不进入actor或critic，实时推理仍只有原61维测量/命令。

BC沿用已验证的双教师模式：独立Adam lr3e−5、5epochs、最多4个均衡minibatch、每批最少512（不足整批时保留一个批）、末段不足64帧时不使用该teacher；教师/原策略anchor系数均1，不做权重搜索。原PPO仍T24/E5/M4、native adaptive LR/.99/.95/.2clip/.01entropy，原Adam/N/LR/clock6144精确恢复，BC Adam首次新建；后续同源resume必须恢复两个Adam及BC计数。

PPO存储按time-major排列；每次act前将native frontier/accum/latch复制到独立GPU缓存，记录同control clock，BC只消费完整24步缓存。原actor输入/现场状态不改，partial reset后的下一帧使用reset后的native上下文。单独保存最后验证段原始context和PPO observations供独立门控/维度核对；老师权重和归一化末尾逐位不变。

## 准入与停止

106原源核对复制后只改3文件：roulade.py算法配置、train_v2_roulade.py学习/迁移保护、新roulade_teacher.py。CPU递归比较环境完全相同、PPO配置仅新增BC算法配置；源码和父run/checkpoint哈希锁定。随后成熟父64×5、真实timeout/shadow successor/ONNX，再实际N16384×16；验证模型不作为正式父。

PPO KL>.5/clip>.8、BC KL>.5、反馈异常、非有限或容量异常停止并保存，不为让预检通过而放宽。正式候选仍需原5s standing/midroll名义与measured评测，必须区分完整滚转/侧滚失败/单纯恢复。新phase-aware teacher可能因隐藏历史与相近actor输入产生冲突，成功尚未证明；只允许此一配方和预算，不用额外phase真值补actor来绕过部署约束。

当前准入进程PID138792/session95340，`run-roulade-teacher-admission-001.py`复用导入串联CPU、64、实际N；输出`roulade-teacher-admission-status.json`，尚未正式。GPU1独立执行GroundPick单次静力求解候选的真实3s保持，不做PPO。

## 2026-10-16T11:59:25+08:00 准入通过与固定配对预算

001在PPO更新前因配置比较时机错误停止：比较到了env/diagnostic包装后的函数引用。保存的构建前run环境与父逐项相同；改为在构建前捕获不可变环境/agent快照，保留failure.json和001源。没有放宽物理/配置门槛。002重新CPU/64/实际N全部通过。

成熟64×5：32真实timeout，ONNX最大4.77e−7、5物理oracle全0、teacher/N不变，BC KLmax.14891。实际N16384×16：6,291,456验证controls、24,576timeout、contact102530/1048576、feedback/nonfinite0，BC KLmax.34699；稳态aggregate105,057/median106,244控制步/s。资格`roulade-teacher-qualification-001.json`锁定当前源和138资产。验证272不继承，正式父始终原256。

为区分继续PPO的自然改善与teacher新增收益，GPU1增加**同父原PPO对照64**，仅显式GPU0→GPU1 RNG恢复映射改变trainer，物理/奖励/PPO不变。新比较明确总额两臂64+64=50,331,648，Roulade全历史累计上限150,994,944；这项预算修改发生在两臂正式开始前，原25,165,824单臂提案仍保留，不冒充没有增加预算。此为固定两臂，不自动扩大或扫描teacher权重。

GPU0 worker001此前仅等待两臂同时准入，已核验无正式目录/0更新后保存receipt结束。两臂没有时间依赖，已合格teacher组由worker002/session11397独立启动；GPU1 control PID160841/session31755独立完成其准入后，root核查双资格SHA再创建pair-ready供control接续。资格、模型、优化器与正式预算均未因调度修改而变化，也没有重启已开始的训练。GPU0旧预加载耗时/等待时间保留于receipt，不混入新worker训练吞吐。
