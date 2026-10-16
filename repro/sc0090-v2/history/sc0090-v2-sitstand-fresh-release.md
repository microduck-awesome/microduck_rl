# SitStand 正确配方独立从零谱系

当前：fresh128正式与双条件12s末评已结束，新增50,331,648、全SitStand历史226,492,416正式控制封顶。两条件身体tail/head到位仍全0，健康相机分母0；源/模型评测不变。未自动第二段。父级另外授权的M16只做16更新验证（6,291,456诊断、正式0），当前GPU1 PID271446/session25772，见数据盘 `sitstand-fresh-m16-plan-001.json`；不能把该诊断计为新正式或提前称M16更好。

本次问题：原SitStand128使用已证实约8.3mm自穿的错误SIT目标；之后ClearSeat256、Lesson4s384、TransitionRelease448都继承同一actor/critic/normalizer/Adam。因此正确ClearSeat+TransitionRelease没有任何一条从零谱系，448的零通过不能说成正确配方从零已失败。旧所有失败与176,160,768正式控制保持记录。

唯一新阶段：沿用 `sitstand-transition-release-worktree` 冻结数值源及同一任务，不改奖励/PPO/物理/DR/命令课程。N16384、T24、同一个seed2026101611，随机actor/critic、normalizer count0、Adam状态空、clock0。没有resume/源迁移参数，外置初始化记录器保存初始完整状态与SHA，禁止runner.load；正式禁止继承验证训练后权重。

正式预算单128更新=50,331,648新增，SitStand全历史硬上限 **226,492,416**。这是一条fresh谱系，不扫描种子/权重，不自动第二段。随机初始化也重启课程/优化时钟，和旧448比较不能视为只隔离某个单一优化因素。

准入：CPU核验冻结源与旧谱系父链、fresh与resume分支；fresh64×5含32真实timeout、物理reward oracle和官方ONNX；fresh16384×16检查原生terminal、PPO、feedback/nonfinite、contact容量和8种溢出日志。吞吐按真实结果报告，低于10万不单独阻塞或启动优化循环。通过后从全新0起正式128，所有验证权重丢弃。

末评：原command-clock-fixed256世界12s，名义及measured实际P04，四初态/命令×反向/返回、头响应、200Hz相机/20ms曝光与健康分母。原384/448全部保留作参考，不能把动作最终恢复或倒地后低角速当健康完成。不部署/轮滑，无实机成功声明。

记录：数据盘 `sitstand-fresh-release-plan-001.json`、`sitstand-fresh-release-cpu-001.json`、各段initialization.json/.pt、资格、formal run/status/report及双条件末评。run allocation_kind清楚区分validation/formal；新谱系inherited_controls=0，总历史另加176,160,768，避免冒继承旧模型或抹除旧计算预算。


CPU父链已实际核验448→384→256→原128；初始actor/critic与所有旧训练检查点不同。fresh64×5已PASS：N两套count0/mean0/var1/std1，Adam空、clock0、LR.001；32原生timeout、8oracle最大8.94e−8、ONNX2.38e−7、contact450/4096，feedback/nonfinite0。同worker已自动进入实际N16384×16，正式尚未启动。原数值源全部SHA与release资格一致。


实际N16384×16正确性PASS：16384真实timeout，contact103400/1048576，8oracle最大8.94e−8、maxKL.01648/clip.22934，feedback/nonfinite0；稳态median100960.76、aggregate99211.14controls/s。资格已签，正式从0开始，同PID248228已到20/128（7,864,320新增、clock480）。正式初始化N两套count0、Adam空、clock0、LR.001，not_validation_weights=True；正式actor/critic与验证16训练后的权重不同。旧176,160,768继续独立计入总预算，不塞入fresh checkpoint的inherited_controls。


## Fresh128收尾与关键CPU核对

fresh正式0→128：50,331,648新增、clock3072，两套normalizer.count均50331648，Adam非空且网络确有改变；没有旧/验证权重继承。完整formal phase501.6248s/100,337.25controls/s（含环境，排除共享imports/准入/评测），稳态median103844.77/aggregate102325.64。整个共享worker768.085s包含导入+两级准入+正式+两条件评测，不能使用正式分子冒称整条pipeline10万。

名义/实测四组×三phase保持/head到位全0；站起点站目标倾角约26°、head约31.5°，坐目标高约98.5mm仍未到75.999mm；坐起点全部曾>45°且尾tilt约68°。相机健康分母0，反馈/非有限0。与448评测初始qpos/qvel/raw actor obs/origins两条件逐位相同。正确fresh本预算也未成功，但初始化与累计更新/clock不等，不能据此将原坏初始化作用完全排除。

CPU关键审计 `sitstand-fresh-key-cpu-audit-001.json`：本次N16384×T24=393216样本/rollout，M4单minibatch98304，E5实际每update20 Adam；128更新日志共2560次/251658240样本visits。历史SitStand文档jngaedcs@1250明确N4096，引用c832147源T24/E5/M4，即单minibatch24576；旧原始W&B config/checkpoint未找到，历史节点1250约25000优化步/122880000控制仅按文档标签，保留legacy off-by-one不确定性。等交互50,331,648若N4096需512更新/10240 Adam，不是当前128等效。

全部10项课程在fresh clock3072与旧448 clock10752都仍stage0，最早升级12000（500更新）；没有把当前失败归因已发生的高难度升级。历史1250进入更晚阶段，但XL330物理/历史12s episode与V2当前4s也不同，不能直接移植性能结论。

当前训练near-complete命令hold（|flag-alpha|<.02）37,425,754/50,331,648=74.36%，其中sit17,494,599、stand19,931,155；健康sample只有176。训练exact g=1未单独存，不能把near-complete代理充作exact。保存评测nominal/measured g=1各90050/90054 of153600；所有末500ms样本g=1，仍保持0，因此尾失败不是模板gate尚未打开。M16是待验证更新密度因素，不称算法bug；无新正式预算。
