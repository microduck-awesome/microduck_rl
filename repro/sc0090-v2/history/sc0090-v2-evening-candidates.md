# V2 今晚起身候选交付台账

更新：2026-10-16 03:59，Asia/Shanghai。用户确认：**现有 Web 上传模型，ESP32-P4 板端推理**。目前是候选准备阶段，不是已完成实机交付。两卡继续起身/站立及对应头部任务。

## 05:10 新候选与范围更新

full-pitch4658是新增取舍候选：nominal1792/1792、共同measured1791/1792、fault0，两条件ordinary standing0/256；[-10,-5]密集1024例倒1（4594倒2），最终均恢复。measured头38(尾20)/115(112)/227(194)/6(4)，±1°尾92/66/217/10。唯一measured失败为仰躺曾站稳但末段未持续通过稳定条件，不能以最终姿态好看代替完整通过。检查点`p04/paired-catch-full-pitch-allocation-001/checkpoint_004658.pt`，所选谱系1,444,282,368控制；本64段50,331,648、完整formal578.245s/87042.11控制/s。旧4594/4466/3762和独立P04候选继续保留。

现在GPU0继续独立起身头控，GPU1推进步行准入以完成原全策略目标；不做P4/host部署、打包或轮滑。以下旧行的“继续/准备”是当时记录，当前状态以current-stage与主台账为准。


## 最新范围覆盖

用户随后明确先不做P4/host部署，只专注模型训练。以下部署准备产物保留，运行时工程暂停；当前优先改善并筛选训练候选，不能继续以今晚部署为由分散两卡训练精力。

## 当前训练候选（03:59，尚无综合全优或实机验收）

以下直接引用保存的训练checkpoint和评测。只更新训练筛选，不做P4/host部署与打包。小幅/当前单轴/两轴/四轴头控T90均采用同时到位并持续.1秒；括号为最后.5秒全程保持数，分母依次128/128/384/64。nominal和measured站姿分别标明，不能混用。

| 保留候选 | 完整P04恢复 | standing倒地 | measured头控T90（尾保持） | 当前用途 |
| --- | --- | --- | --- | --- |
| 随机化基线paired3762 | nominal、measured均1792/1792，fault0 | 原nominal28/256，仍有承接短板 | 78(11) / 128(116) / 351(206) / 53(12) | 保留头控较好的迁移参考，不被最新编号覆盖 |
| P04从零Head2-1664 | nominal、measured均1792/1792，fault0 | 正式0/256 | 3(2) / 97(96) / 156(148) / 3(2) | 独立从零谱系的身体和组合头控参考；完整头控未合格 |
| P04从零低熵Head2-1728 | nominal、measured均1792/1792，fault0 | nominal、measured均0/256 | 2(1) / 102(100) / 158(156) / 5(5) | 同父同64相比普通续训保留更多组合精度；最终四轴仍0，尚非全头控合格 |
| 基线适应Catch2s-3954 | nominal、measured均1792/1792，fault0 | 原nominal5/256；共同measured6/256 | 5(1) / 113(113) / 108(93) / 6(3) | 负pitch承接较好；[−9,−6]配对1024例零倒，头控退化 |
| 基线适应Pitch10-4018 | 原nominal1792/1792、fault0；共同measured1792/1792、fault0 | 原nominal2/256（1先、1后），measured2/256（均后），都负pitch | 3(2) / 99(98) / 86(74) / 7(6) | 新承接候选，头弱于3954，更不能称综合最佳 |
| 基线适应低熵Pitch10-4018 | nominal、measured均1792/1792，fault0 | nominal2/256，measured1/256（后） | 3(2) / 98(94) / 101(82) / 4(1) | 同父同64对照；有限收益，停止同配方 |
| 基线适应head cap-4018 | nominal、measured均1792/1792，fault0 | nominal4/256（2前2后），measured3/256（均前） | 5(3) / 110(105) / 134(121) / 8(5) | 组合有限恢复，仍小幅弱；同配方有界128续训中，不覆盖旧承接候选 |
| 基线适应head cap-4146 | nominal、measured均1792/1792，fault0 | nominal2/256（1前1后），measured1/256（前） | 10(8) / 107(106) / 170(150) / 7(3) | 承接与两轴继续改善，完整头控仍弱；同配方有界续至4274 |
| 基线适应head cap-4274 | nominal、measured均1792/1792，fault0 | 两者均1/256，首次稳定后 | 15(8) / 110(110) / 181(151) / 7(5) | [-10,-7]配对1024例4倒；完整[-10,-5]新配对1024例2倒；T12无综合收益，保留该T24父 |
| 基线适应accuracy-4338 | nominal、measured均1792/1792，fault0 | nominal1/256（后）、measured0/256 | 20(9) / 115(112) / 193(170) / 9(7) | 新有收益头控候选，保留并同源续128；完整负pitch配对尚未测该点 |
| 基线适应accuracy-4466 | nominal、measured均1792/1792，fault0 | 两者均0/256；完整负pitch尚未测此点 | 29(11) / 116(115) / 217(187) / 17(13) | 常用组合保持改善；微幅保持低于中间4416的14，保留两者；同配方续至4594中 |
| 基线适应accuracy-4594 | nominal、measured均1792/1792，fault0 | 两者均0/256；完整负pitch1024例2倒，均首次稳定前 | 30(22) / 116(113) / 204(168) / 7(3) | 小幅保持优于4466，但组合保持回落；保留多检查点，不自动同配方第四段 |
| 基线适应T12-4402 | nominal、measured均1792/1792，fault0 | nominal1/256（前）、measured0/256；整个负pitch1024例倒4 | 21(12) / 90(86) / 136(120) / 3(2) | 同100M对照，小幅好但常用单/组合退步，停止追加，保留T24 |
| P04从零micro1856 | nominal、measured均1792/1792，fault0 | 两者均0/256 | 1(0) / 114(112) / 171(160) / 11(6) | 同初态完整负pitch1024例零倒，当前单轴更完整但微幅弱；roll-only/accuracy1920未综合胜出，1856保留 |
| P04从零head cap005-1920 | nominal、measured均1792/1792，fault0 | 两者均0/256 | 3(3) / 116(114) / 165(150) / 8(3) | 同父同64控制单/双轴保持113(112)/153(138)，cap有局部收益，四轴保持3低于控制4；保留micro1856 |
| P04从零head cap+LR1792 | nominal、measured均1792/1792，fault0 | 两者均0/256 | 3(3) / 105(104) / 164(155) / 11(7) | 独立谱系身体保持，neck正向局部改善；微命令1856尚在验收 |

两网络冻结拼接仅为诊断：虽然到位更多，末段保持变差，未纳入本候选表，也未打包部署。

训练文件（持久根 `/home/l40-vm01/data2/experiments/microduck-v2-20261015/`）：

- `baseline/paired-feedback-allocation-001/checkpoint_003762.pt`。
- `p04/frozen-head2-allocation-001/checkpoint_001664.pt`；所选独立谱系1,308,622,848控制交互，未选实验另计。
- `p04/head2-late-entropy-allocation-001/checkpoint_001728.pt`；所选独立谱系1,358,954,496控制，本段50,331,648；完整Python进程542.35s、92803.1/s，run.json口径102569.3/s分列。未选实验和预检另计。
- `p04/paired-short-catch-allocation-002/checkpoint_003954.pt`；该128更新100,663,296控制，run.json至完成965.44s、104267/s（排除导入和末次旁评）。
- `p04/paired-catch-pitch10-allocation-001/checkpoint_004018.pt`；所选谱系940,965,888控制，本段64新增50,331,648；完整训练Python进程587.05s、85736.69/s，run.json至结束538.24s、93511.30/s，稳态准入130937.37/s分别记录。失败/未选分支预算未抹去。
- `p04/paired-late-entropy-allocation-001/checkpoint_004018.pt` 与 `p04/paired-stdcap-allocation-001/checkpoint_004018.pt`；相同3954父各64更新，不得混淆同编号文件。cap本段50,331,648控制，进程start至完成状态592.22s/84988.26每秒；精确exit未捕获，时间边界见主台账，非完整pipeline速度。
- `p04/paired-stdcap-allocation-002/checkpoint_004146.pt`：所选基线谱系1,041,629,184控制；本128段100,663,296控制，完整formal1000.704s/100592.51控制/s，含两次外部head及共同measured的pipeline1368.558s分列。
- `p04/paired-stdcap-allocation-003/checkpoint_004274.pt`：所选基线谱系1,142,292,480控制；本128段100,663,296，完整formal998.509s/100813.62控制/s，全continuation pipeline1380.557s。
- `p04/paired-rollout12-allocation-001/checkpoint_004402.pt`：与T24-4274同父、同新增100,663,296、所选谱系1,142,292,480控制；正式1011.854s/99484.05控制每秒，包含预检及末评全pipeline1681.572s。不是综合替换候选。
- `p04/paired-head-accuracy-allocation-001/checkpoint_004338.pt`：所选基线谱系1,192,624,128控制；本段50,331,648，完整formal578.220s/87045.78控制每秒，含物理预检和全部末评的pipeline1278.036s。
- `p04/paired-head-accuracy-allocation-002/checkpoint_004466.pt`：所选基线谱系1,293,287,424控制；本128段100,663,296，完整formal992.901s/101383.02控制每秒，包含末评的continuation pipeline1370.033s。中间`checkpoint_004416.pt`也保留。
- `p04/head2-cap005-allocation-001/checkpoint_001920.pt`：所选独立P04谱系1,509,949,440控制；本64段50,331,648，完整formal553.053s/91006.93控制每秒。同编号无改动control/roll-only/accuracy是不同目录、不同策略。
- `p04/paired-head-accuracy-allocation-003/checkpoint_004594.pt`：所选基线谱系1,393,950,720控制；本128段100,663,296，完整formal994.765s/101192.99控制每秒，标准末评全continuation pipeline1371.576s。后续负pitch/执行探针另计。
- `p04/head2-micro-exposure-allocation-001/checkpoint_001856.pt`：所选独立P04谱系1,459,617,792控制；本64段50,331,648，完整formal553.458s/90940.3控制/s。旧低熵1856是不同目录与策略，不能按更新编号混淆。
- `p04/head2-stdcap-lr-allocation-001/checkpoint_001792.pt`：所选独立P04谱系1,409,286,144控制；本64段50,331,648，完整formal544.334s/92464.59控制/s。未选实验预算仍在主台账和专职页。

GPU0保留micro1856与cap0051920等有取舍候选；同cap配方128准备中。GPU1新accuracy4594标准与负pitch配对已完成，后者2/1024倒；新单参数全负pitch覆盖课程已正式开始；以真实头控、承接和保持选优，不只看恢复率、奖励或GPU占用。

## 17:32历史候选与证据（已被以上训练进展覆盖）

| 候选 | 训练谱系/交互 | 已有成绩 | 未通过或待完成 |
|---|---|---|---|
| 随机化基线3756 | 基线根；总738,459,648控制交互 | 自身匹配延迟恢复1792/1792；CPU MuJoCo+ONNX 14/14完整8s；冻结ONNX2048条实际输入最大差2.86e−6 | 正roll T90 0/16；正neck15/16；当前固件完整运行契约尚未对齐 |
| 随机化基线4012 | 3756再256更新，新增50,331,648；该段run.json创建至结束468.25s，含初始化/评测，非纯训练耗时 | 3840/3968/4012自身均1792/1792；修复P04有效原始反馈nominal1780/1792、measured1716/1792，fault0；规范ONNX冻结最大差1.91e−6 | 负neck T90退至6/16、正roll仍0/16，不能按恢复率认定全面胜出 |
| 基线经P04适应3648 | 基线3628父713,293,824 + P04新增15,728,640 | nominal1791/1792、measured1769/1792、fault0；当前8单轴T90均16/16 | standing初态27/256先倒再起；还须规范冻结/完整运行适配；不计为从零P04 |
| P04从零重叠课程 | 随机根，已评测640为当前父；旧失败尾部算力另记 | 新704：907/1792、fault0；standing254、prone252、right251、frontier123、sitting27 | left/supine0；尚不合格，继续训练，不能包装成完成策略 |
| 基线头部精度axis3 | 从3756分叉，保持位置随机化/延迟/PPO/奖励总权重；只将fine std从统一.04改为neck/pitch/yaw .03、roll .012rad | 64×5/真实timeout/ONNX通过；8192×16稳态134535.92控制交互/s；正式session51021、PID2553004已运行 | 计划到4012后自动head与P04 nominal/measured检查；未预写改善结论；与旧4012同更新号但不同目录/配方/哈希 |

3756/4012与适应3648已测当前头命令范围相同：neck/pitch±.39、yaw±.49、roll±.11rad；从零当前范围更小，不能直接比较其头部指标。开发种子重复使用，最终仍需未用于调参的验收条件。

## 已落盘文件

持久根 `/home/l40-vm01/data2/experiments/microduck-v2-20261015/`：

- `baseline/candidates/baseline-randomized-3756/` 和 `baseline/candidates/baseline-randomized-4012/`：checkpoint、带normalizer ONNX、真实训练run/source、导出配置、2048输入oracle、evidence和candidate哈希。保持 simulation candidate 标记；没有伪称现固件兼容。
- `baseline/deployment-prep/baseline3756/`：现有严格转换器生成 `policy.mdp`；生产C推理核普通/packed FMA相对ORT259条检查通过；另2048实际观测最大误差3.34e−6。只是推理核验证，非完整控制环或P4时限实测。
- `baseline/cpu-gpu-3756-eight-seconds/`：同编译V2模型、v5 BAM、CPU足底，原生CPU+ONNX闭环14/14；GPU动作开环在CPU重放只有3/14，轨迹存在漂移，不能称完整物理逐步对齐。两种回放及失败均保留。
- `v2-rise-runtime-requirements.json`：Web/P4交付契约和已知差异。独立 `firmware-v2-runtime-worktree` 从当前干净固件ee77131建立，主固件工作区不变。

## 交付仍需完成

现固件旧HOME非零、动作EMA头.5/腿.7及目标限速，与当前V2训练不一致；原生LS3和P04 O0局部历史也不是同一时序实现。要在独立运行时适配这些已证实差异并验证，再交付可上机候选。不能只因ONNX/MDP形状正确就改标部署通过。硬件每关节零位/符号/比例按用户计划向V2训练对齐；不凭空生成实际校准值。

实机统一记录：每个初态的成功次数/总数、起身到稳定时间、站立不倒时长、姿态倾斜和抖动、头部正负/回中T90/稳态误差/超调、反馈无效与控制超时/推理耗时、供电工况。训练耗时分完整谱系、候选新增段、评测和失败实验；模型/固件/config版本与SHA共同记录，防止换模型时误比不同配置。
