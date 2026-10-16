当前：成熟精度256续段及双条件评测已结束；Ball累计477,757,440正式controls，预算关闭，无自动续训。各原门槛结论见[完整曲线](sc0090-v2-ballkick-target-precision.md)及 `ballkick-target-precision-final-review.json`。

## 原阶段记录

# 当前：BallKick两配方阶段关闭未合格；GPU1已交回根代理

本子代理接管GPU1；GPU0已交回根代理camera准入。Roulade100,663,296控制、两配方关闭未合格；不再占GPU0或追加翻滚。

独立真实worktree：`p04/ballkick-finish-worktree`，detached070d606，从根代理已准入ballkick-repaired-worktree复制src/scripts/tests。保持V2 CAD/v5/S0O0无尖峰门控、61 actor/14动作/critic82、5ms×4、N24576/T24、PPO/E5/M4/原head与pose奖励、实际接触状态和原独立评测。新数值源仅4文件：ball_kick.py增加2项成本、ball_kick_finish.py纯函数、metrics追加独立oracle/各项样本统计、trainer严格跨源来源清单与精确恢复。

父检查点`ballkick-right-allocation-001/checkpoint_000085.pt`，已花50,135,040正式controls；原双条件有效触球均256/256，但完整/尾500ms站稳0。保存轨迹尾body高.109m/tilt约5°，速度.183m/s/角速2.4rad/s、双足约34%，球横速约.45m/s。球正奖约92–93%付给未站稳样本；只看forward不约束方向。

唯一最后修正：原+12 ball-forward/−4 overshoot保持，仅增加负成本：−4*valid*min(abs(ball_lateral),1)；−2*valid*[v²/(v²+.15²)+ω²/(ω²+1.5²)+非双足]。有效触球之前两项严格0；不加正到位奖、timer/历史state或actor输入，不将物理真值喂策略。Cauchy代数实现1−s²/(x²+s²)避免巨大有限速度平方溢出时inf/inf，但独立oracle保留原式在实际范围核对。踢后立即生效，可能影响follow-through/已学触球，需原评测实测，未预报成功。

CPU旧轨迹量级：名义末1s原球项(含overshoot)11.1206，方向−1.8130、quiet−3.6025后部分和5.7051；measured10.9924−1.8065−3.5789=5.6069。仅这4个分项，不是完整return/因果改进。稳定双足且相同球状态会去掉quiet成本；侧速0去掉方向成本。报告`ballkick-finish-proposal-audit.json`。

阶段左右共享上限100,663,296、最多2配方。最后仅85更新=50,135,040，累计100,270,080；剩393,216不足一次N24576更新，不用改名/N补预算、不训练左足。父85 actor/critic/normalizer/Adam/LR/clock2040精确继承，准入学习模型不作正式父。

当前CPU配置diff/纯函数+保存轨迹oracle与成熟64×5在同一解释器串联(session56601)，减少重复import。通过后actual24576×16、最后85和原两条件256例5s评测自动衔接；保持feedback/nonfinite/优化器/8类capacity护栏，真实球/身体/头中性/200Hz相机和健康分母分别报告。吞吐当前旧24576稳态82,388/s，尚未10万，不重复无限性能搜索。

CPU+成熟64×5同解释器完成PASS：train/play仅2项新增cost，旧reward/物理/观测/PPO不变；独立纯函数及保存轨迹最大2.38e−7，输入/RNG不变，valid前0且极大有限值成本有界。原85 actor/critic/N/Adam逐位恢复，LR.0015/clock2040；32真实timeout，官方ONNX实际64obs7.15e−7，无安全guard。GPU1 actual24576×16已PID63032，外置队列session54702；成本/正球收益/overshoot已分别进训练metric，feedback错误下标保持原9:11未错位。

实际24576×16准入PASS：9,437,184验证controls，24,281真实timeouts，fault/nonfinite0，contact192,063/1,572,864，五项reward oracle最大1.43e−6；排除首3稳态median86,947.55/aggregate86,603.07控制/s，未10万目标（资格如实false）。不追加性能档。新资格ballkick-finish-qualification.json已签，最后85正式PID70926/session54702启动初始化，仍从原85父出发，未用validation101权重；完成后原nominal/measured自动。

最后85正式完成：累计170更新/100,270,080 controls；共享上限余393,216不足一整次更新，不追加左足/第三配方。新段50,135,040，whole747.486s/67,071.57控制/s；排除首3稳态median83,925.95/aggregate83,979.53，仍未10万。194,154真实timeout，feedback/nonfinite0，无优化护栏。原名义末评PID83904正在运行，再自动measured；终marker后GPU1也交还根代理camera control，子代理只CPU复盘。

## 最终原始末评与阶段关闭

两个原门槛的nominal/measured各256例：有效正确足触球仍256/256；末1s滚速方向目标名义8→41、实测最终62；末.5s身体保持/身体且头中性/完整踢球均0。clean_motion父子均0，fault/nonfinite都0。最后170两条件原始report保存，actor/normalizer未变、五项物理reward oracle最大9.54e−7。阶段两配方共100,270,080 controls，余393,216不再改N用完；左足未训练。双方检查点/源SHA/原报告/完整时间保存在`p04-ballkick-retained-candidates-20261016.json`，明确UNQUALIFIED/runtime_qualified=false/hardware_tested=false。

只读CPU严格配对：两条件各自initial qpos/qvel/raw actor_obs/origins全部逐位相同。名义尾1s球横速.45547→.21909m/s，线速.18347→.13557，身体角速2.3867→1.6726rad/s，双足样本34.28%→58.25%；实测球横速最终.21830，线速.18340→.13169，角速2.4049→1.6300，双足34.25%→59.02%。这是真实部分减速/方向改善，但不是连续保持，原500ms门槛仍全失败。新成本为一个共同配方，不能拆分归因方向和quiet各自训练效果；旧85未训练新成本，复盘新成本只作相同公式的反事实比较，不称完整return。

头部稳像未改善：回正4–5s全样本camera角速RMS名义7.4073→7.4719、实测7.4187→7.4750rad/s；20ms exposure span P95均值名义.21554→.21368、实测.21553→.21164rad。所有这些窗的健康整窗分母仍0，不能把全样本变化称健康站立画面稳定。身体各轴RMS均下降而相机仍高，保留这一独立头稳缺口，无新增第三配方。

最终marker `ballkick-finish-pipeline-status.json`=BALLKICK_STAGE_COMPLETE_NO_FURTHER_TRAINING，GPU1已交根代理camera control；GPU0之前已交camera mild。子代理GPU任务全部结束，CPU复盘完成，不安排左足/部署/其它GPU动作。
