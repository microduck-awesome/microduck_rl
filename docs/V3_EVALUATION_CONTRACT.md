# V3 起身评测与 HOME 契约

## 模型和 HOME

评测使用本分支的 assembly V3 `46a1b2c1d371def5775d33e67424755f253e9881` 完整触地模型，`robot.xml` SHA256 `45a9ce13c521ac07080381e9d5597b040e1988b98fb9f89c981008a7dac778dd`，质量约 641.906 g。训练环境实际编译后的质量、关节、几何、网格和碰撞字段经 `v3-compiled-model-audit.json` 核对。此模型和评测均不证明实机零位或机械止挡。

HOME 是 assembly 选定的第二组站立姿态，不是把 CAD 关节全设为零。根部初始化高度 0.1285712173 m；左/右 hip roll 为 −4.0123°/+4.0123°，左右 hip pitch 均 −23.6711°、knee 均 +1.0082°、ankle 均 +22.6630°；neck pitch −20.0020°、head pitch +20.0020°；两侧 hip yaw、head yaw、head roll 和独立嘴为 0°。精确 15 关节弧度见 `src/mjlab_microduck/custom_v2/assets/model_metadata.json`。SC0090 固定目标短保持所得站立目标高约 0.1248 m；两者分别是初始化根高和受力后目标高。

## 七类恢复

训练器的评估和 `scripts/evaluate3762_recovery.py` 使用同一 V3 环境、HOME、重置几何保护和接触模型，保留 standing、sitting、prone、frontier、left_side、right_side、supine 七类。正式完整协议为 2 seeds × 每类每 seed 128 条 × 8 秒，无自动重置；当前课程阶段评估使用该阶段 frontier，最终对照另固定 stage 5。原评价是末段连续站稳，不单凭总奖励或一个姿态通过判定全部行为。

`scripts/evaluate_v3_recovery_clearance.py` 在原评估上额外按 200 Hz 记录全部 14 个策略关节距模型限位、连续靠近限位的时间及头壳/嘴、其他身体、双足与地面和自碰接触。末 0.5 秒还逐关节报告相对于 V3 HOME 的平均绝对误差、带符号误差和最大误差，包含 14 个策略关节及独立嘴；5°内样本数仅作描述，不代表自定的资格门槛。恢复期间接触可用于支撑，接触本身不自动判失败；报告包含末 0.5 秒，用于区分恢复过程和最终站立。模型限位不等于实测机械止挡。

两种 seed、每类各 1 条、8 秒的 CPU 端到端验证已完成，确认真实环境使用上述 15 关节 HOME、七类分组、14 关节限位、V3 头部碰撞体，以及每类末段 15 关节相对 HOME 的误差记录。含 HOME 误差的报告在 `/home/l40-vm01/data2/experiments/v3-rise6804-20261101/diagnostic-fixture/cpu-home-e2e/`。其输入是一次性 64×5 smoke 临时模型，不能作为正式策略成绩。

## 头部与回 HOME

`scripts/evaluate_v2_head.py` 保留历史文件名，但从本分支 V3 环境读取模型与 HOME：头四轴物理位置减各自 `default_joint_pos` 后再与相对 HOME 的命令比较，阶段范围也已按各关节的 V3 HOME 和限位计算。协议为 2 秒零命令、2 秒阶跃、2 秒回零，记录关节误差、身体稳定与 200 Hz 相机角运动；该相机位姿仍是 CAD 占位，并非实测相机标定。报告记录模型 SHA、质量、15 关节 HOME 和初始化根高。

头部评测已以一次性 smoke 模型通过 CPU 端到端检查：2 seeds、每个 seed 三个诊断指令（零、正颈俯仰、负头俯仰），核对 15 关节 HOME、根高、阶跃与回零时间、头部奖励计算及 200 Hz 相机轨迹，见 `/home/l40-vm01/data2/experiments/v3-rise6804-20261101/diagnostic-fixture/head-cpu-e2e/fixture-audit.json`。这只验证评测路径，正式完整命令网格及训练后策略评估仍未进行。`repro/compare.py` 是父分支的 V2 严格复现比较，要求 V2 相同头命令范围，不可直接拿来宣称 V3 与 V2 同模型或同 HOME；如需跨版本比较，应标明不同的模型、HOME、命令范围并仅比较共同协议。
