# 2026-10-23：共享舵机动力组的力矩串位

目标：定位8040上机刚启用就姿态/方向异常，区分HOME、策略输出、封装与仿真执行。用户已暂停训练；本次只有源码、已有轨迹分析、CPU静态装配和前向力矩映射验证，没有新增PPO或GPU仿真。

## 已确认根因

引入提交为 `8eafdcfcc594f658acdc0a1f5019c244319caed8`（2026-10-21 20:45:35 +08:00），标题 `Share one battery sag model across all fifteen V2 servos`。该改动将原两个驱动组替换为一个15轴组。共享电池/负载压降是合理建模目标；错误在于合并计算组后沿用BAM的XML目的地收集顺序，缺少逐命名DOF的力矩路由检查。

`sim2real.configure_randomization`为共享3S供电把14策略轴和嘴巴合成一个SC0090V5Actuator。实体关节自然顺序为左腿5、头4、嘴巴、右腿5；XML的actuator顺序为左腿5、头4、右腿5、嘴巴。`sort_actuators=False`时，BAM按前者生成compute(cmd)的力矩列，却按后者收集`_mjs_actuators`，mjlab将力矩直接写入其ctrl_ids。因此后6列发生循环错位：

| 按哪个关节状态/目标计算力矩 | 实际施加到哪个关节 |
|---|---|
| mouth | right_hip_yaw |
| right_hip_yaw | right_hip_roll |
| right_hip_roll | right_hip_pitch |
| right_hip_pitch | right_knee |
| right_knee | right_ankle |
| right_ankle | mouth |

这不只是输出标签错误：一个关节的目标、位置、速度产生的实际力矩作用到另一个关节，属于错误闭环动力学。不能通过改变实机HOME或交换导出输出几列可靠修复。策略可能在错误的plant上优化出较高起身/站立成绩，实机正确接线则不再实现它学到的关系。

## 已有数值证据

8040实际tilt评测中neutral/sample0，首帧右髋yaw原输出−1.88644rad，目标截断到−.992128rad；右roll−.549262rad、右踝−2.42064rad。该样本后续yaw原输出可达约−34rad，目标仍截断，而实际yaw约−1°，right_roll约−27.6°。这些现象由错位驱动解释，不能仅归咎为HOME偏差或奖励不够。

同一理想直立输入（gravity[0,0,−1]，关节/速度/命令/previous_action为0）下，8040右yaw/roll/踝约−108.08/−31.30/−143.30°，旧3762约+4.25/−15.67/+.72°。输入合成对比本身不是闭环动作成功证明；实际保存轨迹和逐轴力矩去向提供了更直接证据。

之前checkpoint/ONNX/MDP数值一致和CRC检查仍成立，但只证明同一网络被完整封装，不能证明它训练时的驱动映射正确。仿真评分读取真实关节姿态，却未核验每列驱动力矩到达同名关节，因此没有识别错误plant。

## 历史与影响范围

旧P04 1856、基线转P04 4466和基线3762的冻结源码均为14轴策略组＋独立嘴巴组，两组内顺序一致；源SHA及原分组见`historical-sources.json`。这三者没有此次合并组导致的串位证据，不能拿本故障解释其历史反馈/异常诊断问题。

确认受影响的已交付模型：起身8040、步行3400，以及早期本轮起身4000、步行4600，均为共享15轴/不排序装配。各原ONNX SHA见`affected-releases.json`。暂停的步行3650同样继承错误plant训练谱系，不作为正确模型可用进展或恢复起点。原文件保留供追溯，撤回其上机候选资格，现有仿真通过率只描述错误plant。

## 本地修复与必要验证

独立分支`fix/v2-actuator-target-order-20261023`，不改历史训练分支。SC0090V5Actuator在BAM完成edit_spec后按计算目标列的名字重排control目的地；initialize要求整个目的地名字列表与target_names精确相同，否则拒绝启动。共享15舵机供电、动力参数、CAD关节顺序、动作/观测语义保持。

新增`tests/test_sc0090_actuator_mapping.py`，装配真实起身/步行配置、每轴注入不同小力矩，CPU MuJoCo `mj_forward`检查每个命名DOF的qfrc_actuator。修复前原配置在mouth检测到.015而非.010Nm；修复后起身/步行×sort开关的4种装配均通过，标准unittest共1个测试方法（4个subTest）。无积分、无GPU、无学习。命令：

```bash
bash scripts/v2_python.sh -m unittest discover -s tests -p test_sc0090_actuator_mapping.py -v
```

证据：本目录的`evidence/`。只做了本缺陷必要回归，不重跑大规模资格或训练。后续如用户授权恢复，应先在正确映射的624.6g模型下检查可信旧策略，再决定重训/迁移；错误plant的checkpoint不能原样视为健康续训起点。当前保持暂停。

本目录同时保留最小源码修复补丁及定向CPU回归。原训练分支保留；本地修复代码提交37e3066e、引入定位文档提交562dea42。尚未进行修复后策略评测或正式训练，不把修复前评分移植为修复后评分。
