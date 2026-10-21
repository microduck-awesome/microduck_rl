# 基线训练的延迟与运行时行为对齐

更新日期：2026-10-21。

面向 SC0090 真机的基线训练，采用位置观测 **5 ms**、速度观测 **20 ms**、
IMU **0 或 20 ms 随机**、动作下发 **10、15、20、25 ms 四档随机**的配置起点。
动作随机化按本次讨论替代先前的固定 10 ms 建议；这些是训练近似与随机化范围，
不是四个精确测得的硬件固有时延。本文件记录方案，训练代码尚未应用这些变更。
末节同时记录官方运行时的目标平滑、越限及缺反馈处理，与本机 V2 路径分别比较。

## 我们的训练配置

物理步长为 5 ms，策略每 4 个物理步运行一次，即 50 Hz。统一以策略观测参考时刻
`t₀` 为仿真决策时刻，观测从过去取值，动作在之后进入执行器控制律。

| 通路 | 配置 | 随机化 | 实现单位 |
|---|---|---|---|
| 关节位置 | 5 ms | 首轮固定该近似值 | 1 个物理子步的观测历史 |
| 关节速度 | 20 ms | 首轮固定该近似值 | 1 个策略步的观测历史 |
| IMU 角速度与重力方向 | 0 或 20 ms | 暂保留官方两档等概率随机，每 64 个策略步更新 | 0 或 1 个策略步 |
| 动作下发 | 10、15、20、25 ms | 四档等概率随机，每个物理步更新，每个环境独立 | 2～5 个物理步 |

动作延迟配置在 BAM 执行器配置对象上：

```python
delay_min_lag = 2
delay_max_lag = 5
delay_update_period = 0
delay_hold_prob = 0.0
```

这里 `delay_update_period=0` 表示每次物理更新重新采样，不表示关闭随机化。
四档平均延迟为 17.5 ms。上限 25 ms 用于扩大鲁棒性训练范围，并非实测发现了
25 ms 下发延迟；策略依然每 20 ms 产生一次动作。

沿用官方实现时，随机化通过每个物理步重新选择历史目标完成，并非给每条策略命令
抽取一次固定延迟后按 FIFO 投递。改变档位可能重复或跳过历史目标。单个执行器组内
的全部关节共用该环境的延迟；若分成多个执行器组，各组默认分别随机。

位置 5 ms 不能直接写成 `ObservationTermCfg.delay_min_lag=1`，因为普通观测延迟
的一档是 20 ms。需要在物理子步保存位置历史，在策略采样时读取上一物理子步；
同时避免再叠加普通观测管理器的整帧位置延迟。

## 官方训练代码的默认值

本次在线核对 `pollen-robotics/microduck_rl` 的 `main`，固定提交
`1e79c29c97d8b38aee9eefde77a545860ba7658e`。以下适用于该版本的
Velocity 与 Standup 配置，不能泛指所有独立任务或历史基线检查点。
该版本锁定 mjlab 1.3.0，默认物理步长 5 ms、策略周期 20 ms。

| 通路 | 官方参数 | 对应时间 | 更新方式 |
|---|---|---|---|
| 关节位置 | 观测项未覆盖延迟，继承 `min=0, max=0` | 0 ms 额外延迟 | 不随机 |
| 关节速度 | 观测项 `min=1, max=1, update_period=0` | 固定 20 ms | 范围相同，实际没有随机变化 |
| IMU 角速度 | 观测项 `min=0, max=1, update_period=64` | 0 或 20 ms | 等概率两档，每个环境周期性重采样 |
| IMU 重力方向 | 同上，独立的观测项 | 0 或 20 ms | 与角速度分别持有延迟缓冲 |
| 动作目标 | BAM `min=3, max=6`，其余沿用默认 | 15、20、25、30 ms | 每个物理步等概率重采样 |

官方当前这两个模板并不是“位置和速度都固定延迟 20 ms”。位置仍有噪声、偏置等
观测处理，但这些不等于时间延迟。
来源：[Velocity 观测配置](https://github.com/pollen-robotics/microduck_rl/blob/1e79c29c97d8b38aee9eefde77a545860ba7658e/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py#L564-L613)、
[Standup 观测配置](https://github.com/pollen-robotics/microduck_rl/blob/1e79c29c97d8b38aee9eefde77a545860ba7658e/src/mjlab_microduck/tasks/microduck_standup_env_cfg.py#L559-L589)、
[BAM 延迟配置](https://github.com/pollen-robotics/microduck_rl/blob/1e79c29c97d8b38aee9eefde77a545860ba7658e/src/mjlab_microduck/robot/microduck_constants.py#L120-L134)。

官方动作下限 15 ms 的直接来源是 `delay_min_lag=3` 乘以 5 ms 物理步。
该配置处没有给出为何选择 3 步的实测数据或推导，不能将其认定为实测推理耗时、
UART 固定耗时或 50 Hz 控制的必然下限。官方配置针对 XL330，
本机 SC0090 与 P4 的训练范围按本机证据另定。

IMU 的 64 步对应 1.28 s。默认每个环境独立采样，且有随机更新相位，因此环境之间
不会同时换档；角速度与重力方向的缓冲也未强制共享随机档位。取值是离散的
0 或 20 ms，不是连续均匀的 0～20 ms，更不是正负 20 ms。
默认 `delay_hold_prob=0`；reset 清空历史，历史不足时按已有历史长度限制实际延迟。
来源：[mjlab 延迟缓冲](https://github.com/mujocolab/mjlab/blob/v1.3.0/src/mjlab/utils/buffers/delay_buffer.py)。

观测延迟的单位是观测更新步，执行器延迟的单位是物理步；不能仅凭相同的
`delay_min_lag` 字段名把两者都乘以 20 ms。
来源：[观测参数](https://github.com/mujocolab/mjlab/blob/v1.3.0/src/mjlab/managers/observation_manager.py)、
[执行器参数](https://github.com/mujocolab/mjlab/blob/v1.3.0/src/mjlab/actuator/actuator.py)、
[基础环境时钟](https://github.com/mujocolab/mjlab/blob/v1.3.0/src/mjlab/tasks/velocity/velocity_env_cfg.py)。

## 实测依据和近似边界

V5 使用原始 848 条训练记录选择时移，844 条验证记录评估；搜索间隔为 1 ms。
内部位置相对读取中点的整体最佳时移约 5 ms，相对读取完成约 6 ms。
LS3 速度相对最新回包约滞后 13 ms，其中三点窗口本身贡献约 10 ms。
P=15 子集得到接近结果。位置加 5 ms 外推的离线近似仍得到约 6 ms 最佳时移。

位置结果随轨迹变化约为 4～27 ms，不能全部解释成固定传感器延迟；
回差、动态误差及比例残差可能混在其中。因此位置 5 ms 是简化起点，
不能视为所有动作下都成立的精确参数。
V5 只有主机读取开始/结束及外编码器 CAN 接收时刻，没有两侧硬件采样时刻，
不能通过扣除整个读取事务时长得到精确内部时延。
见 [V5 分析](../../microduck-firmware/validation/v5-feedback-delay-20261021/summary.json)、
[轨迹分组](../../microduck-firmware/validation/v5-feedback-delay-20261021/profile-summary.json)。

D747A9 的 180 秒固定 count 带扭矩试验中，捕获样本的最新关节回包到 `t₀`
年龄中位数为 5.477 ms。将其与 V5 的 LS3 约 13 ms 相加，得到约 18.5 ms 的
跨实验近似，按 5 ms 物理步长取 20 ms。该试验没有动态关节真值，不能单独验证速度滞后。

同轮试验的 `t₀ → UART 发送完成` 三路中位数为 8.385、8.324、8.422 ms，
全程累计最大值为 10.622、10.496、10.564 ms。本方案的动作四档随机化以此为背景，
主动扩大到 25 ms。UART 发送完成表示主控完成发包，不代表舵机已执行到目标角度。
见 [全链路记录](../../microduck-firmware/validation/feedback-freshness-r2-20261020/fixed-count-180s/summary.json)。

IMU 以 120 Hz 采样。捕获样本中，角速度和姿态的来源年龄中位数约 4.84 ms、
最大约 9.50 ms；固件还会外推角速度、传播姿态到 `t₀`。这些年龄不能证明芯片滤波、
姿态融合后的有效动态延迟为 5 ms。尚无独立动态参考来拆分两者，因此暂保留官方
0/20 ms 的随机化，而不将来源年龄直接改写成 IMU 固有延迟。
本轮 Wi-Fi 捕获存在丢帧，上述年龄分布只覆盖已录样本。

## 避免重复计入延迟

以 `t₀` 为统一训练决策时刻，本方案的位置与速度延迟表示对齐反馈相对 `t₀` 的
有效滞后；动作延迟包含 `t₀` 之后的观测准备、推理、调度和发送，不能再额外叠加推理时间。

如果以后改为策略输入时刻 `t_in` 作为决策时刻，则位置与速度观测延迟分别增加
`t_in - t₀`，动作延迟改为 `t_tx - t_in`。现有捕获没有独立的 `t_in`，
不能精确拆分这两段；不能既增加观测侧的准备时间，又完整保留 `t_tx - t₀`。

若将来训练直接实现 100 Hz 采样、三点 LS3 和外推，应重新核对总观测延迟，
避免在算法自然产生的窗口滞后之外再加一次完整 20 ms。
执行器的力矩建立、摩擦和机械响应属于动力学模型，也不等于额外的通信延迟。

## 官方运行时的输出与缺反馈处理

本次核对官方 `pollen-robotics/microduck` 的 `main`，固定提交
`942e58e3aa6dbb28e7037203fff11866122c7567`，提交日期为 2026-10-19。
这是官方 `robotd` 与 `duck-control` 的运行路径；以下总线读取结论针对 Dynamixel 后端。
本机自研 ESP32-P4 固件的 V2 路径另列，不因同属“基线”而视为相同代码。

### 本机 V2 的网络输出数值就是目标角度

通用转换是 `目标角度 = HOME + action_scale × 网络输出`。本机当前 V2 Rise 模型契约
中 HOME 的 14 个策略关节均为 0，`action_scale=1.0`，固件 `v2_home` 也全部为 0。
因此网络输出在数值上直接等于 CAD 关节坐标中的目标角度，单位 rad；它不是相对
当前实测角度的增量。之后仍需经过关节限位和 O0 仿射标定转换，才能成为舵机 count。
“相对 HOME 的动作”与“数值就是目标角度”在该零 HOME 约定下同时成立。
来源：[V2 模型约定](../../microduck-firmware/esp32p4/tools/contracts/v2-rise-candidates.json)、
[实际 HOME 与目标映射](../../microduck-firmware/esp32p4/components/microduck_core/microduck_v2.c)。

### 策略输出默认有低通平滑

官方先计算 HOME 加缩放后的策略动作，再对目标角度做一阶低通：

```text
filtered[k] = alpha × target[k] + (1 − alpha) × filtered[k−1]
```

头颈四关节的默认 `alpha=0.5`，十个腿部关节默认 `alpha=0.7`，嘴不在这两组内。
配置 `head_lowpass=1.0`、`legs_lowpass=1.0` 可分别关闭；实际行为以设备配置为准。
正常首帧没有历史时直接通过，特定恢复入口可用保持姿态初始化滤波历史。
观测中的 previous action 保存原始网络输出，滤波历史保存目标角度，两者不同。
来源：[目标构造与低通](https://github.com/pollen-robotics/microduck/blob/942e58e3aa6dbb28e7037203fff11866122c7567/robotd/src/control.rs#L647-L739)、
[默认值与关闭方式](https://github.com/pollen-robotics/microduck/blob/942e58e3aa6dbb28e7037203fff11866122c7567/robotd-params/src/lib.rs#L2169-L2170)。

系数来源可追溯到原型运行时的配置沿用。官方设计说明表示保留这些滤波是为了延续已有
运行效果，并将其列为可调参数；未提供从电机参数、传输时延或截止频率推导出
头部 0.5、腿部 0.7 的过程。训练历史也存在不同取值：2026-07-06 的 `9cc8fc1`
加入头腿均为 0.6 的输出 EMA；其后 `69ebeb6` 又从该 Velocity 配置移除了它。
这些提交在本次核对的官方训练主线祖先中，但不能据此认定某个已发布模型经过了哪套训练。
来源：[运行时设计说明](https://github.com/pollen-robotics/microduck/blob/942e58e3aa6dbb28e7037203fff11866122c7567/docs/design/robotd-design.md)、
[加入 0.6 方案的提交](https://github.com/pollen-robotics/microduck_rl/commit/9cc8fc1aa8dad3fe5b5ad1774be3db6d9ed968c2)、
[移除该方案的提交](https://github.com/pollen-robotics/microduck_rl/commit/69ebeb632e061a6f8bfda18e7a135a22da1090d7)。

运行时注释声称这些系数匹配训练；但本次核对的官方 Velocity/Standup 训练模板使用
标准 `JointPositionAction`，动作路径只有缩放、偏移及可选 clip，未发现同样的输出 EMA。
奖励中的 action-rate 惩罚、头部偏置 EMA 不等于执行目标的低通。不能凭注释认定当前
两个仓库或某个历史模型训练时具有相同滤波。
来源：[训练动作入口](https://github.com/pollen-robotics/microduck_rl/blob/1e79c29c97d8b38aee9eefde77a545860ba7658e/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py#L270-L273)、
[实际动作处理](https://github.com/mujocolab/mjlab/blob/v1.3.0/src/mjlab/envs/mdp/actions/actions.py#L154-L225)。

若要采用官方运行时的这组平滑，训练和部署应显式匹配滤波、初始化与 previous-action
语义。低通有频率相关的幅值和相位变化，不能仅把它折算成固定动作延迟。

### 越限截断到执行器范围

官方安全层把有限目标逐项截断到 `[-π, +π]`，记录 Range 状态后仍发送整组目标，
不会仅因目标越界而主动卸力。这个范围是 XL330 的整圈坐标范围，不是每个关节
在 CAD 中的机械上下限；超出 CAD 限位但仍在 ±π 内，不能指望该层截断。

如果任一目标为 NaN 或 Inf，则拒绝整组策略目标并尝试发送保持姿态，不把非法数值
转换成限位角。来源：[安全层边界定义](https://github.com/pollen-robotics/microduck/blob/942e58e3aa6dbb28e7037203fff11866122c7567/duck-control/src/safety.rs#L40-L48)、
[越限与非有限数值处理](https://github.com/pollen-robotics/microduck/blob/942e58e3aa6dbb28e7037203fff11866122c7567/duck-control/src/safety.rs#L360-L428)。

### 单关节缺一帧时沿用上一份完整传感器快照

官方一次 combined sync-read 读取 IMU 和 15 个舵机。任一设备超时、回包缺失或长度
不符会让该次读取整体失败，不会把其余新反馈与某个缺失关节的旧值拼成成功快照。
来源：[Dynamixel 读取路径](https://github.com/pollen-robotics/microduck/blob/942e58e3aa6dbb28e7037203fff11866122c7567/duck-control/src/bus.rs#L513-L579)。

已有成功快照且其他运行条件有效时，`Coast` 允许连续 3 次读失败沿用上一份完整快照，
策略继续推理和发送，previous action 与滤波历史保留，并非只重发上一条动作。
第 4 次连续失败不再向策略提供快照，暂停策略，尝试保持最后成功读取的实测姿态；
该读失败路径本身不主动卸力。读成功即清零缺帧计数。短于 200 ms 的被动暂停后恢复，
不会仅因此重置控制器历史；显式停用等入口有独立规则。

沿用旧快照时保持原读取时间戳，跌倒判定与里程计只接受新鲜读取。
3 次是循环次数窗口，名义上对应 60 ms；总线超时会拉长实际时间，不能视为严格
60 ms 的墙钟截止。
来源：[读取与沿用旧快照](https://github.com/pollen-robotics/microduck/blob/942e58e3aa6dbb28e7037203fff11866122c7567/robotd/src/main.rs#L2364-L2420)、
[Coast 与恢复规则](https://github.com/pollen-robotics/microduck/blob/942e58e3aa6dbb28e7037203fff11866122c7567/robotd/src/main.rs#L3987-L4037)。

### 与本机 V2 固件的差异

| 项目 | 官方上述运行路径 | 本机 ESP32-P4 V2 路径 |
|---|---|---|
| 输出平滑 | 默认头部 0.5、腿部 0.7 的 EMA | D747A9 已安装默认值，Web 可调整并持久保存，1 为直通 |
| 越限目标 | 统一 ±π 的执行器范围截断 | V2 策略目标按 O0 映射后截断到 SC0090 实际 EEPROM 目标范围 20～1003 count |
| 单关节缺反馈 | combined-read 失败后复用整份旧快照 | 只复用缺失关节的上一有效位置/速度，其他关节继续更新 |
| 容忍窗口内 | 用旧快照继续推理 | 仍运行推理，保留原始来源时间，不伪装成新采样 |
| 缺有效观测的容忍窗口 | 最多 3 次连续失败；另有恢复规则 | 最多 3 个控制帧，另有 90 ms 墙钟保护；超过后暂停新推理并保持最后已知姿态 |

本机最后两项只描述可恢复软缺样；硬故障及授权撤销不借此续行。
2026-10-21 首次安装阶段已完成 D747A9 应用升级及默认系数的重启读回，当时未运行运动试验。
记录：[安装与保存验证](../../microduck-firmware/validation/official-behavior-align-20261021/install-summary.json)。
本次部署行为按用户要求更改；已有模型权重和冻结训练元数据保持原样，新训练需显式
使用相同 EMA 及 count 目标边界。训练方案中的动作随机延迟与 EMA 是两个独立处理阶段。
手动测试和 HOME 仍使用原配置限位，策略原始反馈可保留 0～1023 count。
代码入口：
[V2 输出与保持](../../microduck-firmware/esp32p4/main/control/microduck_control.c)、
[V2 限位映射](../../microduck-firmware/esp32p4/components/microduck_core/microduck_v2.c)、
[单关节连续性窗口](../../microduck-firmware/esp32p4/main/include/microduck_control_feedback.h)。


### 2026-10-21 后续实机选择及部署状态

用户在对照后选用基线 3762，4466 的实物评价为“更差”。最新固件已完成三次独立
60 秒基线运行，每轮 3000 次推理、无缺反馈/目标回读不匹配；这不是一次连续
180 秒，也不等于所有场景均可自行起身站稳。汇总见
[固件实测记录](../../microduck-firmware/validation/baseline-selection-20261021/physical-trials.json)。

部署侧电压补偿已启用：策略使用 1 Hz 更新、α=0.1 的电压 EMA，倍率为
`11.1 / clamp(V_filtered, 9.0, 14.25)`，之后进入目标 EMA。原始 10 Hz ADC
仍用于电源保护。参考电压和开关可在 Web 保存。EMA 头部 0.5、腿部 0.7。
策略目标截断 20～1003 count；手动/HOME 额外遵守保存的关节限位。

本文件提出的位置 5 ms、速度 20 ms、IMU 随机延迟及发送 2～5 个子步（10～25 ms）
是后续训练配置建议，不代表现有基线/P04 权重已按此重新训练。


### 2026-10-21：基线 3762 连续 15 分钟结果与设备配置同步

**当前实物效果最好的是 `rise_baseline_paired3762`，作为后续实机基准。**
D747A9 从伏卧、有防倒保护的状态连续运行 900.000776 秒，操作者反馈
“能起身并基本站稳”。45000 次推理、44999 次目标提交；末帧结果越过到期时刻
6.539 ms 而未提交。无故障、无重启、无策略缺反馈，15 台舵机目标回读无不匹配。
电脑状态读取超时 1 次没有卸力，P4 自行到期停止，最终三路全部卸力。
HTTP 采样到的舵机最高温度为 68°C；这不是连续温度或电压瞬态测量。

[实测数据与模型身份](../repro/sc0090-v2/evidence/hardware/baseline3762-15min-20261021.json)
和[当前 D747A9 部署快照](../repro/sc0090-v2/deployment/D747A9-20261021.json)
均保存在本仓库，在线配置已核对与试验前配置、固件默认关节夹具完全一致。
最佳判断来自本轮及此前操作者对照，不是随机化多机成功率实验。

同步规则：

- `zero_count`、带方向的 `radians_per_count`、各关节角度/count 上下限属于实机映射，
  用于部署与实测重放，不直接替换训练模型的 CAD 坐标。
- 当前 V2 HOME 是按明确 15 关节顺序排列的全零弧度，策略使用其中除嘴巴外的 14 维。
  部署快照同时保存对应目标 count；这是计算结果，本次同步未发送 HOME 命令。
- 固件策略目标统一截断到 20～1003 count；手动/HOME 还遵守保存的逐关节上下限。
  训练关节约束、动作截断与实机手动限位各自记录，不混用。
- 修改零位若改变同一 CAD 角度在实机上的物理姿态，会影响部署一致性；不能把
  调整过的 HOME 当作不影响现有 MLP 的任意原点平移。
- 现有模型权重、冻结训练配置及历史 qualification 元数据保持原样。后续仿真复现
  当前实机时须显式加载或实现这些部署行为；保存 JSON 本身不会自动改变训练环境。


同步快照中的两类限位明确分开：`limits.per_joint[].servo_target_rad_range` 是将
20～1003 count 按当前零点、方向和 O0 仿射映射换算后的**舵机目标指令弧度范围**；
`joint_angle_rad_range` 是设备保存的**关节角度上下限**。两者都以同一 CAD 角度
坐标表达，但意义不同，不能互相覆盖。`saved_manual_count_range` 仅记录当前
HOME/手动编码约束，不是舵机完整指令范围。

对于后续训练对齐，动作侧截断对应前者，关节运动/接触限位约束对应后者；采用当前
保存关节限位前还应与具体训练 CAD 限位核对。既有权重和冻结 CAD 未自动修改，
设备保存值也不能冒充独立测得的全部机械硬限位。
