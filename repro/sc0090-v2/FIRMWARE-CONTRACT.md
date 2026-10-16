# 固件与最终 P04 部署契约审计

**结论：当前主固件不能直接视为最终 native-count-v2 P04 的兼容运行时。** 本次只读核对，不改固件、不转换模型、不编译刷写、不训练，无实机验收。

固件：`microduck-firmware/main`，`ee77131564eef52a64575628b0c117d69f583039`，审计前后工作区干净。训练：`groundpick-target-precision-worktree`，实际768 ONNX标识 `microduck-v2-p04-o0-valid-raw-native-counts-v2`，部署字段仍为 `simulation-feedback-repair-validation`。文件路径、SHA和具体行号见JSON。

## 61D 与关节顺序

| 零起点半开槽 | 含义 |
|---|---|
| 0:3 | body angularvelocity rad/s |
| 3:6 | projected gravity bodyframe |
| 6:20 | 14 completedO0 q inV2 HOME radians |
| 20:34 | 14 completedO0 v rad/s |
| 34:48 | previous rawpolicyaction; beforetargetquantization/EMA |
| 48:51 | taskcommand3 |
| 51:55 | headcommand4; allzero inGroundPick |
| 55:61 | bodycommand6; allzero inGroundPick |

14动作顺序：`left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle, neck_pitch, head_pitch, head_yaw, head_roll, right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle`。固件15路wire在index9插入mouth，右腿从wire10开始；训练反馈内部mouth追加在15路末尾。命名映射结构一致，不能直接复制15维数组。GroundPick头4槽和body6槽全部为0，61D相同不表示跨技能命令可互换。

## 关键差异

| 项目 | 最终 P04 | 当前主固件 |
|---|---|
| HOME | 所有关节0，V2 CAD保存装配姿态 | 旧alpha HOME，例如neck/head_pitch各.3491rad；观测减旧HOME、动作加旧HOME |
| 动作 | scale1、无EMA、command_rate=None；机械与count范围约束 | head EMA .5、其它.7；walking默认scale.9；max_rate必须正且≤20，默认1rad/s |
| 原始count/命令 | 读数0..1023与命令20..1001分离 | 常规motion health又要求raw落在配置command范围内；默认150..850 |
| O0 | 100Hz采样，5有效样本准备LS3/LS5；O0只用LS3仿射，无MLP；完成输出后插值查询t−20ms | 默认LS3从native physical history直接拟合t−20ms，不是同一采样/输出查询过程 |
| 预热 | reset生成280ms静姿历史；O0没有O1的19帧/220ms固定重预热 | 默认LS3是3点窗口；neural分支保留19帧及220ms |
| Ground周期 | 4s循环，原验收12s/3周期，回正含phase≥.8 | 默认phase到.7即2.8s结束 |
| 模型身份 | 本轮V2 checkpoint/normalizer/训练契约 | resident ground_pick仍alpha_ground_pick旧模型，runtimeID0x4ac2d0fa |

**尖峰需分路径陈述：** 默认LS3的aligned_history来自raw position，没有消费neural stream的去尖峰结果；不能说当前默认固件统一在去尖峰。neural分支仍对有效跳变做趋势预测替换/持续异常拒绝，因此不能切到该分支冒充已匹配最终无尖峰O0。P04保留坏包、NaN、计数范围、时间、数值速度及查询支持检查，未删除所有健康保护。

## count、归一化与真实标定

O0仿射执行系数约 a=0.0185713563114、b=0.958647012711，r=0.00562459622224735rad/count。训练目标 `round_even(clamp((q−a)/b/r+512,20,1001))`；等效虚拟映射scale约0.00539200236616、zero约508.555758872。这些只是仿真坐标，绝不是每个实机关节的标定。motor API另用v5反馈bias/scale逆变换，不能把三个比例混成一个。固件用signed calibration及lroundf，恰半count舍入与torch.round不同，仍需精确量化边界对照。

实际768 ONNX已只读核验：Sub常量逐位等于checkpoint mean，Div常量逐位等于checkpoint std+0.01；61→512→256→128→14，3个ELU，误差0。固件MDPOL001可表达同一数学图并保留normalizer，但数学格式支持不证明HOME/动作/时序兼容。不得漏归一化或重复归一化。

v5文件冻结SHA `7c35a732575bfad135d5bf7c7eeb44f268ffd96862064f04e9e7a5105bbf6fb4`；实际训练P15/D30/I0、nominal11.1V、goal profile zeroed。原fit文件的P20/10.2V来源信息不是当前加载后的运行参数。当前无真实机器人config，零位/轴向/比例/机械限位/限速/PID/profile/实际供电仍未核验。不能仅据v5数据集称整机sim2real已保证。

## 其余边界

- 原始q/v invalid时训练保留last valid直到episode终止并reset，不用真值或预测静默替代；固件则依健康/时限撤销目标，不能把仿真reset当成实机流程。
- V2默认IMU安装方向与CAD一致为−X,+Y,−Z，安装平移亦对应设计值；物理calibrated标志默认未置。IMU随机化不是现场系统偏置的校准。
- 当前固件converter/核心未找到对microduck_training_contract/native-count-v2语义的消费；统一runtimeID绑定旧模型与代码，不能按shape接受后就称兼容。
- 历史`firmware-v2-runtime-worktree`同HEAD仅两个servo_stream文件有未集成修改；不等于主固件已完成适配，更不能把旧主机测试当实板结果。
- 导出旧joint_stiffness=1/damping=.08字段是结构预览元数据，实际v5/PDI以训练契约和actuator配置为准。

机器可读证据：`firmware-contract-audit.json`。

## 离线格式工具与契约保留

这里只读核对CLI，未执行转换。FP32入口是 `python esp32p4/tools/convert_policy.py SOURCE.onnx DESTINATION.mdp`。量化研究入口 `quantize_policy.py --history HISTORICAL_TOOL_ROOT --sources ORIGINAL_ONNX_DIR --output STUDY_DIR --models registered_asset_names`，随后 `export_policy_quant.py --study STUDY_DIR --report REPORT.json`；后者会写生成的core文件/runtimeID，且现有工具绑定assets身份与历史校准工具，不是可直接套新模型的纯一文件转换。

MDPOL001保留网络、有效归一化分母、源ONNX SHA与CRC，**没有嵌入**microduck_training_contract/runtime/HOME/动作与命令语义；sidecar仍写固定alpha-61标签。源SHA间接绑定原ONNX完整字节，不能代替运行时读取和强制契约。MDPQ1601包装原MDP+量化spec，MDPQ1602只保留normalizer/整数系数及源SHA，均不会补回训练契约。因此交给同事应保留原ONNX、checkpoint、SHA及独立契约，不用转换/上传成功冒充兼容。

## 三个起身现有ONNX核验

未手工导出或重建模型文件。每个现成ONNX的4层weight/bias共8数组及mean/effective denominator逐位等于对应checkpoint；CPU ORT与检查点公式分别使用259个固定合成输入和2048条已保存实际P04 rawobs。后者来自1856轨迹，给所有模型作算术覆盖，不是三模型各自闭环验收。

| 候选 | 合成最大动作误差rad | 实际rawobs最大动作误差rad |
|---|---:|---:|
| rise_p04_micro1856 | 4.76837158e-07 | 1.49011612e-07 |
| rise_baseline_paired3762 | 2.98023224e-07 | 1.49011612e-07 |
| rise_baseline_to_p04_4466 | 2.68220901e-07 | 1.78813934e-07 |

完整路径、checkpoint/ONNX SHA和契约metadata见 `export-parity.json`，三套均通过2e−5数值门槛。该结论不覆盖当前固件接口差异、量化误差、设备延迟或实机安全。

三套原始metadata应保持：1856/4466为`microduck-v2-p04-o0-valid-raw-native-counts-v2`/`simulation-feedback-repair-validation`；paired3762为`microduck-v2-sc0090-official-baseline-v1`/`simulation-baseline-only`。后者虽已有完整P04迁移评测，仍是基线训练谱系，不得重标为P04续训或hardware-qualified。
