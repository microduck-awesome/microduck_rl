# 官方XL330发布策略与V2候选实测对照

更新：2026-10-15 19:15，Asia/Shanghai。用户要求直接验证官方发布模型通过率；本报告记录已实际运行的冻结策略仿真，不代表实机效果。

## 固定来源与执行

- 官方仓库：[pollen-robotics/microduck-policies](https://huggingface.co/pollen-robotics/microduck-policies)，实取revision `665ae4eb20bfe13bca1ef4ea6d4bb6b4a096e448`。
- 模型 `velstand.onnx` v8，SHA256 `2fc7f8ee6acd7d0d3f66a99927b5d5a56589aeed02ec714de329b252d5ccf492`。发布清单标明XL330/50Hz、训练run `0045i2si`、checkpoint2749、源码`3750776`、2026-10-12导出；支持站立、行走、跌倒起身。
- 使用清单指定源码的独立worktree `official-xl330-worktree`；原始XL330 M6、kp200、全碰撞/齿隙本体、原HOME、动作scale1、50Hz/200Hz保留。训练任务Rough-Backlash，当前明确在Flat-Backlash平地评测；保留原生电压/sag/舵机3–6物理步延迟，移除外部推扰及额外观测噪声/encoder偏置事件。
- 官方ONNX已含normalizer；GPU仅逐算子执行同一Sub/Div/Gemm/Elu图。头控576条实际观测与原ONNX Runtime比较，max abs 7.4505806e-07；起身320条max abs 5.66244125e-07。均通过容差，权重/normalizer未变。不是训练出来的本地“官方配方”模型。
- native HOME与ONNX元数据三位小数检查一致（容差.00051rad，实际训练用源码精确HOME），关节顺序/61D命令槽/14D动作尺度通过。头部实际角度是servo+passive backlash的输出侧角度，减本体HOME，并与官方head reward同边界逐步核验。

## 相同头控命令和指标

两种子2026101503/04，每工况8次，81个工况，2s零指令+2s阶跃+2s回零。小幅各轴±.03rad；中幅neck/pitch±.39、yaw±.49、roll±.11rad；组合为对应范围半幅。各自原本体仿真，不能视为同一硬件的因果对照。

T90通过：所有受命令轴同时进入 `max(指令幅度10%, .002rad)` 误差带，持续.1s。另统计末端.5s持续到位，避免只把短时进入误差带当稳定精度。

| 指标 | 官方XL330 velstand v8 | V2基础baseline3506 |
| --- | --- | --- |
| 小幅单轴T90 | 4/128（3.13%） | 47/128（36.72%） |
| 中幅单轴T90 | 36/128（28.13%） | 128/128（100%） |
| 当前范围两轴同时T90 | 12/384（3.13%） | 265/384（69.01%） |
| 当前范围四轴同时T90 | 0/64（0%） | 54/64（84.38%） |
| 最终大幅单轴T90 | 46/128 | 96/128 |
| 小幅末端.5s保持 | 3/128 | 44/128 |
| 中幅末端.5s保持 | 21/128 | 128/128 |
| 两轴末端.5s保持 | 7/384 | 246/384 |
| 四轴末端.5s保持 | 0/64 | 51/64 |

官方中幅单轴示例站立高度约.115m、tilt p95约1.5–3.7°，这些工况stable_fraction均1；不能把头控失败误写成机器人已倒地。官方颈部在零命令时仍有约7.9°偏差。所有162记录无NaN/无invalid；头控达标不自动替代站立保持判据。

结论：**本基础候选在这组静止头控指标上优于官方v8，尤其是中幅和组合响应；两者小幅精度仍有不足。** 尚不能由此宣称综合能力、行走稳像或sim2real优于官方。

## 官方起身通过率

两个种子2026101501/02×七类初态×128，8s零指令。保留官方原HOME/SIT；站立z=.12–.13、坐姿z=.06–.09、倒地z=.05–.09m，末端frontier0–30°；对应逻辑姿态相同，物理初态受本体结构影响，不声称逐轨迹配对。

| 初态 | 官方共同严格判据 |
| --- | --- |
| 站立 | 256/256 |
| 坐姿 | 256/256 |
| 俯卧 | 256/256 |
| 最后前沿角度 | 256/256 |
| 左侧 | 256/256 |
| 右侧 | 254/256 |
| 仰卧 | 256/256 |
| 合计 | **1790/1792（99.89%）** |

共同判据为末端连续.5s：z≥.102m、tilt≤20°、线速度<.15m/s、角速度<1.5rad/s、双脚着地、状态有限。按官方自身z≥.09m/tilt≤25°的姿态门槛及其附加速度/双脚约束版本也都是1790/1792，没有偷偷用更宽门槛改善共同通过数。轨迹独立复算最终25帧逐项一致，保存qpos和health高度差为0。

官方standing初态先倒后起0/256；坐姿途中208/256进入低位倾倒再恢复，须与直接坐站动作品质分开看。日志存在原生Warp EPA horizon=24提示，保留日志；未为提高成绩改碰撞模型或隐藏失败。

P04从零1024在自己的S0/O0/R1测量目标下为1792/1792、standing先倒0/256，已扫完剩余七个前沿窗。官方与P04起身都很高，2条差异及不同硬件模型不足以证明总体优劣。P04从零1024的measured中幅头控仍0/128，正在做head1精度训练，不能把基础baseline3506的头控成绩写到P04模型名下。

## 原始证据

- [官方下载与哈希](/home/l40-vm01/data2/experiments/microduck-v2-20261015/official-xl330-models/provenance.json)
- [官方头控完整报告](/home/l40-vm01/data2/experiments/microduck-v2-20261015/official-xl330-head-v8/report.json)
- [统一头控指标复算](/home/l40-vm01/data2/experiments/microduck-v2-20261015/official-xl330-head-v8-comparison/report.json)
- [官方起身完整报告](/home/l40-vm01/data2/experiments/microduck-v2-20261015/official-xl330-recovery-v8/report.json)
- [起身独立指标复算](/home/l40-vm01/data2/experiments/microduck-v2-20261015/official-xl330-recovery-v8/independent-metric-check.json)
- 每个目录保留source.tar.gz、实际环境配置、原始NPZ；本轮仅训练与评测，没有P4/host部署或实机操作。
