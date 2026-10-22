# 历史起身策略实机对比包（2026-10-22）

本次用户明确要求：把现有值得测试的起身策略交给同事上机比较并反馈，模型使用 **V2格式**。这一授权更新了历史文档中“不做候选实验”的实测范围；不改变主线程正在进行的新训练，也不启动新训练或实机动作。

共12套历史起身候选。Web上传下表的 **`.mdp`**，二进制为 `MDPOL001 version 2`，FP32、61输入/14输出、归一化内置，起身runtime ID为 `0x834335F5`。原始ONNX、训练配置、来源SHA和CPU验证随模型保存。`.pt`不是Web上传文件；本补充包不重复保存训练checkpoint，逐个记录其原文件名、SHA与历史机器路径。

## 模型与测试次序

优先级1先覆盖主要配方；优先级2用于补齐父节点、分支及P04对照，全部已打包。每轮以paired3762开始和结束，模型顺序交错，记录电压与温度变化。不要仅按“4012”“4018”“3762”识别模型。

| 优先级 | 完整模型ID / V2上传文件 | 来源与比较意义 |
|---|---|---|
| 1 | [rise_baseline_paired3762](rise_baseline_paired3762/rise_baseline_paired3762.mdp) | 既有实机参照；head4＋配对反馈误差＋三档电压 |
| 1 | [rise_baseline_foundation3506](rise_baseline_foundation3506/rise_baseline_foundation3506.mdp) | paired3762加入配对随机化之前的父模型 |
| 1 | [rise_baseline_foundation4018](rise_baseline_foundation4018/rise_baseline_foundation4018.mdp) | 3506继续原基础head4配方；未进入PairedError |
| 1 | [rise_baseline_position_dr2_4012](rise_baseline_position_dr2_4012/rise_baseline_position_dr2_4012.mdp) | 独立位置噪声±0.04rad、20ms位置延迟、head2 |
| 1 | [rise_baseline_axis3_4268](rise_baseline_axis3_4268/rise_baseline_axis3_4268.mdp) | 保留位置DR2，收窄逐轴头精度奖励，继续训练后的节点 |
| 2 | [rise_baseline_head_noise3_4012](rise_baseline_head_noise3_4012/rise_baseline_head_noise3_4012.mdp) | 腿噪声±0.04rad，头降至±0.015rad，head2 |
| 2 | [rise_baseline_foundation_dr1_3762](rise_baseline_foundation_dr1_3762/rise_baseline_foundation_dr1_3762.mdp) | 3506→位置噪声±0.02rad、20ms；与paired3762不同 |
| 2 | [rise_baseline_foundation_dr2_4018](rise_baseline_foundation_dr2_4018/rise_baseline_foundation_dr2_4018.mdp) | 上一DR1继续，位置噪声扩大至±0.04rad |
| 2 | [rise_baseline_position_dr2_3756](rise_baseline_position_dr2_3756/rise_baseline_position_dr2_3756.mdp) | position4012、axis3与head-noise3的共同父节点 |
| 2 | [rise_baseline_axis3_4012](rise_baseline_axis3_4012/rise_baseline_axis3_4012.mdp) | axis3较早节点，可与4268比较继续训练影响 |
| 2 | [rise_p04_micro1856](rise_p04_micro1856/rise_p04_micro1856.mdp) | 历史P04从零参照；曾被操作者评价弱于3762 |
| 2 | [rise_baseline_to_p04_4466](rise_baseline_to_p04_4466/rise_baseline_to_p04_4466.mdp) | 历史基线到P04参照；曾被操作者评价更差 |

机器可读入口：[models.json](models.json)、[文件SHA清单](manifest.json)。每个子目录含 `policy.onnx`、同名MDP/JSON、`provenance.json`、`training-run.json`、`validation.json`及已有的原训练评测。来源参数以这些原记录为准，不能拿其它同编号模型的成绩套用。

当前新训练的[起身4000/行走4600快照](../20261022-current/README.md)仍独立保留；本包聚焦上述历史起身分支，不把在训节点或其它技能混入同一比较表。未把全部中间节点机械地加入：3506/4018代表基础分支，3756/4012代表位置DR2，axis3保留4012/4268两个节点。这样覆盖已讨论的不同配方和关键父节点。

## 实测统一口径

1. 使用同一台V2机器人、同一固件和同一标定。沿用已测试3762的设备配置；模型归一化内置，不能重复归一化。50Hz，HOME为CAD全零、action scale1、嘴巴独立，起身twist/body命令为零。头命令初始为零，站稳后再测头控。
2. 固定并记录头/腿EMA、电压补偿和目标限位。历史3762实测使用头EMA0.5、腿0.7及11.1V参考补偿，详见[实机交接](../../FRESH-BASELINE-TRAINING-HANDOFF.md)。本包不修改设备；不能用改过滤波的一个模型与原配置的另一个模型直接排名。
3. 起身按坐姿、伏卧、左/右侧卧、仰卧分别记录。先每个可执行姿态3次筛选，入围后同条件增至每姿态10次；所有失败和中止都留行。记录起身到稳定耗时、随后至少30秒保持、先倒再起和站稳后再倒，不只报告最终站起来。
4. 站立恢复分前、后、左、右，分别观察轻倾和持续轻推。记录施扰方式、估计幅度、最大倾角、首次纠正和回到4°带内时间。手推力度不可精确复现，必须标记为操作者筛选，不将它当标准力控测量。
5. 头部逐轴正负和回零，优先±5°，再按任务需要测±10°；保持在各模型共有命令范围内（neck/pitch±0.39rad、yaw±0.49rad、roll±0.11rad）。记录稳定后误差与抖动，按用户接受的**±2°**解释实用精度。旧仿真T90/1°结果不改写、不直接替代本次实测标准；没有外部角度测量时注明指标来自舵机反馈。
6. 每次测试保留视频和遥测链接、模型ID/MDP SHA、固件commit、设备配置SHA、电压和温度。记录反馈故障、目标饱和、明显抖动、人工扶持/中止原因。尚未测的项目留空，不能填0当成成功。

[hardware-results-template.csv](hardware-results-template.csv)为空模板，一行一试次。同事完成后将结果CSV、设备配置及证据索引提交至本分支 `repro/sc0090-v2/evidence/hardware/rise-comparison-20261022/`（大视频可外链），并在结果说明中给出各模型各姿态成功数/总数、失败类型、耗时分布及主观排序。不要覆盖旧3762实测记录。

## 解释边界与模型身份

- 本次是**现有完整配方筛选**，不是单因素随机化消融。父模型、奖励、随机化和训练预算并不完全相同，不能由排名断言“延迟随机化更重要”。历史20ms关节观测延迟是固定值，不是随机延迟。
- paired3762误差块来自P04仿真采集而非直接整机实测；配对q/v在16步块内相关，误差块与三档供电独立抽样。它与独立位置白噪声分支存在多项差异。
- 历史模型使用约600.94g旧CAD，并未因本次封装自动变成624.6g、修正外壳接触的新模型。原模型、元数据和已有失败结果保留。
- V2二进制格式与训练版本是两回事。基线的原metadata仍标 `official-baseline-v1`，表示训练来源；交付MDP头确实为version2，表示上传格式。不能为消除名称里的v1而篡改训练身份。
- 转换器保留既有起身实验runtime/契约身份（注册表历史目标范围20..1001、无训练EMA）；后来设备快照使用20..1003及头/腿EMA。MDP的44字节扩展不替设备改这些设置。实际部署参数须以同一台设备读回值记录，不能把旧注册表当成当前板端设置，也不能因此声称训练与硬件完全一致。
- 本次只有CPU数学/格式核验，无新增PPO、GPU仿真或实机试验。历史3762较好的判断来自操作者，不是12模型已经完成同条件测试。

## 文件验证与重新封装

```bash
python verify_files.py
# 安装numpy、onnx、onnxruntime后，可从包含的原ONNX重建到新位置：
python tools/convert_policy.py rise_baseline_foundation4018/policy.onnx /tmp/foundation4018-v2.mdp
```

`tools/`复制自此前已交付的固件V2转换器，只扩充本包候选SHA/原metadata注册表；未修改转换或推理算法。`package.py`保留本机来源核对流程，运行它需要历史实验目录和PyTorch。接手上机只需要 `.mdp`，无需运行打包脚本。
