# 2026-10-22 当前效果实机快照

用户要求取当时最新已保存模型上机观察。本包固定为**起身4000、行走4600**，不随后续训练覆盖；两项均未通过完整仿真验收，当前没有这两项模型的实机结果。

| 策略 | 现有Web上传文件（FP32） | 训练检查点 | 原训练导出ONNX |
|---|---|---|---|
| 起身4000 | [rise-4000.mdp](rise-4000/rise-4000.mdp) | [checkpoint.pt](rise-4000/checkpoint.pt) | [policy.onnx](rise-4000/policy.onnx) |
| 行走4600 | [walk-4600.mdp](walk-4600/walk-4600.mdp) | [checkpoint.pt](walk-4600/checkpoint.pt) | [policy.onnx](walk-4600/policy.onnx) |

MDP使用本包固定的固件转换器生成，格式MDPOL001 version 2，含FP32网络与归一化。原checkpoint、ONNX不改权重；逐权重/归一化一致，771组CPU输入（含512条保存的实际观测）最大误差分别7.63e-6、1.91e-6。MDP系数逐位相同，CRC及源ONNX SHA通过。这里没有重新训练、跑GPU评测、刷写固件或操作实机。

## 使用与记录

沿用已用于基线3762的V2本体运行配置，通过现有Web加载对应MDP。61输入/14输出、50Hz、HOME全0、连续CAD弧度目标、action scale1；嘴巴独立保持。归一化已内置，不能再做一遍。详细关节顺序、观测排列和限位见各目录的`runtime-contract.json`；checkpoint不是Web上传格式。

起身先使用零twist/头部/body命令，记录坐姿、伏卧、左右侧卧、仰卧的起身和起身后保持表现。行走从零命令、0.03/0.05m/s直行及±0.15rad/s转向观察当前响应。当前运动课程仍stage0，采样上限为前后0.15m/s、侧向0.10m/s、转向0.5rad/s；头部命令范围为颈pitch±0.05、头pitch±0.05、yaw±0.07、roll±0.015rad。测试时记录实际设备参数和滤波配置，本次交付不修改设备配置。训练没有动作EMA；先前实机3762使用过头0.5/腿0.7，实际使用值应随记录保存，不能混成同一条件。

使用[已有实测记录模板](../../hardware-results-template.csv)，附模型ID或MDP SHA、初始姿态、命令、起身耗时、保持时间、头部/步态视频和遥测。当前目的是观察实机效果，不将本包标为已合格或优于3762。

## 同检查点的仿真结果

- 起身4000：小倾姿1184/1184未倒，429/1184末段身体稳定；四向推力766/768未倒，1条严格静稳；全部完整协议仍0通过。原七类恢复计数见`rise-4000/evaluation/recovery.json`。
- 行走4600：行走922/928未倒、571条出现自碰；直线254/256未倒、204条出现自碰；停车静稳及完整行为仍0通过。不倒不能替代命令跟踪/停车合格。

各目录`evaluation/`为与该checkpoint SHA绑定的原报告，不混用旧模型成绩。`checkpoint-info.json`含课程和精确训练状态；`training-allocation.json`含启动配置；`training-code.tar.gz`包含配方哈希覆盖的代码、姿态bank和任务MJCF，**不包含所有外部网格、误差库和依赖**。它是代码快照，不是独立完整仿真安装包。

`onnx-metadata.json`保留原导出元数据。旧简写压降字段仍写0..0.2V/Nm，实际配置为共享母线0.05..0.15Ω，准确值见checkpoint随机化契约；本包不改原ONNX字节。`converter-original.json`保留当前V2转换器原始输出；`legacy-version1-converter.json`保留首次交付的旧转换记录。训练身份与部署身份分别记录在`runtime-contract.json`。

文件大小、SHA256、源提交、配方身份及验证结果见[manifest.json](manifest.json)。`verify_and_package.py`记录本机打包/验证过程，其中实验源路径用于追溯，不是任意机器的安装入口。

## V2封装修正

首次提交`387ac31b`的MDP缺少二进制V2标记，请重新下载本分支当前两份MDP；旧提交下载链接不会更新。此次只修改version字段和44字节契约扩展，权重、归一化、checkpoint及原ONNX均未改变。起身runtime ID为`0x834335F5`，行走为`0x11D720E4`。详见`v2-header-fix.json`。

`tools/`固定来自固件提交`13eea5de53224b5cb6bd1bf0de91ff4f1f5692de`的转换器与契约校验，注册表增加本次两项模型的精确SHA及原始元数据。未知V2候选或元数据不符会报错，不会退回V1。新检查点须核对任务契约并登记原始身份后导出；不能只改旁边JSON标签。

安装numpy、onnx、onnxruntime后，可单独复现封装（输出到新目录）：

```bash
python tools/convert_policy.py rise-4000/policy.onnx /tmp/rise-4000.mdp
python tools/convert_policy.py walk-4600/policy.onnx /tmp/walk-4600.mdp
```

`verify_and_package.py`使用本包转换器并强制检查V2头和任务runtime。CPU封装检查不代表实机动作验收。
