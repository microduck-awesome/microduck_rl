# 正确映射重训谱系：起身 1400 实机验证候选

用户明确授权推送，用于实机验证 sim2real；不是完整仿真或实机验收通过声明。

**Web 上传：[rise-1400.mdp](rise-1400/rise-1400.mdp)**。配套 [ONNX](rise-1400/policy.onnx)、[checkpoint](rise-1400/checkpoint.pt)、[运行契约](rise-1400/runtime-contract.json)。模型ID `v2_sim2real_rise_1400_20261023_correct_routing`。

来源为 correct-routing-retrain-008/rise/assessed-1400，源码 fe2cc15205583df3508ec86ea9fcc2f9ebeee720，归档精确 recipe 哈希已核对。采用已修复的按关节名称匹配15轴执行器路由，14策略输出排除嘴巴；嘴巴独立保持。不是8040/3400错误映射模型的输出重排。

| 已有仿真评测 | 通过 |
|---|---:|
| 起身恢复（全量随机化、robust阶段） | 1752/1792 |
| 倾斜恢复与保持 | 1081/1184 |
| 四向外力扰动 | 571/768 |
| 中性站立严格保持 | 32/32 |

已进入抗扰训练但**抗扰课程未毕业、头控未合格、实机未验证**。中性站立尾段四头轴平均绝对误差依次约 17.61°, 46.55°, 2.78°, 3.24°，顺序为颈俯仰/头俯仰/yaw/roll。明显头部偏姿是此候选的已知行为，不是保证回 HOME。选择它是为了比较起身与身体保持，不表示推荐该头部姿态。

## 实机使用与记录

沿用现有 V2 Web 上传和 ESP32-P4 推理链路，上传本目录 `.mdp`。FP32 MDPOL001 v2，runtime ID `0x834335F5`；61维输入、14维输出、50Hz、CAD HOME全0、action scale1，归一化内置，不二次归一化。没有动作EMA；若设备另设滤波，实测记录其数值。保留既有固件目标截断和关节映射，不改固件、零位或配置。

这轮测试 twist、head、body 全部置零，分别记录坐姿/俯卧/仰卧/左右侧卧起身及站立保持；记录模型SHA、初态、成功次数、起身时间、再跌倒、关节触限及视频/遥测。使用[实测模板](../../hardware-results-template.csv)。零头命令不能消除上述实际偏姿。既有部署头部指令上限只是接口能力，不是本模型头控认证。

## 交付核验

通过原生 `MjlabOnPolicyRunner.export_policy_to_onnx` CPU导出，包含 checkpoint 的归一化；771组输入（含512条保存的实际观测）与 checkpoint 对照最大误差 1.91e-6。MDP系数逐位一致，CRC、源ONNX SHA、精确候选身份、关节顺序和源码recipe归档核验通过。无新增GPU仿真、无PPO更新、未操作实机。

原评测绑定本checkpoint的SHA，见 `rise-1400/evaluation/`；路由资格、训练配置、来源、校验与文件SHA均随包保存。精确源码在 `training-code.tar.gz`，外部CAD网格/误差库和安装依赖未全部打包。`package_and_verify.py --verify-only` 可只读核验交付包；完整重建依赖其中记录的不可变训练源路径。
