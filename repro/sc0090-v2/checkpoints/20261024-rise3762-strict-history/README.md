# 3762碰撞修正：单GPU完整历史重训结果

已从随机初始化按原10段历史完成3762次更新，单GPU cuda:1、8192环境；机器人模型只补入后背/左右外壳32个零密度碰撞凸块，既有质量、惯量、关节、执行器和碰撞参数保持一致。

**这是实验结果，不是已通过全角度或实机验收的替代基线。** 同一修正碰撞下，新模型起身1792/1792，旧3762为1525/1792；+63°颈部稳态误差约11.76°，旧约10.58°，但新模型在该指令下仍明显失稳（16次中7次出现躯干高度<8cm或倾角>45°）。不要以接近的平均角度误差掩盖稳定性差距。

## 文件

- `policy.onnx`：最终新模型，已内置归一化，61D输入、14D输出。SHA256 `01b5fafde4bde1c33bf5ce1a671b275bd06556354dcacdc9288ba13d47a084e9`。
- `checkpoints/paired-feedback-allocation-001/checkpoint_003762.pt`、`resume_003762.pt`：最终训练及原生恢复文件；两者模型、归一化器和优化器一致。
- `checkpoints/`：十段终点恢复文件，另保留2250/2500/2750用于诊断的评测前检查点。
- `reference-old3762/`：旧模型比较参照，未用于本轮训练初始化。
- `model/robot.xml`与`model/assets/`：修正后的机器人模型及其全部引用网格。完整仿真仍使用源码中的场景和BAM，单独XML不是完整训练环境。
- `source.tar.gz`：最终历史段数值源码、静态资源与评测辅助脚本。各早期段的原源码归档和全量配置在`repro/evidence/`。
- [结果与限制](repro/FINAL_RESULTS.md)、[完成审阅版协议](repro/STRICT_3762_SINGLE_GPU_REVIEWED_FINAL.md)、[实际冻结协议](repro/STRICT_3762_SINGLE_GPU.md)、[完整审计](repro/training-completion-audit.json)。
- `manifest.json`：文件哈希、模型身份和原始数据索引。

## 2500次退化

同条件回放复现右侧卧256→0→256；2500策略卡在侧卧约5cm高度、90°倾角。归一化互换对照中，失败跟随2500网络参数，未定位到某一个具体奖励项。之后按原训练恢复，没有回退或新增救援训练。诊断混合策略没有被训练或作为交付模型保存。

![大角度头部与身体状态](repro/final-evidence/neck63-comparison.png)

![右侧卧回放](repro/final-evidence/regression-right-side.png)

## 可恢复范围

保留checkpoint中的优化器、归一化器、Torch CPU/CUDA RNG、课程状态与计数。恢复语义严格沿用原生新episode，不声称恢复连续物理/电机状态。训练只用cuda:1，但保留原两张可见卡的CUDA RNG结构。

完整运行日志、中间检查点、原始轨迹及失败尝试保留在`/home/l40-vm01/data2/experiments/microduck-v2-20261024-strict3762`；本包保留模型、关键状态、报告和哈希，不重复打包全部原始轨迹。
