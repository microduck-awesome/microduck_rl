# SC0090 V2 sim2real 实验交付

这是供同事开展实机对比的实验包，快照日期 **2026-10-16**。不含轮滑。提供真实模型、精确训练源码、参数与课程、评测证据和固件接口契约；**没有模型已通过本轮实机验收**。

先读 [上机交接说明](HARDWARE-HANDOFF.md) 和 [固件契约差异](FIRMWARE-CONTRACT.md)。现有主固件不能直接认作 V2 兼容运行时；尤其是 HOME、EMA、动作缩放、O0 时间语义及模型身份。现有 `.mdp` 转换器不保留 V2 契约，所以这里交付 checkpoint 和 ONNX，不提供容易被误认为可直接上传的旧身份 `.mdp`。

## 下载与选择模型

[下载模型包及 SHA256](https://github.com/microduck-awesome/microduck_rl/releases/tag/sc0090-v2-sim2real-20261016)。模型独立发布，避免二进制检查点堆积在训练分支历史中。`release-manifest.json` 给出包内文件、模型 ID、SHA256 与用途。

首批重点对比三条起身路线：

| 模型 ID | 路线 | 已有证据 | 使用限制 |
|---|---|---|---|
| `rise_p04_micro1856` | P04 随机初始化谱系 | measured 起身 1792/1792；指定负 pitch 组 1024/1024 | 头部精度未全通过；不是完整技能验收 |
| `rise_baseline_paired3762` | 基线动作 + 配对反馈误差随机化 | 直接迁移到当时完整 P04，measured 起身 1792/1792 | 使用历史前端；没有单独验证量化适应阶段 |
| `rise_baseline_to_p04_4466` | 随机化基线 + P04 续训 | measured 起身 1792/1792 | 头控仍有缺口；不能只凭起身通过率宣告路线胜出 |

以上是开发评测，数据曾用于选择模型，不能称作未见测试集。真实上机结果由同事填写，不从仿真通过率推断。包中另保留其他技能与失败节点的 checkpoint，供复现及诊断；不把 Sit 的 SIT 成功写成 STAND 成功，不把 Ground 局部低头或 Roulade 部分滚动写成完整技能成功。

- [25 个模型的清单、结果和限制](MODELS.md)；[机器可读清单](models.json)
- [复现步骤](REPRODUCE.md)；[各模型训练参数](recipes.json)
- [目标、方案、已做事项、下一步台账](LEDGER.md)
- [PPO、采样规模与课程说明](TRAINING.md)
- [证据索引](evidence-index.json)；[原始历史记录](history/sc0090-v2-training-ledger.md)

## 当前研究判断

默认候选路线重新设为 **基线掌握动作 → 有效随机化和量化适应 → 完整 P04 续训与验收**，保留 P04 从零对照。路线比较尚未完成，不能凭吞吐率决定全部技能的路线。

已提交正式交互累计 **8,802,336,768 个 50 Hz control transitions**，包含失败和被放弃的实验；不含资格验证、诊断及评测。完整研究耗时没有统一记录，保持未知。Ground precision 完整训练阶段约 101,807 controls/s；这不是包括所有准备、等待及评测的总吞吐。Sit 相同口径约 82,936 controls/s。

B0 Ground 基线候选已准备，之前两次停在 CPU 入口/报告序列化，**0 GPU 仿真、0 PPO**；修复后须重新资格验证。没有 B0 成品模型。所有状态以本页/LEDGER 快照为准，`history/` 中的旧 GPU/PID 状态仅供追溯。
