# 3762碰撞修正历史重训：已完成

单GPU按原10段链路完成3762，最终产物和对比评测已完成。

- [结果、性能限制和2500退化复核](FINAL_RESULTS.md)
- [完成审阅版协议](STRICT_3762_SINGLE_GPU_REVIEWED_FINAL.md)
- [实际启动时的冻结协议](STRICT_3762_SINGLE_GPU.md)
- [完整训练审计](training-completion-audit.json)
- [polish单向保持语义](POLISH_STATE_REVIEW_ADDENDUM.md)

新模型起身表现更好，但+63°颈部指令下仍有明显失稳，不能认定为旧基线的全面替代品。2500右侧卧退化已在独立回放中复现，后续恢复；归一化互换对照将该退化定位到网络参数部分。原始失败记录保留。
