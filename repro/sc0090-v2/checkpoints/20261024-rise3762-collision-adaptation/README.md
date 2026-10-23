# 当前新碰撞适配结果：6804（非严格历史复现）

按用户 2026-10-24 指令先归档这轮现有结果。该模型以 3762 冻结源码为起点、修改后背与左右外壳碰撞，但使用双 GPU、从初始化就启用配对误差，并补过头部课程。因此它不是“除碰撞外完整对齐原版历史”的严格复现。接下来单 GPU 严格复现会另行记录，不能混淆。

- [ONNX](candidate-6804/policy.onnx)：61D 输入、14D 输出，含归一化。
- [完整结果](candidate-6804/README.md)、[逐文件 SHA256 清单](candidate-6804/manifest.json)、[包验证](candidate-6804/package-verification.json)。
- [rank0 检查点](candidate-6804/checkpoints/rank0/resume_006804.pt)、[rank1 检查点](candidate-6804/checkpoints/rank1/resume_006804.pt)。
- [碰撞模型](candidate-6804/model/robot.xml)、[完整源码快照](candidate-6804/source.tar.gz)、[起身回放](candidate-6804/recovery-seven-poses.mp4)。

同一新碰撞模型下，起身七桶评测新模型 1792/1792、旧3762 1535/1792；使用原开发种子，不是实机验收。新模型仍有宽范围颈部欠响应：+63° 单指令误差约 47°，旧3762约11°。不能用宽范围平均误差掩盖该失败，也不能把这份归档称为等能力替代。

包内数据保持本地交付包原始字节；归档不授权实机部署。
