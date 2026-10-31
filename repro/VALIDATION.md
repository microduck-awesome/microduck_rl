# 2026-10-31 复现材料验证边界

已完成：

- 全部新增 Python 语法检查，shell wrapper 的 bash -n 检查。
- 原训练源码/资源 SHA256 检查（起身380项（含runtime清单）、行走349项）。
- 保留的历史运行环境包版本、Python3.12和BAM Git来源检查。原始结果见 validation/retained-runtime-check.json；这是对保留环境的检查，不是新机器安装成功的证明。
- 起身8阶段累计6804、父子检查点和预检/正式路径分离；生成的46组历史CLI参数经原 argparse 定义解析通过，不启动训练。
- 行走原训练CLI --help确认 --agent.max-iterations、--agent.seed、--agent.run-name 与 --gpu-ids 参数有效，不启动训练。
- 起身恢复/四组头控参考报告自对照差值为零；行走139个参考checkpoint均可解析，10250自对照score差值为零。
- 改错行走seed或起身头控样本数会被对照脚本拒绝。
- 起身历史6804双卡权重/normalizer/optimizer一致、恢复精确、66输入ONNX/原生parity通过，最大误差1.043081283569336e-6。只做CPU产物验证，不进行机器人仿真回放。

证据JSON见 validation/。此轮没有启动新训练、GPU策略评测、真机动作或完整新环境重训。因此当前证明的是配方/材料/命令/参考对照自洽，效果复现结果须由按README显式执行的新run及同协议报告提供。

补充：两个保留环境的 `uv pip check` 均通过（各142个已安装包）；起身 runtime requirements 的每个适用平台版本约束均与保留环境匹配。新空环境在线dry-run在本轮等待窗口内未完成，离线解析缺少platformdirs缓存，因此不将该尝试计作全新安装验证。
