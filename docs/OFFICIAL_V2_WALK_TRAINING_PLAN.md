# V2 官方基线步行训练方案（当前 GPU0 单卡）

## 目标与来源

初次从随机初始化使用 GPU0 和 GPU1 训练；用户随后要求单卡，当前使用 GPU0，释放 GPU1。以 2026-10-25 实时核验的官方默认分支 develop 为唯一训练基线：
https://github.com/pollen-robotics/microduck_rl/commit/b02fdea4902c2b93b7d34a83a6f05da4ececb7d3

官方任务：`Mjlab-Velocity-Flat-MicroDuck`。本地适配任务：`Mjlab-Velocity-Flat-MicroDuck-V2-Official-Baseline`。
不参考 10 月 24 日和 25 日步行实验，不加载其配置、奖励、课程、策略或优化器状态。

## 仅替换以下三项

1. V2 结构：直接取 `microduck-assembly@88392c4859046976dbee706b4288f02af2e78814` 的 model 文件与 reports/model_metadata.json。总质量 **624.6 g**，已由用户确认；保留结构仓库质量、质心、惯量和关节定义。
2. V2 HOME：使用结构交付的 CAD 保存姿态，15 个关节零位和对应根部高度；14 个策略关节按官方左右腿及头颈顺序排列，mouth 独立保持。相应调整初始化高度和关节选择器。
3. SC0090 V5 BAM：冻结 M6 标定系数，SHA256 `7c35a732575bfad135d5bf7c7eeb44f268ffd96862064f04e9e7a5105bbf6fb4`；P15/D30/I0、11.1 V。电机适配从 10 月 15 日已提交的原始接入代码取用，不使用后续步行实验改版。移除只适用于 XL330 的电机参数随机化，保留官方观测及其他随机化。

## 官方训练过程约束

- 官方任务 MDP、步行配置、奖励权重、命令采样和所有课程阶段保持原样；不裁剪头部课程范围，不增加奖励、阶段、自定义 PPO 或反馈误差库。
- 使用官方 `mjlab.scripts.train`、官方 Microduck runner 和 rsl_rl PPO；不引入旧实验的 reset/command/recompute 流程修改。
- 50 Hz 策略，5 ms 物理步长，4 次 decimation；actor 61 维、14 维动作；网络 512/256/128，观测归一化。
- 官方 PPO：每环境 24 步、5 epochs、4 minibatches、初始学习率 0.001、自适应 KL=0.01；其余直接继承官方配置。
- 当前 GPU0 单卡 4096 环境，保持总采样批量；此前为双卡各 2048 环境。GPU1 留给其他任务。
- 正式训练沿用官方上限 50000 次更新和每 250 次保存一次。日志使用本地 TensorBoard，禁止自动上传。训练产物不能等同于真机验证。

## 执行顺序

1. 保存本方案，完成最小来源/配置检查；修正接入错误，不进行额外配方搜索。
2. 双卡各 64 环境、5 次更新启动短测，确认能训练并保存。
3. 从零启动正式双卡训练，不从短测 checkpoint 续训。
4. 正式训练开始后每 **30 分钟**检查一次进程、迭代、损失、奖励各项、episode length 和 checkpoint。异常退出或完成可提前检查。无依据不频繁自检，不自动调整奖励或课程。
5. 保留日志、配置、模型和本地提交；真实步行效果需在训练后另行评估。

## 启动命令

在本目录执行（环境兼容设置见 scripts/v2_python.sh）：

```bash
bash scripts/v2_python.sh -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-MicroDuck-V2-Official-Baseline \
  --gpu-ids all --env.scene.num-envs 4096 \
  --agent.logger tensorboard --agent.upload-model False \
  --agent.run-name official_v2_sc0090_v5
```

`--gpu-ids all` 在启动脚本限定 `CUDA_VISIBLE_DEVICES=0` 后使用。max_iterations、save_interval 和算法参数直接使用官方默认值。

## 启动前结果

- 来源与配置测试 15/15 通过；完整奖励及课程参数与官方一致。
- 双卡各 64 环境、5 次更新完成，退出码 0，已保存 model_4.pt 及 ONNX。训练中 nan_state=0。日志末尾有 rank1 shutdown traceback 标头，但 launcher 报告双方成功退出，未出现训练异常；保留原日志。
- 正式训练从随机初始化开始；移除旧入口遗留的 no-FMA 编译环境覆盖，使用默认编译行为。
- scripts/run_official_walk.py 只负责启动官方命令和每 1800 秒记录状态，不更改训练配置或训练循环；退出时立即记录。
- 证据位于 evidence/official-walk/，正式启动信息见 launch.json，定时记录见 monitor.jsonl。

## 正式启动

2026-10-25 13:47:03（北京时间）启动，已确认双卡进入 PPO 更新，nan_state=0。代码提交 b315a689。下次常规检查为 14:17:03，此后每 30 分钟一次。

## 单卡切换及碰撞核对（2026-10-25）

按用户要求停止双卡进程，从最近的 model_250.pt 使用官方 resume 恢复至 GPU0 单卡、4096 环境。恢复模型、优化器、迭代和课程 common_step_counter；不重置课程，不改奖励。未保存的 checkpoint 后更新丢弃；环境物理状态按官方 resume 重新初始化，并非逐步完全连续复现。剩余更新 49750，使目标终点仍为 50000。具体命令见 scripts/run_official_walk.py。

编译官方与 V2 Entity 后核对：两者均有 5 个启用碰撞 geom；双足的 contype/conaffinity=1/1、condim=3、priority=1；躯干支撑和双小腿的自碰撞 geom 为 2/2、condim=3、priority=0。摩擦、solref、solimp 相同；均无显式接触 pair 和额外 body exclusion。头壳等不参与地面碰撞，遵循官方 walk 的精简碰撞拓扑，不是全身碰撞模型。

碰撞接触规则相同，几何形状因 V2 CAD 而不同：官方 power_support/leg 对应 V2 trunk_support/shin，不能声称几何完全相同。证据：evidence/official-walk/collision-comparison.json。

单卡日志与半小时记录改存 evidence/official-walk/single-gpu/。首次单卡启动遇到 CLI 对 --gpu-ids 0 的列表解析错误；改用 CUDA_VISIBLE_DEVICES=0 配合 --gpu-ids all，未改训练算法。

单卡于 13:56:15 启动，已确认从 iteration 250 继续更新，GPU1 无本任务计算进程。下次常规检查 14:26:15，此后每半小时一次。
