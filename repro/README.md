# 起身 6804：训练过程与效果复现

目标是从随机初始化重新得到可比较的起身/头控效果，允许网络权重不同。这里固定的是源码、物理模型、实际运行依赖、双卡批量、阶段顺序和评测协议。本文与脚本的补齐不代表已经重新训练验证过；本次只做离线材料、命令计划和参考报告校验。

## 1. 版本与物理身份

```bash
git clone --branch publish/v2-rise6804-training-20261031 https://github.com/microduck-awesome/microduck_rl.git rise6804
cd rise6804
git rev-parse HEAD
python3 repro/check.py
```

保存实际 checkout 的提交号。历史最终训练源码为 `36e0780ca57504f309b46e8b815595a69b6bb108`，分支基于交付源码 `4fb82ab3f043312d79e77c5b912a33964a01f8d9`。`repro/environment.json` 对完整已跟踪 src、模型资源和关键训练脚本做 SHA256 固定；新复现辅助脚本位于 repro，不修改原奖励、课程、训练器或动力学。

- 任务：`Mjlab-StandUp-Flat-MicroDuck-V2-PairedError-Baseline`。
- 质量 **600.941825 g**，SC0090 V5；V2 HOME 为全部关节零位。不要替换为行走分支的 624.6 g 模型。
- 新碰撞沿原模型增加 32 个无质量外壳凸分块，保留原质量/惯量。
- 观测 actor 61D、critic 74D，动作 14D，不含嘴；50 Hz 控制、5 ms 物理步长、4 倍 decimation。
- 两张 NVIDIA L40（每张约 48 GB），每卡 4096 环境，总批量 8192，同步一个策略。PPO 每次采样 24 步，网络 512/256/128。每 rank 的 seed 为 `20261015 + rank`；初始化与 RNG 恢复以 `train3762_ddp.py` 为准。
- 单卡更改全局批量、改电压/动力学/碰撞、改头部课程都属于新实验，不能直接用本说明认定复现。

## 2. 建立匹配实际历史运行的环境

原 `uv.lock` 锁定 MuJoCo 3.10.0，而所有历史正式阶段 `run.json` 记录 **3.13.0**。因此本分支提供单独的 runtime requirements：从原锁文件导出完整生产依赖，唯一版本替换是 MuJoCo 3.10.0 → 3.13.0；BAM 固定到 `62bd8ce12154340be97e06f7f41a0ca8f116d967`。不把当前机器的任意 pip freeze 当成历史环境。

```bash
uv venv --python 3.12 .venv-repro
uv pip install --python .venv-repro/bin/python -r repro/requirements-runtime.txt
uv pip install --python .venv-repro/bin/python --no-deps --editable .
uv pip check --python .venv-repro/bin/python
.venv-repro/bin/python repro/check.py --environment --output repro-output/environment-packages.json
```

核心版本：torch 2.9.1、mjlab 1.3.0、MuJoCo 3.13.0、mujoco-warp 3.8.1、warp-lang 1.12.0、rsl-rl-lib 5.0.1、BAM 1.0.1（并核对上述 Git commit）。通过 `--no-deps --editable .` 注册本项目的 mjlab task entry point，依赖不重新求解；wrapper 同时设置 `PYTHONPATH=src`。仅设置 PYTHONPATH 而不安装项目入口可能无法发现 V2 任务。**不要再用 `uv run` 或 `uv sync` 覆盖这个 runtime 环境**；使用 `bash repro/python.sh ...`。

历史机器驱动为旧版环境，需要 CUDA 12.9 的兼容库。驱动 535/570 下，提供包含 `libcuda.so.1` 的兼容库目录：

```bash
export SC0090_CUDA_COMPAT_DIR=/absolute/path/to/cuda-12.9/compat
export CUDA_VISIBLE_DEVICES=0,1
bash repro/python.sh repro/check.py --environment --gpu --output repro-output/environment-gpu.json
```

兼容库属于系统驱动配套文件，未打包进仓库；不要把路径设成不存在的历史绝对路径。较新驱动由 wrapper 自动跳过这个分支；如环境需要特殊处理，可显式设置 `REPRO_USE_COMPAT=0`，但必须保留实际 driver/CUDA 记录，并通过 GPU smoke。wrapper 使用项目内缓存目录和 `.venv-repro/bin/python`，也可用 `SC0090_REPRO_PYTHON` 指定 Python。历史数值编译设置 `TRITON_DEFAULT_FP_FUSION=0` 已保留。硬件不同不要求权重相同，但必须记录版本、批量与指标差异。

## 3. 完整正式训练阶段

下表是同一策略的连续学习；`updates` 是追加量，`end` 是累计量。结构化记录见 `phases.json`，每个阶段两张卡的原始 run/status 见 `history/`。

| 阶段参数 | 起点 | 追加 updates | 终点 | 头部 recipe | 正式评测间隔 |
| --- | ---: | ---: | ---: | --- | ---: |
| recovery-start | 0 | 500 | 500 | 默认恢复课程 | 500 |
| recovery | 500 | 3262 | 3762 | 默认恢复课程 | 500 |
| head1 | 3762 | 1000 | 4762 | 原 stage 1 | 250 |
| head2 | 4762 | 250 | 5012 | 原 stage 2 | 128 |
| head4 | 5012 | 512 | 5524 | 原 stage 4 | 128 |
| head4-continue | 5524 | 512 | 6036 | 原 stage 4 | 128 |
| coarse2 | 6036 | 512 | 6548 | 原 stage 2 | 128 |
| final4 | 6548 | 256 | 6804 | 原 stage 4 | 128 |

首次历史运行在 update 500 评测后的 DDP 归一化广播出错。正式恢复前仅修复持久 normalizer tensor 的 inference-mode 问题及异常记录；其余阶段记录中的数值源码与本分支匹配。新复现从一开始使用已修复版，在 500 保存/恢复一次以保留历史分段，不故意重现程序崩溃。因此原 `recovery-start` 的 status 是 FAILED，这是历史事实，不是新复现的预期终态。paired-error 从初始化就启用；这是 A6804 的过程，不能用历史旧 3762 权重作起点。

头部训练不是直接从零训练 6804 次，也不是只执行最后 256 次。stage 2 的补充大指令巩固是为纠正正向颈俯仰欠响应，随后返回原 stage 4。阶段 2/4 最终训练范围均为 ±0.39 / ±0.39 / ±0.49 / ±0.11 rad；recipe 参数、采样比例和权重从原 `configure_head_precision_learning` 加载，禁止手工重新编写。

## 4. 可执行阶段命令

以下命令默认只打印完整命令计划，不训练：

```bash
python3 repro/rise.py --phase recovery-start --output-root /absolute/path/to/new-rise6804
```

要实际执行，由操作者显式加 `--execute`。按表格顺序逐阶段运行，例如第一阶段：

```bash
python3 repro/rise.py --phase recovery-start --output-root /absolute/path/to/new-rise6804 --execute
```

完成后查看 `recovery-start/rank-verification.json` 和 formal 两卡的 `status.json`；再执行：

```bash
python3 repro/rise.py --phase recovery --output-root /absolute/path/to/new-rise6804 --execute
python3 repro/rise.py --phase head1 --output-root /absolute/path/to/new-rise6804 --execute
python3 repro/rise.py --phase head2 --output-root /absolute/path/to/new-rise6804 --execute
python3 repro/rise.py --phase head4 --output-root /absolute/path/to/new-rise6804 --execute
python3 repro/rise.py --phase head4-continue --output-root /absolute/path/to/new-rise6804 --execute
python3 repro/rise.py --phase coarse2 --output-root /absolute/path/to/new-rise6804 --execute
python3 repro/rise.py --phase final4 --output-root /absolute/path/to/new-rise6804 --execute
```

每条单独执行、检查该阶段报告后再进入下一条。不要不看报告就把整组作为自动连续任务运行。

每阶段脚本明确执行：包/源码/GPU 审计；64 环境×5 updates smoke；单卡实际 4096 环境×12 updates 预检；双卡实际批量×6 updates、每 2 次评测的边界检查；丢弃所有预检权重；从该阶段的原父检查点执行正式追加量；检查两卡权重、归一化、optimizer 与 recovery stage/polish 一致。恢复阶段正式间隔 500，其余按表执行，边界按累计 update 对齐，包含终点评测。头部阶段与 3762 终点还执行独立恢复/头控评测。正式训练日志和所有命令保存在各阶段目录。

恢复使用两卡各自的 `resume_NNNNNN.pt`，保留 optimizer、normalizer、课程和各卡 RNG；环境物理状态采用原生新 episode，并不是无缝接续旧 qpos。不要换成 `checkpoint_NNNNNN.pt`：后者在评测之前保存，`resume_*` 才包含随后课程决策。边界验证的 6 次更新不能计入正式 lineage。

输出目录必须全新。若进程中断，保留该目录，用最后完整评测后的两个 rank `resume_*` 恢复，并手动把剩余追加量、原评测间隔、原 head stage 和父目录写入新命令；不要重跑已存在目录、不要从单卡副本复制两卡 RNG、不要让脚本静默覆盖。如果恢复课程尚未到 stage 5/polish，头部恢复断言会停止；保留差异报告，不通过改 checkpoint/门槛伪造课程通过。额外训练若必要，应作为单独记录的追加实验。

## 5. 用同一协议核对效果

恢复评测采用固定 stage 5，7 初态×2 seeds（2026101501/2026101502）×128 样本，8 s，无自动重置、确定性 actor；与原评测相同，保留 checkpoint 的反馈/延迟 recipe，关闭额外观测 corruption。稳定判据为末段连续 0.5 s 满足原身高、20° tilt、速度和双脚接触要求，详见 `recovery.sc0090_stable_standing`。不要把训练课程的每桶 70% 进阶门槛当成最终复现成绩。

原 6804 的结果是 **每桶 256/256，总计 1792/1792**。头控 seeds 为 2026101503/2026101504，每指令8次，2 s静止→2 s阶跃→2 s回零，记录 step/return、四轴物理误差、响应时间、无效样本和稳定性。原四轴 step 稳态 MAE（度；颈俯仰/头俯仰/偏航/侧倾）：

| 范围 | neck pitch | head pitch | yaw | roll |
| --- | ---: | ---: | ---: | ---: |
| zero | 0.230 | 0.289 | 0.198 | 0.263 |
| small | 0.417 | 0.356 | 0.369 | 0.285 |
| current | 0.605 | 0.311 | 0.296 | 0.330 |
| final（超出训练范围） | 5.073 | 1.379 | 0.906 | 0.417 |

正向 neck +0.39 rad 的两个 seed 误差为 1.148°/1.375°。超范围颈控仍有明显误差，这是参考模型本身的限制，不能省略。对新训练结果自动列出相对原值的差异：

```bash
python3 repro/compare.py \
  --recovery /absolute/path/to/new-rise6804/final4/recovery/report.json \
  --head /absolute/path/to/new-rise6804/final4/head/report.json \
  --output /absolute/path/to/new-rise6804/comparison.json
```

脚本核对种子、指令、范围、样本量和模型身份，不会用不同协议冒充对照。它报告原值、新值和差值，不凭空设定“误差增大多少也算复现”的容忍线。恢复目标按原每桶 256/256 核对，头部按四轴/各指令报告改善或退化；恢复通过不能单独推出头控效果复现。如果需要统计层面的等价结论，应另行预注册容差与多训练种子实验，现有单次历史训练不支持成功概率保证。

## 6. 导出、保存与复核

`final4` 会检查 rank0 `rank0.onnx` 的 ONNX 格式、61→14维、关节顺序，并用固定66组输入对照 checkpoint 自带 normalizer 的原生网络，容差沿用原验收 `atol=rtol=2e-5`。输出 `artifact-verification.json`；ONNX 必须带观测归一化，不能手工只导出 MLP 权重。

交付新结果时保存：Git commit、环境/GPU审计、完整阶段命令、每 rank 的 run/status/RNG/optimizer/normalizer/checkpoint、课程评测、独立 head/recovery 报告、ONNX与哈希及比较报告。参考 checkpoint 哈希用于辨认历史文件，新训练无需相同哈希。本仓库的训练配方从零可生成所有父检查点；旧历史父权重不是从零复现的前置条件。

现有证据仅支持所列仿真协议，不自动推导真机效果、硬件零点一致或其他质量模型的表现。两个协议使用的是已有开发 seeds，并非留出验证。
