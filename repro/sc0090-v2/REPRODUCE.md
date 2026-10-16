# 复现与核验

本分支根目录 `src/` 是最近完成的 Ground precision P04 源码，便于审阅。**复现其他模型必须使用它自己的快照**；不要直接用根目录代码运行旧 checkpoint。16 份快照覆盖 25 个模型及待验证 B0，每个文件按 SHA256 去重存储在 `objects/`，`snapshots/` 保留原相对路径。训练、评测脚本及 robot/measurement assets 都包含在内。

## 1. 校验、解包模型、还原源码

```bash
python3 repro/sc0090-v2/tools/reproduce.py verify
# 下载 Release，先对照发布的 SHA256SUMS 校验再解包。
sha256sum -c SHA256SUMS
mkdir -p /work/sc0090-artifacts
tar -xzf sc0090-v2-models-20261016.tar.gz -C /work/sc0090-artifacts
python3 repro/sc0090-v2/tools/reproduce.py verify --artifact-root /work/sc0090-artifacts
python3 repro/sc0090-v2/tools/reproduce.py materialize groundpick_precision768 --output /work/ground-source
```

`materialize` 只接受新目录，不覆盖任何训练 checkout。快照完整保留历史 wrapper 的字节；可移植 launcher 通过环境变量覆盖本机 Python/CUDA/cache 路径，不改变训练源码 SHA。包内历史 `run.json`、评测和 source-transition JSON 保留原始绝对路径作为证据，不会被直接执行。

## 2. 环境

在还原的源码目录用 Python **3.12**、`uv sync --frozen` 安装。`uv.lock` 固定 mjlab 1.3.0、torch 2.9.1、warp 1.12.0 及 BAM Git 身份。先确认所装 PyTorch 支持 CUDA、driver/compat 库和本机一致。历史训练使用 no-FMA 设置，wrapper 已保留。

535/570 驱动的历史 wrapper 要求 `SC0090_CUDA_COMPAT_DIR` 指向本机合适的 CUDA compatibility 目录；该机器库不是训练数据，不打入仓库。较新驱动按 wrapper 原逻辑执行。启动例子：

```bash
python3 repro/sc0090-v2/tools/reproduce.py run \
  --source /work/ground-source --python /work/ground-source/.venv/bin/python -- \
  scripts/evaluate_v2_ground_pick.py \
  --checkpoint /work/sc0090-artifacts/p04/groundpick-target-precision-allocation-001/checkpoint_000768.pt \
  --stage measured --seed 2026101633 --num-envs 256 --device cuda:0 \
  --output /work/new-ground768-measured
```

输出目录必须是新的。该例使用 precision 复评的种子；历史默认种子 2607 与 2026101633 不能混为同一配对实验。`models.json` 的每项关联原评测文件及 SHA，复现时按相应记录使用种子、样本数、task/runtime。评测脚本也有自己的快照身份，不只检查训练源。

## 3. 训练与续训

`recipes.json` 含模型 → 精确快照 → trainer → 原 run.json → PPO/seed/原参数的映射；完整 environment、课程、奖励在对应 `run.json` 和 task factory 中。起身早期 launcher 的 CLI 与其他技能不同，以该快照脚本 `--help` 为准，不跨版本套参数。

每次新运行先做 CPU/终止后继检查、64 环境×5 updates smoke，再按实际 N 验证接触缓冲区、数值、PPO KL/clip、normalizer、终止 bootstrap 和吞吐；成功后才进入有限预算的正式训练。归档 `qualification` 是历史证据，不能冒充新机器或新代码的资格验证。新增验证权重不能混入正式初始化。

例如 fresh Ground history 的 smoke 使用 `groundpick_history512` 快照、其 `recipes.json` 中的 task：

```bash
python3 repro/sc0090-v2/tools/reproduce.py materialize groundpick_history512 --output /work/ground-history-source
# 该目录安装依赖后：
python3 repro/sc0090-v2/tools/reproduce.py run \
  --source /work/ground-history-source --python /work/ground-history-source/.venv/bin/python -- \
  scripts/train_v2_ground_pick.py \
  --task Mjlab-GroundPick-Flat-MicroDuck-V2-HealthHistory-Repaired-S0-O0-R1 \
  --num-envs 64 --iterations 5 --validation-only --force-timeouts \
  --seed 2026101636 --device cuda:0 --output /work/new-ground-smoke
```

续训需要原父 checkpoint、teacher（如适用）、optimizer/normalizer/clock 和 source-transition。发布的 25 个 checkpoint 是保留节点，不是全部历史祖先；缺失祖先的名称和 SHA 必须从清单追索原实验存档，不能声称只下载本包就能逐阶段重跑整个研究谱系。父权重不齐时仍可评测已发布模型或按同一 task 从零训练。

迁移机器时，把 source-transition 的 **路径字段**改到新文件位置，保留 from/to 源码 SHA、父权重 SHA、预算和语义；将派生的 transition 另存新文件并记录差异。不能通过改 SHA 绕过不匹配。旧 worker 和审查脚本在 `evidence/` 中作为执行证据保留，包含历史机器路径/PID，不应原样启动。

模型包提供原始 checkpoint（含训练状态）及首批三个起身模型的官方 ONNX。导出必须使用配套官方 runner 路径并带 normalizer；不要手工仅提取 Dense 权重。新导出须进行原 actor 与 ORT 的输入/输出一致性核验。
