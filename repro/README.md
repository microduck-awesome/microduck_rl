# 行走 10250：训练过程与效果复现

复现目标是相同训练条件和固定协议下可比较的行为效果，权重可以不同。本材料明确区分从零训练、历史 checkpoint 回放、以及检查点筛选。本次补齐只做命令/材料/参考报告校验，没有重新启动训练或 GPU 评测，因此尚无“新环境完整重训已通过”的结论。

## 1. 获取分支，固定身份

```bash
git clone --branch publish/v2-walk10250-training-20261031 https://github.com/microduck-awesome/microduck_rl.git walk10250
cd walk10250
git rev-parse HEAD
python3 repro/check.py
```

历史官方基线 `b02fdea4902c2b93b7d34a83a6f05da4ececb7d3`，正式启动源码 `7aec17f231e44ee3aef3fa20201795c9f359c0f6`。完整源码/模型与锁文件的哈希见 `repro/environment.json`。发布分支包含后续筛选代码和证据；训练数值源码相对启动版本不变。

- 任务：`Mjlab-Velocity-Flat-MicroDuck-V2-Official-Baseline`。
- 物理：624.6 g、V2 CAD、全部关节零位 HOME、SC0090 V5 M6 P15/D30/I0；不要替换为起身 6804 的600.941825 g碰撞模型。
- 原配方：一张 NVIDIA L40（约48 GB）、4096环境、seed 42、50Hz控制、5ms物理、decimation 4、每次采样24步、5 epochs、4 minibatches、512/256/128 ELU、actor61D、14D动作、观测归一化。PPO其余参数逐项沿用保存的 `selected/agent.yaml` 和官方 cfg，例如 adaptive LR、gamma0.99、lambda0.95；不自行修改奖励/课程。
- 保留官方所有随机化和延迟：摩擦/armature倍率0.9–1.1、3–6物理步命令延迟、供电10.2–12.4V、10.2V下限、压降gain0–0.2，其余事件以 audit/config 为准。不要因使用固定SC0090标定而关闭训练DR。

## 2. 环境建立与核验

```bash
UV_PROJECT_ENVIRONMENT=.venv-repro uv sync --frozen --no-dev --python 3.12
uv pip check --python .venv-repro/bin/python
.venv-repro/bin/python repro/check.py --environment --output repro-output/environment-packages.json
```

版本从原锁文件直接安装，核心为 MuJoCo **3.10.0**、mjlab1.3.0、torch2.9.1、mujoco-warp3.8.1、warp1.12.0、rsl-rl5.0.1。BAM固定Git `62bd8ce12154340be97e06f7f41a0ca8f116d967`，不能只看其包版本1.0.1。与起身6804的MuJoCo3.13.0环境分开安装。

```bash
export CUDA_VISIBLE_DEVICES=0
# 驱动535/570需要CUDA12.9兼容库；填写实际存在的目录。
export SC0090_CUDA_COMPAT_DIR=/absolute/path/to/cuda-12.9/compat
bash repro/python.sh repro/check.py --environment --gpu --output repro-output/environment-gpu.json
```

旧驱动的兼容库须另行准备，仓库未包含系统驱动二进制。新 wrapper 自动识别535/570，否则使用系统驱动；特殊环境可设置 `REPRO_USE_COMPAT=0`，但应记录实际driver/CUDA并通过smoke。Python默认`.venv-repro/bin/python`，可通过`SC0090_REPRO_PYTHON`覆盖。缓存位于项目内`.repro-cache`。wrapper仅设置执行环境，不修改训练reward/curriculum/model。

## 3. 先做原配置审计和短测

以下命令会写入历史脚本固定的 `docs/evidence/official-walk/corrected/` 路径。请在新clone运行，先把已有参考材料备份到新结果目录，避免覆盖后失去对照；不要对活跃训练工作树运行。

```bash
mkdir -p repro-output/reference
cp -a docs/evidence/official-walk/corrected repro-output/reference/corrected
bash repro/python.sh scripts/audit_official_alignment.py
bash repro/python.sh scripts/smoke_official_v2.py
```

审计要求官方文件、奖励/课程、PPO与runner对齐，除V2结构/HOME/关节选择/标定/电压适配外不修改配方。原smoke为64环境×5更新，检查真实摩擦/armature/电压随机化、原生延迟缓冲、有限loss、checkpoint和ONNX。smoke权重必须丢弃，正式运行不能加载它。

## 4. 从随机初始化训练

直接调用原官方训练CLI，使用唯一新run-name；不使用会碰到历史`formal.log`的旧supervisor。

```bash
mkdir -p repro-output
bash repro/python.sh -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-MicroDuck-V2-Official-Baseline \
  --gpu-ids all --env.scene.num-envs 4096 \
  --agent.seed 42 --agent.max-iterations 10251 \
  --agent.logger tensorboard --agent.upload-model False \
  --agent.run-name reproduce_walk10250 \
  > repro-output/train.log 2>&1
```

仅暴露GPU0再用`--gpu-ids all`，不要把物理GPU编号误当CLI接收的GPU数量。不传`resume`或checkpoint；原生随机初始化、环境噪声与课程保持不变。检查日志和新生成的 `logs/rsl_rl/microduck_v2_official_baseline/<timestamp>_reproduce_walk10250/params/agent.yaml`、`env.yaml`，与参考 `docs/evidence/official-walk/corrected/best-search/selected/` 中保存的配置对照；允许run-name、输出路径及终止上限不同，其余实质参数不应漂移。

**编号说明：**历史文件 `model_10250.pt` 内 `iter=10250`，`common_step_counter=246024=10251×24`，所以是10251次更新，不是10250次。历史run的默认上限是50000，实际暂停后保留了139个checkpoint；10250是事后筛选出的文件编号。上面只训练到目标文件，足以比较这一训练时点，不等于重做完整139个文件的择优过程；改变max-iterations只缩短本次执行上限，原训练采用adaptive学习率而非按总预算线性退火。也不要把原始50000上限误写成发布模型训练了50000次。

若要复现原139个checkpoint的选择实验，需要保存同样的候选集合、截止点和250间隔，再按下节同一排序规则筛选。重新训练后最佳编号可能不是10250，这是允许的，但必须说明是“同训练时点比较”还是“同候选集合择优比较”。额外训练或改变候选范围可能提高成绩，须单独标注，不能混成相同预算结果。

## 5. 同一协议评价新旧模型

先给新运行目录设置变量（填实际timestamp目录，不要选择旧run）：

```bash
REPRO_WALK_RUN=/absolute/path/to/walk10250/logs/rsl_rl/microduck_v2_official_baseline/ACTUAL_TIMESTAMP_reproduce_walk10250
bash repro/python.sh scripts/search_best_walk_checkpoint.py \
  --run "$REPRO_WALK_RUN" --iterations 10250 \
  --output repro-output/eval-new10250 \
  --seconds 20 --replicas 12 --voltages 10.2 11.1 12.4 --seed 2026102603
python3 repro/compare.py --report repro-output/eval-new10250/results.json \
  --output repro-output/comparison-new10250.json
```

脚本无PPO更新。13固定指令，每条12样本，3个电压按样本轮转，**每个电压每指令4个样本，总计156个片段/模型**，不是12×3再乘一次。每片段20s，前2s为过渡，余下统计。指令包含静止、0.10/0.15/0.20/0.30/0.40前进、-0.10后退、±0.10侧移、±0.50原地转向、0.15前进搭配±0.50转向。成对使用相同IMU误差和命令延迟，私有RNG协议`private-rng-v2`。保留脚本中的初态、关闭自动重置/额外noise、固定指令覆盖与倾倒/限位统计，不能用自由重采样轨迹代替。

原参考：未触发倾倒156/156、near_limit_trials0、score **0.4930318938**、balanced_tracking_error0.3471985605、响应覆盖0.7083333333。六类行为误差等权，score=balanced_error+0.5×(1-coverage)，先按倾倒数、限位数，再按score排序。25%指令响应阈值只服务于排名，不代表全工况合格。

所有13条实际速度和原始逐样本值见 `docs/evidence/official-walk/corrected/best-search/SEARCH_RESULT.md` 与 `selected/selection.json`。要保留原模型的短板：0.10m/s前进接近死区，后退/侧移弱，转向不足且不对称。`repro/compare.py` 输出新旧score差值、稳定/限位数和逐指令实际速度，不用权重哈希一致作为复现判据，也不自设允许退化的百分比。若有退化，应按这些指标明确报告，不能仅凭“没摔倒”宣布效果相当。

可先重评仓库内的历史文件来校验环境（仍由操作者显式启动）：

```bash
bash repro/python.sh scripts/search_best_walk_checkpoint.py \
  --run docs/evidence/official-walk/corrected/best-search/selected --iterations 10250 \
  --output repro-output/eval-reference10250 \
  --seconds 20 --replicas 12 --voltages 10.2 11.1 12.4 --seed 2026102603
```

环境变化时，先同时对照“新环境旧权重”与“新环境新权重”，才能区分训练差异和评测运行库差异。历史139个候选的完整筛选报告共21684片段；分批每批最多35个checkpoint并要求nuisance哈希一致。旧`search_all_walk_checkpoints.py`自动查找历史`*dr_fresh`目录，不能直接拿来筛选新run；新run请用带显式`--run`与`--iterations`的CLI分批，保留统一seed/样本/电压并按上述规则汇总，禁止覆盖原参考结果。

## 6. 导出与保存结果

```bash
bash repro/python.sh repro/export.py \
  --checkpoint "$REPRO_WALK_RUN/model_10250.pt" \
  --output repro-output/new10250.onnx
```

调用原官方`run_export`，包含观测归一化；导出后验证61D→14D、关节顺序、ONNX格式和64组输入相对原生actor的数值一致性（沿用原导出校验容差1e-4），写同名`.verification.json`。历史原文件SHA256：checkpoint `311c8a1fe4dfb0bcf8556b4661ed6edae69a99b766d781708e4d2d6d7355f463`，ONNX `290e9c026cc438104590ed1f427892059cd643444afa882fb1d20d4576f036fa`，仅用于辨认参考文件，不要求新权重相同。

保存Git commit、实际GPU/driver/CUDA与依赖、正式CLI、params、日志、checkpoint及normalizer/optimizer、所有评测report和ONNX。一次历史训练不能保证新的随机训练每次得到相同分数；应按相同协议报告差异，若要统计等价应预先规定行为容差并做多训练种子验证。此说明不把奖励、仿真存活或ONNX数值一致性推导成真机资格。
