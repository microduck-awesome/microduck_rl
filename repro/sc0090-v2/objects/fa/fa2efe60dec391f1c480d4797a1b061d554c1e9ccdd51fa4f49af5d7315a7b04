# Microduck RL

## V2 / SC0090 官方配方基线

分支 `codex/v2-sc0090-official-baseline` 直接基于官方 `develop`
`b02fdea4902c2b93b7d34a83a6f05da4ececb7d3`，与
[S0/O0/R1 分支](https://github.com/microduck-awesome/microduck_rl/tree/codex/v2-sc0090-s0-o0-r1)
使用相同的 V2 CAD 和冻结 v5 BAM M6，便于分别对照官方配方与实测反馈方案。

任务：`Mjlab-Velocity-Flat-MicroDuck-V2-Official-Baseline`。
本分支不包含或加载 S0/O0/R1 测量模型、残差库、观测器权重，也不运行 P04 actor。
克隆后仅使用分支内的 CAD 和电机拟合 JSON 即可训练。

| 项目 | 本基线 | S0/O0/R1 分支 |
| --- | --- | --- |
| 机械模型 | V2 CAD，600.941825 g，原始惯量 | 相同 |
| 电机 | 冻结 v5 M6，P15/D30/I0，11.1 V | 同一 M6；nominal 11.1 V，measured 扩大供电工况 |
| 关节反馈 | 仿真关节状态 + 官方噪声、编码器零偏和延迟 | S0 测量、实测残差、O0 LS3 因果查询 |
| 目标路径 | 官方连续位置动作，动作 scale 1.0 | CAD 限位、O0 逆映射、原生计数量化与保护 |
| 学习设置 | 官方网络、PPO、奖励权重、命令及通用随机化 | 尽量保留同一配方，关节观测链替换为实测模型 |
| 测量归档 | 无依赖 | 需要随分支附带的归档 |

基线保留官方 actor 的位置噪声 ±0.001 rad、速度噪声 ±0.25 rad/s、
速度延迟 1 个控制周期（20 ms）、编码器零偏 ±0.015 rad，以及 IMU 噪声/延迟/安装偏差。
训练/回放时是否启用噪声、critic 特权观测均跟随官方工厂配置。
这些是用于对照的官方假设，不代表已经测得的 SC0090 整机反馈特性。

为使用实际硬件，必要差异包括 V2 的 CAD HOME/初始高度、14 策略关节顺序和独立嘴巴、
头颈命令范围裁剪，以及针对 SC0090 的 BAM 力矩/摩擦/armature。
XL330 的摩擦、armature、阻尼和增益随机化被移除，避免改动冻结的 M6；
身体质量/惯量、CoM、接触、推扰和 IMU 随机化保留，围绕 V2 自身物理参数运行。
原官方任务、`uv.lock`、网络与优化器代码保持不变。

两分支还存在目标限位与计数量化等差异，直接比较结果不能解释为只改变 S0/O0 的单因素实验。
本基线保持官方目标可超出机械止挡的训练设置；真实关节运动仍受模型关节范围和电机动力学约束。

```bash
git clone --branch codex/v2-sc0090-official-baseline https://github.com/microduck-awesome/microduck_rl.git
cd microduck_rl
uv sync --frozen
# 首先执行 64 环境、5 次 PPO 更新和带归一化的 ONNX 检查（默认 CPU）
uv run python scripts/smoke_v2_ppo.py --output-dir /tmp/v2-baseline
uv run python scripts/validate_v2_hold.py --report /tmp/v2-baseline-hold.json
# 正式训练需要可用 CUDA 环境
uv run train Mjlab-Velocity-Flat-MicroDuck-V2-Official-Baseline --env.scene.num-envs 4096
# 回放和导出沿用官方入口，并显式选择 V2 任务
uv run play Mjlab-Velocity-Flat-MicroDuck-V2-Official-Baseline --checkpoint-file /path/to/model.pt
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck-V2-Official-Baseline \
  --checkpoint-file /path/to/model.pt --num-envs 1 --device cpu --onnx-file v2-baseline.onnx
```

ONNX 仍包含归一化、61 维 actor 观测、14 个动作及正确的关节元数据，并明确标识
`microduck-v2-sc0090-official-baseline-v1` / `simulation-baseline-only`。
它用于 V2 仿真对照，不能直接当作当前 S0/O0 固件策略部署；部署前需要匹配的反馈/目标链路。

CAD 来源为 assembly `9b66fec`，逐文件 SHA256 见
[`assembly-install.json`](src/mjlab_microduck/custom_v2/assembly-install.json)。
电机 JSON SHA256 为 `7c35a732575bfad135d5bf7c7eeb44f268ffd96862064f04e9e7a5105bbf6fb4`，
与实测链分支一致。适配器是本分支专用版本；assembly 的 S0/O0 安装器会替换它，不能用于更新本基线。

软件检查、PPO 和 HOME 保持结果见
[`reports/v2-baseline/software-verification.json`](reports/v2-baseline/software-verification.json)。
短训练验证训练路径和导出，不代表已学会行走或完成整机 sim2real。

---

以下为固定版本的官方说明，其中原版质量/舵机描述不适用于 V2。


<img width="2215" height="884" alt="image" src="https://github.com/user-attachments/assets/5db7cc83-b3ce-4f7c-83f0-0572a63baed7" />


RL training environments for [Microduck](https://github.com/pollen-robotics/microduck) —
a ~800 g, ~25 cm tall bipedal robot — built on
[mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp) with PPO.
Policies are trained here at 50 Hz, exported to ONNX, and deployed on the real
robot by the runtime in [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck).

<!-- HERO VIDEO — real robot montage: walking, standup, roulade, roller skating.
     Keep it short (~30 s) and real-robot-first: this is the "why should I care" shot. -->

https://github.com/user-attachments/assets/50c3d537-8db2-4005-9d9c-3472faeec4d0

The repo encodes the full sim2real recipe: [BAM](https://github.com/Rhoban/bam)
actuator physics, domain randomization, backlash simulation, and the
reward-design lessons that made it work
(see [AGENTS.md](AGENTS.md) for the distilled playbook).

## Quickstart

Requires a CUDA GPU (training runs through MuJoCo Warp) and [uv](https://docs.astral.sh/uv/).

> **On ARM boxes (DGX Spark / GB10, Jetson):** `uv sync` pulls ~2 GB of CUDA
> wheels on first run and uv's default 30 s HTTP timeout can abort mid-download.
> Export `UV_HTTP_TIMEOUT=600` for the first sync. 

```bash
git clone https://github.com/pollen-robotics/microduck_rl
cd microduck_rl

# train the walking policy (uses your GPU; ~1-2 h for a usable gait at 4096 envs)
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096

# watch a trained policy in the viewer
uv run play Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <entity/project/run_id>

# export to ONNX for deployment
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <...>
uv run publish --onnx output.onnx --repo <user>/microduck-<name> --kind episodic --duration-s 4.0   # share it (see "Publishing a policy")

# drive the exported policy in CPU MuJoCo with the keyboard
uv run scripts/infer_policy.py --walking output.onnx
```

Resume from a checkpoint:

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 \
    --agent.run-name resume --agent.load-checkpoint model_29999.pt --agent.resume True
```

No GPU? Add `--hf-jobs` to any train command to run it on Hugging Face Jobs
instead of locally (see [scripts/hf/README.md](scripts/hf/README.md)).

## Tasks

`uv run list-envs` prints the live registry. Flat/Rough variants exist where noted.

<!-- SHOWCASE GRID — one short GIF per task family (sim or real), 3 per row.
     Priority order if you only record a few: Velocity, VelStand (fall+recover),
     Roulade, SitStand, Rollers/Swizzle, BallKick. -->

| Task id | Terrain | Description |
|---|---|---|
| `Mjlab-Velocity-{Flat,Rough}-MicroDuck` | flat/rough | **The main task**: walking with velocity commands + head-pose commands |
| `Mjlab-VelStand-{Flat,Rough}-MicroDuck` | flat/rough | Walking + fall recovery in one policy |
| `Mjlab-StandUp-{Flat,Rough}-MicroDuck` | flat/rough | Stand up from face-down/face-up/sitting, then hold the stand + body-pose control |
| `Mjlab-SitStand-{Flat,Rough}-MicroDuck` | flat/rough | Commanded sit ↔ stand in one policy, gently, head commandable |
| `Mjlab-GroundPick-{Flat,Rough}-MicroDuck` | flat/rough | Crouch and touch the ground with the mouth tip, return to stand |
| `Mjlab-BallKick-Flat-MicroDuck` | flat | Kick a 70 mm / 15 g ball forward (actor is ball-blind) |
| `Mjlab-Roulade-Flat-MicroDuck` | flat | Forward roll over the head, land back on the feet |
| `Mjlab-Velocity-Flat-MicroDuck-Rollers` | flat | Roller-skate velocity tracking (passive wheels under the feet) |
| `Mjlab-Velocity-Swizzle-MicroDuck` | flat | Classic symmetric swizzle skating |
| `Mjlab-RollerCrouch-Flat-MicroDuck` | flat | Crouch while gliding on rollers |
| `Mjlab-RollerSlope-Flat-MicroDuck` | slope | Glide down slopes on rollers |
| `Mjlab-RollerStandUp-Flat-MicroDuck` | flat | Stand up from the ground onto the wheels |
| `Mjlab-Spin-Flat-MicroDuck` | flat | Fast spin in place on rollers |

At deployment the runtime hot-swaps these policies (walk / recover / trick)
behind a shared 61-dimensional observation contract, so any of them can take
over the robot at any moment. `scripts/infer_policy.py` rehearses exactly that:

```bash
uv run scripts/infer_policy.py --walking walk.onnx --standing stand.onnx \
    --sitstand sitstand.onnx --roulade roulade.onnx --new-cmd-obs
```

Keyboard-driven (velocity commands, `G` ground pick, `Y` sit/stand, `R` roulade,
`K`/`L` kicks); `--debug`, `--save-csv`, `--record` support sim2real comparisons.
The servos are simulated with the same BAM M6 XL330 model the policies are
trained against (voltage control + load-dependent friction, via
`bam.mujoco.MujocoController`); `--vin` / `--vin-drop-gain` / `--kp-fw` pin the
training DR ranges to one value, `--no-bam` falls back to the XML PD actuators.

### Backlash variants

Every main task has a **Backlash** twin that trains on a model with ±1° of gear
play (2° total) in series with each of the 14 servo joints: insert `-Backlash`
before `MicroDuck` in the task id, e.g. `Mjlab-Velocity-Flat-Backlash-MicroDuck`.

The backlash is modeled properly for sim2real: each servo gets an unactuated
`passive_<joint>_backlash` hinge, and because the real encoder sits on the
output side of the play, both the firmware PD emulation
(`BacklashEncoderBamActuator`) and the `joint_pos`/`joint_vel` observations
read *through* the backlash (`qpos[servo] + qpos[backlash]`). Observation and
action dims are unchanged, so ONNX export and the runtime need no changes.
See `src/mjlab_microduck/tasks/backlash.py`.

## Actuator model

All tasks use the [BAM](https://github.com/Rhoban/bam) M6 actuator model for
the Dynamixel XL330 (voltage control law, back-EMF, Coulomb/Stribeck/load-dependent
friction), with per-env domain randomization on battery voltage, voltage sag
under load, command delay, and friction magnitude
(`FrictionDRBamActuator` in `src/mjlab_microduck/actuator/`).

At this scale — tiny servos driving a ~800 g biped — actuator fidelity is most
of the sim2real gap, which is why the actuator is modeled down to its voltage
control law instead of an ideal PD.

## Robot models

MJCF models live in `src/mjlab_microduck/robot/microduck/` and are exported
from Onshape with [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot),
one `config_mjcf_*.json` per model:

| XML | Used by |
|---|---|
| `robot_walk.xml` | Velocity (stripped trunk/head contacts — falling is cheap) |
| `robot_groundcontact.xml` | VelStand, StandUp, SitStand, GroundPick, BallKick, Roulade (curated collision set for the parts that touch the floor — body can physically lie on the ground; formerly `robot_allcollisions.xml`) |
| `robot_groundcontact_rollers.xml` | Roller tasks (passive wheels) |
| `robot_allcollisions.xml` | True full-collision model — every part has a collision geom. No task uses it yet |
| `robot_*_backlash.xml` | Backlash task variants (generated by `add_backlash.py`) |

`scene*.xml` files wrap the robots with a floor + keyframes (STAND/SIT/FOLD)
for quick viewing and for `infer_policy.py`.

<!-- IMAGE — side-by-side render: walk model vs rollers model (or a collision-geom
     visualization). One image here makes the model-variant story instant. -->

## Project structure

```
src/mjlab_microduck/
├── robot/
│   ├── microduck/                    # MJCF exports, export configs, scenes, add_backlash.py
│   └── microduck_constants.py        # robot cfgs, HOME frame, BAM actuator cfg
├── actuator/friction_dr_bam.py       # BAM + friction DR + backlash encoder feedback
├── tasks/
│   ├── __init__.py                   # task registration (base + backlash variants)
│   ├── mdp.py                        # rewards, events, observations, custom classes
│   ├── backlash.py                   # make_backlash_variant() env-cfg wrapper
│   └── microduck_*_env_cfg.py        # one cfg module per task family
├── train_cli.py                      # `train` script (identical to mjlab's)
├── train_hook.py                     # intercepts `train ... --hf-jobs`
└── hf_jobs.py                        # Hugging Face Jobs submission
```

Conventions worth knowing:

- The observation layout is shared across every policy (61-dim actor obs:
  48 proprioception + commands `[twist(3), head_pose(4), body_pose(6)]`), which
  is what makes runtime policy hot-swapping possible. Envs that don't use a
  command slot zero-pad it rather than dropping it.
- Unactuated joints are all named `passive_*` (roller wheels, backlash
  hinges); actuators, joint observations and pose rewards select servo joints
  with `^(?!passive_).*`.
- Domain-randomization toggles are `ENABLE_*` booleans at the top of each
  env cfg file.
- Joint layout (14 servos): 0–4 left leg (hip_yaw, hip_roll, hip_pitch, knee,
  ankle), 5–8 neck/head (neck_pitch, head_pitch, head_yaw, head_roll),
  9–13 right leg.
- The exporter bakes the observation normalizer into the ONNX graph — always
  deploy ONNX produced by `scripts/export.py`, never a hand-converted
  checkpoint, or the policy sees unnormalized observations at runtime.

[AGENTS.md](AGENTS.md) documents the env-building workflow and the reward-design
rules learned across the project (also aimed at AI coding agents working in
this repo).

## Publishing a policy

`uv run publish` puts a policy on the Hugging Face Hub in the shape the robot's
daemon loads: one `policy.onnx` with the observation normalizer baked in, a
`manifest.json` following schema 2 of the
[microduck policy manifest](https://github.com/pollen-robotics/microduck/blob/main/docs/policy-manifest.md),
and a README saying how to run it. Anyone with a microduck can then install it
with one command, no daemon release needed.

```bash
# From a wandb run — exports through the one safe path, then uploads
uv run publish --task Mjlab-PoliteBow-Flat-MicroDuck \
    --wandb-run-path <entity/project/run_id> --checkpoint 3000 \
    --repo <user>/microduck-polite-bow --kind episodic --duration-s 4.0 \
    --description "Bows from a two-foot stand and comes back up."

# From an ONNX you already exported (validated, not re-exported)
uv run publish --onnx output.onnx --repo <user>/microduck-flamingo \
    --kind perpetual --unwind-s 1.5 --twist-help "[flag, side, 0]"

# A new gait for a slot
uv run publish --onnx output.onnx --repo <user>/microduck-my-walk --kind perpetual --slot walk

# Ship a rollout video too: uploaded as replay.mp4, which the Hub's replay widget plays
uv run scripts/render_policy.py output.onnx --seconds 10 --video media/bow.mp4
uv run publish --onnx output.onnx --repo <user>/microduck-bow --kind episodic --duration-s 4.0 \
    --video media/bow.mp4

# A remix (warm-started / fine-tuned from someone's published policy): name the parent so the
# model card sets `base_model` and the Hub links the two
uv run publish --onnx output.onnx --repo <user>/microduck-deep-bow --kind episodic --duration-s 5.0 \
    --base-model <owner>/microduck-polite-bow

# See what would be uploaded without touching the Hub
uv run publish --onnx output.onnx --repo <user>/microduck-bow --kind episodic --duration-s 4.0 --dry-run
```

Then on a robot:

```bash
sudo robotctl policy add polite-bow <user>/microduck-polite-bow   # episodic: length comes from the manifest
sudo robotctl policy add flamingo <user>/microduck-flamingo --hold 5   # held pose: you pick how long
sudo robotctl policy load walk <user>/microduck-my-walk                # gait: into the walk slot
robotctl robot do polite-bow
```

What `--kind` means, and what each needs:

- **episodic** — runs for `--duration-s` and returns itself to a standing pose
  (kicks, roulade, a bow). Add `--chain` if holding the button should repeat it.
- **perpetual** — runs until told otherwise. Two shapes:
  - a **gait** (a new walk or stand): add `--slot walk` (or `stand`) and
    nothing else; the owner installs it with `robotctl policy load walk <repo>`.
  - a **held pose** (the flamingo): give `--unwind-s`, how long the daemon
    drives the idle twist (`--idle`, zeros by default) before handing back to
    the gait, so the robot is not let go of on one foot. The owner runs it as a
    one-shot with `policy add ... --hold <seconds>`.

Before anything is uploaded, `publish` checks the graph is `[1,61] -> [1,14]`
(a 51-D legacy policy is refused with a message), runs it on plausible inputs
and refuses NaNs or a constant output, fills the `training` block from git and
wandb (task, commit, branch, dirty flag, run, checkpoint), and refuses to
overwrite an existing `.onnx` in the repo without `--force`. Repos are created
private; `--no-private` for public, `--tag v1` to tag the revision.

Only constant-command policies are publishable this way. Phase-driven moves
(the ground pick) and the posture-flag sit↔stand are driven by the daemon
itself and live in the official set, `pollen-robotics/microduck-policies`.

## Tests

```bash
uv run --with pytest pytest tests/
```

CPU-only config-invariant and reward-function regression tests — they lock in
joint-index mappings, reward sign conventions, and NaN guards.

## Related projects

- [microduck](https://github.com/pollen-robotics/microduck) — the Microduck project home, including the onboard runtime that runs the exported policies
- [mjlab](https://github.com/mujocolab/mjlab) — the training framework (MuJoCo Warp + rsl_rl)
- [BAM](https://github.com/Rhoban/bam) — better actuator models, by Rhoban

## License

The code in this repository is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

The 3D model files (the `.stl` meshes in `src/mjlab_microduck/robot/microduck/assets/`
and `src/mjlab_microduck/robot/xl330_test_bench/assets/`) are licensed under
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).
See [LICENSE-MODELS.md](LICENSE-MODELS.md) for the exact scope and attribution.
