# SC0090 V4: from scratch or continuation after 6000 updates

V4 makes the same training program usable from random weights, from an existing
SC0090 expert, and after interrupting a V4 run. It adds new task IDs and leaves
the currently running V2 walking / V3 recovery jobs on their existing recipes.
There is no automatic launch when those jobs finish.

| Task | Local launcher | Registered task |
| --- | --- | --- |
| Walking | `walk_v4` | `Mjlab-Velocity-Flat-MicroDuck-SC0090-V4` |
| Recovery and standing posture | `recovery_v4` | `Mjlab-StandUp-Flat-MicroDuck-SC0090-V4` |

## Three startup paths

**From scratch:** omit the checkpoint. Networks, optimizer and observation
normalizers are new; phase 0 starts with small head commands and CoM variation.
The learning-rate default remains 2e-4 with adaptive KL target 0.005. Discovery
must pass evaluation before harder spawns, pushes or posture demands begin.

```bash
env -u MICRODUCK_WARM_START scripts/train_sc0090_local.sh recovery_v4 \
  --env.scene.num-envs 8192 --agent.max-iterations 12000 \
  --agent.run-name fresh_staged
```

**Continue an existing expert after 6000 updates:** provide its exact checkpoint
path. For the old zero-based runs this is normally `model_5999.pt`; use the
actual saved file. V2/V3 may have repeated iteration labels on earlier resumes,
so the global step counter is restored from checkpoint metadata, never inferred
as `iteration * 24`.

```bash
env -u MICRODUCK_WARM_START scripts/train_sc0090_local.sh recovery_v4 \
  --env.scene.num-envs 8192 --agent.max-iterations 2000 \
  --agent.training-program.checkpoint /absolute/path/to/model_5999.pt \
  --agent.run-name after_6000
```

`--agent.max-iterations 2000` means **2000 additional PPO updates**, not a total
iteration target of 2000. It does not promise every phase will finish in that
budget. Use the walking task's own checkpoint with `walk_v4` for walking.

The loader retains actor, critic, optimizer moments, observation normalizers and
global simulation-step count. It restores both the optimizer LR and PPO's
separate adaptive LR scalar. A mature V3 recovery expert / V2 walking expert
enters consolidation at the current global step, then must pass assessment
before strengthening. An unfinished legacy recovery frontier is retained.
The number 6000 is not a curriculum trigger.

**Resume V4:** use the same checkpoint option with a V4 checkpoint. Phase,
phase-entry step, pass streak, evaluation cadence, smoothing reference metrics
and completion state are restored. The next update label is saved explicitly,
so resuming a checkpoint does not repeat its completed iteration label.

```bash
env -u MICRODUCK_WARM_START scripts/train_sc0090_local.sh recovery_v4 \
  --env.scene.num-envs 8192 --agent.max-iterations 2000 \
  --agent.training-program.checkpoint /absolute/path/to/v4/model_7999.pt \
  --agent.run-name continued_staged
```

Do not combine this explicit-path option with `--agent.resume True`; native
`--agent.resume` / `--agent.load-run` / `--agent.load-checkpoint` remain available
for checkpoints inside the same experiment directory. V4 rejects
`MICRODUCK_WARM_START`: starting a new phase does not reset global history.
Use each task's own normalizers. V4 task tags or legacy `params/agent.yaml`
establish provenance; legacy recovery checkpoints also carry a recovery tag.
If a legacy file was copied without its sidecar, explicitly supply
`--agent.training-program.legacy-task walk` or `recovery` after checking its
origin. A conflicting tag is rejected even with that option.

## Phase program

The complete, declarative stage table is built by `make_training_program()` in
`src/mjlab_microduck/tasks/microduck_sc0090_v4_env_cfg.py`. MDP state/application
functions are in `tasks/mdp.py`; `tasks/program_runner.py` handles checkpoints
and assessment processes. Neither the PPO learning loop nor motor arithmetic
has been replaced.

| Phase | Settings |
| --- | --- |
| Foundation | Four head-command / CoM levels, from small ranges to existing full ranges. Recovery initially uses the near-prone frontier. |
| Recovery discovery | Existing roll frontiers 120→90→60→30→0°, one at a time; walking skips these stages. |
| Consolidation | Existing skills under full head/CoM randomization, modest pushes and light smoothing. |
| Pushes | Per-axis velocity increments ±0.10→0.15→0.20→0.30 m/s. These are velocity perturbations, not forces in newtons. |
| Body posture | Height down/up: 5/5→15/5→25/10 mm; roll/pitch ±3→8→12°. Tracking weight 1→2→4. These endpoints remain subject to physical/learned feasibility evaluation. |
| Smoothing | Action-change weight −0.5→−0.7→−1.0, then torque-change weight −0.001; recovery finally increases arrival damping −0.025→−0.05. |

25% of training episodes rehearse without pushes; once body tracking is active,
these episodes use the neutral body command. Existing velocity buckets and all
recovery spawn groups remain represented. Walking posture tracking uses the
locomotion-aware reward; it does not refer moving robots to their spawn origin.
Only height, roll and pitch are trained body-control axes. The x/y/yaw slots
retain their observation layout.

Recovery has two distinct conditions: first reach nominal standing under the
original continuous 0.5 s criterion; then track a posture command. Fixed nominal
height/upright rewards are reduced as commanded-posture reward rises. A legitimate
crouch has its own target-relative stability reward and does not create a false
nominal recovery success. The motor model, 80 rpm active-drive limit, 61D actor
observations, 14 actions and lack of action filtering stay unchanged.

## Evaluation gates and limits

At checkpoint saves, at most once every **200 updates** by default, the runner
launches an isolated deterministic-policy assessment. Configure cadence with
`--agent.training-program.interval-iterations`; actual execution is aligned to
checkpoint saves. Every seed has 128 samples per group **for each** of retention
and challenge. Two seeds therefore provide 256 samples per group/profile.
Recovery has seven spawn groups; walking has eleven fixed velocity commands.
DR and sensor noise remain active. Retention has no push/body command; challenge
uses current body targets and guaranteed cardinal pushes at 3 and 5 seconds,
then measures the final settled behavior. Failed episodes remain failures if
their worlds must be reset to continue the batched simulation.

- Foundation/roll stages require ≥70% success per required group in each seed;
  recovery's yet-unlearned exact side/back groups are still reported.
- Subsequent stages require ≥95% retention and ≥90% current challenge success
  **in every group, for every seed**.
- Two consecutive passing assessments and at least 2400 environment steps in
  the current phase are required. A report can advance only one phase.
- Smoothing additionally requires ≥1% improvement of its relevant measured RMS
  quantity compared with the preceding qualified phase. Recovery's 95th-percentile
  nominal rise time may grow at most 15% + 0.1 s. Thresholds are initial settings
  for evaluation, not claims that all robots can achieve them.
- Reports with a wrong checkpoint/config hash, phase, counter, missing group,
  insufficient samples or nonfinite values cannot promote. Errors are recorded
  separately and leave the phase unchanged.

Reports and frozen inputs are under `program_evaluations/step_XXXXXXXXX/`.
`last_qualified.pt` retains the latest policy that passed the current gate. To
roll back, explicitly resume that checkpoint with the same plan. There is no
silent weight replacement during PPO updates. The `complete` flag records that
all phases have passed; use the latest `last_result.passed` as well when deciding
whether the current policy is acceptable. Release evaluation should additionally
use held-out seeds, intermediate push strengths and videos.

V4 currently requires a single CUDA training GPU with room for the temporary
assessment process. The evaluator bypasses shared MPS; the default timeout is
300 s. In this L40 smoke run, recovery assessments took about 59–64 s, walking
expert assessment 69 s, and random walking policy assessment 153 s (frequent
terminal resets are more expensive). These are observed smoke timings, not
portable performance guarantees. Increasing the interval reduces this overhead.

Changing the phase plan invalidates its checkpoint fingerprint and is rejected
on resume; migrations must be explicit. Changes to hardware parameters require
their own validation. Physics episodes and RNG are not fully serialized by
mjlab, so a resumed trajectory is not claimed to be bitwise identical to an
uninterrupted run. The evaluator itself cannot update the parent's policy,
optimizer, normalizers or environment counter.

At deployment, the recovery command sender must use neutral body commands until
nominal standing is stable, and clear them when the robot falls again. This is
command scheduling, not action filtering; the 61D ONNX interface is unchanged.
The short tests below do not establish hardware readiness or mastery of the
new posture/push ranges.

## Validation

Full regression suite: **331 passed, 1 skipped**. Seven real 64-environment ×
5-update smokes covered fresh walking/recovery, legacy walking/recovery import,
V4 walking/recovery resume, and a synthetic advanced-phase fixture exercising
posture and smoothing rewards. The fixture changes only test checkpoint phase
metadata; it is not a trained advanced expert or a production checkpoint.

Checkpoint validation checks finite tensors, +100 optimizer steps, +7680
normalizer samples, +120 environment steps, correct next iteration labels,
preserved phase clocks/cadence and finite normalized 61→14 ONNX inference.
Across the assessments, policy, critic and optimizer remain bitwise unchanged.
Artifacts: `logs/sc0090_setup/v4_compatibility/{validation.json,full_tests.log}`.
Fresh runs correctly remain in foundation; the walking expert remains in
consolidation when its direction-specific retention fails the stricter gate.
An additional full advanced-phase assessment exercised posture commands and
forced pushes on the synthetic fixture with finite results; its low challenge
success correctly distinguishes nominal recovery from commanded-pose ability.
An actor-only load/rollout check restored the advanced phase without loading
optimizer state or updating observation normalization, covering play/export use.

Before any long V4 launch, run its own 64-environment × 5-update smoke using the
same task, checkpoint and overrides, as required by `AGENTS.md`.
