# SC0090 V4: from scratch or checkpoint continuation

V4 makes the same training program usable from random weights, from an existing
SC0090 expert, and after interrupting a V4 run. It adds new task IDs and leaves
the currently running V2 walking / V3 recovery jobs on their existing recipes.
There is no automatic launch when those jobs finish.

## Budgets, cadence, and learning criteria are different parameters

**6000 is the existing V2/V3 job budget**, not a convergence theorem or a phase
trigger. **2000 and 12000 in the examples are illustrative additional
budgets**, not required amounts. V4 now requires an explicit positive
`--agent.max-iterations`; it no longer inherits V2's default of 6000. Pick a
budget for the available compute, inspect measured progress, then resume if
needed. Exhausting the budget does not mark the program complete.

The current jobs inherit the preceding overnight training. Their V2 warm start
loaded walking `model_4050.pt` and recovery `model_3750.pt`, retaining the
learner while zeroing the iteration and recipe step counters. Those sources
already contained **4051 / 3751 PPO updates**, verified independently from
optimizer steps and observation-normalizer counts at the same 8192×24 rollout
and 5×4 optimizer-update settings. Thus today's 6000-label target is a
continuation budget, not the policy's lifetime training age. Walking
`model_5999.pt` actually contains **10052 cumulative PPO updates**: its retained
optimizer counter is 201040 and normalizer sample count is 1,976,303,616.
The old resume path repeated an update label, so filename arithmetic is not
an authoritative cumulative counter.

Keep three quantities separate when designing schedules: retained cumulative
experience, the checkpoint's current recipe clock, and exposure since entering
the current phase. V4 restores the saved recipe clock; it cannot reconstruct
history that an earlier warm start already removed from that clock. Optimizer
and normalizer counters retain that history in these specific runs. On other
runs, resets, normalizer update limits, or changed batch/epoch settings can make
that conversion invalid; missing history must be reported as unknown.

Existing experts retain their learned frontier, head/DR difficulty and adaptive
learning rate, then pass a baseline assessment. Time spent on newly introduced
push/posture objectives starts at their phase entry. Prior experience informs
the continuation budget and avoids repeating discovery; it is not counted as
exposure to a newly introduced challenge. This also means that a trained
checkpoint need not consume another fixed 6000/2000 updates before assessment.
The audit is saved in `logs/sc0090_setup/v4_review/training_history_audit.json`.

**200 is the configurable default number of PPO updates between assessments**,
chosen to amortize the measured 1–3 minute assessment cost. It is an operational
heuristic, not a learning threshold. With 24 steps/update, it equals 4800 control
steps per environment (96 simulated seconds at 50 Hz); 8192 environments collect
39,321,600 transitions in that interval. Change it with
`--agent.training-program.interval-iterations`. Actual execution waits for the
next checkpoint save; the startup log prints both settings and their units.
The initial saved checkpoint is assessed immediately.

The phase criteria live separately in `make_training_program()`: minimum 2400
steps per environment (48 simulated seconds), two consecutive passing reports,
and explicit success thresholds. These are initial empirical settings, not
universal constants. Changing that phase plan changes its checkpoint fingerprint.
Increasing the assessment interval reduces overhead but delays promotion and
detection of regressions; decreasing it does the reverse.

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
the checkpoint's recipe simulation-step count. It restores both the optimizer LR and PPO's
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
`MICRODUCK_WARM_START`: starting a new phase does not reset the saved recipe clock.
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
After another fall, posture commands require a new nominal hold before being
enabled again. Target changes restart the target-relative hold timer.

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
their worlds must be reset to continue the batched simulation. Automatic resets
use the normal training step path, so healthy worlds do not receive extra
observation-delay updates. Fixed walking commands are installed during command
reset, before observations; timed pushes occur inside the control step.

- Foundation/roll stages require ≥70% success per required group in each seed;
  recovery's yet-unlearned exact side/back groups are still reported.
- Subsequent stages require ≥95% retention and ≥90% current challenge success
  **in every group, for every seed**.
- Two consecutive passing assessments and at least 2400 environment steps in
  the current phase are required. A report can advance only one phase.
- Smoothing additionally requires ≥1% improvement of its relevant measured RMS
  quantity compared with the preceding qualified phase. Recovery's 95th-percentile
  nominal rise time (worst seed) may grow at most 15% + 0.1 s. Thresholds are initial settings
  for evaluation, not claims that all robots can achieve them.
- Reports with a wrong checkpoint/config hash, phase, counter, missing group,
  insufficient samples or nonfinite values cannot promote. Errors are recorded
  separately, break the passing streak, invalidate the latest qualification,
  and leave the phase unchanged.

Torque-change rewards skip the first sample after each episode reset because
there is no within-episode predecessor. Assessment RMS uses only first-episode
transitions, excluding reset discontinuities and retries; independent-seed RMS
values are pooled in squared units. Completing the final phase retains its
entrance baseline, so unchanged qualified performance remains qualified.
Phase changes take effect through the next normal environment reset, avoiding
an action computed from an observation cached under different live commands.

Reports and frozen inputs are under `program_evaluations/step_XXXXXXXXX_<unique>/`.
`last_qualified.pt` retains the latest policy that passed the current gate. To
roll back, explicitly resume that checkpoint with the same plan. There is no
silent weight replacement during PPO updates. The `complete` flag records that
all phases have passed; use the latest `last_result.passed` as well when deciding
whether the current policy is acceptable. Release evaluation should additionally
use held-out seeds, intermediate push strengths and videos.
Checkpoint and ONNX publication use same-directory atomic renames. A nonblocking
run-directory lock prevents two V4 writers from publishing into the same run. The loader
validates and loads one byte snapshot, rejecting invalid counters, nonfinite
state and inconsistent learning rates before overwriting the live learner.

V4 currently requires a single CUDA training GPU with room for the temporary
assessment process. The evaluator bypasses shared MPS; the default timeout is
300 s. Unsupported distributed configuration is rejected before the runner can
initialize NCCL. On timeout or interruption the evaluator process group is
terminated and reaped with a bounded wait; unsuccessful cleanup stops training
with a saved checkpoint. In the initial L40 smoke run, recovery assessments took about 59–64 s, walking
expert assessment 69 s, and random walking policy assessment 153 s (frequent
terminal resets are more expensive). These are observed smoke timings, not
portable performance guarantees. Increasing the interval reduces this overhead.

Changing the phase plan invalidates its checkpoint fingerprint and is rejected
on resume; migrations must be explicit. Changes to hardware parameters require
their own validation. Physics episodes and RNG are not fully serialized by
mjlab, so a resumed trajectory is not claimed to be bitwise identical to an
uninterrupted run. The evaluator itself cannot update the parent's policy,
optimizer, normalizers or environment counter.

The reviewed evaluator uses protocol 3. Initial V4 protocol-2 reports are not
comparable because they could advance delay buffers during manual resets and
include retry episodes in smoothing metrics. Protocol-2 V4 checkpoints without
a smoothing reference can load their model/optimizer/phase, with the old pass
streak cleared and a fresh assessment due. Checkpoints containing such a
reference are rejected for training explicitly; use the original recipe or import
the original V2/V3 expert. Actor-only play/export can still load them, discarding
obsolete assessment metrics. Protocol-3 V4 resumes retain their state normally.

At deployment, the recovery command sender must use neutral body commands until
nominal standing is stable, and clear them when the robot falls again. This is
command scheduling, not action filtering; the 61D ONNX interface is unchanged.
The short tests below do not establish hardware readiness or mastery of the
new posture/push ranges.

## Validation

Reviewed regression suite: **355 passed, 1 skipped**. Eight real 64-environment ×
5-update smokes covered fresh walking/recovery, legacy walking/recovery import,
V4 walking/recovery resume, a synthetic advanced-phase fixture exercising
posture and smoothing rewards, and an actual promotion in the training loop.
The two fixtures change only checkpoint phase metadata; they are not trained
advanced experts or production checkpoints.

Checkpoint validation checks finite tensors, +100 optimizer steps, +7680
normalizer samples, +120 environment steps, correct next iteration labels,
preserved phase clocks/cadence and finite normalized 61→14 ONNX inference.
Across the assessments, policy, critic and optimizer remain bitwise unchanged.
Artifacts: `logs/sc0090_setup/v4_review/{validation.json,full_tests.log}`.
Fresh runs correctly remain in foundation; the walking expert remains in
consolidation when its direction-specific retention fails the stricter gate.
An additional full advanced-phase assessment exercised posture commands and
forced pushes on the synthetic fixture with finite results; its low challenge
success correctly distinguishes nominal recovery from commanded-pose ability.
The suite also checks atomic checkpoint interruption, one-writer locks, bounded
child cleanup, fail-before-NCCL validation, explicit partial/full load modes,
and observation-delay clocks under repeated partial episode failure.
See [the review record](sc0090-v4-review.md) for findings and compatibility notes.

Before any long V4 launch, run its own 64-environment × 5-update smoke using the
same task, checkpoint and overrides, as required by `AGENTS.md`.
