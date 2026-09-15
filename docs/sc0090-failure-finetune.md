# SC0090 failure-focused continuation — 2026-09-15

The baseline evaluation at walk iteration 3900 and recovery iteration 3650
showed a persistent failure, not just an early checkpoint: a 0.1 m/s walking
command produced 0.0034 m/s; supine, supine with roll variation, and both side
starts each produced 0/32 stable recoveries. Sitting was 31/32 and prone 32/32.
Raw rollouts and the diagnostic report are in
`logs/sc0090_setup/weak_skills_20260915_0815/`. The earlier 17-case video gallery
remains at `artifacts/sc0090-videos-20260915/index.html`.

The two V2 tasks derive from their respective original factories. They keep
SC0090 M6 at 12 V, the 80 rpm active-drive limit, 14 actions, the 61D actor
layout, sensor delays/noise, joint geometry and voltage/friction randomization.
Trunk/head CoM ranges are pinned to the learned final ±15/±10 mm conditions,
including at step zero. The original task IDs remain available as baselines.

## Walking

Task: `Mjlab-Velocity-Flat-MicroDuck-SC0090-V2`.

| Exclusive command bucket | Sampling share | Command |
| --- | ---: | --- |
| Idle | 15% | Exact zero |
| Slow forward | 25% | 0.03–0.15 m/s |
| Slow backward | 8% | −0.03 to −0.15 m/s |
| Forward | 12% | 0.15–0.40 m/s |
| Backward | 7% | −0.15 to −0.35 m/s |
| Lateral | 13% | ±0.03–0.20 m/s |
| Turn in place | 13% | ±0.2–1.0 rad/s |
| Mixed | 7% | vx −0.3–0.4, vy ±0.15 m/s; yaw ±0.7 rad/s |

The base sampler's independent forward bucket clamps vx to at least 0.3 m/s;
V2 samples exclusive buckets directly, so it cannot overwrite a slow command
or an idle/turn sample. Tracking uses tolerance `0.035 + 0.45 * |v_command|`:
at 0.1 m/s, stationary tracking drops from 90.48% to 20.96% of its maximum.
Tracking weight rises from 2 to 3. Yaw tracking uses yaw error alone (std 0.3
rad/s), plus a small L1 yaw penalty, while retaining separate small body-motion
penalties. This avoids charging gait roll/pitch as if it were yaw error.

Air-time weight is reduced 3→1.5 and scales down further at slow commands;
the lower rewarded flight-time boundary becomes 75 ms. Action-rate weight is
−0.5 during this discovery run. Pushes are ±0.08 m/s; head commands are exactly
zero on half of resamples and retain the learned nonzero range otherwise.
Per-command-bucket XY and yaw errors are logged under `Tracking/`.

## Recovery

Task: `Mjlab-StandUp-Flat-MicroDuck-SC0090-V2`.

Each 8 s episode samples standing 5%, sitting 10%, prone 10%, a progressive
roll start 40%, exact left side 10%, exact right side 10%, or full supine 15%.
The roll frontier starts 150–180° along the supine-to-prone roll, with both
directions and random yaw. Its lower angle advances 150→120→90→60→30→0°.
These are initial-state distributions; no movement trajectory is prescribed.

Advancement requires at least 256 completed frontier episodes, at least 100
training iterations since the previous assessment, and ≥70% success. Statistics
are cleared after each assessment; episodes from an older frontier cannot
promote the new stage. Left, right and full-back success are counted separately.
At the final frontier, all four hard groups must pass ≥70% (at least 64 episodes
per side/back group) before extra smoothing is enabled.

Success means trunk height ≥0.105 m, tilt ≤20°, body speed <0.15 m/s, angular
speed <1.5 rad/s and both feet grounded continuously for 0.5 s. It counts once
per episode. The stable-standing reward is a continuous rate, not an arrival
bonus. Checkpoints retain frontier, assessment time, trial/win counts and polish
state; volatile episode buffers are discarded on restart.

The old iteration-driven body-pose/polishing schedules are removed from V2.
Body commands retain small nonzero values for observation compatibility, while
the reward targets nominal standing. The full height/upright/composite target
is retained. Discovery reduces joint-pose and action/momentum penalties;
action-rate is −0.2, with arrival damping and torque-rate initially zero.
Gentle-rise/limits/contact penalties remain. Pushes are ±0.05 m/s and head
commands are zero on 60% of resamples.

A signed bounded progress potential favors rolling from back to belly and
then completing the rise. It replaces positive-only upward velocity: holding
any pose pays zero progress, and reversing a move pays the opposite signed
change. A small persistent fallen-state cost discourages parking in the
observed low back-lean basin. Recovery also disables the inherited forward
sampler, which otherwise sends 20% of supposedly neutral twist commands to
vx≥0.3 m/s.

## Validation and continuation

CPU regression suite: 247 passed, 1 skipped. Both V2 tasks passed the mandatory
64-env × 5-iteration smoke from their own source checkpoint, including finite
training, 61D observations, checkpointing and the official normalized ONNX
export. Logs: `logs/sc0090_setup/v2_{unit_tests,full_tests,smoke_walk,smoke_recovery}.log`.
Export comparisons are recorded in `logs/sc0090_setup/v2_export_checks.json`.
These checks establish implementation validity, not learned recovery success.

Continuation seeds are immutable copies of SC0090 walk `model_4050.pt` and
recovery `model_3750.pt`, with SHA256 provenance in
`logs/sc0090_setup/v2_sources.json`. Weights, normalizers and optimizer moments
are preserved. `MICRODUCK_WARM_START=1` restarts the new curriculum counter and
iteration at zero. Initial LR is 2e-4 with adaptive KL target 0.005.

Both jobs use 8192 environments, save every 50 iterations, and have a 6000
iteration continuation budget. The local launch record and source snapshot are
in `logs/sc0090_setup/v2_training_runs.json`.

```bash
MICRODUCK_WARM_START=1 taskset -c 0-15 scripts/train_sc0090_local.sh walk_v2 \
  --env.scene.num-envs 8192 --agent.resume True \
  --agent.load-run source_baseline --agent.load-checkpoint model_4050.pt
MICRODUCK_WARM_START=1 taskset -c 16-31 scripts/train_sc0090_local.sh recovery_v2 \
  --env.scene.num-envs 8192 --agent.resume True \
  --agent.load-run source_baseline --agent.load-checkpoint model_3750.pt
```

On later restart of a V2 run, **unset `MICRODUCK_WARM_START`** and load its run
directory/checkpoint instead of `source_baseline`. This restores the trained
curriculum frontier. Inspect the launch record and processes before launching
another copy. Logs are `logs/sc0090_setup/train_{walk,recovery}_v2.log`.

Use the same fixed-command/spawn evaluation as the baseline for comparisons;
do not infer learned skills from total reward or from curriculum stage alone.
The archived evaluator accepts `--checkpoint` and `--output-dir`; use a fresh
output directory so the original videos and rollout records remain reproducible.

## Recovery V3: evaluated frontier and neck-limit penalty

Task: `Mjlab-StandUp-Flat-MicroDuck-SC0090-V3` (`recovery_v3` in the local launcher).
The V2 iteration-1900 diagnostic found deterministic success on standing/sitting/
prone/frontier at 16/16, 16/16, 15/16, 16/16 with neutral commands. With sampled
actions the same policy reached the posture but did not hold the strict stability
condition for 0.5 s. Consequently the stochastic training gate stayed at stage 0.
Both side starts and full supine remained 0/16. Doubling returned torque without
retraining also gave 0/16 on these hard poses; that counterfactual does not prove
physical feasibility or establish real motor torque requirements.

V3 retains the V2 motor, 80 rpm active-drive limit, observation/action layout,
DR, reset mixture, PPO and learned exploration standard deviations. It changes:

- Every 100 training iterations (at a checkpoint save), a separate process tests
  a frozen deterministic policy for 8 s. Each of two fixed seeds has 128 episodes
  for **each** of seven initial pose groups. Training noise, DR, commands and
  pushes remain enabled. This is stricter than the neutral-command diagnostic.
- Each seed must pass ≥70% on standing, sitting, prone and the current frontier.
  At least 256 frontier trials and 2400 training steps since the previous promotion
  are required. One assessment can advance only one frontier. The final stage
  requires both sides and full supine to pass as well before extra smoothing.
- Stochastic training success stays visible, but cannot advance or polish V3.
  Checkpoints preserve evaluation history, frontier and promotion time. Failed,
  incomplete, stale or mismatched assessments cannot promote. Assessment failures
  are written to `evaluations/iteration_XXXXXX/error.json` and the training console.
- A `neck_pitch` physical-position limit-proximity cost has margin 0.20 rad and
  weight −2.0 (zero in the interior, −0.4 at either hard limit before timestep
  weighting). This targets the observed saturated neck stop. Command ranges and
  motor strength stay unchanged.

The evaluator receives the exact constructor configurations including CLI
overrides, with checkpoint/config SHA256 binding. It loads trusted local inputs
only. Its process cannot update PPO, observation normalizers, environment state
or RNG in the training process. It bypasses a shared MPS server so timeout cleanup
does not terminate an MPS client. V3 currently rejects distributed training; a
single training GPU must have room for the temporary 896-environment evaluator.
The evaluation timeout defaults to 240 s. Retain raw per-seed reports when
interpreting advancement; stage alone is not proof of full recovery.

Resume from the saved V2 iteration 2150 with actor, critic, optimizer moments,
normalizers and step counter intact. Do **not** set `MICRODUCK_WARM_START` here.
The optimizer LR from that checkpoint is passed explicitly because RSL restores
the optimizer LR without restoring its separate adaptive LR scalar. The local
copy in `source_v2` has SHA256 provenance in
`logs/sc0090_setup/recovery_v3_transition/plan.json`.

```bash
scripts/train_sc0090_local.sh recovery_v3 \
  --env.scene.num-envs 64 --agent.max-iterations 5 \
  --agent.resume True --agent.load-run source_v2 \
  --agent.load-checkpoint model_2150.pt \
  --agent.algorithm.learning-rate 0.00011390625000000005
# After the smoke passes, use 8192 environments and 3850 remaining iterations
# for the existing total budget of 6000. Keep checkpoint save interval at 50.
```

For later V3 resumes, load its actual run/checkpoint and current optimizer LR.
The restored evaluation history prevents immediately repeating an assessment;
it also prevents resetting the minimum training dwell between promotions.

Validation on the local L40: 296 tests passed, 1 skipped, including exact motor
graph/reset comparisons and 30 V3 gate/config/restore regressions. V2→V3 and
V3→V3 both passed 64-environment × 5-iteration smokes. In each, optimizer steps
increased by 100, normalizer count by 7680 and training counter by 120, with
finite 61→14 ONNX inference. Across an evaluation the actor, critic and optimizer
were bitwise unchanged. The resumed V3 checkpoint retained stage/history and did
not repeat evaluation. The first V3 smoke assessment took 39.3 s: standing
256/256, sitting 255/256, prone 254/256, frontier 256/256; each side and full
supine 0/256. This validates stage 0→1, not the still-unlearned hard recoveries.
Artifacts: `logs/sc0090_setup/recovery_v3_transition/smoke_validation.json` and
`full_tests.log`.
