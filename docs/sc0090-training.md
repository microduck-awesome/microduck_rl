# SC0090 12 V training

For the failure-focused continuation launched after the initial evaluation,
see [SC0090 V2 training](sc0090-failure-finetune.md). The original runs described
below are preserved as baselines; the current jobs are recorded separately in
`logs/sc0090_setup/v2_training_runs.json`.

The common actuator for all Microduck tasks is now SC0090 M6. The robot still
has 14 actuated joints, the original MJCF geometry, 61 actor observations and
50 Hz actions. XL330 mesh names refer to the unchanged CAD housings only.

The bundled `src/mjlab_microduck/actuator/sc0090_m6.json` is an exact copy of
`../sc0090-datasets/m6-matched-comparison/v3/best-actual48-model.json`:

- SHA256: `6efbe8f2c1a1f7dbb6518f8e43693d1c6e3694eee4b3f2dce009e3bba6b97466`
- Dataset commit: `c4715f3ee061ebdc30420acd10abff8bbc2e845f`
- Toolkit commit: `80e4a870cf1e760c33fefa468fd77c48f9546c40`
- BAM commit: `3505fba49a8aa41f47c9a729c495f0c083757b0b`
- Custom optimizer, selected seed 1, training MAE 1.105610 degrees and development
  holdout MAE 1.167544 degrees. The measured fit used 12.4 V.

Firmware settings are restored from that artifact: P=20, D=30, I=0, deadband=2
counts, PWM limit=1, radians/count=0.005782163964207939. Nominal simulation supply
is 12.0 V. Voltage DR is 10.8–12.6 V, supply resistance DR is 0–0.2 ohm and the
voltage floor is 10.0 V. These DR ranges are simulation assumptions, not new
measurements. The fitted command delay is zero; the old XL330 3–6 substep delay
is removed. All servos share one actuator group for battery current estimation.

The user specified a maximum output speed of 80 rpm (8.37758041 rad/s). A shared
GPU/CPU voltage limiter stops positive motor power at or above that speed while
retaining braking torque. It does not clip MuJoCo velocities: contacts/gravity
can still produce passive overspeed, and a discrete step can cross the boundary.
This is an explicit simulation speed limit, not a claim about the servo's
undocumented internal speed-regulation implementation. Below the limit, the
fitted M6 control/torque equations remain unchanged.

Robot targets and feedback use the same fitted linear encoder mapping, so the
bench bias does not displace robot HOME. The bench gravity offset is retained
in the source artifact but does not alter robot joint coordinates or inertia.
Only reflected motor armature changes. BAM applies friction randomization once;
the robot wrapper restores and samples the scale at each reset.

The source remains `physical_qualification.qualified=false`. The task configs
explicitly permit this candidate for the requested simulation training; no
physical qualification is inferred from PPO training. No hardware commands are
issued. CPU rehearsal in `scripts/infer_policy.py` loads the same model and uses
the same control coordinates and current-based voltage sag.

Install with Python 3.12 and `uv sync --frozen`. The local editable toolkit
dependency is declared in `pyproject.toml`; keep the sibling directory available.
BAM is pinned to the toolkit's exact revision. A declared uv override retains
the RL project's NumPy 2.4.1: the toolkit's NumPy 2.5.2 breaks mediapy 1.2.6 at
import time. This does not rerun or change the archived fit. The frozen lockfile
and an offline sync were checked against the installed environment.

The two training tasks are:

- `Mjlab-Velocity-Flat-MicroDuck`: walking with velocity commands.
- `Mjlab-StandUp-Flat-MicroDuck`: recovery and standing. Its unchanged curriculum
  starts with standing/sitting/prone poses, introduces supine poses at iteration
  600 and reaches the full fallen-pose mix at iteration 2500. Supine pose roll
  noise also covers side starts.

Run each task's 64-environment, 5-iteration smoke test before a long training
run. Training is from scratch: no XL330 checkpoint or optimizer is loaded.
Local runs use TensorBoard logging and retain checkpoints on disk.

## Local L40 runtime

This host has an L40 with 48 GB VRAM, 32 logical CPUs (16 physical cores) and
125 GiB RAM. Its system driver is 535.129.03. The launcher can use NVIDIA's
CUDA 12.9 forward-compatibility libraries from a local extracted package so
MuJoCo/Warp can use CUDA Graphs. The kernel driver and desktop are unchanged.
NVIDIA documents R535 support in its
[forward compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html).

Package provenance:

- [cuda-compat-12-9 575.57.08-0ubuntu1 amd64](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-compat-12-9_575.57.08-0ubuntu1_amd64.deb)
- SHA256: `536cf41fb15fd7d8b39231648379b48c41718d4bf588b44481443ed35f1f6ce9`
- Extracted with `dpkg-deb --extract` into `artifacts/cuda-compat-12.9`.
- The launcher prepends `usr/local/cuda-12.9/compat` within that directory to
  `LD_LIBRARY_PATH` only for supported R535/R570 system drivers. If the local
  directory is absent, the normal installed driver is used.
- PyTorch CUDA Graph capture/replay and both task smoke tests passed with the
  compatibility libraries; Warp reports CUDA driver API 12.9.

## Validation

- Full suite: **236 passed, 1 skipped** (`logs/sc0090_setup/pytest.log`).
- Both tasks: 64 environments, 5 iterations, zero `nan_state` terminations,
  saved checkpoints and ONNX exports. The walking smoke was repeated after
  enabling CUDA Graphs.
- Export checks: ONNX checker and CPU inference, input `(1, 61)`, output
  `(1, 14)`, finite values, observation normalization included.
- CPU HOME hold for 3 seconds at 10.8 / 12.0 / 12.6 V: final trunk height
  0.11564–0.11575 m and tilt 3.93–4.39 degrees. These are equilibrium checks,
  not evidence of a learned gait or recovery success.
- A further nine HOME holds (three seeds at each voltage) perturb every joint
  by up to 0.03 rad and initial trunk roll by up to 0.035 rad. All remain finite
  and standing after 3 seconds: height 0.11517–0.11577 m, tilt 4.05–7.02 degrees.
- Setup logs and detailed JSON results are under `logs/sc0090_setup/`.

## Joint calibration

The simulation retains the original joint zero definitions, axes, limits and
HOME pose. The M6 bench offset is not a replacement for a robot joint's zero.
Before hardware inference, the runtime must map each of the 14 motor IDs to
the correct policy joint and calibrate its encoder zero count and direction.
Its count-to-radian conversion, mechanical limits, firmware P/D settings and
80 rpm output limit must agree with the simulation convention. No per-joint
hardware calibration values were inferred or written during this migration.

## Local training and monitoring

Each throughput benchmark ran for 12 iterations; FPS below is the median after
discarding the first three iterations. Aggregate FPS sums the two task medians.
The GPU utilization column uses samples during active GPU work (at least 65%
utilization), excluding most startup work; VRAM is the peak device-wide value.

| Configuration | Walk steps/s | Recovery steps/s | Total steps/s | GPU use | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Walk only, 4096 envs | 54,782 | — | 54,782 | 72% | 6,602 MiB |
| Walk only, 8192 envs | 75,300 | — | 75,300 | 82% | 12,846 MiB |
| Both, 8192 each | 40,979 | 38,587 | **79,566** | 99.5% | 24,603 MiB |
| Both, 12288 each | 37,118 | 35,457 | 72,575 | 100% | 39,825 MiB |

The 8192 pair is about 9.6% faster than the larger pair. Peak combined host RSS
was 24.2 GiB versus 52.8 GiB; active CPU use was about two logical cores across
the two GPU-driven loops. CPU/RAM capacity is retained for work that needs it;
artificially filling it would not improve this measured training throughput.
These short measurements characterize startup training, not later curriculum
stages. Raw samples and commands are in `logs/sc0090_setup/benchmark_*.json`
and the corresponding named benchmark JSON files.

The selected configuration runs both tasks concurrently with 8192 environments
each. Walking is assigned logical CPUs 0–15 and recovery CPUs 16–31, with
OMP/MKL thread limits of 8 per process. PPO minibatches, rollout length,
observations, rewards and curricula retain their task defaults. The existing
iteration budgets are 50,000 for walking and 15,000 for recovery. Checkpoints
are saved every 50 iterations for closer inspection of the new actuator model.

Equivalent foreground commands, one per terminal:

```bash
taskset -c 0-15 scripts/train_sc0090_local.sh walk \
  --env.scene.num-envs 8192 --agent.save-interval 50
taskset -c 16-31 scripts/train_sc0090_local.sh recovery \
  --env.scene.num-envs 8192 --agent.save-interval 50
```

The already launched processes use independent sessions. Their PIDs, exact
commands, creation times and source-snapshot hash are recorded in
`logs/sc0090_setup/training_runs.json`. Inspect that file and the live processes
before launching another copy. The snapshot includes new source files that a
plain `git diff` would omit.

```bash
tail -F logs/sc0090_setup/train_walk.log logs/sc0090_setup/train_recovery.log
uv run --frozen tensorboard --logdir logs/rsl_rl --host 127.0.0.1 --port 6006
```

Checkpoints and exported policies are under `logs/rsl_rl/sc0090_walk/` and
`logs/rsl_rl/sc0090_recovery/`. To resume after a deliberate stop, use the same
task/environment count with `--agent.resume True --agent.load-run RUN_DIRECTORY`
and `--agent.load-checkpoint CHECKPOINT_FILENAME`; choose the directory and
checkpoint from that task's experiment. A running process should not be
restarted solely because a log viewer or tool call times out.

The initial runs were launched on 2026-09-15 (Asia/Shanghai), both with run
directory `2026-09-15_03-00-03_sc0090_12v_m6`. At the startup verification,
walking had reached iteration 17 and recovery iteration 16; both had finite
`model_0.pt` checkpoints containing actor/critic normalizers and environment
state, with zero `nan_state` terminations. Device use was 99% and 24,603 MiB.
See `logs/sc0090_setup/training_start_verified.json` for this dated observation;
the live logs are authoritative for subsequent progress. Training has started,
and learned walking/recovery performance still requires later evaluation.

## V2 execution optimization and accuracy checks (2026-09-15)

The flat SC0090 V2 walking/recovery tasks remove the terrain's visualization-only
`env_origin_*` sites before compilation. Terrain origin tensors and all seven
robot sites remain intact. With 8192 environments, the old model had 8199 sites,
so each site-transform launch processed 67,166,208 world/site pairs. The new
model processes 57,344 pairs. This also eliminates 3 GiB of redundant site
position/matrix buffers per 8192-environment job. Existing custom scene callbacks
on other configurations are preserved.

The tasks also cache the damping scalar on the GPU and reuse a CUDA Graph for
the `set_const_0` part of reset-time recomputation.
The actuator still writes friction and damping every physics substep and uses
the original BAM calculations. Changes in damping value, device or dtype
refresh the scalar. Model arrays are looked up on each write, including after
field expansion. The original task factories do not enable these optimizations.

The graph runs the pinned MuJoCo Warp function, with its original kernels,
precision and ordering. `set_const_fixed` remains outside capture because this
version reads gravity-compensation flags on the CPU. Recreating mjlab's graphs
also invalidates the recompute cache, so replacement arrays cannot leave stale
captured pointers. CPU/devices without CUDA Graph support retain eager execution.

Accuracy checks exposed a separate storage issue in the original pipeline:
`actuator_acc0` had shape `(1, 14)` and a zero world stride even with 64 worlds;
`stat.meaninertia` was also a singleton. Reset-time kernels write different
values for randomized worlds, so these allocations allowed cross-world write
races. Mean inertia participates in the solver's convergence scaling. V2 now
allocates independent per-world storage for these values and the remaining
camera/light/actuator derived fields. This correctness fix remains active when
graph replay is disabled. It corrects storage ownership, without changing the
motor fit, 80 rpm limit, time step, observations, rewards or curriculum.

Validation after the storage fix:

- **260 tests passed, 1 skipped**. Motor torques and friction/damping writes
  match the original computation exactly, including changed damping values,
  interleaved passive DOFs, partial resets and replacement storage.
- Compiling each real robot with and without the origin markers preserves all
  tested body/joint/DOF/geometry/actuator parameters and sensor references. Across
  100 CPU physics steps per model, qpos/qvel/qacc, actuator/contact forces,
  sensor data and every remaining site's position/matrix are exactly equal.
  The environment origin tensor also remains exactly equal.
- The serial-chain recompute fixture matches exactly for all three recompute
  levels, changed model parameters and graph invalidation after expansion.
- The real walking and recovery models exercise full and alternating partial
  resets with domain randomization. Physical inputs/states match exactly.
  Derived quantities use the same `rtol=1e-6, atol=1e-7` bound for both
  eager-vs-eager and eager-vs-graph comparisons: the original branching-tree
  kernels already contain nondeterministic FP32 atomic sums. This is not a
  claim of bitwise-identical full training trajectories.
- Both tasks completed 64-environment, 5-iteration continuation smoke runs.
  Every saved checkpoint tensor was finite; optimizer state and the recovery
  curriculum were preserved, and each environment counter advanced by 120.
- Normalized ONNX exports retained the 61-input/14-output contract. Compared
  with full FP32 policy inference, maximum absolute errors were `5.97e-7`
  for walking and `9.54e-7` for recovery. The comparison temporarily disables
  TF32 only for validation, then restores the original training precision.

`scripts/benchmark_sc0090.py` runs the real PPO loop in a separate process and
saves only to a new output directory. Variants `reference`, `writes`, `graphs`,
`sites` and `optimized` isolate execution changes; **all include the storage fix**.
Use the same checkpoint, seed and environment count, and bracket optimized
runs with repeated reference runs. The reported wall time includes logging,
synchronization and any intervening checkpoint write. Discard warmup, compare
steps/s, and do not add isolated-task throughput to claim concurrent throughput.

Final single-process measurements used 8192 environments, 3 warmup iterations
and 12 measured iterations per fresh process, loading checkpoint 1050 for
walking and 750 for recovery:

| Task | Reference s/iteration | Markers removed only | All optimizations | Throughput gain |
| --- | ---: | ---: | ---: | ---: |
| Walking | 2.561 | 2.082 | 1.956 | 30.9% |
| Recovery | 2.929 | 2.361 | 2.320 | 26.2% |

Earlier repeated reference medians were 2.549/2.553 s for walking and
2.957/2.918 s for recovery. Marker removal provides most of the improvement;
graph replay and cached writes provide additional benefit after that bottleneck
is removed. These are execution benchmarks, not learned-skill evaluations.

Local validation artifacts are under `logs/sc0090_setup/optimization/`;
`logs/sc0090_setup/optimization_full_tests_sites.log` contains the complete test result.
The original production processes are deliberately paused at the user's
request. `logs/sc0090_setup/optimization_pause.json` records their identities,
last logged iterations and checkpoint hashes. Existing processes have already
loaded the old code; applying the update requires checkpoint continuation in
new processes, rather than simply sending SIGCONT to the paused processes.

### Further optimization after local checkpoint `5188aa3`

The first validated implementation was committed locally before continuing.
The next changes preserve the motor equations, solver settings, PPO batch size,
normalization and curricula:

- `fast_friction_force` extracts active DOF-friction rows with one Warp kernel.
  It keeps the original `row < nefc` and constraint-type mask, accumulates into
  the actual DOF indices and skips inactive zero contributions. Stale invalid
  IDs and NaN forces remain ignored. Active invalid IDs set a device error
  flag checked by `torch._assert_async`, preserving asynchronous failure instead
  of allowing an out-of-bounds write. Other dtypes and CPU retain BAM's path.
- `graph_compute` records the original SC0090 tensor program. It does not use
  compiler fusion or change floating-point settings. Commands and previous duty
  are copied into stable buffers; current DR tensors are read each replay.
  Changes to storage, tensor layout/dtype or scalar model parameters invalidate
  the graph. Full reset follows the original no-previous-duty branch; partial
  reset's updated duty is copied before replay. Friction/damping writes remain
  on the original bridge stream, outside the graph. CPU, calls outside training's
  `inference_mode`, autograd and simulations without CUDA graph support retain
  the eager path. Mode-switch tests cover this fallback; capturing across modes
  exposes an inference-tensor restriction in PyTorch 2.9's graph RNG bookkeeping.
- Both flags default off in the actuator and are enabled by the V2 factories.
  The benchmark's `previous` variant reproduces the execution settings committed
  in `5188aa3`; `optimized` includes the new motor optimizations. Graph capture
  counts are recorded so unintended recapture on every reset is visible.

The new tests compare full motor torque, shared-supply voltage, firmware gains,
duty, friction and damping against the original implementation with **zero
numerical tolerance**, including the 80 rpm boundaries, near-zero velocity,
full/partial resets, in-place randomization, replacement storage and changed
scalar parameters. Constraint extraction tests also cover noncontiguous storage,
ignored NaNs/invalid IDs and repeated indices with exactly representable sums.
A separate kernel test verifies that invalid active IDs cannot write outside the
output array. Parallel floating-point summation retains the limits described in
the preceding validation section.

The final motor regression run passed **266 tests, 1 skipped**. The mode-switch
checks also verify that an independent CUDA capture can follow motor training:
CUDA graph bookkeeping is allocated with inference mode disabled, avoiding
inference-only tensors in PyTorch's shared generator state. This changes tensor
bookkeeping, not the recorded arithmetic or training's numerical precision.
Full output: `logs/sc0090_setup/optimization/motor_final_full_tests.log`.

End-to-end measurements, each task at 8192 environments × 24 steps/iteration:

| Execution | Walking s/iteration | Recovery s/iteration | Combined steps/s |
| --- | ---: | ---: | ---: |
| `5188aa3`, two concurrent processes | 3.412 | 4.712 | 99,355 |
| New motor execution, concurrent | 3.107 | 4.538 | 106,606 |
| New motor execution + MPS | 2.370 | 2.912 | 150,488 |
| MPS repeat in a fresh session | 2.398 | 2.959 | 148,438 |

Both processes warmed up for 4 iterations and ran 16 measured iterations.
Only complete iteration intervals while **both** processes were past warmup
and still running are included; the MPS comparisons have 15 walking and 12
recovery samples each. Each motor graph was captured once. The repeated MPS
result is **49–51% faster than `5188aa3` in concurrent throughput**, and 39–41%
faster than the new motor execution without MPS. These gains must not be added
to the earlier isolated-task percentages.

Single-process results with the new motor path were 1.796 s walking and
2.181 s recovery. Fresh `previous` controls measured 1.980/2.373 s, giving
10.2%/8.8% additional single-process throughput. Relative to the original
corrected-storage reference, the gains are 42.6%/34.3%.

MPS allows kernels from the two processes to execute concurrently while
retaining separate GPU address spaces. See NVIDIA's
[MPS architecture documentation](https://docs.nvidia.com/deploy/mps/architecture.html).
This experiment used a private per-user MPS pipe, the existing CUDA compatibility
libraries, default MPS resource allocation, 8 CPU threads per worker and CPU
affinities `0-15` / `16-31`. GPU compute mode and the system driver were unchanged.
The diagnostic MPS daemons were stopped after their children exited.

`scripts/benchmark_sc0090_pair.py` reproduces the comparison, creates a fresh
output directory and supervises only its own children. `--mps` manages a private
MPS daemon; without that flag, child processes explicitly bypass inherited MPS
settings. Example (inherit the same `LD_LIBRARY_PATH` used by local training):

```bash
.venv/bin/python scripts/benchmark_sc0090_pair.py \
  --walk-checkpoint "$SC0090_WALK_CHECKPOINT" \
  --recovery-checkpoint "$SC0090_RECOVERY_CHECKPOINT" \
  --output logs/sc0090_mps_comparison \
  --walk-cpus 0-15 --recovery-cpus 16-31 --mps
```

MPS passed all **19 execution/accuracy tests**, followed by both 64-environment,
5-iteration continuation runs and normalized ONNX exports. ONNX maximum absolute
errors versus FP32 inference were `4.77e-7` / `1.91e-6`. The environment counters
advanced by 120, optimizer state was retained, recovery curriculum remained at
stage 0, and all diagnostic checkpoint tensors were finite. MPS is a scheduling
optimization, not evidence that long training will converge or that recovery
quality has improved.

The original production PIDs **2080438 / 2080439 remain stopped**, and checkpoint
1050 / 750 hashes are unchanged. Resume requires new processes loading these
checkpoints; to use MPS, both new processes must inherit the same private MPS
pipe. No production run was resumed during optimization.

Final local evidence: `logs/sc0090_setup/optimization/motor_results.json`,
`motor_report.md`, `motor_validation_manifest.json`, `motor_final_full_tests.log`,
`motor_pair_mps_v2_tests.log`, `motor_pair_mps_v2_*` and `mps_repeat/`.
