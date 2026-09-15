# SC0090 walking acceptance and dynamics revision 2

The final V2 walking policy followed ordinary forward commands but stalled near
0.03 m/s and tracked turns poorly. The old CPU video also contained a physics
adapter error. This update corrects numerical implementation and observation
timing before continuing the existing experts.

After the first continuation to 6400, [turning diagnosis and further training](sc0090-turn-diagnosis.md)
compare sampled versus deterministic actions and document the low-speed operating target.

## Confirmed errors

1. Pinned BAM CPU `MujocoController.update()` matched DOF-friction `efc_id`
   against joint IDs. Microduck's free base makes servo joint IDs 1–14 correspond
   to DOFs 6–19. The local controller preserves upstream control, torque and
   supply calculations, then corrects the pure friction budget before stepping
   physics. A real MuJoCo constraint-Jacobian projection independently tests the
   indexing, including floating bases and interleaved passive joints.
2. Pinned BAM's GPU M6 quadratic term omitted the identification model's
   `sign(external) != sign(motor)` condition and treated equal magnitudes as the
   external-load branch. The local tensor implementation now follows BAM's
   original NumPy equation. Tests cover sign combinations, equal loads, zero,
   near-stationary/high speeds, FP32/FP64 and friction randomization. Before
   correction, the explicit test grid (loads up to ±2 N·m, friction scales
   0.5–1.5) differed by up to 0.00354531 N·m. This is a test-grid result, not an
   estimate of the error experienced throughout earlier training.
3. CPU rehearsal used instantaneous joint velocity while these policies train
   with a one-control-step (20 ms) velocity-observation delay. The web demo now
   matches it. History advances only at inference and clears on reset;
   diagnostic reads and policy switches do not consume it.
4. CPU rehearsal now refreshes MuJoCo derived state at each control boundary,
   matching mjlab's post-decimation `forward()`. Previously gyro/gravity could
   describe the preceding 5 ms physics substep. Standalone inference exposes
   `--joint-velocity-obs-lag 0|1`, default 1.

Missing or ambiguous body/joint/sensor names now fail explicitly instead of
silently indexing the final object with `-1`; unique namespaces are supported.
No fitted parameter, joint zero, action filtering, observation dimension or
80 rpm limit was changed.

New V4 checkpoint ABI metadata and web status include dynamics revision 2.
Old V4 checkpoints with different dynamics metadata are rejected for
continuation. Import the original V2/V3 expert through the documented legacy
path and reassess, preserving weights, optimizer, normalizers and recipe clock.
Earlier assessments/videos retain their historical implementation and must not
be presented as revision-2 evidence.

## Walking contract

V4 uses 0.08 m/s as the provisional minimum commanded pure walking speed, based
on final-model measurements. This is a policy target, not a measured hardware
lower bound. The V2 recipe remains available; shared dynamics corrections apply
to all newly created environments.

- Explicit left/right turn buckets each receive 12% of commands. Idle, ordinary
  forward/backward, lateral and mixed motion remain represented. Pure slow
  commands sample 0.08–0.15 m/s; mixed commands still explore smaller components.
- Linear reward width is `0.02 + 0.35 * |command_xy|`, so standing at a 0.08 m/s
  command earns about 6.2% of that term's maximum. Yaw absolute-error weight is
  −1.0. Consolidation keeps action-change weight −0.5; stronger smoothing waits
  for measured qualification.
- Fourteen cases cover idle, low forward/backward, ordinary motion, each turn
  and curve direction, and start/stop. Start changes 0→0.08 m/s at 2 s; stop
  changes 0.1→0 at 2 s. Retention must follow the new command during 3–4 s;
  forced-push challenges settle and are judged during 6–8 s.
- Besides the existing instantaneous-error bounds, mean signed velocity must
  match within 25%, with absolute floors of 0.02 m/s and 0.10 rad/s near idle.
  Standing and half-speed turns cannot pass. Per-group/per-seed randomized
  retention/challenge thresholds remain unchanged.
- Each walking assessment also runs fixed HOME regressions at 12 V, nominal
  friction, no domain randomization/noise, and the trained velocity delay.
  All fixed cases must pass. These are reported separately and are not counted
  as independent randomized trials. They catch policies that move only when
  random perturbations dislodge them from a stationary state.

The sampling/reward/acceptance contract enters the program hash, preventing
silent reuse of obsolete qualification. Fresh training, legacy expert import
and matching revised V4 resume use the same implementation.

## Evidence and limits

Local artifacts: `logs/sc0090_setup/walk_repair/`. A matched CPU/GPU probe uses
the same compiled model, nominal parameters, initial state and commands, with
TF32 disabled for actor comparison. Initial observations match exactly;
physical-array discrepancies fit FP32 representation error. Normalized
actor/ONNX maximum output difference is 4.7684e-7. Corrected CPU/GPU forward
speeds are close (0.08 command: about 0.081/0.082; 0.1: about 0.101/0.102 m/s),
but turning remains sensitive to solver and initial conditions. Closed-loop
trajectories are not claimed bitwise identical between FP64 CPU and FP32 Warp.

Independent tests compare CPU/GPU torque, friction and duty at identical
sampled physics states. CUDA Graph/eager execution is checked exactly through
resets, DR changes and buffer reallocation. Revised randomized walking
assessment still rejects the existing expert's turns and keeps consolidation.
Recovery passes its first revision-2 consolidation assessment, including weak
forced pushes. Neither result establishes hardware readiness.

The initial continuation window is two existing 200-update assessment intervals
plus the first saved update: 401 additional updates. It provides two new
assessment opportunities, not a convergence promise or another 6000-update
assumption. Source checkpoints stay immutable. Before long runs, perform the
64-environment × 5-update smoke and verify finite checkpoint/ONNX, optimizer and
normalizer increments, inherited counters and assessment results.
