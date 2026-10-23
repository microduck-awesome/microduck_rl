# 3762 collision retraining, 2026-10-23

Materialized and hash-verified the complete rise_baseline_paired3762 snapshot. Preserve all historical task, reward, curriculum, actuator and randomization source files. robot.xml gains 32 cached CoACD parts for the identical CAD rear/left/right enclosure. Preserve every inertial element and original 0.600941825 kg mass. Cached collision parts/provenance come from microduck-v2-staged. No later 624.6 g model or learning recipe is substituted.

New train3762_ddp.py uses native RSL-RL gradient synchronization. Each rank has 4096 environments, global N=8192, T=24, E=5, M=4. Both ranks contribute to observation normalization and GAE statistics; independent sampling follows identical network initialization. Rank 0 evaluates the original two seeds and seven recovery buckets, then broadcasts assessment for shared curriculum decisions. Historical single-GPU diagnostics are replaced by native distributed PPO and finite-loss/status recording. Rank-local checkpoints preserve optimizer, normalizers, RNG and curriculum; physics resume remains the original native-new-episodes semantics, not exact physical rollout continuation.

Passed gravity contacts with recorded final support forces; 64x5 smoke with ONNX 61D/14D parity; 4096x12 batch check (9 measured updates); 2x4096x5 distributed check with bitwise identical actor/critic parameters and normalizer buffers. Qualification weights are discarded.

Formal run starts randomly, bounded at 3762 updates, evaluating every 500 updates and at the endpoint with 128 samples per bucket per seed. Recovery does not establish strict standing, push resistance, HOME, head precision or hardware qualification.

Output: /home/l40-vm01/data2/experiments/microduck-v2-20261023/rise3762-collision/formal
Service: microduck-rise3762-collision.service

Historical logs migrate separately to /home/l40-vm01/data2/archive/root-disk-20261023 with checksum verification before replacing each directory by a symlink.

## Update-500 failure and repair

At 21:30:45 the first evaluation completed and promoted the curriculum, but the next learn() failed on both ranks: loading obs_normalizer._std outside inference mode attempted to write an inference tensor. Initial qualification had only one learn segment and missed this boundary. Original status files were stale RUNNING; corrected to FAILED while retaining their original copies and update-500 resume checkpoints.

Fix: run persistent normalizer updates under inference_mode(False) and no_grad. Statistical formulas, batches and training recipe are unchanged. Add rank-specific resume-root selection and FAILED status on training-loop exceptions. Requalified from each rank's saved update-500 checkpoint at the actual 4096 environments per rank, through updates 502, 504 and 506 with evaluations between segments. Restoration of actor/critic/optimizer/normalizers is exact; final actor/critic/optimizer/normalizers are bitwise identical across ranks. Verification weights are discarded.

Resume formal training from original update 500 for 3262 remaining updates (endpoint 3762), with original 500-update full assessment cadence. Output formal-resumed; service microduck-rise3762-collision-resumed.service. Physics restarts native episodes as documented above.

## Completion verification plan

Require the live service to finish 3762 updates, both rank checkpoints to agree on actor/critic/normalizers/optimizer and curriculum, and a valid normalization-baked ONNX. Use evaluate3762_recovery.py on the final checkpoint and frozen historical baseline on this SAME corrected collision model at fixed final stage 5, 128 samples per bucket and each of the two original seeds. Report seven buckets separately; these reused development seeds are not held-out acceptance. Run the existing head-response battery separately on the new model, and preserve all shortfalls rather than treating recovery score as head or hardware qualification. If the bounded allocation has not learned full recovery, the goal remains incomplete and the evidence determines the next training action.

## Required correction: 3762 is a checkpoint identity, not a completion budget

Direct inspection of historical run.json reveals flags absent from the compact recipe recorded_arguments: paired-feedback-allocation-001 resumed head-foundation4 at 3506 and trained 256 updates using head precision stage 4, with evaluations every 128. Its parent route used 1000 updates of head precision stage 1 after recovery mastery, 250 of stage 2, then 256 of stage 4 before the paired-error transition. Exact manifest hashes and configurations are in historical-head-lineage.json.

The new source is collision-only relative to the frozen source, but its current default-head RUN CONFIG is not the full historical 3762 recipe. Earlier wording claiming full recipe preservation was too broad. The current 3762-update allocation and queued evaluation are a recovery foundation and diagnostic comparison only; do not mark the goal complete or label its ONNX a qualified replacement merely because its update counter matches 3762.

After recovery mastery and initial comparison, complete the historical head curriculum (stage 1, stage 2, stage 4 via the existing frozen configure_head_precision_learning function), preserving one policy lineage and corrected collisions. Qualify each transition with smoke/resume checks and evaluate recovery retention plus head response against historical 3762. The new lineage already uses paired-error randomization from initialization; this differs from the historical late transfer and must remain disclosed. No invented head reward or new physical randomization is authorized by this correction.

## Head phase 1 complete

Completed the historical 1000-update head stage 1 at cumulative update 4762. All periodic recovery evaluations and independent final recovery pass 1792/1792. Idle neck MAE drops from foundation 8.316 degrees to 0.563 (old3762 0.299). Common trained-range neck MAE remains 2.188 versus old3762 0.450; head precision work is still incomplete. Stage 2 continues for the historical 250 updates, after independent smoke, actual-batch and DDP boundary qualification. Preserve stage-1 weights and evidence. Historical log migration completed all four directories with checksum verification and original-path symlinks; root now has about 290 GB free.

## Head phase 2 complete

Historical 250-update stage 2 ended at cumulative 5012. Independent recovery is 1792/1792. Idle neck MAE 0.302 degrees, matching old3762 0.299; trained-range neck MAE 1.958 still exceeds old3762 0.450. Continue the existing historical stage-4 recipe for 512 updates (the original combined 256 foundation plus 256 paired-error exposure), retaining the paired-error task already used throughout this new lineage. No new reward formula is introduced; qualification weights are discarded and original-stage behavior is documented.

## Stage 4 at 5524: recovery retained, positive neck response incomplete

Stage 4's 512 updates and independent recovery pass 1792/1792. The artifact parity check passes with maximum ONNX error 5.96e-7. Idle errors improve; however the +0.39 rad neck case still has about 14 degree MAE versus old3762 0.6-0.8 degrees. Aggregate trained-range neck error declines 2.188 -> 1.958 -> 1.376 degrees across head stages but hides this specific gap. This is not a full replacement acceptance.

Keep 5524 frozen and continue exactly the same qualified stage-4 recipe for a bounded 512 updates to 6036. No new reward, range, action filtering or physics is introduced. Reuse qualification only after every parent training-source hash matches; re-evaluate all recovery buckets and the same head cases at the endpoint. Further action depends on actual improvement, not just update count.

## 6036 diagnostic and original-stage coarse practice

512 unchanged stage-4 updates retain recovery 1792/1792 and improve trained-range neck MAE 1.376 -> 1.056 degrees, but the +0.39 rad single-neck case remains 12.35/12.47 degrees off in two seeds. Saved 5524 traces show a fast response followed by persistent undershoot (22.35 degree target, 8.28 degree actual), while old3762 reaches 21.64 degrees in the same corrected geometry. This is not just response delay. Adam LR at 6036 is 1.71e-4 versus old3762 7.59e-5; neck sampling std is 0.0489 versus 0.0630, not collapsed. These do not establish a unique cause.

Freeze 6036. Revisit the original head stage 2 for a bounded 512 updates (6036->6548) before further micro precision: it restores the existing wider-command exposure and unmultipled coarse head tracking term. No new reward formula, optimizer parameter, collision or actuator change. Reuse stage-2 config qualification only with identical parent source hashes, and separately validate the 4->2 transition at actual DDP batch across three evaluation boundaries. Qualification weights are discarded. Full recovery and per-case head evaluation are required at the endpoint.
