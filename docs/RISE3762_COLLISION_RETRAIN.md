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
