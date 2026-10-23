# 3762 collision retraining, 2026-10-23

Materialized and hash-verified the complete rise_baseline_paired3762 snapshot. Preserve all historical task, reward, curriculum, actuator and randomization source files. robot.xml gains 32 cached CoACD parts for the identical CAD rear/left/right enclosure. Preserve every inertial element and original 0.600941825 kg mass. Cached collision parts/provenance come from microduck-v2-staged. No later 624.6 g model or learning recipe is substituted.

New train3762_ddp.py uses native RSL-RL gradient synchronization. Each rank has 4096 environments, global N=8192, T=24, E=5, M=4. Both ranks contribute to observation normalization and GAE statistics; independent sampling follows identical network initialization. Rank 0 evaluates the original two seeds and seven recovery buckets, then broadcasts assessment for shared curriculum decisions. Historical single-GPU diagnostics are replaced by native distributed PPO and finite-loss/status recording. Rank-local checkpoints preserve optimizer, normalizers, RNG and curriculum; physics resume remains the original native-new-episodes semantics, not exact physical rollout continuation.

Passed gravity contacts with recorded final support forces; 64x5 smoke with ONNX 61D/14D parity; 4096x12 batch check (9 measured updates); 2x4096x5 distributed check with bitwise identical actor/critic parameters and normalizer buffers. Qualification weights are discarded.

Formal run starts randomly, bounded at 3762 updates, evaluating every 500 updates and at the endpoint with 128 samples per bucket per seed. Recovery does not establish strict standing, push resistance, HOME, head precision or hardware qualification.

Output: /home/l40-vm01/data2/experiments/microduck-v2-20261023/rise3762-collision/formal
Service: microduck-rise3762-collision.service

Historical logs migrate separately to /home/l40-vm01/data2/archive/root-disk-20261023 with checksum verification before replacing each directory by a symlink.
