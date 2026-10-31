# V2 walking 10250 training source

This branch preserves the training code, V2 CAD/mesh assets, SC0090 calibration, task registration, and original `uv.lock` used for `model_10250.pt`.

- Official baseline: `b02fdea4902c2b93b7d34a83a6f05da4ececb7d3`.
- Recorded training launch source: `7aec17f231e44ee3aef3fa20201795c9f359c0f6`.
- Publication source snapshot: `edf4a681` (includes checkpoint selection and comparison evidence).
- Task: `Mjlab-Velocity-Flat-MicroDuck-V2-Official-Baseline`; 4096 environments, physical GPU0.
- Training entry point: `scripts/run_official_walk.py`; runtime wrapper: `scripts/v2_python.sh`.
- Rebuild dependencies with `uv sync --frozen`. The wrapper uses `.venv/bin/python`; on driver 535/570 it also requires the external CUDA compatibility library documented in the wrapper (`SC0090_CUDA_COMPAT_DIR`). This library is not bundled.
- Saved configuration/checkpoint: `docs/evidence/official-walk/corrected/best-search/selected/` (`agent.yaml`, `env.yaml`, `model_10250.pt`, `policy.onnx`).
- Training plan and provenance: `docs/OFFICIAL_V2_WALK_TRAINING_PLAN.md`.

V2 mass is 624.6 g and HOME is all joint zeros. This publication does not restart training or evaluate policies; the run remains paused. Checkpoint selection is simulation evidence, not hardware qualification.

The historical supervisor opens its existing `formal.log` exclusively; use a fresh log directory before deliberately launching a new run.
