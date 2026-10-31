# V2 rise 6804 training source

This branch preserves the original A6804 collision-adaptation training source and assets.

- Training commit: `36e0780ca57504f309b46e8b815595a69b6bb108`.
- Delivery source / branch base: `4fb82ab3f043312d79e77c5b912a33964a01f8d9`. Changes since the training commit only add replay rendering helpers and documentation/evidence; training source is unchanged.
- All 104 files listed in the final rank0 run source manifest match this branch byte for byte (SHA256). Verification: `docs/evidence/rise6804-source-publication/source-verification.json`.
- Task: `Mjlab-StandUp-Flat-MicroDuck-V2-PairedError-Baseline`.
- Entry points: `scripts/train3762_ddp.py`, `scripts/distributed3762.py`, `scripts/train_v2_standup.py`, and `scripts/v2_python.sh`.
- Final checkpoint identity: `head-final/formal/rank0/resume_006804.pt`; 6804 cumulative updates, two GPUs, 4096 environments per GPU, one synchronized policy.
- Physical identity: 600.941825 g, SC0090 V5, corrected shell collisions, V2 all-zero HOME, 61D actor observation and 14D action.
- Historical training stages and transition evidence: `docs/RISE3762_COLLISION_RETRAIN.md`; result: `docs/RISE3762_COLLISION_RESULT.md`. The number 6804 is the cumulative endpoint of multiple phases, not a single default trainer budget.
- Original delivery manifest and both ranks' final run configurations/status are copied under `docs/evidence/rise6804-source-publication/`. The delivery manifest also describes files from the separate checkpoint bundle; this source branch does not copy that entire bundle.

The original `pyproject.toml` and `uv.lock` are retained. Recorded actual runtime versions are in each rank's `run.json` (including MuJoCo 3.13.0); use those records when assessing environment reproducibility, rather than assuming the lockfile alone proves runtime equivalence. The historical runtime wrapper also refers to local Python/CUDA compatibility paths.

This publication performs no training or policy evaluation. Simulation results do not constitute hardware qualification.

## Training reproduction guide

See [repro/README.md](../repro/README.md) for the full environment setup, phase/launch commands, reference metrics, protocol-matched comparison and export verification. New helper commands do not change the archived numerical training source.
