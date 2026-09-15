# GitHub Pages video gallery

Public URL: <https://microduck-awesome.github.io/microduck_rl/>.

Pages contains recorded videos, playback controls and provenance. Viewing works
without Python, local checkpoints or a connection to the visitor's computer.

- `docs/index.html`, `gallery.css`, `gallery.js`: latest video gallery.
- `docs/videos/latest/`: walking / recovery V4 checkpoint 6400, dynamics revision
  2; 14 walking clips, five recovery clips and one complete 60-second exploration
  replay. The page reports incomplete turn acceptance and recovery stage status.
- `docs/legacy-5999.html`, `docs/videos/legacy-5999/`: archived V2/V3 checkpoint
  5999 videos recorded before the physics and observation-timing corrections.
- `docs/baseline.html` and the original `docs/videos/` files: archived checkpoint
  3450 / 3200 recordings, retaining failures and measurements.
- `docs/.nojekyll`: serve plain static files.

Default branch and Pages source: `sc0090-training-pages`, publication path `/docs`.
Push this branch, then verify the latest Pages build's commit and the live page.

To regenerate the latest content, from the repository root:

```bash
MUJOCO_GL=egl uv run --with imageio-ffmpeg --with pillow python simulator/scripts/render_gallery.py --models-dir logs/sc0090_setup/pages_dynamics2/models
uv run python simulator/scripts/build_gallery.py
```

The model folder above was exported with `simulator/scripts/export_models.py`,
using the walking and recovery V4 task IDs and each continuation run's immutable
`model_6400.pt`. Supply `--models-dir` for a different exported manifest; omitting
it uses the local demo's default policies, which may be older. The recorder checks
model/motor hashes and derives checkpoint labels from the supplied manifest.
`assessment.json` must describe those exact checkpoint hashes. For this release,
the final actors were verified identical to their isolated assessment snapshots.

The recorder uses a separate instance of the same CPU MuJoCo + SC0090 BAM
simulator and normalized ONNX policies. It writes H.264 MP4 at 25 fps / real
simulation time, posters and `gallery.json` with hashes, conditions, reset
timestamps and exploration actions. Review videos and data together before pushing.

The latest clips use fixed nominal parameters and one recording per case. The
historical clips used 32 randomized trials per case; these are different protocols.
Do not present the two sets as a matched quantitative evaluation.
Constant walking commands are measured over seconds 2–8. Start/stop switch at
2 seconds and report the response during seconds 3–4. The 0.03 m/s clip remains
a diagnostic; the provisional pure-walking training minimum is 0.08 m/s.

Local preview:

```bash
python3 -m http.server 8000 --directory docs
```
