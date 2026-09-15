# GitHub Pages video gallery

Public URL: <https://microduck-awesome.github.io/microduck_rl/>.

Pages contains recorded videos, playback controls and provenance. Viewing works
without Python, local checkpoints or a connection to the visitor's computer.

- `docs/index.html`, `gallery.css`, `gallery.js`: latest video gallery.
- `docs/videos/latest/`: walking V2 / recovery V3 checkpoint 5999; 11 walking
  clips, five recovery clips and one complete 60-second exploration replay.
- `docs/baseline.html` and the original `docs/videos/` files: archived checkpoint
  3450 / 3200 recordings, retaining failures and measurements.
- `docs/.nojekyll`: serve plain static files.

Default branch and Pages source: `sc0090-training-pages`, publication path `/docs`.
Push this branch, then verify the latest Pages build's commit and the live page.

To regenerate the latest content, from the repository root:

```bash
MUJOCO_GL=egl uv run --with imageio-ffmpeg --with pillow python simulator/scripts/render_gallery.py
uv run python simulator/scripts/build_gallery.py
```

The recorder uses a separate instance of the same CPU MuJoCo + SC0090 BAM
simulator and normalized ONNX policies. It writes H.264 MP4 at 25 fps / real
simulation time, posters and `gallery.json` with hashes, conditions, reset
timestamps and exploration actions. Review videos and data together before pushing.

The latest clips use fixed nominal parameters and one recording per case. The
historical clips used 32 randomized trials per case; these are different protocols.
Do not present the two sets as a matched quantitative evaluation.

Local preview:

```bash
python3 -m http.server 8000 --directory docs
```
