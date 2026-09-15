# GitHub Pages video gallery

The site entry point is `docs/index.html`; its relative asset links resolve to
`docs/videos/`. The gallery contains the unchanged 2026-09-15 baseline videos:
17 individual scenarios and three comparison videos, with thumbnails and
checkpoint/evaluation metadata. Failures remain visible. This is not a live
dashboard of the ongoing V2 training.

Repository: `microduck-awesome/microduck_rl`.
Default branch and Pages source branch: `sc0090-training-pages`.
Pages publication directory: `/docs`, using deployment from a branch.
`docs/.nojekyll` keeps the gallery as plain static files.

Expected public URL: <https://microduck-awesome.github.io/microduck_rl/>.

To update the gallery, replace the corresponding MP4/JPG/JSON files and
`docs/index.html` together, retain the checkpoint IDs and failure results,
then push this branch. GitHub Pages publishes the updated `/docs` directory.
There is no dependency on the training machine, Python environment or model
checkpoint files at page-view time.

Local preview from the repository root:

```bash
python3 -m http.server 8000 --directory docs
```

Open `http://localhost:8000/`. The gallery supports category filters, playback
speed selection, pause-all and direct MP4 downloads. Video files use H.264 at
25 fps, with playback duration matching the simulation duration.
