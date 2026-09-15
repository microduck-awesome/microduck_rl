# Upstream provenance

- Source: https://huggingface.co/spaces/pollen-robotics/microduck-simulator
- Imported revision: `023172c8a7d629b5258d90364c13bafe013abbfa`
- Retrieved: 2026-09-15
- Original README: `UPSTREAM_README.md`
- Original LFS rules: `upstream.gitattributes`

The official web simulator is published as a Hugging Face Space Git repository,
separate from the official `pollen-robotics/microduck_rl` training repository.
The source and its LFS assets were downloaded before integration. The imported
revision does not contain a top-level LICENSE; existing notices are retained,
and this file does not assign a new license to upstream material.

This integration keeps the original web source under `app/`. The SC0090 entry
is `app/src/sc0090/`; `main.jsx` selects it by default and preserves the original
entry under `?official=1`. The original build-only Dockerfile is archived as
`Dockerfile.upstream`; the supported SC0090 launch path is the Python service
documented in `README.md`.

The small binary assets are materialized as ordinary Git files in this repository,
so cloning the integrated project does not require downloading upstream LFS pointers.
The upstream Git metadata is not nested inside this directory.

For an update, clone the upstream Space into a separate temporary directory with
Git LFS enabled, choose and record its exact revision, then review the changes in
the original `app/` files against this imported revision. Preserve the SC0090
entry and run the frontend build, keyboard checks and CPU simulation tests before
updating this provenance file. Do not replace working files used by a running
training/evaluation process during the comparison.
