# Local Kepler experiment blog

Activate `menv`, then run from the repository root:

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory recipes/kepler-orbits/blog
```

Open http://localhost:8765. Everything runs locally; the paper opens only when
its link is clicked. No installation, build, analytics, or hosting account is
needed. Stop the server with Ctrl-C.

The data directory contains copies of actual remote artifacts, plus `run.json`
for the interactive controls. Checkpoints remain on `skoltech_container`.
Verify the controls and evidence links with:

```bash
node recipes/kepler-orbits/blog/check_blog.cjs
```

The matched zero-step and full-set prediction evaluation lives in `data/controls/`.
Original training artifacts and their source snapshot (`data/training-source/`)
remain unchanged. `data/run.json` bundles both original metrics and the corrected
evaluation for the local interface. The checker validates training and evaluation
against their respective source/artifact hashes. No hosted services are used.

The corrected evaluation was rerun in remote `menv` after fetching code from the
fork, in tmux session `kepler-eval-final-20261012`. Its log and zero exit marker
are included in `data/controls/`. It reuses the preserved 10,000-step checkpoint
without a new full training run.
