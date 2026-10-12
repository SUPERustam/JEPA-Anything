# Kepler orbital scaling reconstruction

This runnable learned-model experiment reconstructs the orbital diagnostic in
[§3.4.2, Figure 9](https://arxiv.org/html/2609.20800v1#S3.SS4.SSS2).
The paper reports a latent-frequency slope of −1.4991. Its orbital architecture,
optimizer, and hosted orbital assets were not available for this reconstruction;
all settings below are our choices. This targets Kepler's third-law frequency
scaling, not all three laws or an exact reproduction of the authors' run.

**Figure correction:** Kepler is Figure 9 in the linked paper; Figure 6 is
continuous-control planning. See [the paper comparison](paper-match.md) for
what matches and what the paper leaves unspecified.

From the repository root:

```bash
eval "$(mamba shell hook --shell zsh)"
mamba activate menv
export MPLCONFIGDIR=/tmp/kepler-matplotlib
python recipes/kepler-orbits/run_recipe.py smoke
pytest recipes/kepler-orbits -q
# Local blog (full training runs only on skoltech_container):
python -m http.server 8765 --bind 127.0.0.1 --directory recipes/kepler-orbits/blog
```

The tests execute five optimizer steps total. No command defaults to sustained
training. `smoke` always
uses five CPU Adam steps, 16 orbits, and batch size 32. A missing subcommand prints
usage without training. Existing training directories are refused to preserve
run evidence. Analysis can regenerate its CSV, JSON, and plots from a checkpoint.
Dependencies are NumPy, PyTorch, Matplotlib, and pytest, already present in `menv`.

Full training belongs in `/home/container/jepa-anything` on `skoltech_container`,
inside tmux. The remote preparation script installs Miniforge beneath that
workspace, verifies its published SHA-256, creates Python 3.12 `menv`, and reuses
compatible installed container packages through a `.pth` file. It does not
change the container's system Python or other experiments.

```bash
# On skoltech_container, sync code through the GitHub fork:
cd /home/container/jepa-anything
git pull --ff-only origin main
bash recipes/kepler-orbits/prepare_remote_menv.sh
tmux new-session -d -s kepler-train-controls 'bash recipes/kepler-orbits/run_remote.sh'
tmux set-option -t kepler-train-controls remain-on-exit on
```

`run_remote.sh` activates `menv` and trains 10,000 steps on CPU, with an exit
marker and environment/source provenance in its log. It refuses an existing
`runs/kepler-orbits/train-10000-controls` directory. For another run, choose a fresh run
directory and tmux session name first. The log includes the Git commit, command,
recipe/core source hashes, environment, start/end times, and exit code. Check the exit marker, metric rows, and
checkpoint before reporting completion.

## Data and learning choices

Seed 42 generates 256 elliptic planar trajectories with μ=1, a uniform in
[0.5,4], e uniform in [0.05,0.3], and random orientation and mean-anomaly phase.
Each has 4,096 uniformly spaced samples at dt=0.05. Newton iteration solves
Kepler's equation analytically for position and velocity; no numerical time
integration is used. Complete trajectory IDs are split 80/20 before transitions
are sampled with replacement. Normalization uses training observations only.
Inputs are normalized [x,y,vx,vy]; targets are the next observation. Semimajor
axis, eccentricity, orientation, and phase remain evaluation metadata and never
enter the model, optimizer, transition sampler, or objective.

The online encoder is 4→128→128→32 with GELU hidden activations. The target
encoder starts as a copy, has no gradients, and updates by EMA with momentum
0.99. Four independent 32→8→8 GELU predictor branches read the shared complete
context state and produce eight coordinates each. The existing learnable
32-dimensional OPF projection starts from identity and uses soft Gram
orthogonality. Target encoding is stopped; target projection remains
differentiable so the projector receives prediction and activity gradients.
The existing composite objective uses prediction plus orthogonality 0.1,
factor activity 0.05, and online encoder variance 0.02, with both standard
deviation floors 0.1. Adam uses lr=0.001. Core APIs are unchanged.

## Analysis and evidence boundary

Evaluation uses seed 43 for 32 fresh held-out trajectories with logarithmically
spaced axes and 16,384 samples each (also in smoke). For each learned factor,
coordinates are demeaned, Hann-windowed, and their FFT powers summed. The
strongest nonzero bin is refined by a three-bin log-power parabola. The same
rule is applied to summed power across all factors to give one aggregate peak
per orbit. No factor, coordinate, harmonic, or frequency band is selected using
the theoretical law. Coordinates are not variance-normalized before power
aggregation. The law is consulted only after peak extraction, for error metrics
and plots. Harmonics can win; flat spectra or collapsed representations make
this diagnostic unreliable. Frequency resolution is 1/(N dt); this is not an
uncertainty estimate. Fits use log10 frequency versus log10 axis, with a log10
intercept and R² in log space.

The plot separates learned-factor results from a raw position–velocity physics
check. Both panels show the μ=1 theoretical ridge f=(2π)⁻¹a⁻³ᐟ²; only the learned
panel shows the aggregate fitted line. `analysis.json` contains aggregate and
per-factor slopes, intercepts, R², and absolute/relative frequency errors.
`frequencies.csv` keeps every extracted factor peak and raw-state peak alongside
orbital metadata. Ignored `runs/kepler-orbits/` contains configuration,
training-only normalization and split IDs, metadata, all per-step training and
validation loss metrics, model/optimizer checkpoint, and PNG/PDF plots. New checkpoints also contain
`initial_model`, the exact zero-step weights. Checkpoints support analysis reload,
not training resume. Re-analysis refuses existing evaluation artifacts; preserve
prior reports by using a fresh output directory:

```bash
python recipes/kepler-orbits/run_recipe.py analyze \
  --checkpoint runs/kepler-orbits/train-10000-menv/checkpoint.pt \
  --output runs/kepler-orbits/controls-evaluation
```

The corrected analysis always evaluates a zero-step model using identical
normalization, trajectories, and FFT extraction. Older checkpoints reconstruct
initialization from the seed and disclose that limitation and both PyTorch
versions in `evaluation-provenance.json`; new checkpoints use saved weights.
`frequencies.csv` includes trained and zero-step aggregate and per-factor peaks.
`analysis.json` flags near-integer harmonics only after label-free extraction;
peaks are never selected or corrected using the law. `spectral_diagnostics.csv`
records temporal RMS and three-bin spectral power concentration. These measure
activity and concentration, not frequency confidence intervals.

Full prediction evaluation scores every adjacent transition on all validation
orbits against the final frozen EMA target, with both EMA-target and online
persistence controls. Errors are in latent factor units, not physical coordinates.
The existing core geometry audit reports rank, conditioning, and approximate
orthogonality separately. Aggregate power over a strictly orthogonal complete
factor partition equals encoder power, so it cannot establish a factorization
benefit. The new plot includes the matched zero-step control.

Tests cover circular trajectories, conserved energy/angular momentum,
deterministic generation, split isolation, training-only normalization, finite
gradients, frozen target gradients, EMA, projector updates, exact checkpoint
reload parity, and spectral recovery on sinusoids and circular observations
(slope tolerance 0.02 around −1.5). Five-step smoke metrics only establish that
the implementation runs. Even a good smoke slope does not establish learned
scientific discovery: an untrained encoder can preserve periodicity. The remote
10,000-step result is explained in the [interactive blog](blog/README.md),
together with downloadable artifacts. The preserved checkpoint's matched zero-step slope is essentially identical
to its trained slope. Full-set predictor MSE is about 0.02399 versus 0.06179 for
EMA-target persistence (208,845 held-out transitions), demonstrating useful
one-step prediction while leaving the scaling diagnostic insensitive to learning.
Multiple trained seeds, standard-JEPA comparisons, and hyperparameter searches
remain absent. A close scaling slope does
not establish scientific discovery or an exact reproduction of the paper's run.

## GitHub synchronization

The code source is `https://github.com/SUPERustam/JEPA-Anything.git`, branch
`main`. Commit and push local changes, then run `git pull --ff-only origin main`
in `/home/container/jepa-anything`; compare `git rev-parse HEAD` on both hosts.
Do not copy code with tar/scp or force-reset remote changes. Remote `runs/`,
`tools/`, and pre-Git workspace backups remain outside the tracked source tree.
Use scp only to retrieve generated analysis artifacts, never to synchronize code.
