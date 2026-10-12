# Figure evidence

## Main: orbital scaling

Files: [PDF](../blog/data/scaling.pdf), [PNG](../blog/data/scaling.png).
Source: remote checkpoint analysis, 32 fresh evaluation trajectories.
Purpose: compare factor/aggregate spectral peaks with the theoretical ridge,
while keeping the raw-state physics check separate.
Variables: semimajor axis and cycles per time unit, both on logarithmic axes.
Caption must state one trained seed, FFT selection without theoretical guidance,
32 orbits, and reconstruction settings. No error bars or uncertainty intervals.
Notice: the aggregate follows −1.5 closely, while factor 0 sometimes selects
a harmonic. Raw observations already exhibit the same law. This supports
preservation of periodicity. The corrected plot in `../blog/data/controls/`
includes a matched zero-step comparison, which retains the same scaling.
Check: do not infer physical factor labels or all-law discovery from the ridge.

## Supporting: training dynamics

Files: `figures/training.pdf`, `figures/training.png`.
Source: the 10,000 rows of [metrics.csv](../blog/data/metrics.csv).
Purpose: show how optimization behaved and expose changes beyond first/last
mini-batches. Variables: optimizer step and objective/prediction MSE. Points
are means of 50 adjacent steps, with no uncertainty bars.
Caption must state that validation is sampled from held-out whole orbits and
targets evolve by EMA. Notice: the run is not a monotonic convergence trace;
individual endpoint losses are incomplete summaries. This prevents claiming
convergence or generalization from two endpoint values.
Check: distinguish objective regularization from prediction error and sampled
batch validation from full-dataset validation.
