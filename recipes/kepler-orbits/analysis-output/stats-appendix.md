# Statistical scope

- Unit for learning claims: independent trained seed. Available n=1 (seed 42).
- Diagnostic rows: 32 fresh in-range orbits generated with seed 43. They are
  evaluations of one fitted model, not 32 independent training runs.
- Training/validation batches: 10,000 sequential random mini-batches of 256
  pairs, sampled with replacement. Batch rows are not independent replicates
  of a training experiment. Validation uses the same 51 held-out trajectories.
- Fit: ordinary least squares of log10 frequency versus log10 semimajor axis;
  reported R² is in log space. Relative error is |f_extracted/f_theory−1|.
- No inferential test, p-value, seed standard deviation, or between-seed 95% CI
  is justified by these artifacts. A matched zero-step control and persistence scores now exist in
  `../blog/data/controls/`; no matched standard-JEPA training baseline exists. Effect sizes for superiority and multiple-comparison
  corrections are therefore not applicable. Only descriptive deviations from
  the theoretical law and the reported paper fit are given.
- No uncertainty interval for individual FFT estimates is available.
  Frequency-bin spacing is 1/(16384×0.05)=0.001220703125; log-power parabolic
  interpolation improves point estimation but is not an uncertainty estimate.
- Aggregate maximum relative frequency error is 0.0981287%. Factor 0 maximum
  relative error is approximately 100.0126%, consistent with a second harmonic.
- A high R² summarizes scaling across the chosen axis range. It does not
  measure next-state prediction accuracy, physical rollout quality, factor
  independence, causal interpretability, or a benefit caused by training.
- No confidence intervals or error bars are drawn. Training-curve smoothing
  uses nonoverlapping 50-step means for readability, not uncertainty estimation.
