# Kepler JEPA reconstruction: completed-run analysis

Question: do the trained factor coordinates retain the third-law orbital
frequency scaling on fresh trajectories? The comparison is descriptive:
one seed (42), 10,000 CPU Adam steps, 32 evaluation orbits from seed 43.
The complete run took 46 seconds on `skoltech_container`, under mamba `menv`
in tmux session `kepler-train-menv`. Exit code was 0. All checkpoint tensors
and 10,000 consecutive metric rows were finite; reloading the checkpoint and
regenerating analysis reproduced its JSON exactly. Source and downloaded
artifact hashes match the [remote manifest](../blog/data/provenance.json).

| Readout | Log–log slope | R² | Mean relative frequency error |
| --- | ---: | ---: | ---: |
| Paper Figure 9 | −1.4991 | 0.9999999 | Not stated |
| Trained aggregate | −1.499881966 | 0.999999925668 | 0.0179393% |
| Trained factor 0 | −1.528294400 | 0.956741990507 | 9.39237% |
| Trained factor 1 | −1.499881939 | 0.999999925650 | 0.0179402% |
| Trained factor 2 | −1.499881950 | 0.999999925654 | 0.0179400% |
| Trained factor 3 | −1.499882009 | 0.999999925700 | 0.0179376% |
| Raw position–velocity | −1.499881942 | 0.999999925649 | 0.0179401% |

The aggregate slope differs from the theoretical −1.5 by +0.000118034 and
from the paper's fitted slope by −0.000781966. This is numerical agreement
of the diagnostic, not a comparison under matched protocols. Factor 0 selects
the second harmonic on three of 32 orbits. The aggregate conceals this behavior;
the per-factor view in the blog exposes it.

Training objective changes from 0.038647663 to 0.017587859; validation objective
changes from 0.036285747 to 0.028419845. Final prediction terms are 0.016132066
and 0.026951674 respectively. The final Gram penalty is 0.014557928 and both
activity penalties are zero. These are individual random mini-batches against
a moving EMA target, not full-dataset performance estimates or convergence proof.

## Claim candidates

- Claim: the trained aggregate preserves orbital periodicity.
  - Evidence: 32 evaluation orbits, aggregate fit and frequency errors in the
    [analysis](../blog/data/analysis.json) and [peaks](../blog/data/frequencies.csv).
  - Allowed wording: the representation retains the generator's Keplerian rhythm.
  - Forbidden stronger wording: JEPA discovered all Kepler laws or outperforms
    raw observations or standard JEPA.
  - Uncertainty: one training seed, in-range trajectories, no matched baseline.
  - Next check: evaluate a saved untrained initialization with identical
    normalization and trajectories, then matched baselines and additional seeds.
  - Decision: keep at descriptive strength.
- Claim: every factor cleanly recovers the fundamental frequency.
  - Evidence: factor 0 has three near-2× peaks and R²≈0.9567.
  - Allowed wording: three factors follow the fundamental; one sometimes prefers
    a harmonic under this estimator.
  - Forbidden stronger wording: four disentangled physical laws or factors.
  - Uncertainty: no factor semantics, independence, or intervention evaluation.
  - Next check: inspect spectra and controlled prediction interventions.
  - Decision: discard the original stronger claim.

The previous system-Python run produced identical metrics and fits with the
same seed and package versions. It is a repeat of the same experiment, not a
second independent training seed. The five-step smoke uses only 16 generated
orbits and different normalization; it cannot isolate the effect of training.

For the source comparison, see [paper-match.md](../paper-match.md). The
[paper](https://arxiv.org/html/2609.20800v1#S3.SS4.SSS2) puts Kepler in Figure 9,
not Figure 6, and omits the orbital-specific training settings needed for an
exact protocol reproduction.

## Corrected evaluation

The original artifacts above remain preserved. The corrected outputs are in
`../blog/data/controls/`: identical-normalization zero-step controls, all 208,845
validation transitions, harmonic flags, spectral concentration, and basis geometry.
Zero-step and trained aggregate slopes agree to about 2e-8, so this score does
not demonstrate a training benefit. Predictor MSE 0.02399 is 61.18% lower than
EMA-target persistence MSE 0.06179. This supports useful one-step latent prediction.
The basis is full rank and well conditioned but fails the strict orthogonality
audit, as expected for a soft penalty; exact orthogonal guarantees are unsupported.
The original checkpoint lacks saved initial weights; its control is reconstructed.
Future training checkpoints include their exact initial weights. Evaluation
provenance records the actual source, environment, and checkpoint/artifact hashes.
