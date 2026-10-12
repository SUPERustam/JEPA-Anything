# Does this match the paper's Kepler experiment?

Checked on 2026-10-12 against the [linked alphaXiv paper](https://www.alphaxiv.org/pdf/2609.20800)
and [arXiv PDF](https://arxiv.org/pdf/2609.20800). The orbital diagnostic is
**Figure 9, §3.4.2**, not Figure 6. Figure 6 evaluates capacity-matched
continuous-control planning in Hopper, Walker2d, and HalfCheetah.

The recipe implements a JEPA-Anything reconstruction of the orbital frequency
diagnostic. It matches the described observation type, shared predictive
learning principle, factor-analysis interface, and third-law scaling test.
The paper does not provide enough orbital-specific detail to certify an exact
reproduction of its training or spectral estimator.

| Component | Paper | This experiment | Verdict |
| --- | --- | --- | --- |
| Data | Simulated position–velocity trajectories, no physical labels | Analytic two-body trajectories; `[x,y,vx,vy]` inputs; orbital parameters kept as metadata | Matches stated observation type |
| Context → target | Shared domain-specific predictive interface | Current observation → next observation at Δt = 0.05 | Consistent choice; orbital sampler unspecified |
| Target encoder | EMA, stopped encoder output | EMA 0.99; `no_grad()` target encoder; projection still differentiable | Matches shared rule |
| Predictive factors | Dedicated factor predictors; complete partition K·r=d | Four predictors reading 32 context coordinates; four eight-dimensional factors | Matches shared structure; orbital dimensions unspecified |
| Objective | Factor MSE, within/cross-projector Gram sum, factor activity, encoder variance | Existing core objective; weights 1, 0.1, 0.05, 0.02; activity floors 0.1 | Matches functional form; orbital weights unspecified |
| Diagnostic | Spectral modes versus semimajor axis | Strongest Hann-windowed nonzero FFT peak, power summed per factor and over factors | Matches diagnostic concept; estimator unspecified |
| Scaling law | f=(2π)⁻¹a⁻³ᐟ², slope −1.4991, R²=0.9999999 | Fit log10(f) against log10(a) on 32 fresh in-range trajectories | Same relation, different extraction and selection protocol |
| Training settings | Orbital architecture, budget, optimizer and dataset sizes unspecified; Figure 9 identifies selected seed 0 | MLP 4→128→128→32, Adam 0.001, 10,000 CPU steps, seed 42, 256×4096 observations | Reconstruction choices |
| Other Kepler laws | Figure tests frequency/axis scaling | Ellipse and conservation checks validate generator; no learned first-/second-law evaluation | Third-law diagnostic only |

Semimajor axis is used to generate the synthetic world and to evaluate its
frequency relation, but never enters the model inputs or optimization loss.
That avoids label supervision; it does not remove the physics already present
in the synthetic generator. Orthogonality of projection bases also does not
imply statistical independence or named physical semantics for factors.

## What a successful result supports

A slope near −1.5 demonstrates that the representation retains the orbital
periodicity of this generator. It does not establish that predictive training
discovered the law: raw observations and a matched zero-step encoder retain the same rhythm.
The corrected evaluation includes this control, full-set frozen-target prediction
with persistence baselines, post-extraction harmonic flags, and basis geometry. The five-step smoke comparison uses a different data
set and normalization and is not a controlled baseline. One trained seed gives
no between-seed uncertainty or superiority claim. The full run is also not a
test of decoded physical forecasting, autoregressive rollout, or all three laws.

See the local blog and its downloadable configuration, metrics, frequencies,
and analysis for actual completed-run evidence.

## Details inside Figure 9

[The figure image](https://arxiv.org/html/2609.20800v1/task/sol_v2.png) identifies
seed 0 selected by lowest run-level median frequency error. Its median error is
0.019%, the scale-compensated maximum deviation is 0.169%, and its symplectic
damping annotation is rho = 1.000000. The text/caption does not specify how
those modes or damping were estimated, nor the candidate run count. Our one
seed and strongest Hann-FFT peaks do not establish that same algorithm.
We report median and maximum relative errors for comparison without claiming
superiority or inventing a symplectic estimator. Physical-label-free training
also does not mean that comparison of extracted frequencies with a is label-free;
a remains generation and evaluation metadata.
