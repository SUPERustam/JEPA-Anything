# Reviewable recipes

This directory contains reviewable structural audits and learned-model
experiments. Each recipe documents its source task, deterministic checks,
outputs, and evidence limits.

## Included recipes

[`kepler-orbits/`](kepler-orbits/) provides a runnable learned-model reconstruction
of the Kepler orbital-frequency scaling diagnostic, with a five-step CPU smoke
run, explicit-step training, checkpoint analysis, and physics/spectral tests.
Its README documents reconstruction choices and scientific limits.

### Structural dynamics fixture

[`synthetic-linear-dynamics/`](synthetic-linear-dynamics/) defines one seeded,
action-conditioned linear dynamical system. Its context and target are drawn
from the same trajectory, all target offsets are strictly in the future, and
the proposed OPF state uses `d=4`, `K=2`, and `r=2` so that `K*r=d`.

Run the dependency-free smoke audit from the repository root:

```bash
python3 recipes/synthetic-linear-dynamics/run_recipe.py \
  --output work/synthetic-linear-dynamics-report.json
```

Once `jepa-anything-skill` is present, validate the compiled design with its
deterministic validator:

```bash
python3 jepa-anything-skill/scripts/validate_design.py \
  recipes/synthetic-linear-dynamics/design.expected.json \
  --output work/synthetic-linear-dynamics-design-audit.json \
  --pretty
```

The smoke audit exits non-zero if a structural invariant fails. Passing means
only that this fixture is internally consistent and that every synthetic oracle
coordinate varies under its fixed seed. It does not establish learned-factor
activity, coordinate semantics, model quality, causal identification, or
superiority over a baseline.

## Recipe contract

Every recipe should make the following reviewable:

- schema version and immutable recipe identifier;
- dataset or generator provenance, seed, split policy, and system identity;
- context/target adapter input fields and source lineage for every descriptor
  and exogenous input, including prediction-time availability;
- state dimensions (`K*r=d`), concatenated coordinate layout, learned-projector
  analysis, and Moore-Penrose pseudoinverse state synthesis;
- a standard-deviation floor for every factor-target coordinate and every
  online-encoder coordinate;
- standard-JEPA and unconstrained-multihead baselines with matched full-model
  trainable-parameter and predictor-FLOP plans;
- intended usage mode: terminal readout, repeated transition, or factor
  analysis;
- structured claims whose usage mode, split, horizons, metrics, baselines, and
  audits are all covered by named experiments, plus explicit non-claims;
- output paths and content hashes where artifacts are published.

Generated reports belong in `work/` (or another ignored run directory), not in
this source tree. Model artifacts are described by metadata-only manifests in
[`../checkpoints/`](../checkpoints/).
