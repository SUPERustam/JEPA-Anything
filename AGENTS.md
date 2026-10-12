# Working instructions

@/Users/superustam/.codex/RTK.md

- Full training runs **only** on SSH host `skoltech_container`, under
  `/home/container/jepa-anything`, inside **tmux**.
  Local execution is limited to tests, five-step smoke checks, analysis:
- For local run: use `mamba activate menv`

## Kepler JEPA experiment

- Primary paper: https://www.alphaxiv.org/pdf/2609.20800. In paper, Kepler scaling is
  **Figure 9, §3.4.2**;
- Reproduce the Kepler **third-law frequency diagnostic**, training a JEPA
  online encoder, stopped EMA target encoder, learned orthogonal predictive
  factors, and factor predictors. Reuse `jepa-anything-core` and
  `recipes/kepler-orbits`; preserve physical labels as evaluation metadata.
- The paper leaves orbital architecture, optimizer, dataset sizes, and exact
  spectral extraction unspecified. Document our choices; describe the result
  as a reconstruction, never an exact reproduction or recovery of all three laws.
  serving the blog. Never run sustained training locally.
- Inspect existing remote files and tmux sessions first. Preserve unrelated
  processes and prior results; use fresh run directories and session names.
  Record source hashes, command, environment, seed, start/end times, exit code,
  configuration, split, metrics, checkpoint, and analysis artifacts.
- Use explicit step counts; 10,000 steps is our reconstruction budget, not a
  paper-specified budget. Verify completed metrics and checkpoint steps before
  reporting completion. A live tmux session alone does not prove completion.
- Explain data generation, training, quantitative results, and what those
  results support in the interactive blog at `recipes/kepler-orbits/blog/`.
  Serve on loopback only; keep assets and interactions local, with no external
  scripts, analytics, uploads, or deployment.
