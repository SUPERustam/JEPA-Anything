#!/usr/bin/env bash
# Start this script in tmux on skoltech_container, never on the local machine.
set -euo pipefail
cd /home/container/jepa-anything
export MAMBA_ROOT_PREFIX="$PWD/tools/miniforge3"
eval "$(tools/miniforge3/bin/mamba shell hook --shell bash)"
mamba activate menv
export MPLCONFIGDIR="$PWD/runs/matplotlib"
task_run=runs/kepler-orbits/train-10000-controls
[[ ! -e "$task_run" ]]
mkdir -p runs/kepler-orbits
exec > "${task_run}.log" 2>&1
trap 'status=$?; date -u +Finished:%Y-%m-%dT%H:%M:%SZ; printf "%s\n" "$status" > "${task_run}.exit-code"' EXIT
date -u +Started:%Y-%m-%dT%H:%M:%SZ
python -c 'import json, sys, torch, numpy, matplotlib; print(json.dumps(dict(python=sys.version, executable=sys.executable, prefix=sys.prefix, torch=torch.__version__, numpy=numpy.__version__, matplotlib=matplotlib.__version__, device="cpu")))'
git rev-parse HEAD
sha256sum recipes/kepler-orbits/run_recipe.py jepa-anything-core/src/jepa_anything_core/*.py
printf '%s\n' "Command: python -u recipes/kepler-orbits/run_recipe.py train --steps 10000 --output $task_run"
python -u recipes/kepler-orbits/run_recipe.py train --steps 10000 --output "$task_run"
