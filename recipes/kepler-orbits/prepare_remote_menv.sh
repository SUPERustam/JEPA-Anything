#!/usr/bin/env bash
# Run only on skoltech_container; reuse its existing Python 3.12 packages.
set -euo pipefail
cd /home/container/jepa-anything
task_prefix="$PWD/tools/miniforge3"
export MAMBA_ROOT_PREFIX="$task_prefix"
if [[ ! -x "$task_prefix/bin/mamba" ]]; then
    mkdir -p tools
    installer=Miniforge3-Linux-x86_64.sh
    base=https://github.com/conda-forge/miniforge/releases/latest/download
    wget -q "$base/$installer" -O "tools/$installer"
    wget -q https://api.github.com/repos/conda-forge/miniforge/releases/latest -O tools/release.json
    /usr/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
release = json.loads(Path('tools/release.json').read_text())
name = f"Miniforge3-{release['tag_name']}-Linux-x86_64.sh"
asset = next(a for a in release['assets'] if a['name'] == name)
actual = 'sha256:' + hashlib.sha256(Path('tools/Miniforge3-Linux-x86_64.sh').read_bytes()).hexdigest()
if actual != asset['digest']:
    raise RuntimeError('Miniforge installer checksum mismatch')
print('Verified installer:', name, actual)
PY
    bash "tools/$installer" -b -p "$task_prefix"
fi
eval "$("$task_prefix/bin/mamba" shell hook --shell bash)"
if [[ ! -x "$task_prefix/envs/menv/bin/python" ]]; then
    mamba create -y -n menv python=3.12
fi
mamba activate menv
# Reuse compatible container packages instead of downloading a second Torch.
/usr/bin/python -c 'import site; print("\n".join(site.getsitepackages()))' > \
    "$(python -c 'import site; print(site.getsitepackages()[0])')/container-packages.pth"
python -c 'import sys, torch, numpy, matplotlib; assert sys.prefix.endswith("/envs/menv"); print(sys.executable); print(torch.__version__, numpy.__version__, matplotlib.__version__)'
