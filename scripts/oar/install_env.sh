#!/bin/bash
# Create the SelectArena virtual environment on calculco (front end; needs internet).
#
#   bash scripts/oar/install_env.sh [VENV_DIR]     # default: ~/venvs/selectarena
#
# The venv lives in $HOME so every compute node sees it at the same path; set `python_path` in
# configs/cluster.yaml to <VENV_DIR>/bin/python. Uses the system Python 3.11 and installs with uv
# (AutoGluon comes from pre-release builds, hence --prerelease=allow). Takes a few minutes and
# about 8 GB (torch + CUDA libraries for the GPU models).
set -euo pipefail

VENV=${1:-$HOME/venvs/selectarena}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# Keep uv's download cache out of the $HOME quota.
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv-cache-$USER}

if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install --quiet --upgrade pip uv
"$VENV/bin/uv" pip install --python "$VENV/bin/python" --prerelease=allow -e "$REPO[models,dev]"

mkdir -p "$REPO/logs/oar"
"$VENV/bin/python" -c "import selectarena, tabflow_slurm; print('selectarena installed in', '$VENV')"
