#!/usr/bin/env bash
# Sets up tabfm_bench.py's GPU dependencies on an NVIDIA GPU VM -- source
# this, don't `bash` it, so the venv activation carries into the benchmark
# script's own command.
# Usage (on the VM; in a subshell, since this turns on set -euo pipefail):
#   bash -c 'source ~/tabfm_gpu_setup.sh && python3 tabfm_bench.py --backend gpu --tag <name>'
#
# Expects an image with the NVIDIA driver preinstalled (Deep Learning VM
# family common-cu129-ubuntu-2204-nvidia-580; jax[cuda13] needs driver >= 580).
# tabfm[jax] alone pulls in CPU-only jax, so it is resolved together with
# jax[cuda13] in a single install. CUDA 13 matches torch 2.13.0's default
# Linux wheel used by tabpfn_bench.py (it resolves to CUDA 13.0 libs), so both
# models run on the same CUDA major.
set -euo pipefail

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
[[ -f "$HOME/.local/bin/env" ]] && source "$HOME/.local/bin/env"

uv venv --allow-existing --python 3.11 "$HOME/tabfm-env"
source "$HOME/tabfm-env/bin/activate"
# The image family name says CUDA 12.9 (cu129). If LD_LIBRARY_PATH points at
# it, jax may load it instead of the pip-installed CUDA 13 libs (unverified on
# this image; unsetting is harmless either way).
unset LD_LIBRARY_PATH

# tabfm/jax pinned to the versions the published GPU and TPU runs used;
# sklearn/pandas/numpy to what the 2026-09-27 GPU runs resolved, so reruns match.
uv pip install "tabfm[jax]==1.0.1" "jax[cuda13]==0.10.2" \
    "scikit-learn==1.9.1" "pandas==3.0.6" "numpy==2.4.6"
# Fail here, not after the benchmark starts, if jax cannot see the GPU.
python3 -c "import jax; assert jax.default_backend() == 'gpu', jax.devices()"
