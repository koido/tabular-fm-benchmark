#!/usr/bin/env bash
# Runs both benchmarks on one NVIDIA GPU VM with the device sampler alongside.
# Run on the VM from $HOME, after copying there: this script, tabfm_bench.py,
# tabpfn_bench.py, bench_datasets.py, tabfm_gpu_setup.sh, sample_device.sh,
# and optionally ~/scikit_learn_data (OpenML cache). To reuse a local TabPFN-3 checkpoint
# instead of letting tabpfn download it, export TABPFN_MODEL_CACHE_DIR (an
# absolute path; this script cd's to $HOME first) before running it.
# Usage (on the VM; nohup so an SSH drop doesn't kill the run):
#   nohup bash gpu_bench_run.sh <tag> > run.log 2>&1 &     e.g. gpu-l4
# Any arguments after <tag> are passed through unchanged to both
# tabfm_bench.py and tabpfn_bench.py, so only options both accept work
# (--seeds, --suite; not --dtype or --tag). Write them as `--suite X` and
# `--seeds a b c` (not `--suite=X`): the final result-count check reads that form. E.g.:
#   nohup bash gpu_bench_run.sh gpu-l4-tas --seeds 42 43 44 --suite tabarena-small > run.log 2>&1 &
# Results: ~/logs/bench_results/<tag>/, logs: ~/logs/<tag>_*.log
set -euo pipefail

TAG="${1:?usage: $0 <tag> [extra args passed through to both benchmarks]}"
shift
EXTRA=("$@")
# Step output goes to per-step logs, so say where to look when a step fails.
ON_ERR='echo "FAILED (line $LINENO); see logs/${TAG}_*.log" >&2'
trap "$ON_ERR" ERR
cd "$HOME"
mkdir -p logs
# Result files have fixed names, so leftovers from an earlier run with the
# same tag would pass the final count check.
if [ -n "$(ls -A "logs/bench_results/$TAG" 2>/dev/null)" ]; then
  echo "logs/bench_results/$TAG is not empty; use a new tag or move it away" >&2
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee "logs/${TAG}_nvidia-smi.log"

bash sample_device.sh gpu > "logs/${TAG}_sample_gpu.log" 2>&1 &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null || true' EXIT

# TabFM: its own venv; the setup script asserts jax sees the GPU. A TabFM
# failure must not skip TabPFN (on T4, TabFM cannot run at all: its bf16
# matmul is unsupported there), so capture the status and go on; the result
# count at the end still fails the run. Not `|| rc=$?`: inside an || list
# bash ignores set -e even when the sourced setup re-enables it, so a failed
# setup would run the benchmark anyway. set +e keeps the subshell's own
# set -e effective; the ERR trap is paused so it doesn't report a failure the
# script deliberately continues past.
trap - ERR
set +e
(
  source tabfm_gpu_setup.sh
  python3 tabfm_bench.py --backend gpu --tag "$TAG" "${EXTRA[@]+"${EXTRA[@]}"}"
) > "logs/${TAG}_tabfm.log" 2>&1
tabfm_rc=$?
set -e
trap "$ON_ERR" ERR
[ "$tabfm_rc" -eq 0 ] || echo "TabFM step exited $tabfm_rc; see logs/${TAG}_tabfm.log" >&2

# TabPFN-3: PEP 723 deps via uv (installed by the TabFM setup above if it was
# missing). If TABPFN_MODEL_CACHE_DIR is already set in the environment, it
# is inherited by `uv run` below and reused instead of re-downloading the
# checkpoint; otherwise tabpfn downloads it itself (needs a license
# acceptance or TABPFN_TOKEN -- see tabpfn_bench.py's header).
export PATH="$HOME/.local/bin:$PATH"
tabpfn_rc=0
env -u LD_LIBRARY_PATH \
  uv run tabpfn_bench.py --device cuda --tag "$TAG" "${EXTRA[@]+"${EXTRA[@]}"}" \
  > "logs/${TAG}_tabpfn.log" 2>&1 || tabpfn_rc=$?
[ "$tabpfn_rc" -eq 0 ] || echo "TabPFN step exited $tabpfn_rc; see logs/${TAG}_tabpfn.log" >&2

# The lock file only appears if tabpfn fell into its download path; only
# checked when TABPFN_MODEL_CACHE_DIR points it at a known directory.
if [ -n "${TABPFN_MODEL_CACHE_DIR:-}" ] && ls "$TABPFN_MODEL_CACHE_DIR"/.*.lock >/dev/null 2>&1; then
  echo "WARNING: tabpfn tried to download the checkpoint" | tee -a "logs/${TAG}_tabpfn.log"
fi
# Expected result-file count: models (tabfm + tabpfn, this script's own two
# steps above) x datasets in the --suite passed in EXTRA (default: 5) x
# seeds passed in EXTRA (default: 1, seed 42). Suite sizes below must track
# bench_datasets.SUITES; duplicated here (not queried at runtime) so this
# check works even if the python env the two steps above used is gone by
# the time this line runs.
SUITE="default"
SEEDS_COUNT=1
i=0
while [ "$i" -lt "${#EXTRA[@]}" ]; do
  case "${EXTRA[$i]}" in
    --suite)
      i=$((i + 1))
      SUITE="${EXTRA[$i]}"
      ;;
    --seeds)
      SEEDS_COUNT=0
      i=$((i + 1))
      while [ "$i" -lt "${#EXTRA[@]}" ] && [[ "${EXTRA[$i]}" != --* ]]; do
        SEEDS_COUNT=$((SEEDS_COUNT + 1))
        i=$((i + 1))
      done
      continue
      ;;
  esac
  i=$((i + 1))
done
case "$SUITE" in
  default) N_DATASETS=5 ;;
  tabarena-small) N_DATASETS=16 ;;
  dates-rerun) N_DATASETS=1 ;;
  *) echo "unknown --suite '$SUITE' for the expected-count check" >&2; exit 1 ;;
esac
EXPECTED=$((N_DATASETS * SEEDS_COUNT * 2))

# Per-dataset failures are only printed by the benchmarks, so count results.
n=$(ls logs/bench_results/"$TAG"/*.json 2>/dev/null | wc -l || true)
if [ "$n" -ne "$EXPECTED" ]; then
  echo "expected $EXPECTED result files ($N_DATASETS datasets x $SEEDS_COUNT seeds x 2 models), got $n" >&2
  exit 1
fi
echo "done: $TAG"
