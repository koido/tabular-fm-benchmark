#!/usr/bin/env bash
# Samples accelerator utilization and memory once per second until killed, so a
# benchmark's per-dataset started_at/finished_at can be lined up against it.
# Memory here is what the framework holds (jax preallocates 75% of
# GPU memory; torch caches), so use the benchmark JSON's device_memory for
# what the model needed, and this log for utilization.
#
# Usage (on the VM, before starting the benchmark):
#   bash sample_device.sh gpu > ~/sample_gpu.log 2>&1 &
#   ... run the benchmark ...
#   kill %1
# "tpu" needs tpu-info on PATH (`uv pip install tpu-info` in the benchmark venv).
# Timestamps are UTC, like the benchmark JSON's started_at/finished_at. Note
# nvidia-smi's utilization.memory is memory *bandwidth* use, not occupancy.
set -euo pipefail

case "${1:-}" in
  gpu)
    exec env TZ=UTC nvidia-smi \
      --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total \
      --format=csv -l 1
    ;;
  tpu)
    while true; do
      date -u +%Y-%m-%dT%H:%M:%S.%3NZ
      # tolerate tpu-info failing while no workload holds the TPU yet
      tpu-info --metric hbm_usage duty_cycle_percent || true
      sleep 1
    done
    ;;
  *)
    echo "usage: $0 gpu|tpu" >&2
    exit 2
    ;;
esac
