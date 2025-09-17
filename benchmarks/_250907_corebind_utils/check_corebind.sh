#!/usr/bin/env bash

# srun -N 32 --ntasks-per-node=8 --cpus-per-task=16 --cpu-bind=cores --gpu-bind=closest bash check_corebind.sh
set -euo pipefail

# Usage (1 node): srun --ntasks-per-node=8 --cpus-per-task=16 --cpu-bind=cores bash check_bind.sh
# Usage (multi):  srun -N 32 --ntasks-per-node=8 --cpus-per-task=16 --cpu-bind=cores bash check_bind.sh

host=${SLURMD_NODENAME:-$(hostname)}
rank=${SLURM_PROCID:-0}
local=${SLURM_LOCALID:-0}
ntasks=${SLURM_NTASKS:-?}
cpu_list="$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)"
thr_per_task=${SLURM_CPUS_PER_TASK:-?}

echo "-----"
echo "Host=${host}  RANK=${rank}  LOCAL_RANK=${local}  NTASKS=${ntasks}"
echo "  cpus-per-task=${thr_per_task}"
echo "  Cpus_allowed_list=${cpu_list}"
# Fallback via taskset
if command -v taskset >/dev/null 2>&1; then
  echo "  taskset_affinity=$(taskset -pc $$ | awk -F': ' '{print $2}')"
fi
# Show NUMA placement if available
if command -v numactl >/dev/null 2>&1; then
  numactl -s | sed 's/^/  /'
fi
# Quick GPU mapping hint
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "  GPU(s) on node:"
  nvidia-smi -L | sed 's/^/    /'
fi