#!/usr/bin/env bash

# srun --jobid=703588 -N 1 -G 0 --ntasks-per-node=8 --cpus-per-task=16 --cpu-bind=cores bash ctx_switch_probe.sh "python -c 'import time; time.sleep(2)'"

set -euo pipefail

read_cs () {
  awk '
    /voluntary_ctxt_switches/ {v=$2}
    /nonvoluntary_ctxt_switches/ {nv=$2}
    END {printf "%s %s\n", v+0, nv+0}
  ' /proc/self/status
}

work="${1:-sleep 1}"

read v0 nv0 < <(read_cs)
# Run user's workload inline in this shell so we keep same PID (/proc/self)
bash -lc "$work"
read v1 nv1 < <(read_cs)

echo "voluntary:+$((v1 - v0))  nonvoluntary:+$((nv1 - nv0))"