#!/usr/bin/env bash
set -euo pipefail

echo "Inside bind_and_exec.sh"

# Resolve rank ids
LR=${LOCAL_RANK:-${SLURM_LOCALID:-0}}
RID=${RANK:-$LR}
HOST=${HOSTNAME:-$(hostname)}

# ---- Per-rank logfile ----
LOGDIR=${LOG_DIR:-./logs}
mkdir -p "$LOGDIR"
LOGFILE="${LOGDIR}/${HOST}.r${RID}.lr${LR}.log"
ENVFOLDER="${LOGDIR}/envs"
ENVFILE="${ENVFOLDER}/${HOST}.r${RID}.lr${LR}.env"
mkdir -p "$ENVFOLDER"
echo "Logging $RID to $LOGFILE"

# ---- Logging + exec ----
# exec > >(tee -a "$LOGFILE") 2>&1
if [ "$RID" -eq 0 ]; then
    # Rank 0: log to file *and* console
    exec > >(tee -a "$LOGFILE") 2>&1
else
    # Other ranks: log to file only
    exec >"$LOGFILE" 2>&1
fi


# # ---- CPU bind (edit slices as you like) ----
# CORE_MAP=( "0-15" "16-31" "32-47" "48-63" "64-79" "80-95" "96-111" "112-127" )
# # CORE_MAP=( "0-15" "16-31" "32-47" "48-63" "64-67" "68-71" "72-75" "76-79" )
# CORES=${CORE_MAP[$LR]}

# # export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
# # If you want, set the CUDA device explicitly:
# # python will usually set this, but being explicit doesn't hurt:
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}  # optional


# ----
env > "$ENVFILE" # to check the nvshmem environment variables
nvidia-smi topo -p2p w >> "$ENVFILE"     # Check nvidia-smi topology

set -x
# exec taskset -c "$CORES" "$@"
exec "$@"
set +x