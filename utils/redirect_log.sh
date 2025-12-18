#!/usr/bin/env bash
set -euo pipefail

# Resolve rank ids
LR=${LOCAL_RANK:-${SLURM_LOCALID:-0}}
RID=${RANK:-$LR}
HOST=${HOSTNAME:-$(hostname)}


# ---- Per-rank logfile ----
LOGDIR=${LOG_DIR:-./logs}
mkdir -p "$LOGDIR"
LOGFILE="${LOGDIR}/${HOST}.r${RID}.lr${LR}.log"
echo "Logging $RID to $LOGFILE"

# ---- Logging + exec ----
exec > >(tee -a "$LOGFILE") 2>&1

exec "$@"