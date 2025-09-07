#!/bin/bash

# Get the directory where this script is located
CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${CURDIR}"

# Generate timestamp in PST
TS=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)
OUTDIR="./profiles/${TS}"

# Create the output directory
mkdir -p "${OUTDIR}"

# Update the "latest" symlink
# -f to force overwrite if exists, -n for native paths
ln -sfn "${TS}" "./profiles/latest"

# Run the profiling command
nsys profile \
  -t cuda,nvtx \
  --capture-range=nvtx \
  --capture-range-end=stop \
  -o "${OUTDIR}/trace" \
  -- python sample_gpu_ops.py \
  > "${OUTDIR}/stdout.log" 2>&1

# Print helpful information
echo "Profile saved to: ${OUTDIR}"
echo "Access via latest symlink: ./profiles/latest"
