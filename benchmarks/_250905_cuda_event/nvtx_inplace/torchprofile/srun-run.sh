TS=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)
OUTDIR="./profiles/${TS}"
mkdir -p "$OUTDIR"

set -x
srun -N1 -G1 --ntasks-per-node=1 --jobid=$JOBID \
  bash -lc "OUTDIR='$OUTDIR' python torch_profiler_cuda.py" |& tee "${OUTDIR}/stdout.log"
set +x

echo "Trace dir: $OUTDIR"
echo "Open TensorBoard:"
echo "  tensorboard --logdir $OUTDIR"