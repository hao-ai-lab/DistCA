# PST/PDT timestamped output + tee logs
TS=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)
OUTDIR="./profiles/${TS}"
mkdir -p "$OUTDIR"

# Run and tee logs to the same folder
OUTDIR="$OUTDIR" python torch_profiler_cuda.py |& tee "${OUTDIR}/stdout.log"

echo "Trace dir: $OUTDIR"
echo "Open TensorBoard:"
echo "  tensorboard --logdir $OUTDIR"