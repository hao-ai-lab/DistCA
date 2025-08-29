set -x

BATCH_SIZE=2 PP_SIZE=1 TP_SIZE=8 MODE=d2 NUM_TOKENS=65536 MAX_SAMPLE_ID=3 NUM_LAYERS=4 \
sbatch test_e2e_combined.slurm.sh

set +x