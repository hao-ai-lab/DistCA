exec > logs/${TS}/rank${RANK}.log 2>&1 
echo LOCAL_RANK=${LOCAL_RANK} RANK=${RANK}
# Example: bind LOCAL_RANK to 16 cores per GPU

CORES_PER_RANK=4
START_CORE=$((LOCAL_RANK * CORES_PER_RANK))
END_CORE=$((START_CORE + CORES_PER_RANK - 1))
echo "Binding rank ${RANK} to cores ${START_CORE}-${END_CORE}"

taskset -c ${START_CORE}-${END_CORE} 
echo "Allocated cores: $(taskset -cp $$)"
python torchsample.py