# Remove transformer engine from pip constraint, becuase we're gonna upgrade it
awk '/^transformer_engine/ {print "#" $0; next} 1' /etc/pip/constraint.txt > temp && mv temp /etc/pip/constraint.txt

pip uninstall transformer_engine
cd TransformerEngine
NVTE_FRAMEWORK=pytorch MAX_JOBS=64 NVTE_BUILD_THREADS_PER_JOB=64 pip install --no-build-isolation -v -v -v '.[pytorch]'

cd Megatron-LM
pip install -e .
cd ..

pip install ray
pip install omegaconf
pip install tensordict
pip install transformers