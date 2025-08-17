set -xe

export NSYS_NVTX_PROFILER_REGISTER_ONLY=0 

# nsys profile --trace=cuda,nvtx --force-overwrite=true --output=fa_run.nsys-rep -s cpu --cudabacktrace=all:5000 \
# python main.py

python main.py


# nsys profile -t cuda,nvtx,osrt -s cpu --cudabacktrace=all:5000 -o fa_run python main.py
# # Kernel summary (see actual FlashAttention kernel names & times)
# nsys stats fa_run.nsys-rep --report cuda_gpu_kern_sum --force-export=true

# # Export the report to JSON (or line-by-line JSON for simpler grepping)
# nsys export --type json --force-overwrite true --output fa_run.json fa_run.nsys-rep


# nsys stats fa_run.nsys-rep --report cuda_gpu_kern_sum --force-export=true --filter-nvtx 'sample_1(repeat=5)' 

# # --capture-stacks=cpu \