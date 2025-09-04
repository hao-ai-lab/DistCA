# README

## `srun_multi_combined.mem.sh`

### How to use the memory script to log memory usage

1. Open `srun_multi_combined.mem.sh` and modify the parameters.
2. Execute the command from `tests/` directory.
```bash
cd tests
bash ../benchmarks/_250903_large_scale_v4/srun_multi_combined.mem.sh
```

3. Open `benchmarks/_250903_large_scale_v4/check_mem.py` - and change the variable folder to the new log folder you generated.
4. Run the script and get the plots.
