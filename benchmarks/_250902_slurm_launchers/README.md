All of these scripts should be launched in the `jd-d2` environment and under the `tests` directory.

```bash
conda activate jd-d2
cd tests
bash ../benchmarks/_250902_slurm_launchers/test_e2e_combined.sbatch.sh
bash ../benchmarks/_250902_slurm_launchers/test_e2e_combined.salloc.mem.sh
bash ../benchmarks/_250902_slurm_launchers/test_e2e_combined.salloc.mem.v2.sh
bash ../benchmarks/_250902_slurm_launchers/test_e2e_combined.salloc.launcher.sh
bash ../benchmarks/_250902_slurm_launchers/test_e2e_combined.group.sh
```