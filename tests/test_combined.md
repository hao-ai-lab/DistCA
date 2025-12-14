

1. conda activate `jd-d2` environment
2. cd `/mnt/weka/home/yonghao.zhuang/jd/d2/tests` directory
3. Run `bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks3/_251103_correctness/run_combined.sh`. If needed, modify that file. This will run the test_e2e_combined.py script in a srun job, once for D2 and another for wlbllm.
4. Then you should be able to see the log in the`logs.v1` directory. Find the latest logs - the log folder name should contain the mode and the number of nodes. 