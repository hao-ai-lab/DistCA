<!-- <p align="center"> -->
  <!-- <img src="assets/distca.logo.png" alt="DistCA" width="180" align="center"> -->
  <!-- <img src="assets/distca.png" alt="DistCA" width="350" align="center"> -->
<!-- </p> -->

<div align="center"><h1>&nbsp;DistCA: Efficient Long-context Language Model Training by Core Attention Disaggregation</h1></div>

<!-- =========================
     Badges + Links
     ========================= -->
<p align="center">
  <a href="https://arxiv.org/abs/2510.18121">
    <img src="https://flat.badgen.net/badge/Paper/arXiv/red" alt="Paper">
  </a>
  <a href="https://hao-ai-lab.github.io/blogs/distca/">
    <img src="https://flat.badgen.net/badge/Blog/DistCA/green" alt="Blog">
  </a>
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://flat.badgen.net/badge/License/Apache--2.0/blue" alt="License">
  </a>
</p>


##

<!-- **DistCA** is an LLM training system that efficiently handles long-context training. DistCA disaggregates the **C**ore **A**ttention (the $\text{softmax}(QK^T)V$ operation) from the rest of the model, fundamentally 

DistCA build on top of [Megatron-LM](https://github.com/NVIDIA/Megatron-LM). -->


**DistCA** is a distributed LLM training system designed for efficient **long-context** training. DistCA introduces **Core Attention Disaggregation (CAD)**, a system-level technique that separates the quadratic **core attention** computation (i.e. $\text{softmax}(QK^T)V$, or the FlashAttention kernel) from the remaining linear components of the model. 



### What does DistCA do?

DistCA addresses a fundamental limitation in long-context LLM training: severe workload imbalance caused by the uneven quadratic cost of core attention across micro-batches. Existing systems and parallelization strategies (DP, PP, CP) colocate core attention with linear layers. As context length and system scale increase, this colocation leads to stragglers, pipeline bubbles, and excessive communication or memory overhead.

DistCA treats core attention (CA, the $\text{softmax}(QK^T)V$ operation) as an independent unit of work and dynamically redistributes CA tasks across GPUs, while keeping the rest of the model execution unchanged. This design enables:

- Balanced core attention execution across DP and PP ranks  
- Elimination of stragglers and pipeline bubbles  
- Significantly lower communication overhead than context parallelism  
- Near-linear scalability to very long context lengths


<p align="center">
  <picture>
    <img src="assets/distca.gif" width="75%" alt="How DistCA works" />
  </picture>
  <br/>
  <i>How DistCA works</i>   
</p>


## Installation

See the [installation guide](./README.Installation.md) for detailed instructions.


## Usage

We provide a preliminary slurm script for training a 8B LLaMA model with 128K context length on 2 nodes:


```bash
sbatch pretrain_llama.sh
```

or using `salloc`: 
```bash
salloc -N 2 -G 16 -t 01:00:00 --job-name=distca 
bash pretrain_llama.sh
# NNODES=2 TP_SIZE=8 PP_SIZE=2 bash pretrain_llama.sh
```

For more details, please refer to the [pretrain_llama.sh](./pretrain_llama.sh) and [pretrain_llama.py](./pretrain_llama.py) scripts.

## Performance tunning

We provide a preliminary scripts for benchmarking and debugging the performance of DistCA. Try running the following script to benchmark 4D DistCA paralleism:
```bash
bash ./benchmarks/example-4d-parallel/run4d.sh
```
or 3D parallelism:
```bash
bash ./benchmarks/example-3d-parallel/run3d.sh
```

The logs and performance results will be saved in the `./benchmarks/example-4d-parallel/logs` and `./benchmarks/example-3d-parallel/logs` directories.

### Environment Variables

We provide a set of environment variables for tuning the performance of DistCA. You can set them in bash scripts to control.

| Environment Variable | Default Value | Description |
|---|---|---|
| `ENABLE_NSYS` | 0 | Whether to enable nsys profiling. |
| `EXPERIMENT_LOG_MEMORY_USAGE` | 0 | Whether to log memory usage. |
| `EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB` | 2 | The size of the NVSHMEM buffer in GB.  |






## Citation
If you find DistCA useful in your research, please consider citing us:
```bibtex
@article{zhuang2025efficient,
  title={Efficient Long-context Language Model Training by Core Attention Disaggregation},
  author={Zhuang, Yonghao and Chen, Junda and Pang, Bo and Gu, Yi and Zhu, Yibo and Jiang, Yimin and Stoica, Ion and Xing, Eric and Zhang, Hao},
  journal={arXiv preprint arXiv:2510.18121},
  year={2025}
}
```
