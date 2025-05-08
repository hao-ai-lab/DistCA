# WLB-LLM (Context Parallelism)

This repository contains an open-source implementation of the `Per-Sequence CP Sharding` baseline and the `Per-Document CP Sharding` methods proposed in the paper **[OSDI'25] "WLB-LLM: Workload-Balanced 4D Parallelism for Large Language Model Training."**


## Repository Structure

```python
WLB-LLM-Context-Parallelism/
├── README.md
├── cp_performance_compare.py        # Code for performance evaluation
├── per_doc_correctness_test.py      # Correctness evaluation for Per-Doc CP
├── per_doc_cp_attn.py               # Per-Doc CP implementation
├── per_seq_correctness_test.py      # Correctness evaluation for Per-Seq CP
├── per_seq_cp_attn.py               # Per-Seq CP implementation
├── run_correctness_check.sh         # Script for correctness evaluation
├── run_cp_performance_compare.sh    # Script for performance evaluation
└── utils.py
```


## Get Started

We recommend using the Docker images available on the [NVIDIA GPU Cloud (NGC) Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch). For example, to run the NGC PyTorch container interactively:

```shell
docker run --gpus all -it --rm --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v <Your Dir>:/workspace/WLB-LLM-CP \
  nvcr.io/nvidia/pytorch:25.01-py3
```


### Correctness Evaluation:

```shell
./run_correctness_check.sh
```



### Performance Evaluation:

```shell
./run_cp_performance_compare.sh
```



## Citation

If you find this work useful, please cite:

> **WLB-LLM: Workload-Balanced 4D Parallelism for Large Language Model Training** 

```bibtex
@inproceedings{wang2025wlb,
  title={WLB-LLM: Workload-Balanced 4D Parallelism for Large Language Model Training},
  author={Wang, Zheng and Cai, Anna and Xie, Xinfeng and Pan, Zaifeng and Guan, Yue and Chu, Weiwei and Wang, Jie and Li, Shikai and Huang, Jianyu and Cai, Chris and others},
  booktitle={19th USENIX Symposium on Operating Systems Design and Implementation (OSDI 25)},
  year={2025}
}
