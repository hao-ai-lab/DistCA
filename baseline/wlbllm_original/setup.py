from setuptools import setup, find_packages

setup(
    name="wlbllm",
    version="0.1.0",
    description="WL-BLLM: Workload-Balanced Large Language Model",
    packages=find_packages(),
    python_requires=">=3.8",
    # install_requires=[
    #     "torch>=2.0.0",
    #     "flash-attn>=2.0.0",
    #     "transformers>=4.20.0",
    #     "accelerate",
    #     "datasets",
    #     "tokenizers",
    #     "numpy",
    #     "tqdm",
    # ],
    # extras_require={
    #     "dev": [
    #         "pytest",
    #         "black",
    #         "flake8",
    #         "isort",
    #     ],
    # },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="large language model, attention, context parallelism, distributed computing",
    author="Research Team",
    author_email="research@example.com",
    url="https://github.com/example/wlbllm",
    license="MIT",
)
