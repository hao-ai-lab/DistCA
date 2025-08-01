# Code under this folder is modified from https://github.com/ppl-ai/pplx-kernels and subject to the MIT License.
# run with : cmake -B build -S ./ -G Ninja -DCMAKE_PREFIX_PATH='/mnt/weka/home/yonghao.zhuang/miniconda3/envs/attn/lib/python3.12/site-packages/torch/share/cmake' -DTORCH_CUDA_ARCH_LIST='Hopper'
cmake_minimum_required(VERSION 3.22)
project(AttnServerComm
    VERSION 0.1
    DESCRIPTION "AttnServer Communication Library"
    LANGUAGES CXX CUDA
)

# Configuration
set(CMAKE_CUDA_ARCHITECTURES 90a CACHE STRING "CUDA architecture to target")

# Cmake Config
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_PREFIX_PATH /mnt/weka/home/yonghao.zhuang/miniconda3/envs/attn/lib/python3.12/site-packages/torch/share/cmake)

# Dependencies
include(FetchContent)
find_package(CUDAToolkit REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Torch REQUIRED)
find_package(NVSHMEM REQUIRED HINTS /mnt/sharefs/software/DeepEP/nvshmem/lib)

# Created imported target for PyTorch
add_library(torch_imported INTERFACE)
add_library(torch::py_limited ALIAS torch_imported)
target_include_directories(torch_imported SYSTEM INTERFACE ${TORCH_INCLUDE_DIRS})
target_link_libraries(torch_imported INTERFACE c10 torch torch_cpu c10_cuda torch_cuda CUDA::cudart)

# Compiler flags
add_compile_options(-Wno-deprecated-declarations)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
add_compile_definitions(Py_LIMITED_API=0x03090000)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# CUDA compile options function
function(set_cuda_compile_options target)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--threads=32 -O3>)
endfunction()

# Add CUDA kernel library
add_library(dispatch_kernels STATIC
    core/in_place_attn_switch.cu
)
target_link_libraries(dispatch_kernels PUBLIC
    CUDA::cudart
)
target_link_libraries(dispatch_kernels PUBLIC
    nvshmem::nvshmem_device
    nvshmem::nvshmem_host
)
set_cuda_compile_options(dispatch_kernels)

add_library(fast_a2a_kernels STATIC
    core/fastalltoall.cu
    core/memcpy.cu
)
target_link_libraries(fast_a2a_kernels PUBLIC
    CUDA::cudart
)
target_link_libraries(fast_a2a_kernels PUBLIC
    nvshmem::nvshmem_device
    nvshmem::nvshmem_host
)
set_cuda_compile_options(fast_a2a_kernels)

add_library(as_comm SHARED
    bindings/bindings.cpp
    bindings/nvshmem.cpp
    bindings/all_to_all.cpp
)
target_link_libraries(as_comm PUBLIC
    dispatch_kernels
    fast_a2a_kernels
    torch::py_limited
    Python::Module
    CUDA::cuda_driver
    CUDA::cudart
    nvshmem::nvshmem_host
    nvshmem::nvshmem_device
)
set_target_properties(as_comm PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../d2/runtime/attn_kernels
    CUDA_SEPARABLE_COMPILATION ON
)
