# Code under this folder is modified from https://github.com/ppl-ai/pplx-kernels and subject to the MIT License.
# run with : cmake -B build -S ./ -G Ninja -DCMAKE_PREFIX_PATH='/mnt/weka/home/yonghao.zhuang/miniconda3/envs/attn/lib/python3.12/site-packages/torch/share/cmake' -DTORCH_CUDA_ARCH_LIST='Hopper'
cmake_minimum_required(VERSION 3.22)
project(AttnServerComm
    VERSION 0.1
    DESCRIPTION "WLB shuffle memcpy Library"
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


execute_process(
    COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(EXISTS "${TORCH_CMAKE_PREFIX_PATH}")
    set(CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
else()
    set(CMAKE_PREFIX_PATH "/mnt/weka/home/yonghao.zhuang/miniconda3/envs/attn/lib/python3.12/site-packages/torch/share/cmake")
endif()

# Dependencies
include(FetchContent)
find_package(CUDAToolkit REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Torch REQUIRED)

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

add_library(memcpy_kernels STATIC
    memcpy.cu
)
target_link_libraries(memcpy_kernels PUBLIC
    CUDA::cudart
)
set_cuda_compile_options(memcpy_kernels)

add_library(wlb_shuffle SHARED
    bindings.cpp
)
target_link_libraries(wlb_shuffle PUBLIC
    memcpy_kernels
    torch::py_limited
    Python::Module
    CUDA::cuda_driver
    CUDA::cudart
)
set_target_properties(wlb_shuffle PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../wlbllm/fastmemcpy
    CUDA_SEPARABLE_COMPILATION ON
)
