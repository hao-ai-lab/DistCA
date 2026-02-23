# FindNVSHMEM.cmake - Find NVSHMEM when using pip package (nvidia-nvshmem-cu12)
# which provides include/ and lib/ but no NVSHMEMConfig.cmake.
# Set NVSHMEM_PREFIX to the package root (e.g. .venv/.../nvidia/nvshmem).

if(DEFINED ENV{NVSHMEM_PREFIX})
  set(NVSHMEM_PREFIX "$ENV{NVSHMEM_PREFIX}")
endif()

if(NOT NVSHMEM_PREFIX OR NOT EXISTS "${NVSHMEM_PREFIX}/include/nvshmem.h")
  set(NVSHMEM_FOUND FALSE)
  return()
endif()

set(NVSHMEM_INCLUDE_DIR "${NVSHMEM_PREFIX}/include")
set(NVSHMEM_LIB_DIR "${NVSHMEM_PREFIX}/lib")

# Pip package: libnvshmem_host.so.3 (no .so symlink), libnvshmem_device.a
set(NVSHMEM_HOST_LIBRARY "${NVSHMEM_LIB_DIR}/libnvshmem_host.so.3")
set(NVSHMEM_DEVICE_LIBRARY "${NVSHMEM_LIB_DIR}/libnvshmem_device.a")

if(NOT EXISTS "${NVSHMEM_HOST_LIBRARY}" OR NOT EXISTS "${NVSHMEM_DEVICE_LIBRARY}")
  set(NVSHMEM_FOUND FALSE)
  return()
endif()

# nvshmem::nvshmem_host (shared)
add_library(nvshmem::nvshmem_host SHARED IMPORTED GLOBAL)
set_target_properties(nvshmem::nvshmem_host PROPERTIES
  IMPORTED_LOCATION "${NVSHMEM_HOST_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${NVSHMEM_INCLUDE_DIR}"
)

# nvshmem::nvshmem_device (static, device code)
add_library(nvshmem::nvshmem_device STATIC IMPORTED GLOBAL)
set_target_properties(nvshmem::nvshmem_device PROPERTIES
  IMPORTED_LOCATION "${NVSHMEM_DEVICE_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${NVSHMEM_INCLUDE_DIR}"
)

set(NVSHMEM_FOUND TRUE)
