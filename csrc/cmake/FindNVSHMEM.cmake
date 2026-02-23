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

# Find NVSHMEM libraries dynamically instead of hardcoding soversions.
# Pip package ships libnvshmem_host.so.3 (no .so symlink) and libnvshmem_device.a.
find_library(NVSHMEM_HOST_LIBRARY
  NAMES nvshmem_host
  PATHS "${NVSHMEM_LIB_DIR}"
  NO_DEFAULT_PATH
)
# Fallback: glob for any libnvshmem_host.so* if find_library fails (no .so symlink)
if(NOT NVSHMEM_HOST_LIBRARY)
  file(GLOB _nvshmem_host_candidates "${NVSHMEM_LIB_DIR}/libnvshmem_host.so*")
  if(_nvshmem_host_candidates)
    list(GET _nvshmem_host_candidates 0 NVSHMEM_HOST_LIBRARY)
  endif()
endif()

find_library(NVSHMEM_DEVICE_LIBRARY
  NAMES nvshmem_device
  PATHS "${NVSHMEM_LIB_DIR}"
  NO_DEFAULT_PATH
)
# Fallback: look for static archive directly
if(NOT NVSHMEM_DEVICE_LIBRARY)
  if(EXISTS "${NVSHMEM_LIB_DIR}/libnvshmem_device.a")
    set(NVSHMEM_DEVICE_LIBRARY "${NVSHMEM_LIB_DIR}/libnvshmem_device.a")
  endif()
endif()

if(NOT NVSHMEM_HOST_LIBRARY OR NOT NVSHMEM_DEVICE_LIBRARY)
  message(STATUS "FindNVSHMEM: could not locate host or device library in ${NVSHMEM_LIB_DIR}")
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
