# Defines the etnp_kernels static library: the repo's bundled reference kernel
# plus any consumer-injected sources, gathered into one archive that callers
# whole-archive into their final binary. Also computes ETNP_KERNEL_EXPECT_TUS,
# the list of static-init TU symbols the nm-guard must find post-link.
#
# Include AFTER find_package(ExecuTorch). Sets etnp_kernels + ETNP_KERNEL_EXPECT_TUS
# in the including scope. Paths resolve relative to THIS file (repo cmake/), so the
# top-level build and native_tests' standalone build agree (mirrors RuntimePin.cmake).
option(ETNP_BUILD_REFERENCE_KERNEL
  "Compile the bundled reference custom kernel (etnp::triple.out)" ON)
set(ETNP_EXTRA_KERNEL_SOURCES "" CACHE STRING
  "Semicolon-separated absolute paths to additional custom-kernel .cpp sources to \
compile and register into the module (see docs/custom-kernels.md)")

set(_etnp_kernel_sources "")
if(ETNP_BUILD_REFERENCE_KERNEL)
  list(APPEND _etnp_kernel_sources
    "${CMAKE_CURRENT_LIST_DIR}/../kernels/reference/etnp_reference_ops.cpp")
endif()
list(APPEND _etnp_kernel_sources ${ETNP_EXTRA_KERNEL_SOURCES})

set(ETNP_KERNEL_EXPECT_TUS "")
if(_etnp_kernel_sources)
  add_library(etnp_kernels STATIC ${_etnp_kernel_sources})
  set_property(TARGET etnp_kernels PROPERTY POSITION_INDEPENDENT_CODE ON)
  # PUBLIC so consumers of etnp_kernels inherit ExecuTorch's include dirs; the
  # core lib carries the kernel-registration API headers.
  target_link_libraries(etnp_kernels PUBLIC executorch)

  foreach(_src IN LISTS _etnp_kernel_sources)
    get_filename_component(_base "${_src}" NAME)
    list(APPEND ETNP_KERNEL_EXPECT_TUS "_GLOBAL__sub_I_${_base}")
  endforeach()
endif()
