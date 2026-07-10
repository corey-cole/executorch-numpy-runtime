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
  "Semicolon-separated absolute paths to additional custom-kernel .cpp/.cc \
sources (registrar-bearing kernels and their aux sources) to \
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
    # Only sources that register a kernel have a static-init registrar TU the
    # nm-guard must find. Aux sources (e.g. SIMD helpers like the LSTM
    # example's lstm_cell.cc) have none — expecting one would false-fail the
    # guard. Detection is configure-time content sniffing for either
    # registration idiom (direct register_kernel() or the EXECUTORCH_LIBRARY()
    # macro); re-run cmake if a source gains/loses its registration.
    file(READ "${_src}" _src_text)
    string(FIND "${_src_text}" "register_kernel(" _reg_pos)
    string(FIND "${_src_text}" "EXECUTORCH_LIBRARY(" _macro_pos)
    get_filename_component(_base "${_src}" NAME)
    if(NOT _reg_pos EQUAL -1 OR NOT _macro_pos EQUAL -1)
      list(APPEND ETNP_KERNEL_EXPECT_TUS "_GLOBAL__sub_I_${_base}")
    endif()
    # Each source's own directory is an include dir: lets sources include
    # sibling headers by bare name, and lets Highway's foreach_target
    # re-include a source via its bare filename (see lstm_cell.cc).
    get_filename_component(_dir "${_src}" DIRECTORY)
    target_include_directories(etnp_kernels PRIVATE "${_dir}")
  endforeach()
  message(STATUS "etnp_kernels: expecting registrar TUs: [${ETNP_KERNEL_EXPECT_TUS}]")
endif()
