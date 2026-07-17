cmake_minimum_required(VERSION 3.24)
project(executorch_numpy_runtime LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(cmake/RuntimePin.cmake)
list(APPEND CMAKE_PREFIX_PATH "${ETNP_RUNTIME_PREFIX}")
find_package(ExecuTorch CONFIG REQUIRED)
if(EXISTS "${ETNP_RUNTIME_PREFIX}/lib/cmake/ETNPExtras/ETNPExtras.cmake")
  include("${ETNP_RUNTIME_PREFIX}/lib/cmake/ETNPExtras/ETNPExtras.cmake")
endif()
include(cmake/Kernels.cmake)

# nanobind via scikit-build-core-provided Python
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                OUTPUT_VARIABLE nanobind_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(nanobind CONFIG REQUIRED)

# NOSTRIP: the POST_BUILD nm guard below must be able to see local symbols (incl. the
# XNNPACK static-init TU); nanobind's default Release build strips the symbol table
# entirely, which would make the guard blind rather than meaningful.
nanobind_add_module(_core STABLE_ABI NB_STATIC NOSTRIP
  src/binding/module.cpp
  src/binding/dtype_map.cpp
  src/et_core/et_core.cpp)

target_include_directories(_core PRIVATE src)

# find_package(ExecuTorch CONFIG REQUIRED) can succeed while defining NO targets: its config
# early-returns on the first find_library miss, setting EXECUTORCH_FOUND (all-caps) -- which is
# NOT the ExecuTorch_FOUND that REQUIRED checks. Without this floor, a config that bailed would
# silently produce an empty link line and advertise kernel_libs=portable for the wrong reason.
foreach(_req executorch xnnpack_backend)
  if(NOT TARGET ${_req})
    message(FATAL_ERROR
      "find_package(ExecuTorch) did not define the '${_req}' target. Its config early-returns on "
      "the first find_library miss WITHOUT failing find_package (it sets EXECUTORCH_FOUND, not "
      "ExecuTorch_FOUND). Prefix: ${ETNP_RUNTIME_PREFIX} -- check it matches this platform and "
      "that its lib/ contains the expected library naming for this toolchain.")
  endif()
endforeach()

# Link what the prefix actually provides: the Windows tarball is core-only (no
# optimized_native_cpu_ops_lib, no quantized_ops_lib). The runtime config self-whole-archives
# the kernel/backend archives via INTERFACE_LINK_OPTIONS, so linking the TARGETS is enough --
# never raw .lib paths, which lose /WHOLEARCHIVE: and silently break backend registration.
target_link_libraries(_core PRIVATE
  executorch xnnpack_backend
  extension_module_static extension_data_loader extension_tensor)

# _etnp_kernel_libs is the SINGLE SOURCE OF TRUTH for the advertised kernel set. Every APPEND
# below must come before the JOIN at the bottom of this block.
# portable is unconditional: portable_ops_lib ships in every tarball on every platform, and
# arrives transitively via `executorch` -- do NOT link it explicitly.
set(_etnp_kernel_libs "portable")

if(TARGET optimized_native_cpu_ops_lib)
  target_link_libraries(_core PRIVATE optimized_native_cpu_ops_lib)
  list(APPEND _etnp_kernel_libs "optimized")
endif()

if(TARGET quantized_ops_lib)
  target_link_libraries(_core PRIVATE quantized_ops_lib)
  list(APPEND _etnp_kernel_libs "quantized")
endif()

# PRESERVED from the current file (do not drop): the custom-kernel seam's archive. On Windows
# this target does not exist -- Task 3 defaults ETNP_BUILD_REFERENCE_KERNEL OFF there, so
# etnp_kernels has no sources and is never created.
if(TARGET etnp_kernels)
  target_link_libraries(_core PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,etnp_kernels>")
endif()

# PRESERVED from the current file (do not drop), with one line added: the "lstm" append lives
# HERE, inside this block and before the JOIN. etnp_extras_whole_archive only exists when the
# prefix ships ETNPExtras, which is exactly when etnp::lstm.out is linked. Windows has no
# ETNPExtras, so "lstm" is absent there.
if(COMMAND etnp_extras_whole_archive)
  etnp_extras_whole_archive(_core)
  list(APPEND _etnp_kernel_libs "lstm")
endif()

# Comma-separated, NOT semicolon: a CMake list would expand into separate compile-define
# arguments. This string is consumed verbatim by info.py. MUST come after every APPEND above.
list(JOIN _etnp_kernel_libs "," _etnp_kernel_libs_str)
message(STATUS "etnp: kernel libs linked = ${_etnp_kernel_libs_str}")

target_compile_definitions(_core PRIVATE ETNP_ET_VERSION="${ETNP_ET_VERSION}")
target_compile_definitions(_core PRIVATE ETNP_KERNEL_LIBS="${_etnp_kernel_libs_str}")

# Post-link kernel-registration guard (fail the BUILD, not runtime). The custom
# kernel seam appends its expected registrar TUs via ETNP_KERNEL_EXPECT_TUS.
add_custom_command(TARGET _core POST_BUILD
  COMMAND ${CMAKE_COMMAND} -DSO=$<TARGET_FILE:_core> -DNM=nm
          "-DEXTRA_TUS=${ETNP_KERNEL_EXPECT_TUS}"
          -P ${CMAKE_SOURCE_DIR}/cmake/assert_kernels_registered.cmake
  VERBATIM)

# Post-link USDT guard (fail the BUILD, not runtime). Self-arms off the runtime's BUILDINFO,
# so this is a no-op on runtimes built without USDT (e.g. windows-x86_64: usdt=n/a).
add_custom_command(TARGET _core POST_BUILD
  COMMAND ${CMAKE_COMMAND} -DSO=$<TARGET_FILE:_core>
          -DPREFIX=${ETNP_RUNTIME_PREFIX}
          -DCHECKER=${CMAKE_SOURCE_DIR}/scripts/check-usdt-notes.sh
          -P ${CMAKE_SOURCE_DIR}/cmake/assert_usdt_probes.cmake
  VERBATIM)

install(TARGETS _core LIBRARY DESTINATION executorch_numpy_runtime)
