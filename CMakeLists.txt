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
# "portable" is unconditional because portable aten coverage exists on every platform -- but it
# arrives via a DIFFERENT library depending on the tarball, which is what the if/else below is for.
set(_etnp_kernel_libs "portable")

# optimized_native_cpu_ops_lib registers the FULL aten set (optimized kernels where available,
# portable elsewhere), so where it exists it already provides portable coverage and
# portable_ops_lib is pure redundancy. Where it does NOT exist (the core-only Windows tarball),
# nothing registers aten ops unless we link portable_ops_lib ourselves -- `executorch` does NOT
# pull it transitively (only the `executorch_kernels` INTERFACE target does).
#
# These are mutually exclusive ON PURPOSE. Linking both makes each aten op register twice ->
# Error::RegistrationAlreadyRegistered (0x16), and register_kernels() is documented "Panics on
# error" because it runs at static-init -> the module would ABORT on import. Double-registering
# the full aten set also risks Error::RegistrationExceedingMaxKernels (0x15) against the
# registry's fixed table. Do not "simplify" this into an unconditional link.
if(TARGET optimized_native_cpu_ops_lib)
  target_link_libraries(_core PRIVATE optimized_native_cpu_ops_lib)
  list(APPEND _etnp_kernel_libs "optimized")
else()
  # Windows path: portable_ops_lib self-whole-archives via its own INTERFACE_LINK_OPTIONS
  # (/WHOLEARCHIVE:), but only once something links it. Without this, a .pte fails at load with
  # "Missing operator: aten::add.out" -- observed for real on the first Windows CI run.
  target_link_libraries(_core PRIVATE portable_ops_lib)
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

# The nm-guard counts one codegen registrar TU per codegen'd ops lib actually linked. Derive it
# from the link line rather than hardcoding 2, which was only right while both libs were always
# present.
set(_etnp_expect_codegen 0)
if(TARGET optimized_native_cpu_ops_lib)
  math(EXPR _etnp_expect_codegen "${_etnp_expect_codegen} + 1")
endif()
if(TARGET quantized_ops_lib)
  math(EXPR _etnp_expect_codegen "${_etnp_expect_codegen} + 1")
endif()

target_compile_definitions(_core PRIVATE ETNP_ET_VERSION="${ETNP_ET_VERSION}")
target_compile_definitions(_core PRIVATE ETNP_KERNEL_LIBS="${_etnp_kernel_libs_str}")

# Post-link kernel-registration guard (fail the BUILD, not runtime). The custom kernel seam
# appends its expected registrar TUs via ETNP_KERNEL_EXPECT_TUS.
#
# GNU/Clang only, deliberately. Under MSVC this guard cannot work for three independent
# reasons: there is no `nm`; MSVC emits ??__E-mangled initializers rather than the
# _GLOBAL__sub_I_* symbols the guard matches; and the codegen TUs it counts come from
# quantized_ops_lib + optimized_native_cpu_ops_lib, which the Windows tarball does not ship.
# Windows substitutes a RUNTIME assertion instead -- XnnpackBackend must appear in
# registered_backends() (see the spec's D5 and tests/test_meta_info.py). This is a known,
# accepted asymmetry, not an oversight.
if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
  add_custom_command(TARGET _core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -DSO=$<TARGET_FILE:_core> -DNM=nm
            "-DEXTRA_TUS=${ETNP_KERNEL_EXPECT_TUS}"
            "-DEXPECT_CODEGEN=${_etnp_expect_codegen}"
            -P ${CMAKE_SOURCE_DIR}/cmake/assert_kernels_registered.cmake
    VERBATIM)
else()
  message(STATUS
    "etnp: assert_kernels_registered SKIPPED -- no nm/GNU symbols under '${CMAKE_CXX_COMPILER_ID}'. "
    "Backend registration is asserted at runtime instead (see tests/test_meta_info.py).")
endif()

# Post-link USDT guard (fail the BUILD, not runtime). Self-arms off the runtime's BUILDINFO,
# so this is a no-op on runtimes built without USDT (e.g. windows-x86_64: usdt=n/a).
add_custom_command(TARGET _core POST_BUILD
  COMMAND ${CMAKE_COMMAND} -DSO=$<TARGET_FILE:_core>
          -DPREFIX=${ETNP_RUNTIME_PREFIX}
          -DCHECKER=${CMAKE_SOURCE_DIR}/scripts/check-usdt-notes.sh
          -P ${CMAKE_SOURCE_DIR}/cmake/assert_usdt_probes.cmake
  VERBATIM)

install(TARGETS _core LIBRARY DESTINATION executorch_numpy_runtime)
