cmake_minimum_required(VERSION 3.24)
project(executorch_numpy_runtime LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(cmake/RuntimePin.cmake)
list(APPEND CMAKE_PREFIX_PATH "${ETNP_RUNTIME_PREFIX}")
find_package(ExecuTorch CONFIG REQUIRED)

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

# Link the whole ExecuTorch stack. The runtime config self-whole-archives the kernel/backend
# archives via INTERFACE_LINK_OPTIONS, so linking these targets is enough.
target_link_libraries(_core PRIVATE
  executorch optimized_native_cpu_ops_lib xnnpack_backend quantized_ops_lib
  extension_module_static extension_data_loader extension_tensor)

target_compile_definitions(_core PRIVATE ETNP_ET_VERSION="${ETNP_ET_VERSION}")

# Post-link kernel-registration guard (fail the BUILD, not runtime).
add_custom_command(TARGET _core POST_BUILD
  COMMAND ${CMAKE_COMMAND} -DSO=$<TARGET_FILE:_core> -DNM=nm
          -P ${CMAKE_SOURCE_DIR}/cmake/assert_kernels_registered.cmake
  VERBATIM)

install(TARGETS _core LIBRARY DESTINATION executorch_numpy_runtime)
