cmake_minimum_required(VERSION 3.24)
project(etnp_leak LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(${CMAKE_SOURCE_DIR}/../cmake/RuntimePin.cmake)
list(APPEND CMAKE_PREFIX_PATH "${ETNP_RUNTIME_PREFIX}")
find_package(ExecuTorch CONFIG REQUIRED)

# Workaround (mirrors top-level CMakeLists.txt): the prebuilt runtime's exported targets
# embed an absolute host path (/usr/lib64/libm.so) baked in at the tarball's build time
# (RHEL/Fedora-style layout). On Debian/Ubuntu that exact path doesn't exist even though
# libm is always available via libc; repoint the affected imported targets at the generic
# "m" library instead. native_tests calls find_package(ExecuTorch) independently of the
# top-level build, so this must be duplicated here or the harness link fails with
# "cannot find /usr/lib64/libm.so".
foreach(_libm_tgt xnnpack-microkernels-prod XNNPACK xnnpack-hardware-config xnnpack-indirection xnnpack-subgraph)
  if(TARGET ${_libm_tgt})
    get_target_property(_ill ${_libm_tgt} INTERFACE_LINK_LIBRARIES)
    if(_ill)
      string(REPLACE "/usr/lib64/libm.so" "m" _ill "${_ill}")
      set_target_properties(${_libm_tgt} PROPERTIES INTERFACE_LINK_LIBRARIES "${_ill}")
    endif()
  endif()
endforeach()

add_executable(leak_harness leak_harness.cpp ${CMAKE_SOURCE_DIR}/../src/et_core/et_core.cpp)
target_include_directories(leak_harness PRIVATE ${CMAKE_SOURCE_DIR}/../src)
target_compile_options(leak_harness PRIVATE -fsanitize=address -g -O1)
target_link_options(leak_harness PRIVATE -fsanitize=address)
target_link_libraries(leak_harness PRIVATE
  executorch optimized_native_cpu_ops_lib xnnpack_backend quantized_ops_lib
  extension_module_static extension_data_loader extension_tensor)
