cmake_minimum_required(VERSION 3.24)
project(etnp_leak LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

include(${CMAKE_SOURCE_DIR}/../cmake/RuntimePin.cmake)
list(APPEND CMAKE_PREFIX_PATH "${ETNP_RUNTIME_PREFIX}")
find_package(ExecuTorch CONFIG REQUIRED)
include(${CMAKE_SOURCE_DIR}/../cmake/Kernels.cmake)

set(ETNP_HARNESS_LIBS
  executorch optimized_native_cpu_ops_lib xnnpack_backend quantized_ops_lib
  extension_module_static extension_data_loader extension_tensor)

# Leak gate (ASan/LSan): proves no memory is leaked.
add_executable(leak_harness leak_harness.cpp ${CMAKE_SOURCE_DIR}/../src/et_core/et_core.cpp)
target_include_directories(leak_harness PRIVATE ${CMAKE_SOURCE_DIR}/../src)
target_compile_options(leak_harness PRIVATE -fsanitize=address -g -O1)
target_link_options(leak_harness PRIVATE -fsanitize=address)
target_link_libraries(leak_harness PRIVATE ${ETNP_HARNESS_LIBS})

# Race gate (TSan): proves race-freedom of et_core's synchronization. Separate binary because
# ThreadSanitizer and AddressSanitizer cannot be combined in one build.
add_executable(race_harness race_harness.cpp ${CMAKE_SOURCE_DIR}/../src/et_core/et_core.cpp)
target_include_directories(race_harness PRIVATE ${CMAKE_SOURCE_DIR}/../src)
target_compile_options(race_harness PRIVATE -fsanitize=thread -g -O1)
target_link_options(race_harness PRIVATE -fsanitize=thread)
target_link_libraries(race_harness PRIVATE ${ETNP_HARNESS_LIBS} Threads::Threads)

# Registration + dispatch gate for the custom-kernel seam. Whole-archives
# etnp_kernels so the registrar TUs survive, then the shared nm-guard asserts
# they did (covers any ETNP_EXTRA_KERNEL_SOURCES injected into this build too).
# Guarded on etnp_kernels existing: with ETNP_BUILD_REFERENCE_KERNEL=OFF and no
# ETNP_EXTRA_KERNEL_SOURCES, there is no kernel to look up, so skip cleanly
# instead of hard-erroring on a nonexistent link target (mirrors the top-level
# CMakeLists.txt's `if(TARGET etnp_kernels)` guard on its _core link line).
if(TARGET etnp_kernels)
  add_executable(kernel_registration_test kernel_registration_test.cpp)
  target_link_libraries(kernel_registration_test PRIVATE
    ${ETNP_HARNESS_LIBS} "$<LINK_LIBRARY:WHOLE_ARCHIVE,etnp_kernels>")

  add_custom_command(TARGET kernel_registration_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -DSO=$<TARGET_FILE:kernel_registration_test> -DNM=nm
            "-DEXTRA_TUS=${ETNP_KERNEL_EXPECT_TUS}"
            -P ${CMAKE_SOURCE_DIR}/../cmake/assert_kernels_registered.cmake
    VERBATIM)

  # Leak variant (ASan/LSan): the exemplar a consumer copies to leak-check an
  # allocating custom kernel. Same source, invoked directly (no .pte, so no export
  # dependency); catches per-invocation allocations the kernel body fails to free.
  # Separate binary because ASan cannot combine with the race harness's TSan.
  add_executable(kernel_leak_test kernel_registration_test.cpp)
  target_compile_options(kernel_leak_test PRIVATE -fsanitize=address -g -O1)
  target_link_options(kernel_leak_test PRIVATE -fsanitize=address)
  target_link_libraries(kernel_leak_test PRIVATE
    ${ETNP_HARNESS_LIBS} "$<LINK_LIBRARY:WHOLE_ARCHIVE,etnp_kernels>")

  add_executable(lstm_kernel_test lstm_kernel_test.cpp)
  target_link_libraries(lstm_kernel_test PRIVATE
    ${ETNP_HARNESS_LIBS} "$<LINK_LIBRARY:WHOLE_ARCHIVE,etnp_kernels>")
  add_custom_command(TARGET lstm_kernel_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -DSO=$<TARGET_FILE:lstm_kernel_test> -DNM=nm
            "-DEXTRA_TUS=${ETNP_KERNEL_EXPECT_TUS}"
            -P ${CMAKE_SOURCE_DIR}/../cmake/assert_kernels_registered.cmake
    VERBATIM)
endif()

# XNNPACK-FC helper unit test (Task 1). Isolated: does NOT need etnp_kernels.
add_executable(xnn_linear_test xnn_linear_test.cpp)
target_include_directories(xnn_linear_test PRIVATE
  ${CMAKE_SOURCE_DIR}/../examples/custom_kernels/lstm)
target_link_libraries(xnn_linear_test PRIVATE ${ETNP_HARNESS_LIBS})
