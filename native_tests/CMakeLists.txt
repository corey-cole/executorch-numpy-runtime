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
