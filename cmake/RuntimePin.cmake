# Pins the ExecuTorch runtime prefix + its expected tarball SHA256.
# Mirrors executorch-runtime-dist's EtRuntimePin.cmake contract.
set(ETNP_ET_VERSION "1.3.1" CACHE STRING "Pinned ExecuTorch version")
# Resolve relative to this file's location (repo-root/cmake/), not the including
# project's CMAKE_SOURCE_DIR, so both the top-level build and native_tests' standalone
# `find_package(ExecuTorch)` build resolve to the same repo-root third_party/ prefix.
set(ETNP_RUNTIME_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../third_party/executorch-runtime-1.3.1-logging-linux-x86_64"
    CACHE PATH "Unpacked ExecuTorch runtime prefix")

if(NOT EXISTS "${ETNP_RUNTIME_PREFIX}/lib/cmake/ExecuTorch/executorch-config.cmake")
  message(FATAL_ERROR
    "ExecuTorch runtime not found at ${ETNP_RUNTIME_PREFIX}. "
    "Unpack executorch-runtime-1.3.1-logging-linux-x86_64.tar.gz into third_party/.")
endif()
