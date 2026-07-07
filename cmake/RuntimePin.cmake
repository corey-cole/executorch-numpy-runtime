# Pins the ExecuTorch runtime prefix + its expected tarball SHA256.
# Mirrors executorch-runtime-dist's EtRuntimePin.cmake contract, as used by
# djl-executorch-engine/native/cmake/EtRuntimePin.cmake: escape hatch
# (ETNP_RUNTIME_PREFIX set by the caller) OR FetchContent the pinned, hash-verified tarball.
# Bump procedure: update ETNP_ET_VERSION/ETNP_RUNTIME_VERSION AND the URL/SHA256 rows below
# together to the next `v<etver>-<pkgrev>` executorch-runtime-dist release. The SHA256 change
# is the supply-chain review gate; CI's separate `gh attestation verify` step covers provenance
# (see docs on CI setup).
#
# NOTE: the URL rows below are intentionally fully-resolved literals, not built from
# ${ETNP_ET_VERSION}/${ETNP_RUNTIME_VERSION} substitution, even though that duplicates the
# version. CI (qa-gate.yml, build-wheels.yml) greps this file's raw text to extract the URL
# before/without invoking CMake; a CMake ${VAR} template would scrape as literal "${...}" text
# and break that. Keep literals and version vars in sync by hand when bumping.
set(ETNP_ET_VERSION "1.3.1" CACHE STRING "Pinned ExecuTorch version")
set(ETNP_RUNTIME_VERSION "1.3.1-2" CACHE STRING "Pinned executorch-runtime-dist package revision")
set(ETNP_RUNTIME_VARIANT "logging" CACHE STRING "Runtime variant: logging (only variant this project ships)")
set(_ETNP_PLATFORM "linux-x86_64")

set(ETNP_RUNTIME_URL_logging_linux-x86_64
  "https://github.com/measly-java-learning/executorch-runtime-dist/releases/download/v1.3.1-2/executorch-runtime-1.3.1-logging-linux-x86_64.tar.gz")
set(ETNP_RUNTIME_SHA256_logging_linux-x86_64 "79456966eafc280506eed60eb9327c8dfbf48fcc9e5bed06a20bc45b9061e57a")

# Resolve relative to this file's location (repo-root/cmake/), not the including project's
# CMAKE_SOURCE_DIR, so both the top-level build and native_tests' standalone
# `find_package(ExecuTorch)` build resolve to the same prefix.
set(ETNP_RUNTIME_PREFIX "" CACHE PATH
  "Explicit ExecuTorch install prefix (escape hatch); empty => fetch the pinned tarball")

if(NOT ETNP_RUNTIME_PREFIX)
  set(_ETNP_URL "${ETNP_RUNTIME_URL_${ETNP_RUNTIME_VARIANT}_${_ETNP_PLATFORM}}")
  set(_ETNP_SHA256 "${ETNP_RUNTIME_SHA256_${ETNP_RUNTIME_VARIANT}_${_ETNP_PLATFORM}}")
  if(NOT _ETNP_URL)
    message(FATAL_ERROR
      "No pin row for variant='${ETNP_RUNTIME_VARIANT}' platform='${_ETNP_PLATFORM}' in RuntimePin.cmake")
  endif()

  include(FetchContent)
  FetchContent_Declare(etnp_runtime URL "${_ETNP_URL}" URL_HASH "SHA256=${_ETNP_SHA256}")
  FetchContent_MakeAvailable(etnp_runtime)
  # FetchContent strips the tarball's single top-level dir on extraction, so SOURCE_DIR IS the install root.
  set(ETNP_RUNTIME_PREFIX "${etnp_runtime_SOURCE_DIR}" CACHE PATH "" FORCE)
endif()

if(NOT EXISTS "${ETNP_RUNTIME_PREFIX}/lib/cmake/ExecuTorch/executorch-config.cmake")
  message(FATAL_ERROR
    "ExecuTorch runtime not found at ${ETNP_RUNTIME_PREFIX}. "
    "Set ETNP_RUNTIME_PREFIX to a pre-unpacked prefix, or leave it empty to auto-fetch.")
endif()
