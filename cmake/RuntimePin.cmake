# Pins the ExecuTorch runtime prefix + its expected tarball SHA256.
# Mirrors executorch-runtime-dist's EtRuntimePin.cmake contract, as used by
# djl-executorch-engine/native/cmake/EtRuntimePin.cmake: escape hatch
# (ETNP_RUNTIME_PREFIX set by the caller) OR FetchContent the pinned, hash-verified tarball.
# Bump procedure: update ETNP_ET_VERSION/ETNP_RUNTIME_VERSION AND the URL/SHA256 rows below
# together to the next `v<etver>-<pkgrev>` executorch-runtime-dist release. The SHA256 change
# is the supply-chain review gate; CI's separate `gh attestation verify` step covers provenance
# (see docs on CI setup). Also re-vendor scripts/check-usdt-notes.sh from the same upstream
# release and update docs/usdt-tracepoints.md's probe table if the USDT probe contract
# changed -- otherwise assert_usdt_probes.cmake keeps asserting the OLD contract, stays green
# against surviving old probes, and a new/changed probe ships undocumented and unguarded.
#
# NOTE: the URL rows below are intentionally fully-resolved literals, not built from
# ${ETNP_ET_VERSION}/${ETNP_RUNTIME_VERSION} substitution, even though that duplicates the
# version. CI (qa-gate.yml, build-wheels.yml) greps this file's raw text to extract the URL
# before/without invoking CMake; a CMake ${VAR} template would scrape as literal "${...}" text
# and break that. Keep literals and version vars in sync by hand when bumping.
set(ETNP_ET_VERSION "1.3.1" CACHE STRING "Pinned ExecuTorch version")
set(ETNP_RUNTIME_VERSION "1.3.1-6" CACHE STRING "Pinned executorch-runtime-dist package revision")
set(ETNP_RUNTIME_VARIANT "logging" CACHE STRING "Runtime variant: logging (only variant this project ships)")
# Derive the runtime platform slug from the build's target architecture so the correct
# per-arch pin row (below) is chosen automatically on both x86_64 and aarch64 CI runners.
# CMAKE_SYSTEM_PROCESSOR is populated by project()/the toolchain and is the target arch
# (equals the host arch for the native builds this project does). A caller may still
# pre-set _ETNP_PLATFORM (e.g. -D for a cross-build) to bypass detection.
if(NOT _ETNP_PLATFORM)
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _etnp_arch)
  if(_etnp_arch MATCHES "^(x86_64|amd64)$")
    set(_ETNP_PLATFORM "linux-x86_64")
  elseif(_etnp_arch MATCHES "^(aarch64|arm64)$")
    set(_ETNP_PLATFORM "linux-aarch64")
  else()
    message(FATAL_ERROR
      "Unsupported target architecture '${CMAKE_SYSTEM_PROCESSOR}' for the ExecuTorch runtime pin; "
      "expected x86_64 or aarch64. Set _ETNP_PLATFORM explicitly to override.")
  endif()
endif()

set(ETNP_RUNTIME_URL_logging_linux-x86_64
  "https://github.com/measly-java-learning/executorch-runtime-dist/releases/download/v1.3.1-6/executorch-runtime-1.3.1-logging-linux-x86_64.tar.gz")
set(ETNP_RUNTIME_SHA256_logging_linux-x86_64 "e1c29f4fe7d0e108bfc3a4dc6f0bfb98eb5af97a175b5bae95da61446d8542cd")

set(ETNP_RUNTIME_URL_logging_linux-aarch64
  "https://github.com/measly-java-learning/executorch-runtime-dist/releases/download/v1.3.1-6/executorch-runtime-1.3.1-logging-linux-aarch64.tar.gz")
set(ETNP_RUNTIME_SHA256_logging_linux-aarch64 "feea21ea4d18673601bc7ce231ede25e19a48a1a0ba67d0b02dd490f6ce11eb5")

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
