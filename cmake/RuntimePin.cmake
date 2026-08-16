# Selects and fetches the pinned ExecuTorch runtime prefix.
#
# The pin itself -- URLs, SHA256s, versions -- lives in the GENERATED cmake/EtRuntimePin.cmake,
# vendored verbatim from an executorch-runtime-dist release. This file holds only what is ours:
# platform detection, the row selection, the fetch, and the escape hatch.
#
# Bump procedure:
#   gh release download <tag> --repo measly-java-learning/executorch-runtime-dist \
#     --pattern 'EtRuntimePin.cmake' --dir cmake/ --clobber
#   ./scripts/check-pin-rows.sh
# The SHA256 change in that diff is the supply-chain review gate; CI's separate
# `gh attestation verify` step covers provenance. Also re-check whether the USDT probe contract
# moved upstream (scripts/check-usdt-notes.sh, docs/usdt-tracepoints.md) -- if it did and you skip
# it, assert_usdt_probes.cmake keeps asserting the OLD contract, stays green against surviving old
# probes, and a new probe ships undocumented and unguarded.
#
# The generated file's URLs are intentionally literal, which CI depends on: qa-gate.yml greps this
# pin's raw text for a URL before/without invoking CMake. scripts/check-pin-rows.sh guards that
# contract and the /MD-vs-/MT row choice; run it after any bump.
include(${CMAKE_CURRENT_LIST_DIR}/EtRuntimePin.cmake)

# Derived, deliberately NOT cached: a cached value would silently survive a bump in an existing
# build directory and report the previous release's version.
set(ETNP_ET_VERSION "${ET_RUNTIME_ET_VERSION}")
set(ETNP_RUNTIME_VERSION "${ET_RUNTIME_VERSION}")

set(ETNP_RUNTIME_VARIANT "logging" CACHE STRING "Runtime variant: logging (only variant this project ships)")

# Derive the runtime platform slug from the build's target architecture so the correct
# per-arch pin row is chosen automatically on both x86_64 and aarch64 CI runners.
# CMAKE_SYSTEM_PROCESSOR is populated by project()/the toolchain and is the target arch
# (equals the host arch for the native builds this project does). A caller may still
# pre-set _ETNP_PLATFORM (e.g. -D for a cross-build) to bypass detection.
if(NOT _ETNP_PLATFORM)
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _etnp_arch)
  # CMAKE_SYSTEM_PROCESSOR is "AMD64" on Windows, "x86_64" on Linux.
  if(_etnp_arch MATCHES "^(x86_64|amd64)$")
    set(_etnp_machine "x86_64")
  elseif(_etnp_arch MATCHES "^(aarch64|arm64)$")
    set(_etnp_machine "aarch64")
  else()
    message(FATAL_ERROR
      "Unsupported target architecture '${CMAKE_SYSTEM_PROCESSOR}' for the ExecuTorch runtime pin; "
      "expected x86_64 or aarch64. Set _ETNP_PLATFORM explicitly to override.")
  endif()

  if(WIN32)
    set(_etnp_os "windows")
  elseif(UNIX AND NOT APPLE)
    set(_etnp_os "linux")
  else()
    message(FATAL_ERROR
      "Unsupported target OS for the ExecuTorch runtime pin (Windows and Linux only). "
      "Set _ETNP_PLATFORM explicitly to override.")
  endif()

  # "windows-x86_64" is the /MD (dynamic CRT) row, deliberately: a CPython extension must match
  # CPython's own CRT. The pin also carries "windows-x86_64-static" (/MT), which exists for JNI
  # consumers -- never select it here. Guarded by scripts/check-pin-rows.sh.
  set(_ETNP_PLATFORM "${_etnp_os}-${_etnp_machine}")
endif()

# Resolve relative to this file's location (repo-root/cmake/), not the including project's
# CMAKE_SOURCE_DIR, so both the top-level build and native_tests' standalone
# `find_package(ExecuTorch)` build resolve to the same prefix.
set(ETNP_RUNTIME_PREFIX "" CACHE PATH
  "Explicit ExecuTorch install prefix (escape hatch); empty => fetch the pinned tarball")

if(NOT ETNP_RUNTIME_PREFIX)
  # Fails at configure time naming both values if the combination was never published, rather
  # than expanding to "" and surfacing later as a confusing empty-URL FetchContent error.
  et_runtime_dist_url("${ETNP_RUNTIME_VARIANT}" "${_ETNP_PLATFORM}" _ETNP_URL _ETNP_SHA256)

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
