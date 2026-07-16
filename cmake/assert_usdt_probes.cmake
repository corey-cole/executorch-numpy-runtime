# Post-link guard: prove the etnp USDT probe contract survived the final .so link.
# Sibling of assert_kernels_registered.cmake -- same philosophy: fail the BUILD, not runtime.
# Probes live in libetnp_ops_lstm.a (an ETNPExtras archive), NOT in ExecuTorch's own libs, and
# reach _core only because etnp_extras_whole_archive() whole-archives it. A stripped wheel or a
# regressed whole-archive would otherwise produce a silently un-traceable .so.
#
# SELF-ARMING: reads ${PREFIX}/BUILDINFO and enforces only when the runtime tarball records
# `usdt=on`. windows-x86_64 records `usdt=n/a`, and releases before v1.3.1-5 carry no usdt key
# at all -- both disarm with no platform conditional here.
#
# Invoked: cmake -DSO=<lib> -DPREFIX=<runtime-prefix> -DCHECKER=<script> -P assert_usdt_probes.cmake
if(NOT SO OR NOT EXISTS "${SO}")
  message(FATAL_ERROR "assert_usdt_probes: SO not found: '${SO}'")
endif()
if(NOT CHECKER OR NOT EXISTS "${CHECKER}")
  message(FATAL_ERROR "assert_usdt_probes: CHECKER not found: '${CHECKER}'")
endif()

set(_buildinfo "${PREFIX}/BUILDINFO")
if(NOT EXISTS "${_buildinfo}")
  message(STATUS "assert_usdt_probes: no BUILDINFO at '${_buildinfo}' -- USDT guard disarmed")
  return()
endif()

file(READ "${_buildinfo}" _bi)
if(NOT _bi MATCHES "(^|\n)usdt=on([\r\n]|$)")
  message(STATUS "assert_usdt_probes: runtime does not record usdt=on -- USDT guard disarmed")
  return()
endif()

find_program(_bash bash REQUIRED)
execute_process(COMMAND "${_bash}" "${CHECKER}" --expect on "${SO}"
                RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR
    "USDT probe contract broken in ${SO} (rc=${_rc}).\n${_out}${_err}\n"
    "The pinned runtime records usdt=on, so probes MUST reach the linked .so. Likely causes: "
    "the link or wheel was stripped (NOSTRIP removed, CMAKE_INSTALL_DO_STRIP set, linker -s), "
    "or etnp_ops_lstm stopped being whole-archived.")
endif()
message(STATUS "assert_usdt_probes: etnp USDT probe contract present in ${SO}")
