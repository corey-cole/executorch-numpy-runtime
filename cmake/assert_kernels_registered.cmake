# Post-link guard: prove backend/kernel static-initializer TUs survived the final .so link.
# XNNPACK + quantized + optimized register from pure static-init TUs; --gc-sections silently
# drops them if whole-archive regressed, giving "backend/op not found" only at model-load time.
# Invoked: cmake -DSO=<lib> -DNM=<nm> -P assert_kernels_registered.cmake
if(NOT SO OR NOT EXISTS "${SO}")
  message(FATAL_ERROR "assert_kernels_registered: SO not found: '${SO}'")
endif()
if(NOT NM)
  set(NM "nm")
endif()
execute_process(COMMAND "${NM}" "${SO}" OUTPUT_VARIABLE _syms
                RESULT_VARIABLE _rc ERROR_VARIABLE _err)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "assert_kernels_registered: '${NM} ${SO}' failed (rc=${_rc}): ${_err}")
endif()
foreach(_tu "_GLOBAL__sub_I_XNNPACKBackend")
  string(FIND "${_syms}" "${_tu}" _pos)
  if(_pos EQUAL -1)
    message(FATAL_ERROR
      "Kernel/backend registration '${_tu}' was dropped from ${SO}: static-init TU absent. "
      "whole-archive regressed at final link -> backend/op not found at model-load time.")
  endif()
endforeach()

# quantized_ops_lib and optimized_native_cpu_ops_lib each codegen their own kernel
# registration TU from the same upstream template, so both land in the binary under the
# IDENTICAL local-symbol name "_GLOBAL__sub_I_RegisterCodegenUnboxedKernelsEverything.cpp"
# (confirmed via `nm` on a correct build: two distinct addresses, same symbol text -- see
# task-final-fixes-report.md for the raw nm output). A plain string(FIND) can't tell "both
# present" from "only one survived", so count occurrences instead.
#
# EXPECT_CODEGEN is supplied by the caller from the real link line (quantized_ops_lib and
# optimized_native_cpu_ops_lib each codegen their own registration TU from the same upstream
# template, so both land under the IDENTICAL local-symbol name -- count, don't string(FIND)).
if(NOT DEFINED EXPECT_CODEGEN)
  set(EXPECT_CODEGEN 2)  # back-compat for callers predating the derived count
endif()
string(REGEX MATCHALL "_GLOBAL__sub_I_RegisterCodegenUnboxedKernelsEverything\\.cpp"
       _codegen_matches "${_syms}")
list(LENGTH _codegen_matches _codegen_count)
if(_codegen_count LESS ${EXPECT_CODEGEN})
  message(FATAL_ERROR
    "Expected ${EXPECT_CODEGEN} kernel-registration TU(s) named "
    "'_GLOBAL__sub_I_RegisterCodegenUnboxedKernelsEverything.cpp' in ${SO}, found "
    "${_codegen_count}: whole-archive regressed at final link for one of the kernel libs -> "
    "op not found at model-load time.")
endif()

# Caller-supplied extra static-init TUs that must also survive the final link
# (custom kernels compiled into etnp_kernels and whole-archived into the target).
# EXTRA_TUS is a CMake list of "_GLOBAL__sub_I_<source-basename>" symbol names.
foreach(_tu IN LISTS EXTRA_TUS)
  if(_tu STREQUAL "")
    continue()
  endif()
  string(REGEX REPLACE "\\." "\\\\." _tu_escaped "${_tu}")
  string(REGEX MATCHALL "${_tu_escaped}" _tu_matches "${_syms}")
  list(LENGTH _tu_matches _tu_count)
  if(_tu_count LESS 1)
    message(FATAL_ERROR
      "Custom-kernel registration '${_tu}' was dropped from ${SO}: static-init TU absent. "
      "whole-archive regressed for etnp_kernels -> custom op not found at model-load time.")
  endif()
endforeach()

message(STATUS "assert_kernels_registered: XNNPACK + quantized + optimized registration TUs present in ${SO}")
