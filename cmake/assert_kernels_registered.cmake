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
# present" from "only one survived", so count occurrences and require at least 2.
string(REGEX MATCHALL "_GLOBAL__sub_I_RegisterCodegenUnboxedKernelsEverything\\.cpp"
       _codegen_matches "${_syms}")
list(LENGTH _codegen_matches _codegen_count)
if(_codegen_count LESS 2)
  message(FATAL_ERROR
    "Expected 2 kernel-registration TUs (quantized_ops_lib + optimized_native_cpu_ops_lib) "
    "named '_GLOBAL__sub_I_RegisterCodegenUnboxedKernelsEverything.cpp' in ${SO}, found "
    "${_codegen_count}: whole-archive regressed at final link for one of the kernel libs -> "
    "op not found at model-load time.")
endif()

message(STATUS "assert_kernels_registered: XNNPACK + quantized + optimized registration TUs present in ${SO}")
