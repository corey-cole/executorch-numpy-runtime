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
message(STATUS "assert_kernels_registered: XNNPACK registration TU present in ${SO}")
