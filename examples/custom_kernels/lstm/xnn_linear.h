// out[b,:] = input[b,:] @ weight^T + bias  — weight is [OC, IC] (PyTorch F.linear layout).
// RAII wrapper over XNNPACK's f32 fully-connected operator: the same tuned microkernels
// the XNNPACK delegate runs for lowered linears. No new dependency; single-threaded here.
#pragma once
#include <limits>

#include <pthreadpool.h>
#include <xnnpack.h>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

namespace etnp {
using executorch::runtime::Error;

// XNNPACK needs one process-wide init before any operator is created. C++11 static-local
// init is thread-safe and runs exactly once.
inline Error ensure_xnn_initialized() {
  static const xnn_status s = xnn_initialize(/*allocator=*/nullptr);
  return s == xnn_status_success ? Error::Ok : Error::Internal;
}

class XnnLinear {
 public:
  static executorch::runtime::Result<XnnLinear> create(
      size_t in_ch, size_t out_ch,
      const float* weight,          // [out_ch, in_ch]; packed into the op at create(),
                                    // pointer need not outlive this call
      const float* bias /*nullable*/) {
    xnn_operator_t op = nullptr;
    const xnn_status s = xnn_create_fully_connected_nc_f32(
        in_ch, out_ch,
        /*input_stride=*/in_ch, /*output_stride=*/out_ch,
        weight, bias,
        -std::numeric_limits<float>::infinity(),  // output_min: no fused activation
        +std::numeric_limits<float>::infinity(),  // output_max
        /*flags=*/0,                              // weight already [OC, IC]
        /*weights_cache=*/nullptr,
        &op);
    if (s != xnn_status_success) return Error::Internal;
    return XnnLinear(op);
  }

  // `out` must hold batch*out_ch floats. tp may be null (single-threaded SIMD).
  Error run(size_t batch, const float* input, float* out, pthreadpool_t tp) {
    if (xnn_reshape_fully_connected_nc_f32(op_, batch, tp) != xnn_status_success)
      return Error::Internal;
    if (xnn_setup_fully_connected_nc_f32(op_, input, out) != xnn_status_success)
      return Error::Internal;
    if (xnn_run_operator(op_, tp) != xnn_status_success) return Error::Internal;
    return Error::Ok;
  }

  ~XnnLinear() { if (op_) xnn_delete_operator(op_); }
  XnnLinear(XnnLinear&& o) noexcept : op_(o.op_) { o.op_ = nullptr; }
  XnnLinear(const XnnLinear&) = delete;
  XnnLinear& operator=(const XnnLinear&) = delete;
  XnnLinear& operator=(XnnLinear&&) = delete;

 private:
  explicit XnnLinear(xnn_operator_t op) : op_(op) {}
  xnn_operator_t op_ = nullptr;
};
} // namespace etnp
