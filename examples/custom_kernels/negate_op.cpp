// Worked example of a consumer-supplied custom kernel injected via
// ETNP_EXTRA_KERNEL_SOURCES (see docs/custom-kernels.md). Not compiled into the
// default build. Registers etnp::negate.out computing out = -input (float32).
#include <cstddef>

#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace etnp {
namespace {

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::KernelRuntimeContext;

Tensor& negate_out(KernelRuntimeContext& ctx, const Tensor& input, Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::resize_tensor(out, input.sizes()) == Error::Ok,
      InvalidArgument,
      out);
  ET_KERNEL_CHECK(
      ctx, input.scalar_type() == ScalarType::Float, InvalidArgument, out);

  const float* in_data = input.const_data_ptr<float>();
  float* out_data = out.mutable_data_ptr<float>();
  for (size_t i = 0; i < static_cast<size_t>(input.numel()); ++i) {
    out_data[i] = -in_data[i];
  }
  return out;
}

} // namespace
} // namespace etnp

EXECUTORCH_LIBRARY(etnp, "negate.out", etnp::negate_out);
