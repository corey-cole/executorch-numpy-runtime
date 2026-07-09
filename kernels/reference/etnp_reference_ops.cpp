// Reference custom kernel + CI-tested proof of the kernel-registration seam.
//
// Registers etnp::triple.out, a deliberately trivial elementwise op whose only
// job is to keep the compile + EXECUTORCH_LIBRARY registration + whole-archive
// link path continuously exercised. Consumers copy this file's shape to add
// their own kernels (see docs/custom-kernels.md); real kernels (e.g. nn.LSTM)
// live in the consumer's build via ETNP_EXTRA_KERNEL_SOURCES, not here.
#include <cstddef>

#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace etnp {
namespace {

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::KernelRuntimeContext;

// out = 3 * input, elementwise, float32. Trivial by design.
Tensor& triple_out(KernelRuntimeContext& ctx, const Tensor& input, Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::resize_tensor(out, input.sizes()) == Error::Ok,
      InvalidArgument,
      out);
  ET_KERNEL_CHECK(
      ctx, input.scalar_type() == ScalarType::Float, InvalidArgument, out);
  ET_KERNEL_CHECK(
      ctx, out.scalar_type() == ScalarType::Float, InvalidArgument, out);

  const float* in_data = input.const_data_ptr<float>();
  float* out_data = out.mutable_data_ptr<float>();
  for (size_t i = 0; i < static_cast<size_t>(input.numel()); ++i) {
    out_data[i] = in_data[i] * 3.0f;
  }
  return out;
}

} // namespace
} // namespace etnp

EXECUTORCH_LIBRARY(etnp, "triple.out", etnp::triple_out);
