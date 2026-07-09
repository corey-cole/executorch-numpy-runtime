// etnp::lstm.out — single-layer, unidirectional, batch_first=False, float32 LSTM over a
// full sequence. Gate projections reuse XNNPACK FC (created once, run per timestep); the
// gate activations + cell/hidden update are fused into one pass per timestep.
//
// Schema:
//   etnp::lstm.out(Tensor input, Tensor h0, Tensor c0, Tensor w_ih, Tensor w_hh,
//                  Tensor? b_ih, Tensor? b_hh, *,
//                  Tensor(a!) output, Tensor(b!) hn, Tensor(c!) cn)
//                  -> (Tensor(a!), Tensor(b!), Tensor(c!))
//   input [T,B,I]  h0/c0 [B,H]  w_ih [4H,I]  w_hh [4H,H]  b_ih/b_hh [4H]
//   output [T,B,H]  hn/cn [B,H].  Gate row order: i,f,g,o (PyTorch).
#include <cmath>
#include <cstring>
#include <optional>
#include <tuple>

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/kernel/operator_registry.h>

#include "xnn_linear.h"

namespace etnp {
namespace {
using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::KernelRuntimeContext;

inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

std::tuple<Tensor&, Tensor&, Tensor&> lstm_out(
    KernelRuntimeContext& ctx,
    const Tensor& input, const Tensor& h0, const Tensor& c0,
    const Tensor& w_ih, const Tensor& w_hh,
    const std::optional<Tensor>& b_ih, const std::optional<Tensor>& b_hh,
    Tensor& output, Tensor& hn, Tensor& cn) {
  auto ret = std::tie(output, hn, cn);

  const int64_t T = input.size(0);
  const int64_t B = input.size(1);
  const int64_t I = input.size(2);
  const int64_t H = h0.size(1);

  ET_KERNEL_CHECK(ctx, input.scalar_type() == ScalarType::Float, InvalidArgument, ret);
  ET_KERNEL_CHECK(ctx, w_ih.size(0) == 4 * H && w_ih.size(1) == I, InvalidArgument, ret);
  ET_KERNEL_CHECK(ctx, w_hh.size(0) == 4 * H && w_hh.size(1) == H, InvalidArgument, ret);

  const SizesType osz[3] = {static_cast<SizesType>(T), static_cast<SizesType>(B),
                            static_cast<SizesType>(H)};
  ET_KERNEL_CHECK(ctx,
      executorch::runtime::resize_tensor(output, {osz, 3}) == Error::Ok, InvalidArgument, ret);
  ET_KERNEL_CHECK(ctx,
      executorch::runtime::resize_tensor(hn, h0.sizes()) == Error::Ok, InvalidArgument, ret);
  ET_KERNEL_CHECK(ctx,
      executorch::runtime::resize_tensor(cn, c0.sizes()) == Error::Ok, InvalidArgument, ret);
  ET_KERNEL_CHECK(ctx, ensure_xnn_initialized() == Error::Ok, Internal, ret);

  // Create the two projections ONCE (weights are constant across the sequence).
  auto lin_ih = XnnLinear::create(
      I, 4 * H, w_ih.const_data_ptr<float>(),
      b_ih.has_value() ? b_ih->const_data_ptr<float>() : nullptr);
  auto lin_hh = XnnLinear::create(
      H, 4 * H, w_hh.const_data_ptr<float>(),
      b_hh.has_value() ? b_hh->const_data_ptr<float>() : nullptr);
  ET_KERNEL_CHECK(ctx, lin_ih.ok() && lin_hh.ok(), Internal, ret);

  const size_t g_bytes = static_cast<size_t>(B) * 4 * H * sizeof(float);
  auto g_ih_r = ctx.allocate_temp(g_bytes);
  auto g_hh_r = ctx.allocate_temp(g_bytes);
  ET_KERNEL_CHECK(ctx, g_ih_r.ok() && g_hh_r.ok(), MemoryAllocationFailed, ret);
  float* g_ih = static_cast<float*>(g_ih_r.get());
  float* g_hh = static_cast<float*>(g_hh_r.get());

  // Running state lives in hn/cn, seeded from h0/c0.
  float* h = hn.mutable_data_ptr<float>();
  float* c = cn.mutable_data_ptr<float>();
  std::memcpy(h, h0.const_data_ptr<float>(), static_cast<size_t>(B) * H * sizeof(float));
  std::memcpy(c, c0.const_data_ptr<float>(), static_cast<size_t>(B) * H * sizeof(float));

  const float* in = input.const_data_ptr<float>();
  float* out = output.mutable_data_ptr<float>();

  for (int64_t t = 0; t < T; ++t) {
    // Projections: g_ih = x_t @ W_ih^T + b_ih ; g_hh = h_{t-1} @ W_hh^T + b_hh.
    // lin_hh reads the current h fully into g_hh before the fused loop overwrites h.
    ET_KERNEL_CHECK(ctx,
        lin_ih->run(B, in + t * B * I, g_ih, /*tp=*/nullptr) == Error::Ok, Internal, ret);
    ET_KERNEL_CHECK(ctx,
        lin_hh->run(B, h, g_hh, /*tp=*/nullptr) == Error::Ok, Internal, ret);

    float* out_t = out + t * B * H;
    for (int64_t b = 0; b < B; ++b) {
      const float* aih = g_ih + b * 4 * H;  // gate blocks: [i|f|g|o], each length H
      const float* ahh = g_hh + b * 4 * H;
      for (int64_t j = 0; j < H; ++j) {
        const float ig = sigmoidf(aih[0 * H + j] + ahh[0 * H + j]);
        const float fg = sigmoidf(aih[1 * H + j] + ahh[1 * H + j]);
        const float gg = std::tanh(aih[2 * H + j] + ahh[2 * H + j]);
        const float og = sigmoidf(aih[3 * H + j] + ahh[3 * H + j]);
        const float cnew = fg * c[b * H + j] + ig * gg;
        const float hnew = og * std::tanh(cnew);
        c[b * H + j] = cnew;
        h[b * H + j] = hnew;
        out_t[b * H + j] = hnew;
      }
    }
  }
  return ret;  // hn/cn already hold the final state
}

// Boxed trampoline: the pinned runtime's auto-unboxing macro caps outputs at 1, so we register
// directly. Stack order matches the schema: input,h0,c0,w_ih,w_hh,b_ih?,b_hh?,output,hn,cn.
void lstm_boxed(KernelRuntimeContext& ctx, executorch::runtime::Span<EValue*> stack) {
  auto opt = [](EValue* e) -> std::optional<Tensor> {
    return e->isNone() ? std::optional<Tensor>{} : std::optional<Tensor>(e->toTensor());
  };
  lstm_out(ctx,
      stack[0]->toTensor(), stack[1]->toTensor(), stack[2]->toTensor(),
      stack[3]->toTensor(), stack[4]->toTensor(),
      opt(stack[5]), opt(stack[6]),
      stack[7]->toTensor(), stack[8]->toTensor(), stack[9]->toTensor());
}

// File-scope registrar → dynamic-init TU `_GLOBAL__sub_I_etnp_lstm.cpp` (whole-archived + nm-guarded).
const auto etnp_lstm_registrar = executorch::runtime::register_kernel(
    executorch::runtime::Kernel("etnp::lstm.out", lstm_boxed));
}  // namespace
}  // namespace etnp
