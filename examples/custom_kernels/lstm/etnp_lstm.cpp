// etnp::lstm.out — single-layer, unidirectional, batch_first=False, float32 LSTM over a
// full sequence. The input projection for ALL timesteps runs as ONE batched XNNPACK FC
// on the shared runtime threadpool; packed FC operators are cached across executes
// (xnn_linear_cache.h); the per-timestep recurrent projection is a single-threaded FC
// and the gate/cell/hidden update is a Highway-SIMD fused pass (lstm_cell.cc).
//
// Schema:
//   etnp::lstm.out(Tensor input, Tensor h0, Tensor c0, Tensor w_ih, Tensor w_hh,
//                  Tensor? b_ih, Tensor? b_hh, *,
//                  Tensor(a!) output, Tensor(b!) hn, Tensor(c!) cn)
//                  -> (Tensor(a!), Tensor(b!), Tensor(c!))
//   input [T,B,I]  h0/c0 [B,H]  w_ih [4H,I]  w_hh [4H,H]  b_ih/b_hh [4H]
//   output [T,B,H]  hn/cn [B,H].  Gate row order: i,f,g,o (PyTorch).
#include <cstring>
#include <optional>
#include <tuple>

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/kernel/operator_registry.h>

#include "xnn_linear.h"
#include <executorch/extension/threadpool/threadpool.h>

#include "lstm_cell.h"
#include "xnn_linear_cache.h"

namespace etnp {
namespace {
using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::KernelRuntimeContext;

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

  // Packed projections come from the process-wide cache: weights live in the
  // .pte constant segment (pointer-stable per loaded program), so packing
  // happens once per weight set, not once per execute.
  auto ent_ih = XnnLinearCache::get(
      static_cast<size_t>(I), static_cast<size_t>(4 * H),
      w_ih.const_data_ptr<float>(),
      b_ih.has_value() ? b_ih->const_data_ptr<float>() : nullptr);
  auto ent_hh = XnnLinearCache::get(
      static_cast<size_t>(H), static_cast<size_t>(4 * H),
      w_hh.const_data_ptr<float>(),
      b_hh.has_value() ? b_hh->const_data_ptr<float>() : nullptr);
  ET_KERNEL_CHECK(ctx, ent_ih != nullptr && ent_hh != nullptr, Internal, ret);

  const size_t g_ih_bytes =
      static_cast<size_t>(T) * B * 4 * H * sizeof(float);
  const size_t g_hh_bytes = static_cast<size_t>(B) * 4 * H * sizeof(float);
  auto g_ih_r = ctx.allocate_temp(g_ih_bytes);
  auto g_hh_r = ctx.allocate_temp(g_hh_bytes);
  ET_KERNEL_CHECK(ctx, g_ih_r.ok() && g_hh_r.ok(), MemoryAllocationFailed, ret);
  float* g_ih_all = static_cast<float*>(g_ih_r.get());
  float* g_hh = static_cast<float*>(g_hh_r.get());

  // Running state lives in hn/cn, seeded from h0/c0.
  float* h = hn.mutable_data_ptr<float>();
  float* c = cn.mutable_data_ptr<float>();
  std::memcpy(h, h0.const_data_ptr<float>(), static_cast<size_t>(B) * H * sizeof(float));
  std::memcpy(c, c0.const_data_ptr<float>(), static_cast<size_t>(B) * H * sizeof(float));

  const float* in = input.const_data_ptr<float>();
  float* out = output.mutable_data_ptr<float>();

  // Input projection for ALL timesteps in one GEMM: input [T,B,I] is
  // contiguous, hence also a valid [T*B, I] matrix. This is the only
  // multi-threaded step — same pool the XNNPACK delegate uses.
  ET_KERNEL_CHECK(ctx,
      run_cached(*ent_ih, static_cast<size_t>(T) * B, in, g_ih_all,
                 executorch::extension::threadpool::get_pthreadpool()) ==
          Error::Ok,
      Internal, ret);

  for (int64_t t = 0; t < T; ++t) {
    // Recurrent projection: g_hh = h_{t-1} @ W_hh^T + b_hh. Reads h fully
    // before lstm_cell_update overwrites it. Single-threaded: at [B,H]x[H,4H]
    // scale the pool dispatch overhead exceeds the win (measured).
    ET_KERNEL_CHECK(ctx,
        run_cached(*ent_hh, static_cast<size_t>(B), h, g_hh,
                   /*tp=*/nullptr) == Error::Ok,
        Internal, ret);
    lstm_cell_update(static_cast<size_t>(B), static_cast<size_t>(H),
                     g_ih_all + static_cast<size_t>(t) * B * 4 * H, g_hh, c, h,
                     out + static_cast<size_t>(t) * B * H);
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
