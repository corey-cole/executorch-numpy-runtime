// Analytic correctness for etnp::lstm.out with zero weights/bias (no PyTorch needed):
// c_t = 0.5*c_{t-1}, h_t = 0.5*tanh(c_t). Invoked through the operator registry.
#include <cmath>
#include <cstdio>
#include <vector>

#include <cstdint>

#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::extension::make_tensor_ptr;
using executorch::runtime::EValue;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::OpFunction;
using executorch::runtime::Span;

int main() {
  executorch::runtime::runtime_init();
  static constexpr char kOp[] = "etnp::lstm.out";
  if (!executorch::runtime::registry_has_op_function(kOp)) {
    std::fprintf(stderr, "FAIL: %s not registered\n", kOp); return 1;
  }
  auto fn_r = executorch::runtime::get_op_function_from_registry(kOp, {});
  if (!fn_r.ok()) { std::fprintf(stderr, "FAIL: lookup\n"); return 1; }
  OpFunction fn = fn_r.get();

  const int64_t T = 3, B = 1, I = 2, H = 2;
  std::vector<float> in(T * B * I, 0.0f);
  std::vector<float> w_ih(4 * H * I, 0.0f), w_hh(4 * H * H, 0.0f);
  std::vector<float> b_ih(4 * H, 0.0f), b_hh(4 * H, 0.0f);
  std::vector<float> h0(B * H, 0.0f), c0(B * H, 1.0f);
  std::vector<float> out(T * B * H, -1.0f), hn(B * H, -1.0f), cn(B * H, -1.0f);

  auto t_in = make_tensor_ptr({T, B, I}, in.data());
  auto t_h0 = make_tensor_ptr({B, H}, h0.data());
  auto t_c0 = make_tensor_ptr({B, H}, c0.data());
  auto t_wih = make_tensor_ptr({4 * H, I}, w_ih.data());
  auto t_whh = make_tensor_ptr({4 * H, H}, w_hh.data());
  auto t_bih = make_tensor_ptr({4 * H}, b_ih.data());
  auto t_bhh = make_tensor_ptr({4 * H}, b_hh.data());
  auto t_out = make_tensor_ptr({T, B, H}, out.data());
  auto t_hn = make_tensor_ptr({B, H}, hn.data());
  auto t_cn = make_tensor_ptr({B, H}, cn.data());

  EValue ev[] = {EValue(*t_in), EValue(*t_h0), EValue(*t_c0), EValue(*t_wih),
                 EValue(*t_whh), EValue(*t_bih), EValue(*t_bhh),
                 EValue(*t_out), EValue(*t_hn), EValue(*t_cn)};
  EValue* args[] = {&ev[0], &ev[1], &ev[2], &ev[3], &ev[4],
                    &ev[5], &ev[6], &ev[7], &ev[8], &ev[9]};
  // The kernel allocates scratch via ctx.allocate_temp, so supply a temp allocator.
  // The restructured kernel allocates T*B*4H floats for the batched input
  // projection; size the arena generously so shape bumps don't hit the wall.
  std::vector<uint8_t> temp_buf(4 * 1024 * 1024);
  MemoryAllocator temp_alloc(
      static_cast<uint32_t>(temp_buf.size()), temp_buf.data());
  KernelRuntimeContext ctx(/*event_tracer=*/nullptr, &temp_alloc);
  fn(ctx, Span<EValue*>(args, 10));

  float c = 1.0f;
  for (int64_t t = 0; t < T; ++t) {
    c = 0.5f * c;
    const float h = 0.5f * std::tanh(c);
    for (int64_t j = 0; j < H; ++j) {
      if (std::abs(out[t * B * H + j] - h) > 1e-5f) {
        std::fprintf(stderr, "FAIL: out[t=%lld]=%f expected %f\n",
                     (long long)t, out[t * B * H + j], h);
        return 1;
      }
    }
  }
  std::printf("OK: etnp::lstm.out analytic recurrence correct\n");
  return 0;
}
