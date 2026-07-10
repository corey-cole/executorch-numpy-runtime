// TSan gate: two threads invoke etnp::lstm.out concurrently with the SAME
// weight buffers (=> shared XnnLinearCache entries) and private activations.
// Passes iff TSan reports no race and both threads produce identical output.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

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

namespace {
constexpr int64_t T = 4, B = 1, I = 8, H = 8;
constexpr int kIters = 200;

// Shared, read-only weights (the contended cache keys).
std::vector<float> g_wih(4 * H * I, 0.01f), g_whh(4 * H * H, 0.02f);
std::vector<float> g_bih(4 * H, 0.1f), g_bhh(4 * H, -0.1f);

void worker(OpFunction fn, std::vector<float>* out_final) {
  std::vector<float> in(T * B * I, 0.5f), h0(B * H, 0.1f), c0(B * H, 0.2f);
  std::vector<float> out(T * B * H), hn(B * H), cn(B * H);
  auto t_in = make_tensor_ptr({T, B, I}, in.data());
  auto t_h0 = make_tensor_ptr({B, H}, h0.data());
  auto t_c0 = make_tensor_ptr({B, H}, c0.data());
  auto t_wih = make_tensor_ptr({4 * H, I}, g_wih.data());
  auto t_whh = make_tensor_ptr({4 * H, H}, g_whh.data());
  auto t_bih = make_tensor_ptr({4 * H}, g_bih.data());
  auto t_bhh = make_tensor_ptr({4 * H}, g_bhh.data());
  auto t_out = make_tensor_ptr({T, B, H}, out.data());
  auto t_hn = make_tensor_ptr({B, H}, hn.data());
  auto t_cn = make_tensor_ptr({B, H}, cn.data());
  EValue ev[] = {EValue(*t_in), EValue(*t_h0), EValue(*t_c0), EValue(*t_wih),
                 EValue(*t_whh), EValue(*t_bih), EValue(*t_bhh),
                 EValue(*t_out), EValue(*t_hn), EValue(*t_cn)};
  EValue* args[10];
  for (int i = 0; i < 10; ++i) args[i] = &ev[i];
  std::vector<uint8_t> temp_buf(1 * 1024 * 1024);
  for (int it = 0; it < kIters; ++it) {
    MemoryAllocator temp_alloc(static_cast<uint32_t>(temp_buf.size()),
                               temp_buf.data());
    KernelRuntimeContext ctx(/*event_tracer=*/nullptr, &temp_alloc);
    fn(ctx, Span<EValue*>(args, 10));
  }
  *out_final = out;
}
}  // namespace

int main() {
  executorch::runtime::runtime_init();
  auto fn_r =
      executorch::runtime::get_op_function_from_registry("etnp::lstm.out", {});
  if (!fn_r.ok()) { std::fprintf(stderr, "FAIL: not registered\n"); return 1; }
  OpFunction fn = fn_r.get();

  std::vector<float> out_a, out_b;
  std::thread ta(worker, fn, &out_a);
  std::thread tb(worker, fn, &out_b);
  ta.join();
  tb.join();
  for (size_t i = 0; i < out_a.size(); ++i) {
    if (out_a[i] != out_b[i]) {
      std::fprintf(stderr, "FAIL: thread outputs diverge at %zu\n", i);
      return 1;
    }
  }
  std::printf("OK: concurrent etnp::lstm.out via shared cache is race-free\n");
  return 0;
}
