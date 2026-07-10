// Parity of etnp::lstm.out against torch.nn.LSTM (golden fixture). No PyTorch at run time.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/runtime.h>

#include "lstm_golden.h"

using executorch::extension::make_tensor_ptr;
using executorch::runtime::EValue;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::OpFunction;
using executorch::runtime::Span;
namespace g = lstm_golden;

static bool close(const std::vector<float>& got, const float* ref, int n, const char* what) {
  for (int i = 0; i < n; ++i) {
    const float a = got[i], b = ref[i];
    if (std::abs(a - b) > 1e-4f + 1e-4f * std::abs(b)) {
      std::fprintf(stderr, "FAIL %s[%d]: got %f ref %f\n", what, i, a, b);
      return false;
    }
  }
  return true;
}

int main() {
  executorch::runtime::runtime_init();
  auto fn_r = executorch::runtime::get_op_function_from_registry("etnp::lstm.out", {});
  if (!fn_r.ok()) { std::fprintf(stderr, "FAIL: lstm not registered\n"); return 1; }
  OpFunction fn = fn_r.get();

  std::vector<float> in(g::input, g::input + g::T * g::B * g::I);
  std::vector<float> h0(g::h0, g::h0 + g::B * g::H), c0(g::c0, g::c0 + g::B * g::H);
  std::vector<float> wih(g::w_ih, g::w_ih + 4 * g::H * g::I);
  std::vector<float> whh(g::w_hh, g::w_hh + 4 * g::H * g::H);
  std::vector<float> bih(g::b_ih, g::b_ih + 4 * g::H), bhh(g::b_hh, g::b_hh + 4 * g::H);
  std::vector<float> out(g::T * g::B * g::H, 0.f), hn(g::B * g::H, 0.f), cn(g::B * g::H, 0.f);

  auto t_in = make_tensor_ptr({g::T, g::B, g::I}, in.data());
  auto t_h0 = make_tensor_ptr({g::B, g::H}, h0.data());
  auto t_c0 = make_tensor_ptr({g::B, g::H}, c0.data());
  auto t_wih = make_tensor_ptr({4 * g::H, g::I}, wih.data());
  auto t_whh = make_tensor_ptr({4 * g::H, g::H}, whh.data());
  auto t_bih = make_tensor_ptr({4 * g::H}, bih.data());
  auto t_bhh = make_tensor_ptr({4 * g::H}, bhh.data());
  auto t_out = make_tensor_ptr({g::T, g::B, g::H}, out.data());
  auto t_hn = make_tensor_ptr({g::B, g::H}, hn.data());
  auto t_cn = make_tensor_ptr({g::B, g::H}, cn.data());

  EValue ev[] = {EValue(*t_in), EValue(*t_h0), EValue(*t_c0), EValue(*t_wih), EValue(*t_whh),
                 EValue(*t_bih), EValue(*t_bhh), EValue(*t_out), EValue(*t_hn), EValue(*t_cn)};
  EValue* args[10];
  for (int i = 0; i < 10; ++i) args[i] = &ev[i];
  // Kernel allocates gate scratch via ctx.allocate_temp — supply a temp allocator
  // (same wiring as lstm_kernel_test.cpp; a bare KernelRuntimeContext fails at run).
  // The restructured kernel allocates T*B*4H floats for the batched input
  // projection; size the arena generously so shape bumps don't hit the wall.
  std::vector<uint8_t> temp_buf(4 * 1024 * 1024);
  MemoryAllocator temp_alloc(static_cast<uint32_t>(temp_buf.size()), temp_buf.data());
  KernelRuntimeContext ctx(/*event_tracer=*/nullptr, &temp_alloc);
  fn(ctx, Span<EValue*>(args, 10));

  bool ok = close(out, g::output, g::T * g::B * g::H, "output")
          & close(hn, g::hn, g::B * g::H, "hn")
          & close(cn, g::cn, g::B * g::H, "cn");
  if (!ok) return 1;
  std::printf("OK: etnp::lstm.out matches torch.nn.LSTM (rtol/atol 1e-4)\n");
  return 0;
}
