// TSan gate for concurrent etnp::lstm.out. Two phases, each hammering the op
// from two threads:
//
//   Phase A (SHARED entries): both threads pass the SAME weight buffers, so
//     they resolve to the SAME XnnLinearCache entries and serialize on each
//     entry's run_mutex. Verifies that shared-cache reuse is race-free and
//     deterministic (both threads must produce identical output).
//
//   Phase B (DISTINCT entries): each thread passes its OWN weight buffers, so
//     they resolve to DISTINCT cache entries (distinct run_mutexes). Nothing in
//     the cache serializes them — this is the production parallel-Runtime
//     topology (one Runtime/weights per thread). The only shared resource left
//     is the process-wide singleton threadpool that the batched input
//     projection drives via get_pthreadpool(). Each thread's concurrent output
//     is checked bit-exact against a golden it produced solo beforehand, so a
//     pool-driven corruption would surface as a mismatch.
//
// HONESTY CAVEAT: TSan instruments only THIS binary's code (the cache, the
// kernel, our test). The prebuilt ExecuTorch / XNNPACK / pthreadpool libraries
// are NOT instrumented, so a clean TSan run proves our cache/kernel locking is
// race-free and that concurrent distinct-entry execution stays deterministic —
// it does NOT prove the singleton pthreadpool is safe to drive concurrently on
// the raw-pointer path. That is a property of ExecuTorch's shared pool (the
// XNNPACK delegate drives the same pool the same way); see the pool-use comment
// in examples/custom_kernels/lstm/etnp_lstm.cpp.
#include <atomic>
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

// One weight set (ih/hh projections + biases). Distinct instances => distinct
// buffer pointers => distinct XnnLinearCache keys/entries.
struct Weights {
  std::vector<float> wih, whh, bih, bhh;
  explicit Weights(float s)
      : wih(4 * H * I, 0.01f * s),
        whh(4 * H * H, 0.02f * s),
        bih(4 * H, 0.1f * s),
        bhh(4 * H, -0.1f * s) {}
};

// A start-gate so both threads reach the timing loop before either proceeds,
// maximizing overlap (a bare launch can let thread A finish before B starts).
std::atomic<int> g_arrived{0};
std::atomic<bool> g_go{false};

// Runs the op kIters times with the given weights, into out_final. When
// gate==true, waits at the start-gate so the two threads overlap; solo golden
// runs pass gate==false.
void worker(OpFunction fn, const Weights* w, std::vector<float>* out_final,
            bool gate) {
  std::vector<float> in(T * B * I, 0.5f), h0(B * H, 0.1f), c0(B * H, 0.2f);
  std::vector<float> out(T * B * H), hn(B * H), cn(B * H);
  auto t_in = make_tensor_ptr({T, B, I}, in.data());
  auto t_h0 = make_tensor_ptr({B, H}, h0.data());
  auto t_c0 = make_tensor_ptr({B, H}, c0.data());
  auto t_wih = make_tensor_ptr({4 * H, I}, const_cast<float*>(w->wih.data()));
  auto t_whh = make_tensor_ptr({4 * H, H}, const_cast<float*>(w->whh.data()));
  auto t_bih = make_tensor_ptr({4 * H}, const_cast<float*>(w->bih.data()));
  auto t_bhh = make_tensor_ptr({4 * H}, const_cast<float*>(w->bhh.data()));
  auto t_out = make_tensor_ptr({T, B, H}, out.data());
  auto t_hn = make_tensor_ptr({B, H}, hn.data());
  auto t_cn = make_tensor_ptr({B, H}, cn.data());
  EValue ev[] = {EValue(*t_in), EValue(*t_h0), EValue(*t_c0), EValue(*t_wih),
                 EValue(*t_whh), EValue(*t_bih), EValue(*t_bhh),
                 EValue(*t_out), EValue(*t_hn), EValue(*t_cn)};
  EValue* args[10];
  for (int i = 0; i < 10; ++i) args[i] = &ev[i];
  std::vector<uint8_t> temp_buf(1 * 1024 * 1024);
  if (gate) {
    g_arrived.fetch_add(1, std::memory_order_acq_rel);
    while (!g_go.load(std::memory_order_acquire)) { /* spin */ }
  }
  for (int it = 0; it < kIters; ++it) {
    MemoryAllocator temp_alloc(static_cast<uint32_t>(temp_buf.size()),
                               temp_buf.data());
    KernelRuntimeContext ctx(/*event_tracer=*/nullptr, &temp_alloc);
    fn(ctx, Span<EValue*>(args, 10));
  }
  *out_final = out;
}

// Releases the start-gate once both gated threads have arrived, then joins.
void run_gated(std::thread& ta, std::thread& tb) {
  while (g_arrived.load(std::memory_order_acquire) < 2) { /* spin */ }
  g_go.store(true, std::memory_order_release);
  ta.join();
  tb.join();
}

// Bit-exact float comparison is valid here because XNNPACK fully-connected
// parallelizes over output/batch rows, not the contraction — so each element's
// reduction order is thread-count-independent and reproducible. (This
// assumption is FC-specific; a kernel that parallelized a reduction would need
// a tolerance instead.)
bool identical(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i] != b[i]) return false;
  return true;
}
}  // namespace

int main() {
  executorch::runtime::runtime_init();
  auto fn_r =
      executorch::runtime::get_op_function_from_registry("etnp::lstm.out", {});
  if (!fn_r.ok()) { std::fprintf(stderr, "FAIL: not registered\n"); return 1; }
  OpFunction fn = fn_r.get();

  // Phase A: shared weights => shared cache entries (run_mutex-serialized).
  {
    Weights shared(1.0f);
    std::vector<float> out_a, out_b;
    g_arrived.store(0);
    g_go.store(false);
    std::thread ta(worker, fn, &shared, &out_a, true);
    std::thread tb(worker, fn, &shared, &out_b, true);
    run_gated(ta, tb);
    if (!identical(out_a, out_b)) {
      std::fprintf(stderr, "FAIL(A): shared-entry thread outputs diverge\n");
      return 1;
    }
  }

  // Phase B: distinct weights => distinct cache entries (NO cache-level
  // serialization) => the only shared resource is the batched projection's
  // singleton threadpool. Golden = each weight set run solo first; concurrent
  // output must reproduce it bit-exact.
  {
    Weights wa(1.0f), wb(2.5f);
    std::vector<float> gold_a, gold_b, out_a, out_b;
    worker(fn, &wa, &gold_a, /*gate=*/false);  // solo goldens
    worker(fn, &wb, &gold_b, /*gate=*/false);
    if (identical(gold_a, gold_b)) {
      std::fprintf(stderr, "FAIL(B): distinct weights gave identical output "
                           "(test would be vacuous)\n");
      return 1;
    }
    g_arrived.store(0);
    g_go.store(false);
    std::thread ta(worker, fn, &wa, &out_a, true);
    std::thread tb(worker, fn, &wb, &out_b, true);
    run_gated(ta, tb);
    if (!identical(out_a, gold_a) || !identical(out_b, gold_b)) {
      std::fprintf(stderr, "FAIL(B): concurrent distinct-entry output differs "
                           "from solo golden (shared-pool corruption?)\n");
      return 1;
    }
  }

  std::printf("OK: concurrent etnp::lstm.out race-free for shared AND distinct "
              "cache entries\n");
  return 0;
}
