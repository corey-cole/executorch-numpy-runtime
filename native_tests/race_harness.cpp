// TSan race harness over et_core (no Python). Build with -fsanitize=thread; a data race ->
// TSan report -> non-zero exit (TSan's default exitcode). Complements the LSan leak gate:
// LSan proves no leaks, this proves race-freedom of et_core's synchronization.
//
// Two scenarios:
//   A) many threads sharing ONE Runtime, hammering run_method + method_meta + method_names.
//      This is the surface the per-Runtime mutex must serialize (the F2 metadata-race class).
//   B) many threads each with their OWN Runtime — independent instances must not interfere.
//
// NOTE: the prebuilt ExecuTorch libs are NOT TSan-instrumented. TSan sees our synchronization
// (via pthread interceptors) but not plain memory accesses inside ExecuTorch/XNNPACK/pthreadpool,
// so this gate verifies race-freedom of OUR locking, not races internal to the runtime. Use
// native_tests/tsan_suppressions.txt (TSAN_OPTIONS=suppressions=...) to quiet known-noisy
// uninstrumented frames. See the README "Race QA gate (TSan)" section.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include "et_core/et_core.h"

using namespace etnp;

static size_t dtype_size(int8_t st) {
  switch (st) { case 6: case 3: return 4; case 7: case 4: return 8;
    case 5: case 2: case 15: return 2; case 1: case 0: case 11: return 1; default: return 4; }
}

// 1-filled inputs derived from a method's metadata; owns the backing buffers (model-agnostic).
struct Inputs {
  std::vector<std::vector<uint8_t>> bufs;
  std::vector<InputDesc> descs;
};
static Inputs make_inputs(const MethodMeta& meta) {
  Inputs in;
  in.bufs.resize(meta.inputs.size());
  for (size_t i = 0; i < meta.inputs.size(); ++i) {
    if (meta.inputs[i].scalar_type < 0) continue;
    size_t count = 1; for (int64_t d : meta.inputs[i].shape) count *= (size_t)d;
    size_t bytes = count * dtype_size(meta.inputs[i].scalar_type);
    in.bufs[i].assign(bytes ? bytes : 1, 0);
    if (meta.inputs[i].scalar_type == 6) {        // float32 -> 1.0f
      float one = 1.0f;
      for (size_t b = 0; b + 4 <= bytes; b += 4) std::memcpy(in.bufs[i].data() + b, &one, 4);
    } else std::memset(in.bufs[i].data(), 1, bytes);
    in.descs.push_back(InputDesc{in.bufs[i].data(), meta.inputs[i].shape,
                                 meta.inputs[i].scalar_type});
  }
  return in;
}

int main(int argc, char** argv) {
  const char* pte = (argc > 1) ? argv[1] : "tests/models/add.pte";
  const int nthreads = (argc > 2) ? std::atoi(argv[2]) : 8;
  const int iters = (argc > 3) ? std::atoi(argv[3]) : 200;
  std::atomic<int> empty_outputs{0};  // gross-breakage sanity, model-agnostic

  // Scenario A: one SHARED Runtime; concurrent execute + metadata (the mutex-protected surface).
  // Deliberately do NOT pre-load "forward" on the main thread — each thread queries method_meta
  // itself so the threads race on the runtime's lazy first-load of the method, which is exactly
  // the F2 surface (method_meta/method_names mutate Module state concurrently with run_method).
  {
    auto rt = Runtime::load_path(pte);
    std::vector<std::thread> ts;
    for (int t = 0; t < nthreads; ++t) {
      ts.emplace_back([&] {
        Inputs in = make_inputs(rt->method_meta("forward"));  // concurrent lazy-load path
        (void)rt->method_names();
        for (int i = 0; i < iters; ++i) {
          auto res = rt->run_method("forward", in.descs);
          if (res.outputs().empty()) empty_outputs.fetch_add(1, std::memory_order_relaxed);
          if ((i & 7) == 0) (void)rt->method_meta("forward");
        }
      });
    }
    for (auto& th : ts) th.join();
  }

  // Scenario B: each thread owns its OWN Runtime (independent instances).
  {
    std::vector<std::thread> ts;
    for (int t = 0; t < nthreads; ++t) {
      ts.emplace_back([&] {
        auto rt = Runtime::load_path(pte);
        auto meta = rt->method_meta("forward");
        Inputs in = make_inputs(meta);
        for (int i = 0; i < iters; ++i) {
          auto res = rt->run_method("forward", in.descs);
          if (res.outputs().empty()) empty_outputs.fetch_add(1, std::memory_order_relaxed);
        }
      });
    }
    for (auto& th : ts) th.join();
  }

  if (empty_outputs.load() != 0) {
    std::printf("race_harness: FAIL - %d runs produced no outputs\n", empty_outputs.load());
    return 1;
  }
  std::printf("race_harness: %d threads x %d iters (shared + per-thread Runtime) OK\n",
              nthreads, iters);
  return 0;
}
