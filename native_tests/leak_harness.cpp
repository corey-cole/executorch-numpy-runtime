// ASan/LSan leak harness over et_core (no Python). LSan reports unfreed allocations at
// exit; a leak -> non-zero exit. Model-agnostic: inputs derived from method_meta().
// Also runs every zero-input method (e.g. constant_methods emitting scalar
// int/double/bool or tensor EValues) so the scalar-output path is leak-gated;
// this is a no-op for models whose only method is forward(). Point it at
// tests/models/const_methods.pte to cover the scalar branch.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "et_core/et_core.h"

using namespace etnp;

static size_t dtype_size(int8_t st) {
  switch (st) { case 6: case 3: return 4; case 7: case 4: return 8;
    case 5: case 2: case 15: return 2; case 1: case 0: case 11: return 1; default: return 4; }
}

int main(int argc, char** argv) {
  const char* pte = (argc > 1) ? argv[1] : "tests/models/add.pte";
  const int outer = (argc > 2) ? std::atoi(argv[2]) : 500;
  const int fwd_per_load = 4;
  // Discover zero-input (constant) method names once. method_meta() probes each
  // output's TensorInfo, and ExecuTorch logs a line per non-tensor (scalar)
  // output, so doing this per-iteration would flood stderr for no added
  // coverage -- the method set is invariant across loads of the same .pte.
  std::vector<std::string> const_methods;
  {
    auto rt = Runtime::load_path(pte);
    for (const auto& mname : rt->method_names()) {
      if (mname == "forward") continue;                    // exercised below
      if (rt->method_meta(mname).inputs.empty()) const_methods.push_back(mname);
    }
  }
  for (int it = 0; it < outer; ++it) {
    auto rt = Runtime::load_path(pte);              // load/destroy balance
    auto meta = rt->method_meta("forward");
    std::vector<std::vector<uint8_t>> bufs(meta.inputs.size());
    std::vector<InputDesc> inputs;
    for (size_t i = 0; i < meta.inputs.size(); ++i) {
      if (meta.inputs[i].scalar_type < 0) continue;
      size_t count = 1; for (int64_t d : meta.inputs[i].shape) count *= (size_t)d;
      size_t bytes = count * dtype_size(meta.inputs[i].scalar_type);
      bufs[i].assign(bytes ? bytes : 1, 0);
      if (meta.inputs[i].scalar_type == 6) {        // float32 -> 1.0f
        float one = 1.0f;
        for (size_t b = 0; b + 4 <= bytes; b += 4) std::memcpy(bufs[i].data()+b, &one, 4);
      } else std::memset(bufs[i].data(), 1, bytes);
      inputs.push_back(InputDesc{bufs[i].data(), meta.inputs[i].shape,
                                 meta.inputs[i].scalar_type});
    }
    for (int f = 0; f < fwd_per_load; ++f) {
      auto res = rt->run_method("forward", inputs);  // per-forward allocations
      (void)res.outputs();
    }
    // Exercise every zero-input method: constant_methods surface as scalar
    // (int/double/bool) or tensor EValues through run_method's output path.
    for (const auto& mname : const_methods) {
      auto res = rt->run_method(mname, {});
      (void)res.outputs();
    }
  }
  std::printf("leak_harness: %d iters OK\n", outer);
  return 0;
}
