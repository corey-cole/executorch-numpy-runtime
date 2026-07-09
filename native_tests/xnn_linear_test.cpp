// Proves we can drive XNNPACK's f32 fully-connected operator directly (outside the
// ExecuTorch delegate) and get input @ weight^T + bias. Oracle: a naive triple loop.
#include <cmath>
#include <cstdio>
#include <vector>

#include "xnn_linear.h"

using etnp::XnnLinear;
using etnp::ensure_xnn_initialized;
using executorch::runtime::Error;

int main() {
  if (ensure_xnn_initialized() != Error::Ok) {
    std::fprintf(stderr, "FAIL: xnn_initialize\n");
    return 1;
  }
  const size_t IC = 3, OC = 4, B = 2;
  // weight [OC, IC], row-major (PyTorch F.linear layout)
  const std::vector<float> W = {1, 2, 3,  0, 1, 0,  -1, 0, 1,  2, -1, 0};
  const std::vector<float> bias = {0.5f, -0.5f, 1.0f, 0.0f};
  const std::vector<float> in = {1, 0, -1,   2, 1, 0}; // [B, IC]

  auto lin = XnnLinear::create(IC, OC, W.data(), bias.data());
  if (!lin.ok()) { std::fprintf(stderr, "FAIL: create\n"); return 1; }

  std::vector<float> out(B * OC, 0.0f);
  if (lin->run(B, in.data(), out.data(), /*tp=*/nullptr) != Error::Ok) {
    std::fprintf(stderr, "FAIL: run\n"); return 1;
  }

  // Oracle: out[b,o] = bias[o] + sum_k in[b,k]*W[o,k]
  for (size_t b = 0; b < B; ++b) {
    for (size_t o = 0; o < OC; ++o) {
      float ref = bias[o];
      for (size_t k = 0; k < IC; ++k) ref += in[b * IC + k] * W[o * IC + k];
      const float got = out[b * OC + o];
      if (std::abs(got - ref) > 1e-5f) {
        std::fprintf(stderr, "FAIL: out[%zu,%zu]=%f ref=%f\n", b, o, got, ref);
        return 1;
      }
    }
  }
  std::printf("OK: XnnLinear matches naive matmul\n");
  return 0;
}
