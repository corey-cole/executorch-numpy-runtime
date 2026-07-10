// lstm_cell_update vs a double-precision scalar reference, over randomized
// 256-step chains (state feeds back, so per-step error must not compound).
// Covers the scalar SIMD-tail path (H=13) and B>1.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "lstm_cell.h"

namespace {
void cell_ref(std::size_t B, std::size_t H, const float* g_ih_t,
              const float* g_hh, double* c, double* h, double* out_t) {
  auto sg = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
  for (std::size_t b = 0; b < B; ++b) {
    const float* aih = g_ih_t + b * 4 * H;
    const float* ahh = g_hh + b * 4 * H;
    for (std::size_t j = 0; j < H; ++j) {
      const double ig = sg((double)aih[j] + ahh[j]);
      const double fg = sg((double)aih[H + j] + ahh[H + j]);
      const double gg = std::tanh((double)aih[2 * H + j] + ahh[2 * H + j]);
      const double og = sg((double)aih[3 * H + j] + ahh[3 * H + j]);
      const double cn = fg * c[b * H + j] + ig * gg;
      const double hh = og * std::tanh(cn);
      c[b * H + j] = cn; h[b * H + j] = hh; out_t[b * H + j] = hh;
    }
  }
}
}  // namespace

int main() {
  const std::size_t T = 256;
  const double TOL = 1e-5;
  std::mt19937 rng(0);
  std::normal_distribution<float> dist(0.f, 2.5f);
  int failures = 0;
  for (std::size_t H : {std::size_t(13), std::size_t(32), std::size_t(128)}) {
    for (std::size_t B : {std::size_t(1), std::size_t(3)}) {
      std::vector<float> g_ih(T * B * 4 * H), g_hh(T * B * 4 * H);
      for (auto& v : g_ih) v = dist(rng);
      for (auto& v : g_hh) v = dist(rng);
      std::vector<float> c(B * H, 0.2f), h(B * H, 0.1f), out(T * B * H);
      std::vector<double> cr(B * H, 0.2), hr(B * H, 0.1), outr(T * B * H);
      for (std::size_t t = 0; t < T; ++t) {
        etnp::lstm_cell_update(B, H, g_ih.data() + t * B * 4 * H,
                               g_hh.data() + t * B * 4 * H, c.data(), h.data(),
                               out.data() + t * B * H);
        cell_ref(B, H, g_ih.data() + t * B * 4 * H,
                 g_hh.data() + t * B * 4 * H, cr.data(), hr.data(),
                 outr.data() + t * B * H);
      }
      double m = 0;
      for (std::size_t k = 0; k < T * B * H; ++k)
        m = std::max(m, std::abs((double)out[k] - outr[k]));
      for (std::size_t k = 0; k < B * H; ++k) {
        m = std::max(m, std::abs((double)c[k] - cr[k]));
        m = std::max(m, std::abs((double)h[k] - hr[k]));
      }
      std::printf("H=%zu B=%zu: max|diff| = %.3e %s\n", H, B, m,
                  m <= TOL ? "OK" : "FAIL");
      if (m > TOL) ++failures;
    }
  }
  if (failures) { std::fprintf(stderr, "FAIL: %d config(s)\n", failures); return 1; }
  std::printf("OK: lstm_cell_update matches f64 reference (tol %.0e)\n", TOL);
  return 0;
}
