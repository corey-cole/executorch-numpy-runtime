// Per-component cost of the restructured LSTM at each config:
//   input GEMM (batched, tp) | T recurrent GEMVs | T scalar cell updates |
//   T vectorized-ish cell updates (fast sigmoid/tanh, no libm)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include <pthreadpool.h>
#include <xnnpack.h>

using clk = std::chrono::steady_clock;

static xnn_operator_t make_fc(size_t ic, size_t oc, const float* w, const float* b) {
  xnn_operator_t op = nullptr;
  xnn_create_fully_connected_nc_f32(ic, oc, ic, oc, w, b,
      -std::numeric_limits<float>::infinity(),
      +std::numeric_limits<float>::infinity(), 0, nullptr, &op);
  return op;
}
static void run_fc(xnn_operator_t op, size_t batch, const float* in, float* out,
                   pthreadpool_t tp) {
  xnn_reshape_fully_connected_nc_f32(op, batch, tp);
  xnn_setup_fully_connected_nc_f32(op, in, out);
  xnn_run_operator(op, tp);
}
static inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

static void cell_scalar(size_t H, const float* aih, const float* ahh,
                        float* c, float* h, float* out_t) {
  for (size_t j = 0; j < H; ++j) {
    const float ig = sigmoidf(aih[0 * H + j] + ahh[0 * H + j]);
    const float fg = sigmoidf(aih[1 * H + j] + ahh[1 * H + j]);
    const float gg = std::tanh(aih[2 * H + j] + ahh[2 * H + j]);
    const float og = sigmoidf(aih[3 * H + j] + ahh[3 * H + j]);
    const float cnew = fg * c[j] + ig * gg;
    const float hnew = og * std::tanh(cnew);
    c[j] = cnew; h[j] = hnew; out_t[j] = hnew;
  }
}

// Branch-free polynomial tanh (|err| < ~1e-6 rel on [-9,9]), auto-vectorizable.
static inline float fast_tanh(float x) {
  x = std::max(-9.0f, std::min(9.0f, x));
  const float x2 = x * x;
  float p = 2.76076847742355e-16f;
  p = p * x2 + -1.09410437460867e-14f;
  p = p * x2 + 3.03186217236766e-13f;
  p = p * x2 + -1.38228809955933e-11f;
  p = p * x2 + 8.57027868358909e-10f;
  p = p * x2 + 1.48572235717979e-08f;
  p = p * x2 + 6.37261928875436e-04f;
  p = p * x2 + 4.89352455891786e-03f;
  p = p * x;
  float q = 1.19825839466702e-06f;
  q = q * x2 + 1.18534705686654e-04f;
  q = q * x2 + 2.26843463243900e-03f;
  q = q * x2 + 4.89352518554385e-03f;
  return p / q;
}
static inline float fast_sigmoid(float x) { return 0.5f * fast_tanh(0.5f * x) + 0.5f; }

static void cell_fast(size_t H, const float* aih, const float* ahh,
                      float* c, float* h, float* out_t) {
  for (size_t j = 0; j < H; ++j) {
    const float ig = fast_sigmoid(aih[0 * H + j] + ahh[0 * H + j]);
    const float fg = fast_sigmoid(aih[1 * H + j] + ahh[1 * H + j]);
    const float gg = fast_tanh(aih[2 * H + j] + ahh[2 * H + j]);
    const float og = fast_sigmoid(aih[3 * H + j] + ahh[3 * H + j]);
    const float cnew = fg * c[j] + ig * gg;
    const float hnew = og * fast_tanh(cnew);
    c[j] = cnew; h[j] = hnew; out_t[j] = hnew;
  }
}

template <typename F>
static double median_ms(F&& fn, int iters = 30) {
  for (int r = 0; r < 5; ++r) fn();
  std::vector<double> s(iters);
  for (int r = 0; r < iters; ++r) {
    auto t0 = clk::now();
    fn();
    s[r] = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
  }
  std::nth_element(s.begin(), s.begin() + iters / 2, s.end());
  return s[iters / 2];
}

int main() {
  xnn_initialize(nullptr);
  pthreadpool_t tp = pthreadpool_create(0);

  printf("%-14s | %-10s %-10s %-12s %-10s | %-10s\n",
         "config", "inGEMM_ms", "recGEMV_ms", "cell_scal_ms", "cell_fast", "acc(fast)");
  for (size_t H : {32, 64, 128}) {
    for (size_t T : {16, 64, 256}) {
      const size_t I = H;
      std::vector<float> w_ih(4 * H * I, 0.01f), w_hh(4 * H * H, 0.02f),
          b(4 * H, 0.1f);
      std::vector<float> x(T * I, 0.5f), g_all(T * 4 * H), g_hh(4 * H);
      std::vector<float> h(H, 0.1f), c(H, 0.2f), out(T * H);

      xnn_operator_t ih = make_fc(I, 4 * H, w_ih.data(), b.data());
      xnn_operator_t hh = make_fc(H, 4 * H, w_hh.data(), b.data());

      double t_in = median_ms([&] { run_fc(ih, T, x.data(), g_all.data(), tp); });
      double t_rec = median_ms([&] {
        for (size_t t = 0; t < T; ++t) run_fc(hh, 1, h.data(), g_hh.data(), nullptr);
      });
      double t_cs = median_ms([&] {
        for (size_t t = 0; t < T; ++t)
          cell_scalar(H, g_all.data() + t * 4 * H, g_hh.data(), c.data(), h.data(),
                      out.data() + t * H);
      });
      // accuracy check fast vs scalar on same inputs
      std::vector<float> c1(H, 0.2f), h1(H, 0.1f), o1(H), c2(H, 0.2f), h2(H, 0.1f), o2(H);
      cell_scalar(H, g_all.data(), g_hh.data(), c1.data(), h1.data(), o1.data());
      cell_fast(H, g_all.data(), g_hh.data(), c2.data(), h2.data(), o2.data());
      float md = 0;
      for (size_t j = 0; j < H; ++j) md = std::max(md, std::abs(o1[j] - o2[j]));

      double t_cf = median_ms([&] {
        for (size_t t = 0; t < T; ++t)
          cell_fast(H, g_all.data() + t * 4 * H, g_hh.data(), c.data(), h.data(),
                    out.data() + t * H);
      });

      printf("T%-4zu_H%-6zu | %8.4f   %8.4f   %10.4f   %8.4f   | %.2e\n",
             T, H, t_in, t_rec, t_cs, t_cf, md);
      xnn_delete_operator(ih);
      xnn_delete_operator(hh);
    }
  }
  pthreadpool_destroy(tp);
  return 0;
}
