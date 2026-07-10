// End-to-end LSTM kernel-structure comparison (B=1, I=H, PyTorch gate order):
//   CURRENT   : per-execute create; per-timestep: FC(x_t) + FC(h) + fused update; no tp.
//   RESTRUCT  : per-execute create; ONE batched FC over all T for the input projection
//               (threadpool), then per-timestep FC(h) + fused update.
//   RESTRUCT+C: same, but operator creation amortized (cached across executes).
// Reports median ms over ITERS, matching tools/bench_lstm.py methodology.
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
  xnn_create_fully_connected_nc_f32(
      ic, oc, ic, oc, w, b,
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

// Fused gate update for one timestep (B=1).
static void cell_update(size_t H, const float* aih, const float* ahh,
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

struct Weights {
  std::vector<float> w_ih, w_hh, b_ih, b_hh;
  Weights(size_t I, size_t H)
      : w_ih(4 * H * I, 0.01f), w_hh(4 * H * H, 0.02f),
        b_ih(4 * H, 0.1f), b_hh(4 * H, 0.05f) {}
};

// CURRENT kernel structure (mirrors etnp_lstm.cpp).
static void lstm_current(size_t T, size_t H, const Weights& W, const float* x,
                         float* h, float* c, float* out,
                         float* g_ih, float* g_hh) {
  const size_t I = H;
  xnn_operator_t ih = make_fc(I, 4 * H, W.w_ih.data(), W.b_ih.data());
  xnn_operator_t hh = make_fc(H, 4 * H, W.w_hh.data(), W.b_hh.data());
  for (size_t t = 0; t < T; ++t) {
    run_fc(ih, 1, x + t * I, g_ih, nullptr);
    run_fc(hh, 1, h, g_hh, nullptr);
    cell_update(H, g_ih, g_hh, c, h, out + t * H);
  }
  xnn_delete_operator(ih);
  xnn_delete_operator(hh);
}

// RESTRUCTURED: batched input projection (+tp), sequential recurrent part.
// If cached_ih/hh non-null, reuse them (simulates cross-execute op cache).
static void lstm_restruct(size_t T, size_t H, const Weights& W, const float* x,
                          float* h, float* c, float* out,
                          float* g_ih_all, float* g_hh, pthreadpool_t tp,
                          xnn_operator_t cached_ih, xnn_operator_t cached_hh) {
  const size_t I = H;
  xnn_operator_t ih = cached_ih ? cached_ih : make_fc(I, 4 * H, W.w_ih.data(), W.b_ih.data());
  xnn_operator_t hh = cached_hh ? cached_hh : make_fc(H, 4 * H, W.w_hh.data(), W.b_hh.data());
  run_fc(ih, T, x, g_ih_all, tp);  // ALL timesteps in one GEMM
  for (size_t t = 0; t < T; ++t) {
    run_fc(hh, 1, h, g_hh, nullptr);  // GEMV: too small to thread at these H
    cell_update(H, g_ih_all + t * 4 * H, g_hh, c, h, out + t * H);
  }
  if (!cached_ih) xnn_delete_operator(ih);
  if (!cached_hh) xnn_delete_operator(hh);
}

int main() {
  xnn_initialize(nullptr);
  pthreadpool_t tp = pthreadpool_create(0);
  const int ITERS = 30;

  printf("%-14s | %-11s %-11s %-13s | %-8s %-8s\n",
         "config", "current_ms", "restruct_ms", "restruct+c_ms", "spd(re)", "spd(re+c)");
  for (size_t H : {32, 64, 128}) {
    for (size_t T : {16, 64, 256}) {
      const size_t I = H;
      Weights W(I, H);
      std::vector<float> x(T * I, 0.5f), out(T * H);
      std::vector<float> h(H), c(H), g_ih(T * 4 * H), g_hh(4 * H);
      std::vector<float> out_ref(T * H);

      // correctness cross-check: current vs restructured must match bitwise-ish
      std::fill(h.begin(), h.end(), 0.1f); std::fill(c.begin(), c.end(), 0.2f);
      lstm_current(T, H, W, x.data(), h.data(), c.data(), out_ref.data(),
                   g_ih.data(), g_hh.data());
      std::fill(h.begin(), h.end(), 0.1f); std::fill(c.begin(), c.end(), 0.2f);
      lstm_restruct(T, H, W, x.data(), h.data(), c.data(), out.data(),
                    g_ih.data(), g_hh.data(), tp, nullptr, nullptr);
      float md = 0;
      for (size_t k = 0; k < T * H; ++k) md = std::max(md, std::abs(out[k] - out_ref[k]));

      auto median = [&](auto&& fn) {
        for (int r = 0; r < 5; ++r) fn();  // warmup
        std::vector<double> s(ITERS);
        for (int r = 0; r < ITERS; ++r) {
          std::fill(h.begin(), h.end(), 0.1f); std::fill(c.begin(), c.end(), 0.2f);
          auto t0 = clk::now();
          fn();
          s[r] = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        }
        std::nth_element(s.begin(), s.begin() + ITERS / 2, s.end());
        return s[ITERS / 2];
      };

      double cur = median([&] {
        lstm_current(T, H, W, x.data(), h.data(), c.data(), out.data(),
                     g_ih.data(), g_hh.data());
      });
      double res = median([&] {
        lstm_restruct(T, H, W, x.data(), h.data(), c.data(), out.data(),
                      g_ih.data(), g_hh.data(), tp, nullptr, nullptr);
      });
      xnn_operator_t cih = make_fc(I, 4 * H, W.w_ih.data(), W.b_ih.data());
      xnn_operator_t chh = make_fc(H, 4 * H, W.w_hh.data(), W.b_hh.data());
      double resc = median([&] {
        lstm_restruct(T, H, W, x.data(), h.data(), c.data(), out.data(),
                      g_ih.data(), g_hh.data(), tp, cih, chh);
      });
      xnn_delete_operator(cih);
      xnn_delete_operator(chh);

      printf("T%-4zu_H%-6zu | %9.3f   %9.3f   %11.3f   | %6.2fx  %6.2fx  (max|diff|=%.2e)\n",
             T, H, cur, res, resc, cur / res, cur / resc, md);
    }
  }
  pthreadpool_destroy(tp);
  return 0;
}
