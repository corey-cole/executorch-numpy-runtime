// Hypothesis tests for the etnp::lstm perf issue, using raw XNNPACK FC ops
// (same API as examples/custom_kernels/lstm/xnn_linear.h):
//   H1: xnn_create_fully_connected_nc_f32 (weight packing) cost per call, vs H.
//   H2: T separate batch=1 runs vs ONE batch=T run of the same FC.
//   H3: threadpool effect on the batched run.
#include <chrono>
#include <cstdio>
#include <limits>
#include <vector>

#include <pthreadpool.h>
#include <xnnpack.h>

using clk = std::chrono::steady_clock;

static double ms_since(clk::time_point t0) {
  return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

static xnn_operator_t make_fc(size_t ic, size_t oc, const float* w, const float* b) {
  xnn_operator_t op = nullptr;
  xnn_create_fully_connected_nc_f32(
      ic, oc, ic, oc, w, b,
      -std::numeric_limits<float>::infinity(),
      +std::numeric_limits<float>::infinity(), 0, nullptr, &op);
  return op;
}

int main() {
  xnn_initialize(nullptr);
  const int REPS = 50;

  printf("%-6s %-6s | %-12s | %-14s %-14s %-14s | %-14s\n",
         "H", "T", "create_ms", "T x b1 (ms)", "1 x bT (ms)", "1xbT+tp (ms)", "GEMV/GEMM");
  for (size_t H : {32, 64, 128, 256}) {
    const size_t I = H, OC = 4 * H;
    std::vector<float> w_ih(OC * I, 0.01f), w_hh(OC * H, 0.01f), bias(OC, 0.1f);

    // H1: create cost (both projections, like the kernel does per execute).
    double create_ms = 0;
    for (int r = 0; r < REPS; ++r) {
      auto t0 = clk::now();
      xnn_operator_t a = make_fc(I, OC, w_ih.data(), bias.data());
      xnn_operator_t b = make_fc(H, OC, w_hh.data(), bias.data());
      create_ms += ms_since(t0);
      xnn_delete_operator(a);
      xnn_delete_operator(b);
    }
    create_ms /= REPS;

    for (size_t T : {16, 64, 256}) {
      std::vector<float> x(T * I, 0.5f), g(T * OC);
      xnn_operator_t fc = make_fc(I, OC, w_ih.data(), bias.data());

      // H2a: T separate batch=1 runs (what etnp_lstm.cpp does for the input proj).
      auto run = [&](size_t batch, const float* in, float* out, pthreadpool_t tp) {
        xnn_reshape_fully_connected_nc_f32(fc, batch, tp);
        xnn_setup_fully_connected_nc_f32(fc, in, out);
        xnn_run_operator(fc, tp);
      };
      // warmup
      for (size_t t = 0; t < T; ++t) run(1, x.data() + t * I, g.data() + t * OC, nullptr);

      double t_b1 = 0;
      for (int r = 0; r < REPS; ++r) {
        auto t0 = clk::now();
        for (size_t t = 0; t < T; ++t) run(1, x.data() + t * I, g.data() + t * OC, nullptr);
        t_b1 += ms_since(t0);
      }
      t_b1 /= REPS;

      // H2b: one batch=T run, single-threaded.
      run(T, x.data(), g.data(), nullptr);  // warmup
      double t_bT = 0;
      for (int r = 0; r < REPS; ++r) {
        auto t0 = clk::now();
        run(T, x.data(), g.data(), nullptr);
        t_bT += ms_since(t0);
      }
      t_bT /= REPS;

      // H3: one batch=T run with a threadpool (default size = hw threads).
      pthreadpool_t tp = pthreadpool_create(0);
      run(T, x.data(), g.data(), tp);  // warmup
      double t_bT_tp = 0;
      for (int r = 0; r < REPS; ++r) {
        auto t0 = clk::now();
        run(T, x.data(), g.data(), tp);
        t_bT_tp += ms_since(t0);
      }
      t_bT_tp /= REPS;
      pthreadpool_destroy(tp);
      xnn_delete_operator(fc);

      printf("%-6zu %-6zu | %10.4f   | %12.4f   %12.4f   %12.4f   | %10.1fx\n",
             H, T, create_ms, t_b1, t_bT, t_bT_tp, t_b1 / t_bT);
    }
  }
  return 0;
}
