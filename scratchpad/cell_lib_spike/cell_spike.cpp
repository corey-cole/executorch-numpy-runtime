// Library spike for the LSTM fused cell update: scalar libm vs hand-rolled poly
// vs Eigen fused expressions vs Highway (static + dynamic dispatch).
// Reports median per-step time and max abs error vs a double-precision reference.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <Eigen/Core>

using clk = std::chrono::steady_clock;

namespace spike {
void cell_hwy_dynamic(size_t H, const float* aih, const float* ahh, float* c,
                      float* h, float* out_t);
void cell_hwy_static(size_t H, const float* aih, const float* ahh, float* c,
                     float* h, float* out_t);
const char* hwy_dynamic_target_name();
}

static inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

static void cell_scalar(size_t H, const float* aih, const float* ahh,
                        float* c, float* h, float* out_t) {
  for (size_t j = 0; j < H; ++j) {
    const float ig = sigmoidf(aih[j] + ahh[j]);
    const float fg = sigmoidf(aih[H + j] + ahh[H + j]);
    const float gg = std::tanh(aih[2 * H + j] + ahh[2 * H + j]);
    const float og = sigmoidf(aih[3 * H + j] + ahh[3 * H + j]);
    const float cn = fg * c[j] + ig * gg;
    const float hh = og * std::tanh(cn);
    c[j] = cn; h[j] = hh; out_t[j] = hh;
  }
}

// double-precision reference for accuracy
static void cell_ref(size_t H, const float* aih, const float* ahh,
                     double* c, double* h, double* out_t) {
  for (size_t j = 0; j < H; ++j) {
    auto sg = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
    const double ig = sg((double)aih[j] + ahh[j]);
    const double fg = sg((double)aih[H + j] + ahh[H + j]);
    const double gg = std::tanh((double)aih[2 * H + j] + ahh[2 * H + j]);
    const double og = sg((double)aih[3 * H + j] + ahh[3 * H + j]);
    const double cn = fg * c[j] + ig * gg;
    const double hh = og * std::tanh(cn);
    c[j] = cn; h[j] = hh; out_t[j] = hh;
  }
}

// hand-rolled branch-free rational tanh (the earlier probe poly)
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

static void cell_poly(size_t H, const float* aih, const float* ahh,
                      float* c, float* h, float* out_t) {
  for (size_t j = 0; j < H; ++j) {
    const float ig = fast_sigmoid(aih[j] + ahh[j]);
    const float fg = fast_sigmoid(aih[H + j] + ahh[H + j]);
    const float gg = fast_tanh(aih[2 * H + j] + ahh[2 * H + j]);
    const float og = fast_sigmoid(aih[3 * H + j] + ahh[3 * H + j]);
    const float cn = fg * c[j] + ig * gg;
    const float hh = og * fast_tanh(cn);
    c[j] = cn; h[j] = hh; out_t[j] = hh;
  }
}

static void cell_eigen(size_t H, const float* aih, const float* ahh,
                       float* c, float* h, float* out_t) {
  using namespace Eigen;
  Map<const ArrayXf> ai(aih, H), af(aih + H, H), ag(aih + 2 * H, H), ao(aih + 3 * H, H);
  Map<const ArrayXf> bi(ahh, H), bf(ahh + H, H), bg(ahh + 2 * H, H), bo(ahh + 3 * H, H);
  Map<ArrayXf> C(c, H), Hm(h, H), O(out_t, H);
  C = (af + bf).logistic() * C + (ai + bi).logistic() * (ag + bg).tanh();
  O = (ao + bo).logistic() * C.tanh();
  Hm = O;
}

template <typename F>
static double bench_step_us(F&& cell, size_t T, size_t H,
                            const std::vector<float>& g_ih,
                            const std::vector<float>& g_hh,
                            std::vector<float>& c, std::vector<float>& h,
                            std::vector<float>& out) {
  const int ITERS = 30;
  auto once = [&] {
    std::fill(c.begin(), c.end(), 0.2f);
    std::fill(h.begin(), h.end(), 0.1f);
    for (size_t t = 0; t < T; ++t)
      cell(H, g_ih.data() + t * 4 * H, g_hh.data() + t * 4 * H,
           c.data(), h.data(), out.data() + t * H);
  };
  for (int r = 0; r < 5; ++r) once();
  std::vector<double> s(ITERS);
  for (int r = 0; r < ITERS; ++r) {
    auto t0 = clk::now();
    once();
    s[r] = std::chrono::duration<double, std::micro>(clk::now() - t0).count();
  }
  std::nth_element(s.begin(), s.begin() + ITERS / 2, s.end());
  return s[ITERS / 2] / T;
}

template <typename F>
static double max_err(F&& cell, size_t T, size_t H,
                      const std::vector<float>& g_ih, const std::vector<float>& g_hh) {
  std::vector<float> c(H, 0.2f), h(H, 0.1f), out(T * H);
  std::vector<double> cr(H, 0.2), hr(H, 0.1), outr(T * H);
  for (size_t t = 0; t < T; ++t) {
    cell(H, g_ih.data() + t * 4 * H, g_hh.data() + t * 4 * H,
         c.data(), h.data(), out.data() + t * H);
    cell_ref(H, g_ih.data() + t * 4 * H, g_hh.data() + t * 4 * H,
             cr.data(), hr.data(), outr.data() + t * H);
  }
  double m = 0;
  for (size_t k = 0; k < T * H; ++k) m = std::max(m, std::abs(out[k] - outr[k]));
  return m;
}

int main() {
  printf("hwy dynamic target: %s | Eigen SIMD: %d-wide float\n",
         spike::hwy_dynamic_target_name(),
         (int)Eigen::internal::packet_traits<float>::size);
  printf("%-10s | %-9s %-9s %-9s %-9s %-9s | per-step us (T=256 chain)\n",
         "H", "scalar", "poly", "eigen", "hwy_stat", "hwy_dyn");
  const size_t T = 256;
  std::mt19937 rng(0);
  std::normal_distribution<float> dist(0.f, 2.5f);
  for (size_t H : {32, 64, 128}) {
    std::vector<float> g_ih(T * 4 * H), g_hh(T * 4 * H);
    for (auto& v : g_ih) v = dist(rng);
    for (auto& v : g_hh) v = dist(rng);
    std::vector<float> c(H), h(H), out(T * H);

    double t_sc = bench_step_us(cell_scalar, T, H, g_ih, g_hh, c, h, out);
    double t_po = bench_step_us(cell_poly, T, H, g_ih, g_hh, c, h, out);
    double t_ei = bench_step_us(cell_eigen, T, H, g_ih, g_hh, c, h, out);
    double t_hs = bench_step_us(spike::cell_hwy_static, T, H, g_ih, g_hh, c, h, out);
    double t_hd = bench_step_us(spike::cell_hwy_dynamic, T, H, g_ih, g_hh, c, h, out);
    printf("%-10zu | %9.3f %9.3f %9.3f %9.3f %9.3f\n", H, t_sc, t_po, t_ei, t_hs, t_hd);
    printf("  max err vs f64 ref: scalar %.1e  poly %.1e  eigen %.1e  hwy %.1e\n",
           max_err(cell_scalar, T, H, g_ih, g_hh),
           max_err(cell_poly, T, H, g_ih, g_hh),
           max_err(cell_eigen, T, H, g_ih, g_hh),
           max_err(spike::cell_hwy_dynamic, T, H, g_ih, g_hh));
  }
  return 0;
}
