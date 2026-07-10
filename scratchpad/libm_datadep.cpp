#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
using clk = std::chrono::steady_clock;
int main() {
  const size_t N = 512, REPS = 20000;
  std::vector<float> cst(N, 0.64f), rnd(N), out(N);
  std::mt19937 g(0); std::normal_distribution<float> d(0.f, 2.f);
  for (auto& v : rnd) v = d(g);
  for (auto* src : {&cst, &rnd}) {
    auto t0 = clk::now();
    float acc = 0;
    for (size_t r = 0; r < REPS; ++r)
      for (size_t j = 0; j < N; ++j) {
        float s = 1.f/(1.f+std::exp(-(*src)[j]));
        acc += s + std::tanh((*src)[j]);
      }
    double ms = std::chrono::duration<double,std::milli>(clk::now()-t0).count();
    printf("%s: %.1f ns per (sigmoid+tanh) pair  (acc=%g)\n",
           src==&cst?"const ":"random", ms*1e6/(REPS*N), acc);
  }
}
