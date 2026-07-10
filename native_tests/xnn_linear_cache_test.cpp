// XnnLinearCache: hit/miss accounting, fingerprint invalidation on content
// change at the same address, LRU bound, and numeric correctness of cached runs.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "xnn_linear_cache.h"

using etnp::XnnLinearCache;

static bool expect(bool cond, const char* what) {
  if (!cond) std::fprintf(stderr, "FAIL: %s\n", what);
  return cond;
}

// y = W x + b for the 2x3 fixture below, computed by hand in the asserts.
int main() {
  if (etnp::ensure_xnn_initialized() != etnp::Error::Ok) {
    std::fprintf(stderr, "FAIL: xnn init\n");
    return 1;
  }
  XnnLinearCache::clear();
  bool ok = true;

  // Fixture: in_ch=3, out_ch=2, W=[[1,2,3],[4,5,6]], b=[0.5,-0.5], x=[1,1,1].
  std::vector<float> w = {1, 2, 3, 4, 5, 6}, b = {0.5f, -0.5f}, x = {1, 1, 1};
  std::vector<float> y(2, 0.f);

  // 1) miss then hit on identical key.
  auto e1 = XnnLinearCache::get(3, 2, w.data(), b.data());
  ok &= expect(e1 != nullptr, "create");
  auto e2 = XnnLinearCache::get(3, 2, w.data(), b.data());
  ok &= expect(e1.get() == e2.get(), "second get returns same entry");
  auto s = XnnLinearCache::stats();
  ok &= expect(s.hits == 1 && s.misses == 1 && s.size == 1, "stats after hit");
  ok &= expect(etnp::run_cached(*e2, 1, x.data(), y.data(), nullptr) ==
                   etnp::Error::Ok, "run");
  ok &= expect(std::abs(y[0] - 6.5f) < 1e-6f && std::abs(y[1] - 14.5f) < 1e-6f,
               "y = Wx+b");

  // 2) same content at a DIFFERENT address -> distinct entry (pointer keyed).
  std::vector<float> w_copy = w;
  auto e3 = XnnLinearCache::get(3, 2, w_copy.data(), b.data());
  s = XnnLinearCache::stats();
  ok &= expect(e3.get() != e1.get() && s.size == 2, "different ptr -> new entry");

  // 3) mutate content at the ORIGINAL address -> fingerprint mismatch, repack.
  w[0] = 100.0f;
  auto e4 = XnnLinearCache::get(3, 2, w.data(), b.data());
  ok &= expect(e4.get() != e1.get(), "fingerprint mismatch -> repack");
  ok &= expect(etnp::run_cached(*e4, 1, x.data(), y.data(), nullptr) ==
                   etnp::Error::Ok, "run repacked");
  ok &= expect(std::abs(y[0] - 105.5f) < 1e-4f, "repacked weights take effect");

  // 4) LRU bound: 17 distinct keys -> size stays <= 16.
  std::vector<std::vector<float>> many(17, std::vector<float>(6, 1.0f));
  for (auto& mw : many) XnnLinearCache::get(3, 2, mw.data(), nullptr);
  s = XnnLinearCache::stats();
  ok &= expect(s.size <= 16, "LRU bound holds");

  if (!ok) return 1;
  std::printf("OK: XnnLinearCache hit/miss/fingerprint/LRU behave\n");
  return 0;
}
