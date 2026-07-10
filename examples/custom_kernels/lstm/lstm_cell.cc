// SIMD LSTM fused cell update via Google Highway, runtime dynamic dispatch.
// This TU is re-included once per SIMD target by foreach_target.h; the bare
// filename below resolves because Kernels.cmake adds each kernel source's own
// directory as an include dir. Validated template: scratchpad/cell_lib_spike/.
#include "lstm_cell.h"

#include <cmath>
#include <cstddef>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lstm_cell.cc"
#include "hwy/foreach_target.h"  // must precede highway.h
#include "hwy/highway.h"
#include "hwy/contrib/math/math-inl.h"

HWY_BEFORE_NAMESPACE();
namespace etnp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

void CellUpdateImpl(std::size_t B, std::size_t H,
                    const float* HWY_RESTRICT g_ih_t,
                    const float* HWY_RESTRICT g_hh, float* HWY_RESTRICT c,
                    float* HWY_RESTRICT h, float* HWY_RESTRICT out_t) {
  const hn::ScalableTag<float> d;
  const std::size_t N = hn::Lanes(d);
  const auto one = hn::Set(d, 1.0f);
  const auto sigmoid = [&](auto v) {
    return hn::Div(one, hn::Add(one, hn::CallExp(d, hn::Neg(v))));
  };
  for (std::size_t b = 0; b < B; ++b) {
    const float* aih = g_ih_t + b * 4 * H;  // gate blocks [i|f|g|o]
    const float* ahh = g_hh + b * 4 * H;
    float* cb = c + b * H;
    float* hb = h + b * H;
    float* ob = out_t + b * H;
    std::size_t j = 0;
    for (; j + N <= H; j += N) {
      const auto ig =
          sigmoid(hn::Add(hn::LoadU(d, aih + j), hn::LoadU(d, ahh + j)));
      const auto fg = sigmoid(
          hn::Add(hn::LoadU(d, aih + H + j), hn::LoadU(d, ahh + H + j)));
      const auto gg = hn::CallTanh(d, hn::Add(hn::LoadU(d, aih + 2 * H + j),
                                              hn::LoadU(d, ahh + 2 * H + j)));
      const auto og = sigmoid(hn::Add(hn::LoadU(d, aih + 3 * H + j),
                                      hn::LoadU(d, ahh + 3 * H + j)));
      const auto cn = hn::MulAdd(fg, hn::LoadU(d, cb + j), hn::Mul(ig, gg));
      const auto hh = hn::Mul(og, hn::CallTanh(d, cn));
      hn::StoreU(cn, d, cb + j);
      hn::StoreU(hh, d, hb + j);
      hn::StoreU(hh, d, ob + j);
    }
    for (; j < H; ++j) {  // scalar tail: H need not be a lane multiple
      const auto sg = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
      const float ig = sg(aih[j] + ahh[j]);
      const float fg = sg(aih[H + j] + ahh[H + j]);
      const float gg = std::tanh(aih[2 * H + j] + ahh[2 * H + j]);
      const float og = sg(aih[3 * H + j] + ahh[3 * H + j]);
      const float cn = fg * cb[j] + ig * gg;
      const float hh = og * std::tanh(cn);
      cb[j] = cn;
      hb[j] = hh;
      ob[j] = hh;
    }
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace etnp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace etnp {
HWY_EXPORT(CellUpdateImpl);

void lstm_cell_update(std::size_t B, std::size_t H, const float* g_ih_t,
                      const float* g_hh, float* c, float* h, float* out_t) {
  HWY_DYNAMIC_DISPATCH(CellUpdateImpl)(B, H, g_ih_t, g_hh, c, h, out_t);
}
}  // namespace etnp
#endif  // HWY_ONCE
