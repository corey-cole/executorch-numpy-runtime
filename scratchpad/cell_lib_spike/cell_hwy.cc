// LSTM fused cell update via Highway with per-target codegen + dynamic dispatch.
#include <cmath>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cell_hwy.cc"
#include "hwy/foreach_target.h"  // must come before highway.h
#include "hwy/highway.h"
#include "hwy/contrib/math/math-inl.h"

HWY_BEFORE_NAMESPACE();
namespace spike {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

void CellUpdateImpl(size_t H, const float* HWY_RESTRICT aih,
                    const float* HWY_RESTRICT ahh, float* HWY_RESTRICT c,
                    float* HWY_RESTRICT h, float* HWY_RESTRICT out_t) {
  const hn::ScalableTag<float> d;
  const size_t N = hn::Lanes(d);
  const auto one = hn::Set(d, 1.0f);
  auto sigmoid = [&](auto v) {
    return hn::Div(one, hn::Add(one, hn::CallExp(d, hn::Neg(v))));
  };
  size_t j = 0;
  for (; j + N <= H; j += N) {
    const auto ig = sigmoid(hn::Add(hn::LoadU(d, aih + j), hn::LoadU(d, ahh + j)));
    const auto fg = sigmoid(hn::Add(hn::LoadU(d, aih + H + j), hn::LoadU(d, ahh + H + j)));
    const auto gg = hn::CallTanh(d, hn::Add(hn::LoadU(d, aih + 2 * H + j),
                                            hn::LoadU(d, ahh + 2 * H + j)));
    const auto og = sigmoid(hn::Add(hn::LoadU(d, aih + 3 * H + j),
                                    hn::LoadU(d, ahh + 3 * H + j)));
    const auto cn = hn::MulAdd(fg, hn::LoadU(d, c + j), hn::Mul(ig, gg));
    const auto hh = hn::Mul(og, hn::CallTanh(d, cn));
    hn::StoreU(cn, d, c + j);
    hn::StoreU(hh, d, h + j);
    hn::StoreU(hh, d, out_t + j);
  }
  // scalar tail (H not multiple of lanes)
  for (; j < H; ++j) {
    auto sg = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    const float ig = sg(aih[j] + ahh[j]);
    const float fg = sg(aih[H + j] + ahh[H + j]);
    const float gg = std::tanh(aih[2 * H + j] + ahh[2 * H + j]);
    const float og = sg(aih[3 * H + j] + ahh[3 * H + j]);
    const float cn = fg * c[j] + ig * gg;
    const float hh = og * std::tanh(cn);
    c[j] = cn; h[j] = hh; out_t[j] = hh;
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace spike
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace spike {
HWY_EXPORT(CellUpdateImpl);

void cell_hwy_dynamic(size_t H, const float* aih, const float* ahh, float* c,
                      float* h, float* out_t) {
  HWY_DYNAMIC_DISPATCH(CellUpdateImpl)(H, aih, ahh, c, h, out_t);
}
void cell_hwy_static(size_t H, const float* aih, const float* ahh, float* c,
                     float* h, float* out_t) {
  HWY_STATIC_DISPATCH(CellUpdateImpl)(H, aih, ahh, c, h, out_t);
}
const char* hwy_dynamic_target_name() {
  // Best (lowest-bit) target the dispatcher will pick on this CPU.
  const int64_t sup = hwy::SupportedTargets();
  return hwy::TargetName(sup & -sup);
}
}  // namespace spike
#endif
