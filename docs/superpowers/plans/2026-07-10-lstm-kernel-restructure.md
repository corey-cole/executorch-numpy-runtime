# LSTM Kernel Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `etnp::lstm.out` beat the naive LSTM decomposition at every benchmarked config by batching the input projection, vectorizing the cell update with Google Highway, and caching packed XNNPACK operators.

**Architecture:** The kernel keeps its frozen 10-slot boxed schema. Internally it changes from "create 2 XNNPACK FC ops + run 2 GEMVs + scalar gate math per timestep" to "fetch 2 cached packed FC ops, run ONE batched input GEMM over all timesteps on the shared threadpool, then per timestep one recurrent GEMV + a Highway SIMD fused cell update". Spec: `docs/superpowers/specs/2026-07-10-lstm-kernel-restructure-design.md` — read it first; it contains the background, evidence, and pitfalls this plan assumes.

**Tech Stack:** C++17, ExecuTorch 1.3.1 (pinned dist tarball), XNNPACK (from the dist), Google Highway 1.4.0 (new, FetchContent-pinned), CMake >= 3.24.

## Global Constraints

- Op name/schema FROZEN: `etnp::lstm.out`, 10 stack slots, boxed trampoline registration (`register_kernel(Kernel("etnp::lstm.out", lstm_boxed))`). Never touch the trampoline, schema, or `tools/etnp_lstm_op.py`.
- Scope FROZEN: single-layer, unidirectional, `batch_first=False`, f32, contiguous. Gate row order **i, f, g, o**.
- This repo is torch-free. No torch imports anywhere. Exports run ONLY in `/home/corey/workspace/executorch/.venv` (needs `flatc` from its `bin/` on PATH).
- Numerical gate: `tools/bench_lstm.py` rtol/atol 1e-4 cross-check must PASS; `lstm_parity_test` tolerance 1e-4 must PASS. No fast-math flags, no cheaper approximations.
- Highway pin: release 1.4.0, tarball `https://github.com/google/highway/archive/refs/tags/1.4.0.tar.gz`, SHA256 `e72241ac9524bb653ae52ced768b508045d4438726a303f10181a38f764a453c`.
- Threadpool: ONLY `executorch::extension::threadpool::get_pthreadpool()` (header `executorch/extension/threadpool/threadpool.h`), ONLY for the batched input projection. Recurrent GEMV and cell update stay single-threaded. Never `pthreadpool_create`.
- Benchmark discipline: the local i7 thermal-throttles; NEVER compare absolute ms across runs — only the speedup column within one run.
- Shell caveat: this machine proxies some commands through `rtk`, which can filter output oddly. If a grep/cat result looks impossible, retry with `rtk proxy <cmd>` or different tooling.
- All commands run from the repo root `/home/corey/workspace/executorch-numpy-runtime` unless stated otherwise.

---

### Task 1: Kernels.cmake — registrar-aware nm-guard expectations + per-source include dirs

The nm-guard currently expects a `_GLOBAL__sub_I_<basename>` static-init symbol
for EVERY injected kernel source. Task 3 adds `lstm_cell.cc`, which has NO
registrar, so the expectation must become conditional on the source actually
calling `register_kernel(`. Also add each source's directory as an include dir
(needed so `lstm_cell.cc` can re-include itself by bare filename for Highway's
`foreach_target`, and so sources find sibling headers).

**Files:**
- Modify: `cmake/Kernels.cmake` (the `foreach(_src ...)` block at the end)

**Interfaces:**
- Consumes: existing `_etnp_kernel_sources` list logic (unchanged).
- Produces: `ETNP_KERNEL_EXPECT_TUS` now contains only registrar-bearing
  sources; `etnp_kernels` gains PRIVATE include dirs for each source's parent
  directory; a `message(STATUS "etnp_kernels: expecting registrar TUs: ...")`
  configure line (Tasks 3+ and humans verify against it).

- [ ] **Step 1: Edit `cmake/Kernels.cmake`**

Replace the existing block:

```cmake
  foreach(_src IN LISTS _etnp_kernel_sources)
    get_filename_component(_base "${_src}" NAME)
    list(APPEND ETNP_KERNEL_EXPECT_TUS "_GLOBAL__sub_I_${_base}")
  endforeach()
```

with:

```cmake
  foreach(_src IN LISTS _etnp_kernel_sources)
    # Only sources that register a kernel have a static-init registrar TU the
    # nm-guard must find. Aux sources (e.g. SIMD helpers like the LSTM
    # example's lstm_cell.cc) have none — expecting one would false-fail the
    # guard. Detection is configure-time content sniffing; re-run cmake if a
    # source gains/loses its register_kernel() call.
    file(READ "${_src}" _src_text)
    string(FIND "${_src_text}" "register_kernel(" _reg_pos)
    get_filename_component(_base "${_src}" NAME)
    if(NOT _reg_pos EQUAL -1)
      list(APPEND ETNP_KERNEL_EXPECT_TUS "_GLOBAL__sub_I_${_base}")
    endif()
    # Each source's own directory is an include dir: lets sources include
    # sibling headers by bare name, and lets Highway's foreach_target
    # re-include a source via its bare filename (see lstm_cell.cc).
    get_filename_component(_dir "${_src}" DIRECTORY)
    target_include_directories(etnp_kernels PRIVATE "${_dir}")
  endforeach()
  message(STATUS "etnp_kernels: expecting registrar TUs: [${ETNP_KERNEL_EXPECT_TUS}]")
```

Also update the `ETNP_EXTRA_KERNEL_SOURCES` cache-variable docstring from
"additional custom-kernel .cpp sources" to "additional custom-kernel .cpp/.cc
sources (registrar-bearing kernels and their aux sources)".

- [ ] **Step 2: Verify expectations with the reference kernel only**

```bash
rm -rf /tmp/knl-probe && cmake -S native_tests -B /tmp/knl-probe 2>&1 | grep "expecting registrar"
```
Expected: `-- etnp_kernels: expecting registrar TUs: [_GLOBAL__sub_I_etnp_reference_ops.cpp]`

- [ ] **Step 3: Verify a no-registrar aux source generates NO expectation**

```bash
printf 'namespace etnp_probe { int f() { return 1; } }\n' > /tmp/aux_probe.cpp
rm -rf /tmp/knl-probe && cmake -S native_tests -B /tmp/knl-probe \
  -DETNP_EXTRA_KERNEL_SOURCES=/tmp/aux_probe.cpp 2>&1 | grep "expecting registrar"
```
Expected: same single-entry list as Step 2 — `aux_probe.cpp` must NOT appear.

- [ ] **Step 4: Verify the guard still passes AND still fails (both ways)**

```bash
cmake --build /tmp/knl-probe --target kernel_registration_test -j   # guard runs POST_BUILD
```
Expected: build succeeds, POST_BUILD prints `assert_kernels_registered: ... present`.

```bash
cmake -DSO=/tmp/knl-probe/kernel_registration_test -DNM=nm \
  "-DEXTRA_TUS=_GLOBAL__sub_I_bogus_kernel.cpp" \
  -P cmake/assert_kernels_registered.cmake
```
Expected: `CMake Error ... '_GLOBAL__sub_I_bogus_kernel.cpp' was dropped` (nonzero exit). This proves the guard still detects missing registrars.

- [ ] **Step 5: Commit**

```bash
git add cmake/Kernels.cmake
git commit -m "build: nm-guard expectations only for registrar-bearing kernel sources"
```

---

### Task 2: Kernels.cmake — pinned Highway dependency behind `ETNP_KERNELS_USE_HIGHWAY`

**Files:**
- Modify: `cmake/Kernels.cmake`

**Interfaces:**
- Produces: CMake option `ETNP_KERNELS_USE_HIGHWAY` (default OFF). When ON and
  `etnp_kernels` exists, target `hwy` (static lib) is available and linked
  PRIVATE into `etnp_kernels` (CMake propagates it LINK_ONLY to final
  binaries). Consumers of `etnp_kernels` need no extra wiring.

- [ ] **Step 1: Add the option near the top of `cmake/Kernels.cmake`** (next to the existing `option()`/`set(... CACHE ...)` declarations)

```cmake
option(ETNP_KERNELS_USE_HIGHWAY
  "FetchContent google/highway (hash-pinned) and link it into etnp_kernels. \
Needed by kernels using Highway SIMD, e.g. the LSTM example's lstm_cell.cc." OFF)
```

- [ ] **Step 2: Add the FetchContent block INSIDE the `if(_etnp_kernel_sources)` block, after `target_link_libraries(etnp_kernels PUBLIC executorch)`**

```cmake
  if(ETNP_KERNELS_USE_HIGHWAY)
    # Hash-pinned like RuntimePin.cmake: the SHA256 change is the supply-chain
    # review gate on bumps. contrib/ math is header-only, so no contrib libs.
    # PIC is inherited from the including project's CMAKE_POSITION_INDEPENDENT_CODE.
    set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
    set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    include(FetchContent)
    FetchContent_Declare(highway
      URL "https://github.com/google/highway/archive/refs/tags/1.4.0.tar.gz"
      URL_HASH "SHA256=e72241ac9524bb653ae52ced768b508045d4438726a303f10181a38f764a453c")
    FetchContent_MakeAvailable(highway)
    target_link_libraries(etnp_kernels PRIVATE hwy)
  endif()
```

- [ ] **Step 3: Verify the dependency builds and links**

```bash
rm -rf /tmp/hwy-probe && cmake -S native_tests -B /tmp/hwy-probe -DETNP_KERNELS_USE_HIGHWAY=ON \
  && cmake --build /tmp/hwy-probe --target hwy -j
```
Expected: configure fetches highway 1.4.0 (hash verified), `libhwy.a` builds.

- [ ] **Step 4: Verify the option is inert when OFF**

```bash
rm -rf /tmp/knl-probe && cmake -S native_tests -B /tmp/knl-probe 2>&1 | grep -ci highway
```
Expected: `0` (no Highway mention in a default configure).

- [ ] **Step 5: Commit**

```bash
git add cmake/Kernels.cmake
git commit -m "build: optional hash-pinned Google Highway dep for SIMD kernel sources"
```

---

### Task 3: Highway fused cell update (`lstm_cell.h` / `lstm_cell.cc`) + `lstm_cell_test`

**Files:**
- Create: `examples/custom_kernels/lstm/lstm_cell.h`
- Create: `examples/custom_kernels/lstm/lstm_cell.cc`
- Create: `native_tests/lstm_cell_test.cpp`
- Modify: `native_tests/CMakeLists.txt` (new test target)

**Interfaces:**
- Produces (consumed by Task 5):
  ```cpp
  namespace etnp {
  void lstm_cell_update(std::size_t B, std::size_t H,
                        const float* g_ih_t, const float* g_hh,
                        float* c, float* h, float* out_t);
  }
  ```
  `g_ih_t`/`g_hh` are per-row gate pre-activations, row stride `4*H`, block
  order `[i|f|g|o]`; `c`,`h` are `[B,H]` updated in place; `out_t` is `[B,H]`.
- Template: `scratchpad/cell_lib_spike/cell_hwy.cc` is the numerically
  validated Highway skeleton — same structure, without the B loop.

- [ ] **Step 1: Write the header `examples/custom_kernels/lstm/lstm_cell.h`**

```cpp
// Fused LSTM gate/cell/hidden update for one timestep, SIMD-vectorized via
// Google Highway with runtime dynamic dispatch (best ISA chosen at run time
// from a baseline-flags build). Pure math: no XNNPACK/ExecuTorch types, so it
// is unit-testable standalone (native_tests/lstm_cell_test.cpp).
#pragma once
#include <cstddef>

namespace etnp {
// One timestep for all B batch rows.
//   g_ih_t, g_hh: gate pre-activations, row stride 4*H per batch row, gate
//                 block order [i|f|g|o], each block length H (PyTorch order).
//   c, h:         running cell/hidden state [B,H], updated in place.
//   out_t:        this timestep's output rows [B,H]; receives the new h.
// Per element: i,f,o = sigmoid(g_ih+g_hh), g = tanh(g_ih+g_hh);
//              c' = f*c + i*g;  h' = o*tanh(c').
void lstm_cell_update(std::size_t B, std::size_t H,
                      const float* g_ih_t, const float* g_hh,
                      float* c, float* h, float* out_t);
}  // namespace etnp
```

- [ ] **Step 2: Write the failing test `native_tests/lstm_cell_test.cpp`**

```cpp
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
```

- [ ] **Step 3: Add the test target to `native_tests/CMakeLists.txt`** (after the `xnn_linear_test` block at the end)

```cmake
# Highway fused cell-update unit test. Only meaningful when the LSTM example's
# lstm_cell.cc is among the injected kernel sources (direct link-time call).
if("${ETNP_EXTRA_KERNEL_SOURCES}" MATCHES "lstm_cell\\.cc")
  add_executable(lstm_cell_test lstm_cell_test.cpp)
  target_include_directories(lstm_cell_test PRIVATE
    ${CMAKE_SOURCE_DIR}/../examples/custom_kernels/lstm)
  target_link_libraries(lstm_cell_test PRIVATE ${ETNP_HARNESS_LIBS} etnp_kernels)
endif()
```

- [ ] **Step 4: Run the test to verify it fails (no implementation yet)**

Create a stub `examples/custom_kernels/lstm/lstm_cell.cc` is NOT needed — the
build must fail at link. Configure with both LSTM sources injected:

```bash
rm -rf build/lstm-rt && cmake -S native_tests -B build/lstm-rt \
  -DETNP_KERNELS_USE_HIGHWAY=ON \
  -DETNP_EXTRA_KERNEL_SOURCES="$PWD/examples/custom_kernels/lstm/etnp_lstm.cpp;$PWD/examples/custom_kernels/lstm/lstm_cell.cc" \
  && cmake --build build/lstm-rt --target lstm_cell_test -j
```
Expected: FAILS — either at configure (`lstm_cell.cc` missing) or link
(undefined `etnp::lstm_cell_update`). That is the red state.

- [ ] **Step 5: Write `examples/custom_kernels/lstm/lstm_cell.cc`**

This file is textually re-included once per SIMD target by Highway's
`foreach_target.h`. Structural rules (violating them = weird compile errors):
system includes BEFORE `foreach_target.h`; per-target code between
`HWY_BEFORE_NAMESPACE()/HWY_AFTER_NAMESPACE()`; once-only code under
`#if HWY_ONCE`. Dynamic dispatch (NOT static — static was 5x slower at
baseline flags in the spike).

```cpp
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
```

- [ ] **Step 6: Build and run the test to verify it passes**

```bash
cmake --build build/lstm-rt --target lstm_cell_test -j && ./build/lstm-rt/lstm_cell_test
```
Expected output (values approximate, all `OK`):
```
H=13 B=1: max|diff| = ~e-07 OK
... (6 lines)
OK: lstm_cell_update matches f64 reference (tol 1e-05)
```
If any `max|diff|` is ~1e-3 or worse: gate order or aliasing bug, NOT a
tolerance problem — fix the code.

- [ ] **Step 7: Commit**

```bash
git add examples/custom_kernels/lstm/lstm_cell.h examples/custom_kernels/lstm/lstm_cell.cc \
  native_tests/lstm_cell_test.cpp native_tests/CMakeLists.txt
git commit -m "feat: Highway SIMD fused LSTM cell update + chain-accuracy test"
```

---

### Task 4: XNNPACK operator cache (`xnn_linear_cache.h`) + `xnn_linear_cache_test`

**Files:**
- Create: `examples/custom_kernels/lstm/xnn_linear_cache.h` (header-only, so
  the wheel-injection source list stays at two files)
- Create: `native_tests/xnn_linear_cache_test.cpp`
- Modify: `native_tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `etnp::XnnLinear` and `etnp::ensure_xnn_initialized()` from
  `examples/custom_kernels/lstm/xnn_linear.h` (unchanged).
- Produces (consumed by Task 5):
  ```cpp
  namespace etnp {
  class XnnLinearCache {
    struct Entry;  // opaque to callers except via run_cached
    struct Stats { std::size_t hits, misses, size; };
    static std::shared_ptr<Entry> get(std::size_t in_ch, std::size_t out_ch,
                                      const float* weight, const float* bias);
    static Stats stats();
    static void clear();
  };
  Error run_cached(XnnLinearCache::Entry& e, std::size_t batch,
                   const float* input, float* out, pthreadpool_t tp);
  }
  ```

- [ ] **Step 1: Write the failing test `native_tests/xnn_linear_cache_test.cpp`**

```cpp
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
```

- [ ] **Step 2: Add the test target to `native_tests/CMakeLists.txt`** (next to `xnn_linear_test` — same isolation: no etnp_kernels needed, XNNPACK symbols come from the harness libs)

```cmake
add_executable(xnn_linear_cache_test xnn_linear_cache_test.cpp)
target_include_directories(xnn_linear_cache_test PRIVATE
  ${CMAKE_SOURCE_DIR}/../examples/custom_kernels/lstm)
target_link_libraries(xnn_linear_cache_test PRIVATE ${ETNP_HARNESS_LIBS})
```

- [ ] **Step 3: Build to verify it fails** (header doesn't exist yet)

```bash
cmake --build build/lstm-rt --target xnn_linear_cache_test -j
```
Expected: FAIL — `xnn_linear_cache.h: No such file or directory`.

- [ ] **Step 4: Write `examples/custom_kernels/lstm/xnn_linear_cache.h`**

```cpp
// Process-wide cache of packed XnnLinear (XNNPACK fully-connected) operators,
// keyed by weight identity. Eliminates per-execute weight repacking
// (xnn_create_fully_connected_nc_f32 repacks: measured 92us/exec at H=128).
//
// SAFETY ARGUMENT (do not simplify away): the cache NEVER dereferences stored
// pointers. Fingerprints are always computed from the pointers the CALLER
// passes in — its own live kernel arguments — and compared as plain values
// against the stored fingerprint. An entry whose backing memory was freed
// (program unload) is inert: it can only be looked up again by a caller
// presenting the same pointer value, which then refers to the caller's OWN
// live memory, and the fingerprint decides reuse vs repack.
//
// CONCURRENCY: a global mutex guards the map. Each entry carries its own
// mutex, held across reshape+setup+run (XNNPACK operators are mutated by
// those calls) — two Runtime instances sharing an entry cannot race.
#pragma once
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "xnn_linear.h"

namespace etnp {

class XnnLinearCache {
 public:
  struct Entry {
    XnnLinear op;
    uint64_t fingerprint;
    uint64_t stamp = 0;    // LRU tick of last use
    std::mutex run_mutex;  // serializes reshape/setup/run on this operator
    Entry(XnnLinear&& o, uint64_t fp) : op(std::move(o)), fingerprint(fp) {}
  };
  struct Stats {
    std::size_t hits = 0, misses = 0, size = 0;
  };
  static constexpr std::size_t kMaxEntries = 16;

  // Packed FC for (in_ch, out_ch, weight, bias). Creates + inserts on miss or
  // on fingerprint mismatch (same address, different content -> repack).
  // Returns nullptr if XNNPACK operator creation fails.
  static std::shared_ptr<Entry> get(std::size_t in_ch, std::size_t out_ch,
                                    const float* weight, const float* bias) {
    State& s = state();
    const uint64_t fp = fingerprint(weight, in_ch * out_ch, bias, out_ch);
    const Key key{weight, bias, in_ch, out_ch};
    std::lock_guard<std::mutex> lock(s.mu);
    auto it = s.map.find(key);
    if (it != s.map.end()) {
      if (it->second->fingerprint == fp) {
        ++s.hits;
        it->second->stamp = ++s.tick;
        return it->second;
      }
      s.map.erase(it);  // same address, new content: stale packing
    }
    ++s.misses;
    auto op_r = XnnLinear::create(in_ch, out_ch, weight, bias);
    if (!op_r.ok()) return nullptr;
    auto entry = std::make_shared<Entry>(std::move(op_r.get()), fp);
    entry->stamp = ++s.tick;
    if (s.map.size() >= kMaxEntries) {
      auto oldest = s.map.begin();
      for (auto jt = s.map.begin(); jt != s.map.end(); ++jt)
        if (jt->second->stamp < oldest->second->stamp) oldest = jt;
      s.map.erase(oldest);  // shared_ptr keeps in-flight users alive
    }
    s.map.emplace(key, entry);
    return entry;
  }

  static Stats stats() {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    return Stats{s.hits, s.misses, s.map.size()};
  }

  static void clear() {  // test hook
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    s.map.clear();
    s.hits = s.misses = 0;
    s.tick = 0;
  }

 private:
  struct Key {
    const float* w;
    const float* b;
    std::size_t ic, oc;
    bool operator==(const Key& o) const {
      return w == o.w && b == o.b && ic == o.ic && oc == o.oc;
    }
  };
  struct KeyHash {
    std::size_t operator()(const Key& k) const {
      uint64_t x = reinterpret_cast<uintptr_t>(k.w);
      x = x * 1099511628211ULL ^ reinterpret_cast<uintptr_t>(k.b);
      x = x * 1099511628211ULL ^ k.ic;
      x = x * 1099511628211ULL ^ k.oc;
      return static_cast<std::size_t>(x);
    }
  };
  struct State {
    std::mutex mu;
    std::unordered_map<Key, std::shared_ptr<Entry>, KeyHash> map;
    uint64_t tick = 0;
    std::size_t hits = 0, misses = 0;
  };
  static State& state() {
    static State s;  // process lifetime; reachable at exit (not an LSan leak)
    return s;
  }

  // FNV-1a over a fixed sample of the weight (first 8 + last 8 floats) and
  // bias (first 4), plus the element count. Cheap and catches address reuse.
  static uint64_t fingerprint(const float* w, std::size_t n_w, const float* b,
                              std::size_t n_b) {
    uint64_t h = 1469598103934665603ULL;
    auto mix = [&h](const void* p, std::size_t bytes) {
      const unsigned char* q = static_cast<const unsigned char*>(p);
      for (std::size_t i = 0; i < bytes; ++i) h = (h ^ q[i]) * 1099511628211ULL;
    };
    mix(&n_w, sizeof(n_w));
    const std::size_t head = n_w < 8 ? n_w : 8;
    mix(w, head * sizeof(float));
    if (n_w > 8) {
      const std::size_t tail = (n_w - 8) < 8 ? (n_w - 8) : 8;
      mix(w + n_w - tail, tail * sizeof(float));
    }
    if (b) {
      const std::size_t bh = n_b < 4 ? n_b : 4;
      mix(b, bh * sizeof(float));
    }
    return h;
  }
};

// Locked run on a cached entry (see CONCURRENCY note above).
inline Error run_cached(XnnLinearCache::Entry& e, std::size_t batch,
                        const float* input, float* out, pthreadpool_t tp) {
  std::lock_guard<std::mutex> lock(e.run_mutex);
  return e.op.run(batch, input, out, tp);
}

}  // namespace etnp
```

- [ ] **Step 5: Build and run the test to verify it passes**

```bash
cmake --build build/lstm-rt --target xnn_linear_cache_test -j && ./build/lstm-rt/xnn_linear_cache_test
```
Expected: `OK: XnnLinearCache hit/miss/fingerprint/LRU behave`

- [ ] **Step 6: Commit**

```bash
git add examples/custom_kernels/lstm/xnn_linear_cache.h \
  native_tests/xnn_linear_cache_test.cpp native_tests/CMakeLists.txt
git commit -m "feat: fingerprint-verified LRU cache for packed XNNPACK FC operators"
```

---

### Task 5: Restructure `etnp_lstm.cpp` (batched input GEMM + cache + Highway cell)

**Files:**
- Modify: `examples/custom_kernels/lstm/etnp_lstm.cpp` (kernel body)
- Modify: `native_tests/lstm_kernel_test.cpp:58` (arena 64KiB → 4MiB)
- Modify: `native_tests/lstm_parity_test.cpp:66` (arena 64KiB → 4MiB)

**Interfaces:**
- Consumes: `etnp::lstm_cell_update` (Task 3), `etnp::XnnLinearCache::get` /
  `etnp::run_cached` (Task 4),
  `executorch::extension::threadpool::get_pthreadpool()` (dist header
  `executorch/extension/threadpool/threadpool.h`).
- Produces: same registered op, same schema — nothing downstream changes.

- [ ] **Step 0: Link `extension_threadpool` into `etnp_kernels`**

The kernel now references `executorch::extension::threadpool::get_pthreadpool()`.
Today the symbol would resolve only transitively via `xnnpack_backend` — make
it explicit (spec §4.3(3)). In `cmake/Kernels.cmake` change:

```cmake
  target_link_libraries(etnp_kernels PUBLIC executorch)
```
to:
```cmake
  # extension_threadpool: kernels may use the shared runtime threadpool
  # (executorch::extension::threadpool::get_pthreadpool), e.g. the LSTM
  # example's batched input projection.
  target_link_libraries(etnp_kernels PUBLIC executorch extension_threadpool)
```

- [ ] **Step 1: Confirm the existing tests pass BEFORE the change (green baseline)**

```bash
cmake --build build/lstm-rt -j && ./build/lstm-rt/lstm_kernel_test && ./build/lstm-rt/lstm_parity_test
```
Expected: `OK: etnp::lstm.out analytic recurrence correct` and
`OK: etnp::lstm.out matches torch.nn.LSTM (rtol/atol 1e-4)`.

- [ ] **Step 2: Enlarge both test arenas**

In `native_tests/lstm_kernel_test.cpp` and `native_tests/lstm_parity_test.cpp`
change:
```cpp
  std::vector<uint8_t> temp_buf(64 * 1024);
```
to:
```cpp
  // The restructured kernel allocates T*B*4H floats for the batched input
  // projection; size the arena generously so shape bumps don't hit the wall.
  std::vector<uint8_t> temp_buf(4 * 1024 * 1024);
```

- [ ] **Step 3: Rewrite the kernel body**

In `examples/custom_kernels/lstm/etnp_lstm.cpp`:

(a) Update the header comment's first paragraph to:
```cpp
// etnp::lstm.out — single-layer, unidirectional, batch_first=False, float32 LSTM over a
// full sequence. The input projection for ALL timesteps runs as ONE batched XNNPACK FC
// on the shared runtime threadpool; packed FC operators are cached across executes
// (xnn_linear_cache.h); the per-timestep recurrent projection is a single-threaded FC
// and the gate/cell/hidden update is a Highway-SIMD fused pass (lstm_cell.cc).
```

(b) Replace the includes block addition — after `#include "xnn_linear.h"` add:
```cpp
#include <executorch/extension/threadpool/threadpool.h>

#include "lstm_cell.h"
#include "xnn_linear_cache.h"
```
and delete the now-unused `#include <cmath>` and the `sigmoidf` helper
(lines 12 and 33 of the current file — the scalar gate math moves to
lstm_cell.cc).

(c) Replace everything from the `// Create the two projections ONCE ...`
comment down to (and including) the `for (int64_t t ...) { ... }` loop with:

```cpp
  // Packed projections come from the process-wide cache: weights live in the
  // .pte constant segment (pointer-stable per loaded program), so packing
  // happens once per weight set, not once per execute.
  auto ent_ih = XnnLinearCache::get(
      static_cast<size_t>(I), static_cast<size_t>(4 * H),
      w_ih.const_data_ptr<float>(),
      b_ih.has_value() ? b_ih->const_data_ptr<float>() : nullptr);
  auto ent_hh = XnnLinearCache::get(
      static_cast<size_t>(H), static_cast<size_t>(4 * H),
      w_hh.const_data_ptr<float>(),
      b_hh.has_value() ? b_hh->const_data_ptr<float>() : nullptr);
  ET_KERNEL_CHECK(ctx, ent_ih != nullptr && ent_hh != nullptr, Internal, ret);

  const size_t g_ih_bytes =
      static_cast<size_t>(T) * B * 4 * H * sizeof(float);
  const size_t g_hh_bytes = static_cast<size_t>(B) * 4 * H * sizeof(float);
  auto g_ih_r = ctx.allocate_temp(g_ih_bytes);
  auto g_hh_r = ctx.allocate_temp(g_hh_bytes);
  ET_KERNEL_CHECK(ctx, g_ih_r.ok() && g_hh_r.ok(), MemoryAllocationFailed, ret);
  float* g_ih_all = static_cast<float*>(g_ih_r.get());
  float* g_hh = static_cast<float*>(g_hh_r.get());

  // Running state lives in hn/cn, seeded from h0/c0.
  float* h = hn.mutable_data_ptr<float>();
  float* c = cn.mutable_data_ptr<float>();
  std::memcpy(h, h0.const_data_ptr<float>(), static_cast<size_t>(B) * H * sizeof(float));
  std::memcpy(c, c0.const_data_ptr<float>(), static_cast<size_t>(B) * H * sizeof(float));

  const float* in = input.const_data_ptr<float>();
  float* out = output.mutable_data_ptr<float>();

  // Input projection for ALL timesteps in one GEMM: input [T,B,I] is
  // contiguous, hence also a valid [T*B, I] matrix. This is the only
  // multi-threaded step — same pool the XNNPACK delegate uses.
  ET_KERNEL_CHECK(ctx,
      run_cached(*ent_ih, static_cast<size_t>(T) * B, in, g_ih_all,
                 executorch::extension::threadpool::get_pthreadpool()) ==
          Error::Ok,
      Internal, ret);

  for (int64_t t = 0; t < T; ++t) {
    // Recurrent projection: g_hh = h_{t-1} @ W_hh^T + b_hh. Reads h fully
    // before lstm_cell_update overwrites it. Single-threaded: at [B,H]x[H,4H]
    // scale the pool dispatch overhead exceeds the win (measured).
    ET_KERNEL_CHECK(ctx,
        run_cached(*ent_hh, static_cast<size_t>(B), h, g_hh,
                   /*tp=*/nullptr) == Error::Ok,
        Internal, ret);
    lstm_cell_update(static_cast<size_t>(B), static_cast<size_t>(H),
                     g_ih_all + static_cast<size_t>(t) * B * 4 * H, g_hh, c, h,
                     out + static_cast<size_t>(t) * B * H);
  }
```

Everything after (the `return ret;`, the boxed trampoline, the registrar)
stays byte-identical.

- [ ] **Step 4: Rebuild and run ALL native tests**

```bash
cmake --build build/lstm-rt -j \
  && ./build/lstm-rt/lstm_cell_test \
  && ./build/lstm-rt/xnn_linear_cache_test \
  && ./build/lstm-rt/lstm_kernel_test \
  && ./build/lstm-rt/lstm_parity_test \
  && ./build/lstm-rt/xnn_linear_test \
  && ./build/lstm-rt/kernel_registration_test
```
Expected: every binary prints its `OK:` line; the POST_BUILD nm-guard passed
during the build (registrar TU expected ONLY for `etnp_lstm.cpp`). The parity
test is the real gate: golden fixture has B=2 and H=5, so it exercises the
batch loop AND the SIMD tail through the full kernel.

- [ ] **Step 5: ASan leak check** (the cache is static state — LSan must see it as reachable, not leaked)

```bash
cmake --build build/lstm-rt --target kernel_leak_test -j \
  && ASAN_OPTIONS=detect_leaks=1 ./build/lstm-rt/kernel_leak_test
```
Expected: `OK` line, no leak report.

- [ ] **Step 6: Commit**

```bash
git add examples/custom_kernels/lstm/etnp_lstm.cpp cmake/Kernels.cmake \
  native_tests/lstm_kernel_test.cpp native_tests/lstm_parity_test.cpp
git commit -m "perf: batched input projection + cached packed FCs + Highway cell in etnp::lstm"
```

---

### Task 6: Concurrency gate — TSan test for the shared cache

The existing `race_harness` covers et_core only; the op cache is new shared
state reachable from concurrently-executing Runtime instances. Add a TSan
binary that hammers the op from two threads with the SAME weight pointers
(shared cache entries) and different activations.

**Files:**
- Create: `native_tests/lstm_cache_race_test.cpp`
- Modify: `native_tests/CMakeLists.txt`

**Interfaces:**
- Consumes: the registered op via
  `executorch::runtime::get_op_function_from_registry("etnp::lstm.out", {})`
  (same invocation pattern as `lstm_kernel_test.cpp`).

- [ ] **Step 1: Write `native_tests/lstm_cache_race_test.cpp`**

```cpp
// TSan gate: two threads invoke etnp::lstm.out concurrently with the SAME
// weight buffers (=> shared XnnLinearCache entries) and private activations.
// Passes iff TSan reports no race and both threads produce identical output.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::extension::make_tensor_ptr;
using executorch::runtime::EValue;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::OpFunction;
using executorch::runtime::Span;

namespace {
constexpr int64_t T = 4, B = 1, I = 8, H = 8;
constexpr int kIters = 200;

// Shared, read-only weights (the contended cache keys).
std::vector<float> g_wih(4 * H * I, 0.01f), g_whh(4 * H * H, 0.02f);
std::vector<float> g_bih(4 * H, 0.1f), g_bhh(4 * H, -0.1f);

void worker(OpFunction fn, std::vector<float>* out_final) {
  std::vector<float> in(T * B * I, 0.5f), h0(B * H, 0.1f), c0(B * H, 0.2f);
  std::vector<float> out(T * B * H), hn(B * H), cn(B * H);
  auto t_in = make_tensor_ptr({T, B, I}, in.data());
  auto t_h0 = make_tensor_ptr({B, H}, h0.data());
  auto t_c0 = make_tensor_ptr({B, H}, c0.data());
  auto t_wih = make_tensor_ptr({4 * H, I}, g_wih.data());
  auto t_whh = make_tensor_ptr({4 * H, H}, g_whh.data());
  auto t_bih = make_tensor_ptr({4 * H}, g_bih.data());
  auto t_bhh = make_tensor_ptr({4 * H}, g_bhh.data());
  auto t_out = make_tensor_ptr({T, B, H}, out.data());
  auto t_hn = make_tensor_ptr({B, H}, hn.data());
  auto t_cn = make_tensor_ptr({B, H}, cn.data());
  EValue ev[] = {EValue(*t_in), EValue(*t_h0), EValue(*t_c0), EValue(*t_wih),
                 EValue(*t_whh), EValue(*t_bih), EValue(*t_bhh),
                 EValue(*t_out), EValue(*t_hn), EValue(*t_cn)};
  EValue* args[10];
  for (int i = 0; i < 10; ++i) args[i] = &ev[i];
  std::vector<uint8_t> temp_buf(1 * 1024 * 1024);
  for (int it = 0; it < kIters; ++it) {
    MemoryAllocator temp_alloc(static_cast<uint32_t>(temp_buf.size()),
                               temp_buf.data());
    KernelRuntimeContext ctx(/*event_tracer=*/nullptr, &temp_alloc);
    fn(ctx, Span<EValue*>(args, 10));
  }
  *out_final = out;
}
}  // namespace

int main() {
  executorch::runtime::runtime_init();
  auto fn_r =
      executorch::runtime::get_op_function_from_registry("etnp::lstm.out", {});
  if (!fn_r.ok()) { std::fprintf(stderr, "FAIL: not registered\n"); return 1; }
  OpFunction fn = fn_r.get();

  std::vector<float> out_a, out_b;
  std::thread ta(worker, fn, &out_a);
  std::thread tb(worker, fn, &out_b);
  ta.join();
  tb.join();
  for (size_t i = 0; i < out_a.size(); ++i) {
    if (out_a[i] != out_b[i]) {
      std::fprintf(stderr, "FAIL: thread outputs diverge at %zu\n", i);
      return 1;
    }
  }
  std::printf("OK: concurrent etnp::lstm.out via shared cache is race-free\n");
  return 0;
}
```

- [ ] **Step 2: Add the TSan target to `native_tests/CMakeLists.txt`** (inside the existing `if(TARGET etnp_kernels)` block, after `lstm_parity_test`)

```cmake
  # TSan gate for the op cache: shared XnnLinearCache entries hammered from
  # two threads. Separate binary: TSan cannot combine with the ASan targets.
  add_executable(lstm_cache_race_test lstm_cache_race_test.cpp)
  target_compile_options(lstm_cache_race_test PRIVATE -fsanitize=thread -g -O1)
  target_link_options(lstm_cache_race_test PRIVATE -fsanitize=thread)
  target_link_libraries(lstm_cache_race_test PRIVATE
    ${ETNP_HARNESS_LIBS} "$<LINK_LIBRARY:WHOLE_ARCHIVE,etnp_kernels>" Threads::Threads)
```

- [ ] **Step 3: Build and run under TSan**

```bash
cmake --build build/lstm-rt --target lstm_cache_race_test -j \
  && TSAN_OPTIONS="suppressions=$PWD/native_tests/tsan_suppressions.txt" \
     ./build/lstm-rt/lstm_cache_race_test
```
Expected: `OK: concurrent etnp::lstm.out via shared cache is race-free`, exit 0.

TRIAGE RULE if TSan reports a race: read the stack. If BOTH stacks are inside
third-party frames (`pthreadpool_*`, `xnn_*`) with no `etnp::` frame between,
it's threadpool-internal noise — add a narrowly-scoped suppression to
`native_tests/tsan_suppressions.txt` with a comment. If ANY frame is in
`etnp::` (cache, cell, kernel), it is OUR bug — fix the locking, never
suppress it.

- [ ] **Step 4: Rerun the pre-existing harnesses (must stay clean)**

```bash
cmake --build build/lstm-rt --target race_harness leak_harness -j \
  && TSAN_OPTIONS="suppressions=$PWD/native_tests/tsan_suppressions.txt" ./build/lstm-rt/race_harness \
  && ASAN_OPTIONS=detect_leaks=1 ./build/lstm-rt/leak_harness
```
Expected: both print their success lines, exit 0.

- [ ] **Step 5: Commit**

```bash
git add native_tests/lstm_cache_race_test.cpp native_tests/CMakeLists.txt
git commit -m "test: TSan gate for concurrent lstm executes sharing the op cache"
```

---

### Task 7: End-to-end benchmark verification + documentation

**Files:**
- Modify: `docs/custom-kernels.md` (bench recipe + internals note)
- Modify: `problem-statement.md` (post-restructure results section)
- Modify: `docs/superpowers/specs/2026-07-09-upstream-lstm-productionization-handoff.md`

**Interfaces:**
- Consumes: everything above; `.pte` fixtures in `/tmp/lstm_ptes` (re-export
  per the command below only if missing).

- [ ] **Step 1: Ensure fixtures exist**

```bash
ls /tmp/lstm_ptes/*.pte | wc -l   # expect 16; if fewer, re-export:
PATH="/home/corey/workspace/executorch/.venv/bin:$PATH" \
  /home/corey/workspace/executorch/.venv/bin/python tools/export_lstm_bench_models.py /tmp/lstm_ptes
```
Note: naive T256_H128 never exports (known feasibility wall — script and bench
both skip it). T256_H32 takes minutes; be patient.

- [ ] **Step 2: Build the bench wheel with the restructured kernel**

```bash
CMAKE_ARGS="-DETNP_EXTRA_KERNEL_SOURCES=$PWD/examples/custom_kernels/lstm/etnp_lstm.cpp;$PWD/examples/custom_kernels/lstm/lstm_cell.cc -DETNP_KERNELS_USE_HIGHWAY=ON" \
  uv build --wheel -o /tmp/lstm_bench_wheel2
```
CAUTION: scikit-build-core splits `CMAKE_ARGS` on SPACES, so the semicolon
inside the first `-D` is safe — but verify the configure log shows BOTH
sources. If the list got mangled, write the two paths into a response file
approach: `-DETNP_EXTRA_KERNEL_SOURCES=<path1>\;<path2>` (escaped semicolon).

- [ ] **Step 3: Fresh venv + run the bench**

```bash
rm -rf /tmp/lstm_bench_venv2 && uv venv /tmp/lstm_bench_venv2 --python 3.12
VIRTUAL_ENV=/tmp/lstm_bench_venv2 uv pip install /tmp/lstm_bench_wheel2/*.whl numpy
/tmp/lstm_bench_venv2/bin/python tools/bench_lstm.py /tmp/lstm_ptes
```

Success criteria (from the spec — ALL must hold):
- `rtol 0.0001 cross-check ... (PASS)` — the honesty gate.
- Speedup column >= 1.00x for EVERY config row. Baseline that must flip:
  T16_H64 (was 0.90x), T64_H64 (0.86x), T16_H128 (0.47x), T64_H128 (0.86x).
  H=32 rows must stay >= 1.0x.
If a row is below 1.0x: rerun once (thermal noise), and if it persists,
profile before changing anything — do NOT tune blindly.

- [ ] **Step 4: Update `docs/custom-kernels.md`**

In the "Benchmarking the LSTM example" section, replace the step-2 build line:
```
    # 2. Build a wheel WITH the LSTM kernel compiled in.
    CMAKE_ARGS="-DETNP_EXTRA_KERNEL_SOURCES=$PWD/examples/custom_kernels/lstm/etnp_lstm.cpp" \
      uv build --wheel -o /tmp/lstm_bench_wheel
```
with:
```
    # 2. Build a wheel WITH the LSTM kernel compiled in. The kernel is two
    #    sources (op + Highway SIMD cell update) and needs the pinned Highway dep.
    CMAKE_ARGS="-DETNP_EXTRA_KERNEL_SOURCES=$PWD/examples/custom_kernels/lstm/etnp_lstm.cpp;$PWD/examples/custom_kernels/lstm/lstm_cell.cc -DETNP_KERNELS_USE_HIGHWAY=ON" \
      uv build --wheel -o /tmp/lstm_bench_wheel
```
and append a short subsection after the benchmark description:
```
### LSTM kernel internals (post 2026-07 restructure)

The op computes the input projection for ALL timesteps as one batched XNNPACK
FC on the shared runtime threadpool (`extension_threadpool` — the same pool
the XNNPACK delegate uses), keeps packed FC operators in a process-wide
fingerprint-verified LRU cache (`xnn_linear_cache.h`) so weights are packed
once per weight set instead of once per execute, and fuses the per-timestep
gate/cell/hidden update into a Highway-SIMD pass with runtime dynamic dispatch
(`lstm_cell.cc`; Highway is FetchContent-pinned in `cmake/Kernels.cmake`
behind `-DETNP_KERNELS_USE_HIGHWAY=ON`). Design + evidence:
`docs/superpowers/specs/2026-07-10-lstm-kernel-restructure-design.md`.
```

- [ ] **Step 5: Append results to `problem-statement.md`**

Add at the end (paste the ACTUAL table from Step 3, not the baseline):
```
## Post-restructure results (2026-07-XX)

Kernel restructured per docs/superpowers/specs/2026-07-10-lstm-kernel-restructure-design.md
(batched input projection on the shared threadpool + Highway SIMD cell update
+ cached packed FC operators). Same host, same fixtures:

<paste the full bench_lstm.py output table here>

All configs now >= 1.0x with the rtol 1e-4 cross-check passing.
```

- [ ] **Step 6: Amend the upstream handoff spec**

In `docs/superpowers/specs/2026-07-09-upstream-lstm-productionization-handoff.md`:

(a) In the "four faces" table, face 1's "Port from" cell: change
`examples/custom_kernels/lstm/etnp_lstm.cpp`, `examples/custom_kernels/lstm/xnn_linear.h`
to
`examples/custom_kernels/lstm/etnp_lstm.cpp`, `lstm_cell.{h,cc}`, `xnn_linear.h`, `xnn_linear_cache.h`.

(b) Replace Gotcha B's body with:
```
**B. Projections reuse XNNPACK's public FC API, not BLAS — restructured 2026-07-10.**
`$SRC/examples/custom_kernels/lstm/xnn_linear.h` wraps
`xnn_create/reshape/setup/run_fully_connected_nc_f32` (weight layout [4H,I]/[4H,H],
`flags=0`). Packed operators are CACHED process-wide across executes
(`xnn_linear_cache.h`: pointer+dims key, content fingerprint, LRU 16, per-entry
run mutex — read its safety comment before porting). The input projection runs
as ONE batched FC over all T timesteps on the shared `extension_threadpool`
pool; the per-timestep recurrent FC is single-threaded; the fused cell update
is Highway SIMD (`lstm_cell.cc`, dynamic dispatch) — so the upstream build must
also carry the pinned Highway dependency (see `$SRC/cmake/Kernels.cmake`,
`ETNP_KERNELS_USE_HIGHWAY`, Highway 1.4.0,
SHA256 e72241ac9524bb653ae52ced768b508045d4438726a303f10181a38f764a453c).
```

(c) In "Evidence this op is worth shipping", add a line under the Speed bullet:
```
- **Speed (post-restructure 2026-07):** superseded — see the post-restructure
  table in `$SRC/problem-statement.md`; the custom op now wins at every
  benchmarked (T,H) on the reference host, so the "reach for it at narrow-H
  only" guidance below is obsolete for latency (size/feasibility wins unchanged).
```

- [ ] **Step 7: Final full-suite re-run + commit**

```bash
cmake --build build/lstm-rt -j && ./build/lstm-rt/lstm_cell_test && ./build/lstm-rt/xnn_linear_cache_test \
  && ./build/lstm-rt/lstm_kernel_test && ./build/lstm-rt/lstm_parity_test && ./build/lstm-rt/xnn_linear_test \
  && ./build/lstm-rt/kernel_registration_test
git add docs/custom-kernels.md problem-statement.md \
  docs/superpowers/specs/2026-07-09-upstream-lstm-productionization-handoff.md
git commit -m "docs: post-restructure LSTM bench results + updated recipes and handoff"
```

---

## Self-review checklist (already run by the plan author)

- Spec coverage: §4.1→Task 5, §4.2→Task 3, §4.3→Tasks 1–2, §4.4→Task 4,
  §4.5→Task 5, §6.1→Task 3, §6.2→Task 4, §6.3/6.4→Task 5, §6.5→Task 6,
  §7/§3→Task 7 steps 1–3, §9→Task 7 steps 4–6. §9.4 (scratchpad README line)
  was already done before this plan.
- Type consistency: `lstm_cell_update(std::size_t B, std::size_t H, const
  float*, const float*, float*, float*, float*)` and
  `run_cached(Entry&, std::size_t, const float*, float*, pthreadpool_t)` are
  used with identical signatures in Tasks 3/4/5.
- The `Error` type in `xnn_linear_cache.h` is `executorch::runtime::Error`
  via the existing `using` in `xnn_linear.h` (namespace `etnp`).
