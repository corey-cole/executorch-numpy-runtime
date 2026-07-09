# Sequence-level LSTM Kernel Implementation Plan

**Tracking issue:** [#4](https://github.com/corey-cole/executorch-numpy-runtime/issues/4) — reference in the PR (`Closes #4`) to close the loop.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a single-layer, unidirectional `etnp::lstm` custom kernel that runs a whole sequence in C++, reusing XNNPACK's fully-connected microkernels for the gate projections (create-once / run-per-timestep) and fusing the gate activations + cell/hidden update — validated for numerical parity against `torch.nn.LSTM`.

**Architecture:** The op computes the classic LSTM recurrence over `T` timesteps. The two learned projections (`input @ W_ih^T + b_ih`, `h @ W_hh^T + b_hh`) are delegated to XNNPACK's `xnn_*_fully_connected_nc_f32` operator — the *same* tuned kernels the XNNPACK delegate uses — created ONCE (weights are constant across the sequence) and run per timestep. The pointwise tail (four gate activations, `c' = f*c + i*g`, `h' = o*tanh(c')`, and the `g_ih + g_hh` add) is fused into a single pass per timestep, avoiding materializing intermediates. It rides the custom-kernel seam from PR #3: registered via `EXECUTORCH_LIBRARY`, compiled via `ETNP_EXTRA_KERNEL_SOURCES`, and covered by the whole-archive nm-guard.

**Tech Stack:** C++17, CMake ≥3.24, the pinned ExecuTorch 1.3.1 runtime (`third_party/…`), XNNPACK's public C API (`xnnpack.h`, already shipped + linked), the custom-kernel seam (`cmake/Kernels.cmake`, `native_tests/`), and PyTorch (in the export venv) to generate the committed golden fixture (Task 3) and, for the MVP probe (Tasks 4–6), to export the benchmark `.pte`s.

## Dependency

This plan builds on the custom-kernel seam (branch `feature/custom-kernel-seam`, PR #3). Branch from `main` AFTER that merges, or stack on the seam branch. It relies on: `cmake/Kernels.cmake` (`ETNP_EXTRA_KERNEL_SOURCES` → `etnp_kernels`, `ETNP_KERNEL_EXPECT_TUS`), the generalized `cmake/assert_kernels_registered.cmake` guard, and the `native_tests` include of `Kernels.cmake`.

## Global Constraints

- **This is an EXAMPLE kernel, NOT bundled in the default wheel.** It lives under `examples/custom_kernels/lstm/` and is compiled only when injected via `ETNP_EXTRA_KERNEL_SOURCES` (same posture as `examples/custom_kernels/negate_op.cpp`). The default `_core` wheel must not gain an LSTM op.
- **Scope: single layer, unidirectional, `batch_first=False`, float32, contiguous.** Input `[T, B, I]`, `h0`/`c0` `[B, H]`, `w_ih` `[4H, I]`, `w_hh` `[4H, H]`, optional biases `[4H]`; outputs `output` `[T, B, H]`, `hn`/`cn` `[B, H]`. No bidirectional, no `num_layers>1`, no dropout, no packed sequences.
- **Gate order is PyTorch's: rows `[i, f, g, o]`** (input, forget, cell, output) in `w_ih`/`w_hh`/biases. Must match so parity holds.
- **Weight layout is `[out_channels, in_channels]`** = PyTorch `F.linear` layout, so XNNPACK is created with `flags=0` (no `XNN_FLAG_TRANSPOSE_WEIGHTS`).
- **Reuse XNNPACK, don't add a BLAS.** Projections go through `xnn_*_fully_connected_nc_f32`; do not hand-roll GEMM (except as a test oracle) and do not introduce OpenBLAS/MKL.
- **CMake ≥3.24**; whole-archive only via `$<LINK_LIBRARY:WHOLE_ARCHIVE,etnp_kernels>`. Native test targets are guarded by `if(TARGET etnp_kernels)` and are added to `native_tests/CMakeLists.txt`.
- **Registration name `etnp::lstm.out` must equal the string a consumer's `.pte` serializes.** Registration is static-init; the seam's nm-guard requires `_GLOBAL__sub_I_etnp_lstm.cpp`.
- **XNNPACK single-threaded for now:** pass `pthreadpool_t = nullptr` to reshape/run. Multicore via the runtime pthreadpool is a later enhancement (avoid two competing thread pools).

## File Structure

- `examples/custom_kernels/lstm/xnn_linear.h` — header-only RAII wrapper over XNNPACK f32 fully-connected (`create/reshape/setup/run/delete`) + `ensure_xnn_initialized()`. Included by the kernel and by the Task-1 unit test.
- `examples/custom_kernels/lstm/etnp_lstm.cpp` — the `etnp::lstm.out` kernel + `EXECUTORCH_LIBRARY` registration.
- `native_tests/xnn_linear_test.cpp` — Task 1: isolated proof that we can drive XNNPACK FC directly (vs a naive-matmul oracle).
- `native_tests/lstm_kernel_test.cpp` — Task 2: analytic-case correctness (zero weights) via the operator registry, no PyTorch.
- `tools/gen_lstm_golden.py` — Task 3: generates the committed golden header from `torch.nn.LSTM`.
- `native_tests/lstm_golden.h` — Task 3: generated golden fixture (committed).
- `native_tests/lstm_parity_test.cpp` — Task 3: parity of `etnp::lstm.out` vs the golden.
- `native_tests/CMakeLists.txt` — modified: three new test targets, guarded by `if(TARGET etnp_kernels)`.
- `.github/workflows/qa-gate.yml` — modified: build+run the LSTM tests with the example injected.

---

### Task 1: XNNPACK fully-connected helper + isolated proof

De-risks the whole approach: prove we can create and run an XNNPACK FC operator *directly* (outside the delegate) and get a correct `input @ W^T + bias`.

**Files:**
- Create: `examples/custom_kernels/lstm/xnn_linear.h`
- Create: `native_tests/xnn_linear_test.cpp`
- Modify: `native_tests/CMakeLists.txt`

**Interfaces:**
- Produces: `etnp::XnnLinear` (header-only) with `static Result<XnnLinear> create(size_t in_ch, size_t out_ch, const float* weight /*[out_ch,in_ch]*/, const float* bias /*nullable*/)` and `Error run(size_t batch, const float* input, float* out, pthreadpool_t tp)`; and `inline Error etnp::ensure_xnn_initialized()`.
- Consumes: shipped `xnnpack.h` / `pthreadpool.h` (include dirs come from the `executorch` imported target) and the XNNPACK symbols linked transitively via `${ETNP_HARNESS_LIBS}` (which includes `xnnpack_backend`).

- [ ] **Step 1: Write the failing test**

Create `native_tests/xnn_linear_test.cpp`:

```cpp
// Proves we can drive XNNPACK's f32 fully-connected operator directly (outside the
// ExecuTorch delegate) and get input @ weight^T + bias. Oracle: a naive triple loop.
#include <cstdio>
#include <vector>

#include "xnn_linear.h"

using etnp::XnnLinear;
using etnp::ensure_xnn_initialized;
using executorch::runtime::Error;

int main() {
  if (ensure_xnn_initialized() != Error::Ok) {
    std::fprintf(stderr, "FAIL: xnn_initialize\n");
    return 1;
  }
  const size_t IC = 3, OC = 4, B = 2;
  // weight [OC, IC], row-major (PyTorch F.linear layout)
  const std::vector<float> W = {1, 2, 3,  0, 1, 0,  -1, 0, 1,  2, -1, 0};
  const std::vector<float> bias = {0.5f, -0.5f, 1.0f, 0.0f};
  const std::vector<float> in = {1, 0, -1,   2, 1, 0}; // [B, IC]

  auto lin = XnnLinear::create(IC, OC, W.data(), bias.data());
  if (!lin.ok()) { std::fprintf(stderr, "FAIL: create\n"); return 1; }

  std::vector<float> out(B * OC, 0.0f);
  if (lin->run(B, in.data(), out.data(), /*tp=*/nullptr) != Error::Ok) {
    std::fprintf(stderr, "FAIL: run\n"); return 1;
  }

  // Oracle: out[b,o] = bias[o] + sum_k in[b,k]*W[o,k]
  for (size_t b = 0; b < B; ++b) {
    for (size_t o = 0; o < OC; ++o) {
      float ref = bias[o];
      for (size_t k = 0; k < IC; ++k) ref += in[b * IC + k] * W[o * IC + k];
      const float got = out[b * OC + o];
      if (std::abs(got - ref) > 1e-5f) {
        std::fprintf(stderr, "FAIL: out[%zu,%zu]=%f ref=%f\n", b, o, got, ref);
        return 1;
      }
    }
  }
  std::printf("OK: XnnLinear matches naive matmul\n");
  return 0;
}
```

- [ ] **Step 2: Add the test target and run to verify it fails to compile/link (header absent)**

In `native_tests/CMakeLists.txt`, at the end, add:

```cmake
# XNNPACK-FC helper unit test (Task 1). Isolated: does NOT need etnp_kernels.
add_executable(xnn_linear_test xnn_linear_test.cpp)
target_include_directories(xnn_linear_test PRIVATE
  ${CMAKE_SOURCE_DIR}/../examples/custom_kernels/lstm)
target_link_libraries(xnn_linear_test PRIVATE ${ETNP_HARNESS_LIBS})
```

Run:
```bash
cmake -S native_tests -B build/lstm && cmake --build build/lstm --target xnn_linear_test
```
Expected: FAIL — `xnn_linear.h` does not exist yet.

- [ ] **Step 3: Write the helper**

Create `examples/custom_kernels/lstm/xnn_linear.h`:

```cpp
// out[b,:] = input[b,:] @ weight^T + bias  — weight is [OC, IC] (PyTorch F.linear layout).
// RAII wrapper over XNNPACK's f32 fully-connected operator: the same tuned microkernels
// the XNNPACK delegate runs for lowered linears. No new dependency; single-threaded here.
#pragma once
#include <limits>

#include <pthreadpool.h>
#include <xnnpack.h>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

namespace etnp {
using executorch::runtime::Error;

// XNNPACK needs one process-wide init before any operator is created. C++11 static-local
// init is thread-safe and runs exactly once.
inline Error ensure_xnn_initialized() {
  static const xnn_status s = xnn_initialize(/*allocator=*/nullptr);
  return s == xnn_status_success ? Error::Ok : Error::Internal;
}

class XnnLinear {
 public:
  static executorch::runtime::Result<XnnLinear> create(
      size_t in_ch, size_t out_ch,
      const float* weight,          // [out_ch, in_ch]; not copied — must outlive the op
      const float* bias /*nullable*/) {
    xnn_operator_t op = nullptr;
    const xnn_status s = xnn_create_fully_connected_nc_f32(
        in_ch, out_ch,
        /*input_stride=*/in_ch, /*output_stride=*/out_ch,
        weight, bias,
        -std::numeric_limits<float>::infinity(),  // output_min: no fused activation
        +std::numeric_limits<float>::infinity(),  // output_max
        /*flags=*/0,                              // weight already [OC, IC]
        /*weights_cache=*/nullptr,
        &op);
    if (s != xnn_status_success) return Error::Internal;
    return XnnLinear(op);
  }

  // `out` must hold batch*out_ch floats. tp may be null (single-threaded SIMD).
  Error run(size_t batch, const float* input, float* out, pthreadpool_t tp) {
    if (xnn_reshape_fully_connected_nc_f32(op_, batch, tp) != xnn_status_success)
      return Error::Internal;
    if (xnn_setup_fully_connected_nc_f32(op_, input, out) != xnn_status_success)
      return Error::Internal;
    if (xnn_run_operator(op_, tp) != xnn_status_success) return Error::Internal;
    return Error::Ok;
  }

  ~XnnLinear() { if (op_) xnn_delete_operator(op_); }
  XnnLinear(XnnLinear&& o) noexcept : op_(o.op_) { o.op_ = nullptr; }
  XnnLinear(const XnnLinear&) = delete;
  XnnLinear& operator=(const XnnLinear&) = delete;
  XnnLinear& operator=(XnnLinear&&) = delete;

 private:
  explicit XnnLinear(xnn_operator_t op) : op_(op) {}
  xnn_operator_t op_ = nullptr;
};
} // namespace etnp
```

- [ ] **Step 4: Run to verify it passes**

Run:
```bash
cmake --build build/lstm --target xnn_linear_test && ./build/lstm/xnn_linear_test; echo "exit=$?"
```
Expected: prints `OK: XnnLinear matches naive matmul`, `exit=0`. If linking fails on `xnn_*` symbols, add `XNNPACK` to the target's link libraries explicitly and re-run (document if needed).

- [ ] **Step 5: Commit**

```bash
git add examples/custom_kernels/lstm/xnn_linear.h native_tests/xnn_linear_test.cpp native_tests/CMakeLists.txt
git commit -m "feat: XNNPACK fully-connected helper (XnnLinear) + isolated test

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Sequence-level `etnp::lstm` kernel + analytic correctness

**Files:**
- Create: `examples/custom_kernels/lstm/etnp_lstm.cpp`
- Create: `native_tests/lstm_kernel_test.cpp`
- Modify: `native_tests/CMakeLists.txt`

**Interfaces:**
- Produces: registered op `etnp::lstm.out` with kernel signature
  `std::tuple<Tensor&,Tensor&,Tensor&> lstm_out(KernelRuntimeContext&, const Tensor& input[T,B,I], const Tensor& h0[B,H], const Tensor& c0[B,H], const Tensor& w_ih[4H,I], const Tensor& w_hh[4H,H], const std::optional<Tensor>& b_ih, const std::optional<Tensor>& b_hh, Tensor& output[T,B,H], Tensor& hn[B,H], Tensor& cn[B,H])`.
- Consumes: `etnp::XnnLinear` + `ensure_xnn_initialized()` (Task 1); the seam's `EXECUTORCH_LIBRARY` macro and `etnp_kernels` compilation path; the operator-registry lookup API (`registry_has_op_function`, `get_op_function_from_registry`, `OpFunction = void(*)(KernelRuntimeContext&, Span<EValue*>)`).

- [ ] **Step 1: Write the failing test (analytic zero-weight case)**

With `w_ih=w_hh=0`, `b_ih=b_hh=0`: every gate pre-activation is 0, so `i=f=o=sigmoid(0)=0.5`, `g=tanh(0)=0`. Then `c_t = 0.5*c_{t-1}` and `h_t = 0.5*tanh(c_t)`, independent of input. With `c0=1`, `h0=0`, `T=3`: `c = {0.5, 0.25, 0.125}`, `h_t = 0.5*tanh(c_t)`.

Create `native_tests/lstm_kernel_test.cpp`:

```cpp
// Analytic correctness for etnp::lstm.out with zero weights/bias (no PyTorch needed):
// c_t = 0.5*c_{t-1}, h_t = 0.5*tanh(c_t). Invoked through the operator registry.
#include <cmath>
#include <cstdio>
#include <vector>

#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::extension::make_tensor_ptr;
using executorch::runtime::EValue;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::OpFunction;
using executorch::runtime::Span;

int main() {
  executorch::runtime::runtime_init();
  static constexpr char kOp[] = "etnp::lstm.out";
  if (!executorch::runtime::registry_has_op_function(kOp)) {
    std::fprintf(stderr, "FAIL: %s not registered\n", kOp); return 1;
  }
  auto fn_r = executorch::runtime::get_op_function_from_registry(kOp, {});
  if (!fn_r.ok()) { std::fprintf(stderr, "FAIL: lookup\n"); return 1; }
  OpFunction fn = fn_r.get();

  const int64_t T = 3, B = 1, I = 2, H = 2;
  std::vector<float> in(T * B * I, 0.0f);
  std::vector<float> w_ih(4 * H * I, 0.0f), w_hh(4 * H * H, 0.0f);
  std::vector<float> b_ih(4 * H, 0.0f), b_hh(4 * H, 0.0f);
  std::vector<float> h0(B * H, 0.0f), c0(B * H, 1.0f);
  std::vector<float> out(T * B * H, -1.0f), hn(B * H, -1.0f), cn(B * H, -1.0f);

  auto t_in = make_tensor_ptr({T, B, I}, in.data());
  auto t_h0 = make_tensor_ptr({B, H}, h0.data());
  auto t_c0 = make_tensor_ptr({B, H}, c0.data());
  auto t_wih = make_tensor_ptr({4 * H, I}, w_ih.data());
  auto t_whh = make_tensor_ptr({4 * H, H}, w_hh.data());
  auto t_bih = make_tensor_ptr({4 * H}, b_ih.data());
  auto t_bhh = make_tensor_ptr({4 * H}, b_hh.data());
  auto t_out = make_tensor_ptr({T, B, H}, out.data());
  auto t_hn = make_tensor_ptr({B, H}, hn.data());
  auto t_cn = make_tensor_ptr({B, H}, cn.data());

  EValue ev[] = {EValue(*t_in), EValue(*t_h0), EValue(*t_c0), EValue(*t_wih),
                 EValue(*t_whh), EValue(*t_bih), EValue(*t_bhh),
                 EValue(*t_out), EValue(*t_hn), EValue(*t_cn)};
  EValue* args[] = {&ev[0], &ev[1], &ev[2], &ev[3], &ev[4],
                    &ev[5], &ev[6], &ev[7], &ev[8], &ev[9]};
  KernelRuntimeContext ctx;
  fn(ctx, Span<EValue*>(args, 10));

  float c = 1.0f;
  for (int64_t t = 0; t < T; ++t) {
    c = 0.5f * c;
    const float h = 0.5f * std::tanh(c);
    for (int64_t j = 0; j < H; ++j) {
      if (std::abs(out[t * B * H + j] - h) > 1e-5f) {
        std::fprintf(stderr, "FAIL: out[t=%lld]=%f expected %f\n",
                     (long long)t, out[t * B * H + j], h);
        return 1;
      }
    }
  }
  std::printf("OK: etnp::lstm.out analytic recurrence correct\n");
  return 0;
}
```

- [ ] **Step 2: Add the target; run to verify it fails (op not registered)**

In `native_tests/CMakeLists.txt`, inside the existing `if(TARGET etnp_kernels)` block (add if not present — mirror the seam's kernel-test block), add:

```cmake
  add_executable(lstm_kernel_test lstm_kernel_test.cpp)
  target_link_libraries(lstm_kernel_test PRIVATE
    ${ETNP_HARNESS_LIBS} "$<LINK_LIBRARY:WHOLE_ARCHIVE,etnp_kernels>")
  add_custom_command(TARGET lstm_kernel_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -DSO=$<TARGET_FILE:lstm_kernel_test> -DNM=nm
            "-DEXTRA_TUS=${ETNP_KERNEL_EXPECT_TUS}"
            -P ${CMAKE_SOURCE_DIR}/../cmake/assert_kernels_registered.cmake
    VERBATIM)
```

Run (inject the LSTM source so `etnp_kernels` contains it):
```bash
cmake -S native_tests -B build/lstm \
  -DETNP_EXTRA_KERNEL_SOURCES="$PWD/examples/custom_kernels/lstm/etnp_lstm.cpp"
cmake --build build/lstm --target lstm_kernel_test
```
Expected: FAIL — `etnp_lstm.cpp` doesn't exist yet (configure error or the guard fails on the missing TU).

- [ ] **Step 3: Write the kernel**

Create `examples/custom_kernels/lstm/etnp_lstm.cpp`:

```cpp
// etnp::lstm.out — single-layer, unidirectional, batch_first=False, float32 LSTM over a
// full sequence. Gate projections reuse XNNPACK FC (created once, run per timestep); the
// gate activations + cell/hidden update are fused into one pass per timestep.
//
// Schema:
//   etnp::lstm.out(Tensor input, Tensor h0, Tensor c0, Tensor w_ih, Tensor w_hh,
//                  Tensor? b_ih, Tensor? b_hh, *,
//                  Tensor(a!) output, Tensor(b!) hn, Tensor(c!) cn)
//                  -> (Tensor(a!), Tensor(b!), Tensor(c!))
//   input [T,B,I]  h0/c0 [B,H]  w_ih [4H,I]  w_hh [4H,H]  b_ih/b_hh [4H]
//   output [T,B,H]  hn/cn [B,H].  Gate row order: i,f,g,o (PyTorch).
#include <cmath>
#include <cstring>
#include <optional>
#include <tuple>

#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#include "xnn_linear.h"

namespace etnp {
namespace {
using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::KernelRuntimeContext;

inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

std::tuple<Tensor&, Tensor&, Tensor&> lstm_out(
    KernelRuntimeContext& ctx,
    const Tensor& input, const Tensor& h0, const Tensor& c0,
    const Tensor& w_ih, const Tensor& w_hh,
    const std::optional<Tensor>& b_ih, const std::optional<Tensor>& b_hh,
    Tensor& output, Tensor& hn, Tensor& cn) {
  auto ret = std::tie(output, hn, cn);

  const int64_t T = input.size(0);
  const int64_t B = input.size(1);
  const int64_t I = input.size(2);
  const int64_t H = h0.size(1);

  ET_KERNEL_CHECK(ctx, input.scalar_type() == ScalarType::Float, InvalidArgument, ret);
  ET_KERNEL_CHECK(ctx, w_ih.size(0) == 4 * H && w_ih.size(1) == I, InvalidArgument, ret);
  ET_KERNEL_CHECK(ctx, w_hh.size(0) == 4 * H && w_hh.size(1) == H, InvalidArgument, ret);

  const SizesType osz[3] = {static_cast<SizesType>(T), static_cast<SizesType>(B),
                            static_cast<SizesType>(H)};
  ET_KERNEL_CHECK(ctx,
      executorch::runtime::resize_tensor(output, {osz, 3}) == Error::Ok, InvalidArgument, ret);
  ET_KERNEL_CHECK(ctx,
      executorch::runtime::resize_tensor(hn, h0.sizes()) == Error::Ok, InvalidArgument, ret);
  ET_KERNEL_CHECK(ctx,
      executorch::runtime::resize_tensor(cn, c0.sizes()) == Error::Ok, InvalidArgument, ret);
  ET_KERNEL_CHECK(ctx, ensure_xnn_initialized() == Error::Ok, Internal, ret);

  // Create the two projections ONCE (weights are constant across the sequence).
  auto lin_ih = XnnLinear::create(
      I, 4 * H, w_ih.const_data_ptr<float>(),
      b_ih.has_value() ? b_ih->const_data_ptr<float>() : nullptr);
  auto lin_hh = XnnLinear::create(
      H, 4 * H, w_hh.const_data_ptr<float>(),
      b_hh.has_value() ? b_hh->const_data_ptr<float>() : nullptr);
  ET_KERNEL_CHECK(ctx, lin_ih.ok() && lin_hh.ok(), Internal, ret);

  const size_t g_bytes = static_cast<size_t>(B) * 4 * H * sizeof(float);
  auto g_ih_r = ctx.allocate_temp(g_bytes);
  auto g_hh_r = ctx.allocate_temp(g_bytes);
  ET_KERNEL_CHECK(ctx, g_ih_r.ok() && g_hh_r.ok(), MemoryAllocationFailed, ret);
  float* g_ih = static_cast<float*>(g_ih_r.get());
  float* g_hh = static_cast<float*>(g_hh_r.get());

  // Running state lives in hn/cn, seeded from h0/c0.
  float* h = hn.mutable_data_ptr<float>();
  float* c = cn.mutable_data_ptr<float>();
  std::memcpy(h, h0.const_data_ptr<float>(), static_cast<size_t>(B) * H * sizeof(float));
  std::memcpy(c, c0.const_data_ptr<float>(), static_cast<size_t>(B) * H * sizeof(float));

  const float* in = input.const_data_ptr<float>();
  float* out = output.mutable_data_ptr<float>();

  for (int64_t t = 0; t < T; ++t) {
    // Projections: g_ih = x_t @ W_ih^T + b_ih ; g_hh = h_{t-1} @ W_hh^T + b_hh.
    // lin_hh reads the current h fully into g_hh before the fused loop overwrites h.
    ET_KERNEL_CHECK(ctx,
        lin_ih->run(B, in + t * B * I, g_ih, /*tp=*/nullptr) == Error::Ok, Internal, ret);
    ET_KERNEL_CHECK(ctx,
        lin_hh->run(B, h, g_hh, /*tp=*/nullptr) == Error::Ok, Internal, ret);

    float* out_t = out + t * B * H;
    for (int64_t b = 0; b < B; ++b) {
      const float* aih = g_ih + b * 4 * H;  // gate blocks: [i|f|g|o], each length H
      const float* ahh = g_hh + b * 4 * H;
      for (int64_t j = 0; j < H; ++j) {
        const float ig = sigmoidf(aih[0 * H + j] + ahh[0 * H + j]);
        const float fg = sigmoidf(aih[1 * H + j] + ahh[1 * H + j]);
        const float gg = std::tanh(aih[2 * H + j] + ahh[2 * H + j]);
        const float og = sigmoidf(aih[3 * H + j] + ahh[3 * H + j]);
        const float cnew = fg * c[b * H + j] + ig * gg;
        const float hnew = og * std::tanh(cnew);
        c[b * H + j] = cnew;
        h[b * H + j] = hnew;
        out_t[b * H + j] = hnew;
      }
    }
  }
  return ret;  // hn/cn already hold the final state
}
}  // namespace
}  // namespace etnp

EXECUTORCH_LIBRARY(etnp, "lstm.out", etnp::lstm_out);
```

- [ ] **Step 4: Build and verify the analytic test passes**

Run:
```bash
rm -rf build/lstm
cmake -S native_tests -B build/lstm \
  -DETNP_EXTRA_KERNEL_SOURCES="$PWD/examples/custom_kernels/lstm/etnp_lstm.cpp"
cmake --build build/lstm --target lstm_kernel_test
./build/lstm/lstm_kernel_test; echo "exit=$?"
```
Expected: guard passes (`_GLOBAL__sub_I_etnp_lstm.cpp` present); prints `OK: etnp::lstm.out analytic recurrence correct`; `exit=0`.

- [ ] **Step 5: Commit**

```bash
git add examples/custom_kernels/lstm/etnp_lstm.cpp native_tests/lstm_kernel_test.cpp native_tests/CMakeLists.txt
git commit -m "feat: sequence-level etnp::lstm kernel (XNNPACK projections + fused tail)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Numerical parity against `torch.nn.LSTM`

Golden data generated once in the export venv (has PyTorch), emitted as a committed C++ header, checked in a native test that needs no PyTorch at run time.

**Files:**
- Create: `tools/gen_lstm_golden.py`
- Create: `native_tests/lstm_golden.h` (generated, committed)
- Create: `native_tests/lstm_parity_test.cpp`
- Modify: `native_tests/CMakeLists.txt`
- Modify: `.github/workflows/qa-gate.yml`

**Interfaces:**
- Consumes: registered `etnp::lstm.out` (Task 2); the registry-invoke pattern (Task 2).
- Produces: `lstm_golden.h` exposing `namespace lstm_golden { constexpr int T,B,I,H; constexpr float input[], w_ih[], w_hh[], b_ih[], b_hh[], h0[], c0[], output[], hn[], cn[]; }`.

- [ ] **Step 1: Write the golden generator**

Create `tools/gen_lstm_golden.py`:

```python
"""Generate native_tests/lstm_golden.h from torch.nn.LSTM (single layer, unidirectional,
batch_first=False). Run in the ExecuTorch export venv:
  /home/corey/workspace/executorch/.venv/bin/python tools/gen_lstm_golden.py
"""
import torch

T, B, I, H = 4, 2, 3, 5
torch.manual_seed(0)
lstm = torch.nn.LSTM(I, H, num_layers=1, bias=True, batch_first=False, bidirectional=False).eval()
x = torch.randn(T, B, I)
h0 = torch.randn(B, H)
c0 = torch.randn(B, H)
with torch.no_grad():
    out, (hn, cn) = lstm(x, (h0.unsqueeze(0), c0.unsqueeze(0)))

def arr(name, t):
    flat = ", ".join(f"{v:.8e}f" for v in t.flatten().tolist())
    return f"constexpr float {name}[] = {{{flat}}};"

lines = [
    "// GENERATED by tools/gen_lstm_golden.py — do not edit by hand.",
    "#pragma once",
    "namespace lstm_golden {",
    f"constexpr int T = {T}, B = {B}, I = {I}, H = {H};",
    arr("input", x),
    arr("h0", h0), arr("c0", c0),
    arr("w_ih", lstm.weight_ih_l0), arr("w_hh", lstm.weight_hh_l0),
    arr("b_ih", lstm.bias_ih_l0), arr("b_hh", lstm.bias_hh_l0),
    arr("output", out), arr("hn", hn.squeeze(0)), arr("cn", cn.squeeze(0)),
    "}  // namespace lstm_golden",
]
open("native_tests/lstm_golden.h", "w").write("\n".join(lines) + "\n")
print("wrote native_tests/lstm_golden.h")
```

- [ ] **Step 2: Generate the golden header**

Run:
```bash
/home/corey/workspace/executorch/.venv/bin/python tools/gen_lstm_golden.py
```
Expected: writes `native_tests/lstm_golden.h`. (If that venv path differs, use any Python env with `torch` installed; the header is committed, so this runs once.)

- [ ] **Step 3: Write the parity test**

Create `native_tests/lstm_parity_test.cpp`:

```cpp
// Parity of etnp::lstm.out against torch.nn.LSTM (golden fixture). No PyTorch at run time.
#include <cmath>
#include <cstdio>
#include <vector>

#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/runtime.h>

#include "lstm_golden.h"

using executorch::extension::make_tensor_ptr;
using executorch::runtime::EValue;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::OpFunction;
using executorch::runtime::Span;
namespace g = lstm_golden;

static bool close(const std::vector<float>& got, const float* ref, int n, const char* what) {
  for (int i = 0; i < n; ++i) {
    const float a = got[i], b = ref[i];
    if (std::abs(a - b) > 1e-4f + 1e-4f * std::abs(b)) {
      std::fprintf(stderr, "FAIL %s[%d]: got %f ref %f\n", what, i, a, b);
      return false;
    }
  }
  return true;
}

int main() {
  executorch::runtime::runtime_init();
  auto fn_r = executorch::runtime::get_op_function_from_registry("etnp::lstm.out", {});
  if (!fn_r.ok()) { std::fprintf(stderr, "FAIL: lstm not registered\n"); return 1; }
  OpFunction fn = fn_r.get();

  std::vector<float> in(g::input, g::input + g::T * g::B * g::I);
  std::vector<float> h0(g::h0, g::h0 + g::B * g::H), c0(g::c0, g::c0 + g::B * g::H);
  std::vector<float> wih(g::w_ih, g::w_ih + 4 * g::H * g::I);
  std::vector<float> whh(g::w_hh, g::w_hh + 4 * g::H * g::H);
  std::vector<float> bih(g::b_ih, g::b_ih + 4 * g::H), bhh(g::b_hh, g::b_hh + 4 * g::H);
  std::vector<float> out(g::T * g::B * g::H, 0.f), hn(g::B * g::H, 0.f), cn(g::B * g::H, 0.f);

  auto t_in = make_tensor_ptr({g::T, g::B, g::I}, in.data());
  auto t_h0 = make_tensor_ptr({g::B, g::H}, h0.data());
  auto t_c0 = make_tensor_ptr({g::B, g::H}, c0.data());
  auto t_wih = make_tensor_ptr({4 * g::H, g::I}, wih.data());
  auto t_whh = make_tensor_ptr({4 * g::H, g::H}, whh.data());
  auto t_bih = make_tensor_ptr({4 * g::H}, bih.data());
  auto t_bhh = make_tensor_ptr({4 * g::H}, bhh.data());
  auto t_out = make_tensor_ptr({g::T, g::B, g::H}, out.data());
  auto t_hn = make_tensor_ptr({g::B, g::H}, hn.data());
  auto t_cn = make_tensor_ptr({g::B, g::H}, cn.data());

  EValue ev[] = {EValue(*t_in), EValue(*t_h0), EValue(*t_c0), EValue(*t_wih), EValue(*t_whh),
                 EValue(*t_bih), EValue(*t_bhh), EValue(*t_out), EValue(*t_hn), EValue(*t_cn)};
  EValue* args[10];
  for (int i = 0; i < 10; ++i) args[i] = &ev[i];
  KernelRuntimeContext ctx;
  fn(ctx, Span<EValue*>(args, 10));

  bool ok = close(out, g::output, g::T * g::B * g::H, "output")
          & close(hn, g::hn, g::B * g::H, "hn")
          & close(cn, g::cn, g::B * g::H, "cn");
  if (!ok) return 1;
  std::printf("OK: etnp::lstm.out matches torch.nn.LSTM (rtol/atol 1e-4)\n");
  return 0;
}
```

- [ ] **Step 4: Add the target, build, and verify parity**

In `native_tests/CMakeLists.txt`, inside the `if(TARGET etnp_kernels)` block, add:

```cmake
  add_executable(lstm_parity_test lstm_parity_test.cpp)
  target_link_libraries(lstm_parity_test PRIVATE
    ${ETNP_HARNESS_LIBS} "$<LINK_LIBRARY:WHOLE_ARCHIVE,etnp_kernels>")
```

Run:
```bash
cmake -S native_tests -B build/lstm \
  -DETNP_EXTRA_KERNEL_SOURCES="$PWD/examples/custom_kernels/lstm/etnp_lstm.cpp"
cmake --build build/lstm --target lstm_parity_test
./build/lstm/lstm_parity_test; echo "exit=$?"
```
Expected: prints `OK: etnp::lstm.out matches torch.nn.LSTM (rtol/atol 1e-4)`; `exit=0`.

- [ ] **Step 5: Wire the LSTM tests into CI**

In `.github/workflows/qa-gate.yml`, after the existing custom-kernel steps, add:

```yaml
      - name: Build + run LSTM example kernel tests
        run: |
          cmake -S native_tests -B build/lstm \
            -DETNP_EXTRA_KERNEL_SOURCES="$PWD/examples/custom_kernels/lstm/etnp_lstm.cpp"
          cmake --build build/lstm --target xnn_linear_test lstm_kernel_test lstm_parity_test
          ./build/lstm/xnn_linear_test
          ./build/lstm/lstm_kernel_test
          ./build/lstm/lstm_parity_test
```

- [ ] **Step 6: Commit**

```bash
git add tools/gen_lstm_golden.py native_tests/lstm_golden.h native_tests/lstm_parity_test.cpp \
        native_tests/CMakeLists.txt .github/workflows/qa-gate.yml
git commit -m "test: LSTM parity vs torch.nn.LSTM via committed golden fixture

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## MVP Performance Probe (Tasks 4–6) — relaxed QC

> **Strategy:** per the [LSTM op delivery strategy](../specs/2026-07-09-lstm-op-delivery-strategy.md), the op's permanent home is the upstream relocatable-library tarball, and this repo will eventually just consume it. Tasks 4–6 are **not** that productionization — they are a throwaway **MVP** whose only deliverable is a *performance signal*: does the custom op beat naive decomposition on `.pte` size and execution latency, and can it emit a `.pte` the naive path can't?
>
> **Relaxed quality bar (explicit):** these tasks build throwaway tooling. Per-task review checks only that (a) it runs and (b) the reported numbers are *honest* — same weights and inputs on both sides, a correctness cross-check at rtol 1e-4, no cherry-picked config. It does NOT hold the export tooling, the bench scripts, or the injected-kernel bench build to permanent-change standards (no full dtype/shape matrix, no polish, no durable artifacts, no CI wiring). The correctness backing is the existing Task 1–3 tests. Anything that would need to be bulletproof to *ship* is deferred to **Deferred: upstream productionization** below.

---

### Task 4 (MVP): produce a `.pte` that references `etnp::lstm.out`

The benchmark needs a `.pte` carrying the custom op, so register `etnp::lstm` for `torch.export` and confirm it survives lowering (not decomposed). This is throwaway export tooling: the eager reference impl only needs to be correct enough to trace and shape-propagate — it is **not** the shipping AOT definition (that lives upstream later, per the strategy doc). Highest-uncertainty step; validate in the ExecuTorch export venv.

**Files:**
- Create: `tools/etnp_lstm_op.py` — registers `etnp::lstm` with `torch.library` (schema + reference impl + fake/meta + out-variant) so `to_edge_transform_and_lower` keeps it as one opaque op.

**Interfaces:**
- Produces: a Python-importable module that, on import, registers the `etnp::lstm` operator with a schema matching the C++ kernel: `lstm(Tensor input, Tensor h0, Tensor c0, Tensor w_ih, Tensor w_hh, Tensor? b_ih, Tensor? b_hh) -> (Tensor output, Tensor hn, Tensor cn)` plus its `.out` variant. Semantics (gate order i,f,g,o; shapes) match `examples/custom_kernels/lstm/etnp_lstm.cpp` exactly.

- [ ] **Step 1: Write the op registration + reference implementation**

Create `tools/etnp_lstm_op.py`. Use a functional reference (composed from torch ops) as both the eager implementation and the tracing semantics, a `register_fake` for shape propagation, and register the out-variant so lowering emits `etnp::lstm.out`:

```python
"""Registers etnp::lstm for torch.export so a model calling it lowers to a .pte that
references etnp::lstm.out (matching examples/custom_kernels/lstm/etnp_lstm.cpp).
Import this module before exporting. Run/validate in the ExecuTorch export venv.
"""
import torch
from torch.library import Library, impl, register_fake

_lib = Library("etnp", "DEF")
_lib.define(
    "lstm(Tensor input, Tensor h0, Tensor c0, Tensor w_ih, Tensor w_hh, "
    "Tensor? b_ih, Tensor? b_hh) -> (Tensor, Tensor, Tensor)")

@impl(_lib, "lstm", "CompositeExplicitAutograd")
def _lstm_impl(input, h0, c0, w_ih, w_hh, b_ih, b_hh):
    T, B, _ = input.shape
    H = h0.shape[1]
    h, c = h0, c0
    outs = []
    for t in range(T):
        g = input[t] @ w_ih.t() + h @ w_hh.t()
        if b_ih is not None: g = g + b_ih
        if b_hh is not None: g = g + b_hh
        i, f, gg, o = g[:, 0:H], g[:, H:2*H], g[:, 2*H:3*H], g[:, 3*H:4*H]
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(gg)
        h = torch.sigmoid(o) * torch.tanh(c)
        outs.append(h)
    return torch.stack(outs, 0), h, c

@register_fake("etnp::lstm")
def _lstm_fake(input, h0, c0, w_ih, w_hh, b_ih, b_hh):
    T, B, _ = input.shape
    H = h0.shape[1]
    return (input.new_empty((T, B, H)), h0.new_empty((B, H)), c0.new_empty((B, H)))
```

- [ ] **Step 2: Verify the op survives lowering to a `.pte` (export venv)**

Run in the export venv and confirm `etnp::lstm.out` appears in the lowered program (NOT decomposed):

```bash
/home/corey/workspace/executorch/.venv/bin/python - <<'PY'
import torch, tools.etnp_lstm_op  # noqa: registers the op
from torch.export import export
from executorch.exir import to_edge_transform_and_lower

class M(torch.nn.Module):
    def forward(self, x, h0, c0, wih, whh, bih, bhh):
        return torch.ops.etnp.lstm(x, h0, c0, wih, whh, bih, bhh)

T,B,I,H = 4,2,3,5
ex = (torch.randn(T,B,I), torch.randn(B,H), torch.randn(B,H),
      torch.randn(4*H,I), torch.randn(4*H,H), torch.randn(4*H), torch.randn(4*H))
prog = to_edge_transform_and_lower(export(M().eval(), ex)).to_executorch()
ops = prog.executorch_program.execution_plan[0].operators
names = [f"{o.name}.{o.overload}" if o.overload else o.name for o in ops]
print(names)
assert any("etnp::lstm" in n for n in names), f"op decomposed away: {names}"
print("OK: etnp::lstm survived lowering")
PY
```
Expected: prints a list containing `etnp::lstm.out` and `OK: etnp::lstm survived lowering`. **If it was decomposed**, iterate here (register the op as non-decomposable / adjust the out-variant registration) — this is the crux of custom-op export and is expected to need a pass or two. Document the working recipe in the module docstring.

- [ ] **Step 3: Commit**

```bash
git add tools/etnp_lstm_op.py
git commit -m "feat: torch.library registration of etnp::lstm for AOT export

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5 (MVP): Dual-`.pte` size + execution-time benchmark

Export the SAME logical LSTM two ways — naive decomposition vs our custom op — with identical weights, and compare `.pte` size and wall-clock over a sweep of (T, H, B). This is the headline MVP number.

**Files:**
- Create: `tools/export_lstm_bench_models.py` — emits `lstm_naive_<cfg>.pte` and `lstm_custom_<cfg>.pte` pairs with shared weights; prints a size table.
- Create: `tools/bench_lstm.py` — loads each `.pte` in this runtime, times N iterations, prints size + median-latency table.
- Modify: `README`/`docs/custom-kernels.md` — a short "benchmarking the LSTM example" note (how to build the benchmark extension + run).

**Interfaces:**
- Consumes: `tools.etnp_lstm_op` (Task 4); the built extension WITH the LSTM kernel injected (see Step 2); `executorch_numpy_runtime`'s existing load+run API (as used by `tools/bench.py`).

- [ ] **Step 1: Write the dual-export script**

`tools/export_lstm_bench_models.py`: for each config, build one `torch.nn.LSTM`, export it the normal way (→ naive/decomposed `.pte`), then build a wrapper that calls `torch.ops.etnp.lstm` with that LSTM's *same* weights and export it (→ custom `.pte`). Write both to an output dir and print `config, naive_bytes, custom_bytes`.

- [ ] **Step 2: Build a benchmark extension with the kernel injected, then run both `.pte`s**

The default wheel does not bundle the LSTM op, so build a benchmark-only extension that does, install it into a throwaway venv, and run `bench_lstm.py`:

```bash
CMAKE_ARGS="-DETNP_EXTRA_KERNEL_SOURCES=$PWD/examples/custom_kernels/lstm/etnp_lstm.cpp" \
  uv build --wheel -o /tmp/lstm_bench_wheel
# install that wheel into a venv, then:
python tools/export_lstm_bench_models.py /tmp/lstm_ptes
python tools/bench_lstm.py /tmp/lstm_ptes
```

- [ ] **Step 3: Write the timing harness and record the numbers**

`tools/bench_lstm.py`: for each config, load `lstm_naive` and `lstm_custom`, warm up, time N runs on identical input, and print a table of `config | naive_size | custom_size | naive_ms | custom_ms | size_ratio | speedup`. **The one hard gate (honesty, not polish):** the two paths must use identical weights and inputs, and `custom` vs `naive` outputs must agree at rtol 1e-4 (reuses Task 3's tolerance) — a benchmark comparing two different computations is worthless. Report the `size_ratio`/`speedup` trend vs T as observed; a config where custom does *not* win is a finding to surface, not a failed assertion (MVP: we're measuring, not proving). Do not tune configs to manufacture a win.

Run:
```bash
python tools/bench_lstm.py /tmp/lstm_ptes
```
Expected: a table showing custom `.pte`s smaller and faster, with the gap widening as T grows; the correctness assertion passes.

- [ ] **Step 4: Commit**

```bash
git add tools/export_lstm_bench_models.py tools/bench_lstm.py docs/custom-kernels.md README.md
git commit -m "bench: LSTM custom-op vs naive decomposition (.pte size + latency)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6 (MVP): Feasibility crossover — a `.pte` only the custom op can produce

Demonstrate the size/complexity wall of naive decomposition: sweep T upward until naive export fails or blows a declared practicality budget, while the custom export stays constant and succeeds. This is the "impossible for naive" existence proof from the strategy doc — an artifact + a crossover `T*`, not a polished tool.

**Files:**
- Create: `tools/lstm_feasibility.py` — sweeps T, records naive export outcome (success / size / wall-time / failure) vs custom, and reports the crossover.

**Interfaces:**
- Consumes: `tools.etnp_lstm_op` and the two export paths from Task 5.

- [ ] **Step 1: Write the crossover sweep**

`tools/lstm_feasibility.py`: declare an explicit practicality budget (e.g. naive export must finish under `EXPORT_TIME_BUDGET_S` and produce a `.pte` under `SIZE_BUDGET_MB`). For `T` in an increasing schedule (e.g. 16, 64, 256, 1024, 4096, …), time+attempt the naive export inside a guard that catches export failure/OOM, and always attempt the custom export. Record for each T: naive `{ok, seconds, bytes | error}`, custom `{ok, seconds, bytes}`.

- [ ] **Step 2: Run and capture the crossover**

```bash
python tools/lstm_feasibility.py
```
Expected: a table where, past some `T*`, the naive path exceeds the budget or errors while the custom path still emits a small, near-constant `.pte`. Save the custom `.pte` for the first `T` where naive fails/exceeds budget as the artifact.

- [ ] **Step 3: Document the result and commit**

Add a short "Feasibility" subsection to `docs/custom-kernels.md` stating the observed `T*`, the budget used, and that "impossible" here means *exceeds the declared practicality budget* (size/time/memory), not a universal impossibility.

```bash
git add tools/lstm_feasibility.py docs/custom-kernels.md
git commit -m "bench: LSTM naive-decomposition feasibility crossover (T*)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Deferred: upstream productionization (post-MVP)

Not part of this plan's execution — captured here so the MVP boundary is explicit. Detailed in the [LSTM op delivery strategy](../specs/2026-07-09-lstm-op-delivery-strategy.md). Once the MVP confirms the win:

- **Move the op upstream** into the relocatable-library project (`executorch-runtime-dist`): the runtime kernel (torch-free C++), a *shipping* AOT definition, and a **live** torch round-trip test (export → lower → run → compare to eager) — which supersedes Task 3's committed-golden workaround. Keep the `etnp::lstm.out` name; always-on in every variant.
- **Adopt an upstream `extras` bundle convention** (kernel + op-name-as-single-source-of-truth + AOT def + mandatory round-trip test + `variants:` metadata); generalize the nm-guard into a per-op manifest check. LSTM is extra #1 / the template — build the convention, defer the framework until ops #2–3 exist.
- **In this repo (the shim):** bump `cmake/RuntimePin.cmake` to the new tarball; keep the generic custom-kernel seam; **replace** the Task 1–3 native LSTM tests with a small torch-free *consumer smoke test* (load a committed `.pte` using `etnp::lstm.out`, run it, assert outputs). The `examples/custom_kernels/lstm/` sources can then retire or become a pointer to upstream.
- **Optional, parallel, slow:** an ExecuTorch RFC-appetite issue (design-only, led with the MVP benchmark numbers). Watch the ExecuTorch **1.4.0** release (~2026-07-23) for a surprise LSTM/RNN improvement before investing.

## Self-review notes

- **Spec coverage:** XNNPACK-FC helper + isolated proof (T1) · sequence-level `etnp::lstm` kernel with create-once/run-many projections + fused tail (T2) · registration via the seam + build-guard TU (T2) · analytic correctness without PyTorch (T2) · torch parity via committed golden (T3) · CI wiring (T3) · custom-op AOT export (T4) · dual-`.pte` size + latency benchmark vs naive decomposition (T5) · feasibility crossover / a `.pte` only the custom op can produce (T6). Constraints (example-not-bundled, single-layer/unidirectional/float32, PyTorch gate order, XNNPACK-not-BLAS, whole-archive, `if(TARGET etnp_kernels)`) each map to a task.
- **Task grouping / scope check:** T1–T3 (the kernel + native validation) are DONE and are the permanent, reviewed correctness backing. T4–T6 are a **throwaway MVP performance probe under a relaxed quality bar** (see the banner above Task 4): their only deliverable is an honest size/latency signal + the "impossible-for-naive" existence proof, not shippable code. The real productionization — moving the op upstream, the `extras` mechanism, the shim's consumer smoke test — is **deferred** (see "Deferred: upstream productionization"). T4 is still the highest-risk step (custom-op export must survive lowering) and gates T5–T6.
- **Benchmark honesty (must hold during T5–T6):** the naive path's gate matmuls may ALSO be XNNPACK-delegated, so the execution-time win comes from fewer dispatch boundaries + fused pointwise tail + create-once, not a faster GEMM — the sweep must vary T and H to show this. "Impossible" in T6 means *exceeds a declared practicality budget* (size/time/memory), demonstrated as a crossover T*, not a universal claim.
- **Type consistency:** op name `etnp::lstm.out`, TU `_GLOBAL__sub_I_etnp_lstm.cpp`, kernel signature (arg order input,h0,c0,w_ih,w_hh,b_ih,b_hh,output,hn,cn), and the 10-EValue boxed arg order are identical across the kernel, both native tests, and the parity marshaling. `XnnLinear::create/run` and `ensure_xnn_initialized()` signatures match between the header, the Task-1 test, and the kernel.
- **Known adjustment points (verify at implementation time):** (a) whether `xnn_linear_test`/the kernel need `XNNPACK` added explicitly to link libs or inherit it transitively via `xnnpack_backend` in `${ETNP_HARNESS_LIBS}`; (b) the `ET_KERNEL_CHECK` failure-return form for a 3-tuple `std::tie` return; (c) that `resize_tensor(output, {osz, 3})` selects the `SizesType` overload cleanly; (d) `make_tensor_ptr({...})` brace overloads for 1-D and 3-D shapes. Each surfaces at first compile and is local to fix.
