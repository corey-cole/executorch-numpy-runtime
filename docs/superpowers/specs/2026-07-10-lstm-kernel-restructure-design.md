# Design: restructure `etnp::lstm.out` for performance parity + wins at all H

**Date:** 2026-07-10
**Status:** approved design, ready for implementation planning
**Audience:** the implementing agent. This spec is self-contained — read it fully
before touching code. Background evidence lives in `problem-statement.md`
(root-cause verdict) and `scratchpad/` (probe benches + README); you should not
need to re-derive any measurement.

---

## 1. Background: what is broken and why (read this even if you skip everything else)

`examples/custom_kernels/lstm/etnp_lstm.cpp` implements `etnp::lstm.out`, a
single-layer, unidirectional, `batch_first=False`, float32 LSTM over a full
sequence, as an ExecuTorch custom kernel. It is benchmarked against the "naive"
path (plain `torch.nn.LSTM` exported and lowered through the XNNPACK
partitioner) by `tools/bench_lstm.py`. The custom kernel currently **wins at
H=32 but loses at H>=64**, and loses everywhere on a 16-thread host.

The 2026-07-10 investigation (full verdict appended to `problem-statement.md`)
found three quantified causes. All numbers below were measured on the local
4c/8t i7-1185G7:

1. **Per-execute weight packing.** The kernel calls `XnnLinear::create` (which
   wraps `xnn_create_fully_connected_nc_f32` — XNNPACK repacks the weight
   matrix inside this call) on EVERY inference. Cost for the two projections:
   2µs at H=32, 92µs at H=128, 465µs at H=256 (~H² growth). The XNNPACK
   delegate that the naive path uses packs once at model load.
2. **Input projection not batched.** PyTorch's LSTM decomposition hoists
   `x @ W_ih^T` out of the time loop into ONE `[T·B, I] × [I, 4H]` GEMM; the
   custom kernel instead runs T batch-1 GEMVs. One batched run is 1.5–2.5×
   faster single-threaded, and with a threadpool another 2–3.4× on 8 threads.
3. **Scalar libm cell update — the dominant cost.** The per-timestep gate loop
   makes 5·H scalar `std::exp`/`std::tanh` calls (~30ns per sigmoid+tanh pair
   ⇒ ~10µs/step at H=128, ~60% of the kernel's 17µs/step). The naive path gets
   XNNPACK's vectorized sigmoid/tanh microkernels.

Additionally the kernel passes `pthreadpool_t = nullptr` everywhere
(single-threaded), while the delegate uses the shared ExecuTorch threadpool
sized to all hardware threads.

A library spike (`scratchpad/cell_lib_spike/`, results in §5.2) chose **Google
Highway** for the vectorized cell update and rejected hand-rolled polynomials
(catastrophic accuracy failure: max err 0.68 through a 256-step chain) and
Eigen (compile-time SIMD only — the wheel builds at baseline x86-64/SSE2,
Highway's runtime dispatch reaches AVX-512 from the same baseline build).

## 2. Hard constraints — violating any of these is a failed implementation

- **The op name and schema are FROZEN.** `etnp::lstm.out` with the 10-slot
  boxed signature (input, h0, c0, w_ih, w_hh, b_ih?, b_hh?, output, hn, cn).
  Existing `.pte` files bake this in; the upstream handoff
  (`docs/superpowers/specs/2026-07-09-upstream-lstm-productionization-handoff.md`)
  forbids renaming. The AOT side (`tools/etnp_lstm_op.py`) and export scripts
  are NOT touched.
- **Scope stays MVP:** single-layer, unidirectional, `batch_first=False`, f32,
  contiguous. Shapes: `input [T,B,I]`, `h0/c0 [B,H]`, `w_ih [4H,I]`,
  `w_hh [4H,H]`, biases `[4H]` optional, `output [T,B,H]`, `hn/cn [B,H]`.
  Gate row order **i, f, g, o** (PyTorch). Do not widen.
- **This repo is torch-free.** Nothing in the kernel, tests, or build may
  import torch. Exports happen in the separate export venv (§7.1).
- **Numerical gate:** `tools/bench_lstm.py` cross-checks custom vs naive at
  rtol 1e-4 / atol 1e-4 and this must PASS. Highway's math is ~3e-7 off libm
  through a 256-step chain (measured), so this passes with huge margin — but
  any shortcut that degrades accuracy (fast-math on the whole TU, a cheaper
  approximation) risks the gate. Do not substitute approximations.
- **Registration must keep working.** The kernel registers via a hand-rolled
  boxed trampoline (`register_kernel(Kernel("etnp::lstm.out", lstm_boxed))`),
  NOT `EXECUTORCH_LIBRARY` — the pinned 1.3.1 runtime's auto-unboxing macro
  static_asserts at 1 mutable output and we have 3. Keep the trampoline as-is.
- **The nm post-link guard must keep passing** (see §4.3).

## 3. Success criteria (definition of done)

1. Rebuilt bench wheel + rerun of `tools/bench_lstm.py /tmp/lstm_ptes` shows
   **speedup >= 1.0× for EVERY config present** (H=32/64/128 × T=16/64/256
   pairs that have `.pte`s) on the local i7, with the rtol 1e-4 cross-check
   passing. (Ryzen validation is a follow-up, not a gate.)
2. All existing native tests pass (`lstm_kernel_test`, `lstm_parity_test`,
   `xnn_linear_test`, `kernel_registration_test`), plus the new unit tests
   (§6), plus the TSan race harness and leak harness.
3. The default wheel (without the LSTM kernel) still builds and its test suite
   passes — the Kernels.cmake changes must be inert when no extra kernel is
   injected.

## 4. Design

### 4.1 Kernel flow (`examples/custom_kernels/lstm/etnp_lstm.cpp`)

Replace the body of `lstm_out` with:

1. Validation + output resizing: unchanged from today.
2. Obtain the two packed FC operators from the **op cache** (§4.4) — do NOT
   create them per call. Cache key inputs: weight pointer, bias pointer (may
   be null), in_ch, out_ch.
3. **Batched input projection.** `input` is `[T,B,I]` contiguous, so it is
   also a valid `[T·B, I]` matrix. Run the `lin_ih` operator ONCE with
   `batch = T·B`, writing to a temp buffer `g_ih_all` of `T·B·4H` floats from
   `ctx.allocate_temp`. Pass the **shared threadpool** (§4.5) to this run.
   - Temp allocator note: the numpy runtime's `Module` uses ExecuTorch's
     default malloc-backed `MallocMemoryAllocator` (verified in
     `libextension_module_static.a`), so multi-MB requests are fine at
     runtime. The NATIVE TEST harness wires a fixed 64KiB arena — it must be
     enlarged (§6).
4. Keep a second temp buffer `g_hh` of `B·4H` floats (as today).
5. Time loop, `t = 0..T-1`:
   a. Run `lin_hh` with `batch = B`, input = current `h`, output = `g_hh`,
      threadpool = **nullptr** (a `[1,H]×[H,4H]` GEMV is ~1.8µs at H=128;
      pthreadpool dispatch overhead exceeds the gain — measured decision, do
      not "improve" it).
   b. Call the **Highway fused cell update** (§4.2) with
      `g_ih_t = g_ih_all + t·B·4H`, `g_hh`, and the running `c`, `h`,
      `out_t = out + t·B·H` pointers.
6. `hn`/`cn` already hold the final state (running state lives in them,
   seeded from `h0`/`c0` by memcpy — unchanged from today).

Ordering hazard kept from the original code: `lin_hh` must consume `h`
completely before the cell update overwrites it. The structure above preserves
this (projection first, then fused update).

### 4.2 Highway fused cell update (new files)

New files in `examples/custom_kernels/lstm/`:
- `lstm_cell.h` — plain declaration, no Highway includes:
  ```cpp
  namespace etnp {
  // One timestep for all B rows. Layout per row: gate blocks [i|f|g|o],
  // each length H, in both g_ih_t and g_hh (strides 4H per row).
  // c,h are [B,H]; out_t is the output row block for this timestep [B,H].
  void lstm_cell_update(size_t B, size_t H,
                        const float* g_ih_t, const float* g_hh,
                        float* c, float* h, float* out_t);
  }
  ```
- `lstm_cell.cc` — the Highway per-target TU. **Copy the structure from the
  working spike at `scratchpad/cell_lib_spike/cell_hwy.cc`** — it compiles and
  is numerically validated. Key structural rules a naive port gets wrong:
  - `#include <cmath>` BEFORE `foreach_target.h` (the file is re-included once
    per SIMD target; un-guarded system includes must come first).
  - `#undef HWY_TARGET_INCLUDE` / `#define HWY_TARGET_INCLUDE
    "examples/custom_kernels/lstm/lstm_cell.cc"` — the path must be reachable
    from the compiler's include paths (add the REPO ROOT as an include dir for
    this target, or use the filename form and add the file's own dir; the
    spike used `-I.` with the bare filename).
  - Math: sigma(x) = 1 / (1 + Exp(-x)) using `hwy/contrib/math/math-inl.h`
    `CallExp`; tanh via `CallTanh`. Cell math per lane:
    `c' = f·c + i·g; h' = o·tanh(c')` with gates
    `i = sigma(a_i), f = sigma(a_f), g = tanh(a_g), o = sigma(a_o)` where
    `a_x = g_ih_t[x·H + j] + g_hh[x·H + j]` (per batch row).
  - Scalar tail loop for `H % Lanes != 0` (H is not guaranteed to be a
    multiple of the vector width; the spike has this).
  - Public entry points under `#if HWY_ONCE`: `HWY_EXPORT` +
    `HWY_DYNAMIC_DISPATCH`. Use dynamic dispatch (NOT static — static was 5×
    slower at baseline flags).
  - No dynamic initializers in this TU (keeps the nm-guard logic simple;
    HWY_EXPORT tables are constexpr).

Measured expectation (i7, baseline flags, per step): 0.12µs at H=32, 0.46µs at
H=128, max err vs f64 reference ~3e-7 through a 256-step chain. If your
implementation is >2× off these, something is wrong (check that dispatch
reports an AVX target, not scalar/EMU128).

### 4.3 Build wiring (`cmake/Kernels.cmake` + consumers)

Current state: `ETNP_EXTRA_KERNEL_SOURCES` (semicolon-separated .cpp paths) is
compiled into the `etnp_kernels` static lib; for EVERY source, the post-link
nm-guard (`cmake/assert_kernels_registered.cmake`, invoked from the top-level
`CMakeLists.txt` and `native_tests/CMakeLists.txt`) expects a
`_GLOBAL__sub_I_<basename>` static-init symbol to survive in the final binary.
`lstm_cell.cc` has NO registrar, so injecting it today would make the guard
fail. Changes:

1. **Guard expectation becomes conditional:** in `Kernels.cmake`, compute
   `ETNP_KERNEL_EXPECT_TUS` only for sources whose content contains
   `register_kernel(` — use `file(READ ...)` + `string(FIND ...)` at configure
   time. No new user-facing variables; sources without registrars simply stop
   generating expectations. Verify the guard still FAILS if the real registrar
   TU is dropped (that's its purpose — don't break it).
2. **New option `ETNP_KERNELS_USE_HIGHWAY` (default OFF):** when ON,
   FetchContent Google Highway **pinned by release-tarball URL + SHA256**
   (mirror the hash-pinning style of `cmake/RuntimePin.cmake`; pick the latest
   stable release at implementation time and record the hash — the spike used
   git master @ `9d5b12611fcfe145f988771c45e7bae9f78cb7fa`, 2026-07-09, which
   is known-good). Set `HWY_ENABLE_TESTS=OFF`, `HWY_ENABLE_EXAMPLES=OFF`,
   `HWY_ENABLE_CONTRIB=OFF` (contrib *math* is header-only and needs no built
   lib — verify `hwy/contrib/math/math-inl.h` is installed/reachable from the
   source tree include dir), `BUILD_TESTING=OFF`. Link target `hwy` into
   `etnp_kernels` (PRIVATE is fine).
3. **`extension_threadpool`** from the dist is added to `etnp_kernels`'s link
   libraries (the dist's `ExecuTorchTargets.cmake` exports it; the XNNPACK
   backend already depends on it, but link it explicitly so the standalone
   `native_tests` build resolves `get_pthreadpool()`).
4. Bench-wheel build command becomes (document in `docs/custom-kernels.md`):
   ```
   CMAKE_ARGS="-DETNP_EXTRA_KERNEL_SOURCES=$PWD/examples/custom_kernels/lstm/etnp_lstm.cpp;$PWD/examples/custom_kernels/lstm/lstm_cell.cc -DETNP_KERNELS_USE_HIGHWAY=ON" \
     uv build --wheel -o /tmp/lstm_bench_wheel
   ```
   CAUTION: the semicolon inside the -D value must survive the shell and
   scikit-build-core's CMAKE_ARGS splitting (it splits on SPACES, so the
   semicolon is safe inside one -D token — but verify on the first build; if
   it mangles, fall back to a `;`-escaped or list-file approach).
5. When `ETNP_KERNELS_USE_HIGHWAY=OFF` and no extra sources are given, the
   configure/build must be bit-identical in behavior to today (inert change).

### 4.4 XNNPACK operator cache (`examples/custom_kernels/lstm/xnn_linear.h`, extended or a new `xnn_linear_cache.h`)

Purpose: eliminate the per-execute `xnn_create_fully_connected_nc_f32` weight
repack. Design (agreed with the user — do implement the cache, not the
per-call fallback):

- Process-wide cache: `key = (weight_ptr, bias_ptr, in_ch, out_ch)`;
  value = `{ packed XnnLinear op, content fingerprint, per-entry std::mutex,
  LRU stamp }`.
- **Fingerprint:** cheap hash of a fixed sample of the weight data (e.g. first
  8 floats, last 8 floats, and out_ch·in_ch) read from the CALLER's pointer at
  lookup time. Rationale: `.pte` constant-segment pointers are stable for a
  loaded program's lifetime, but a program can be unloaded and another loaded
  at the same address. The fingerprint catches that and forces a repack.
  SAFETY ARGUMENT (do not "simplify" this away): the cache NEVER dereferences
  a stored pointer. Fingerprints are always computed from the pointer the
  caller just passed in (guaranteed live — they are the kernel's own weight
  args); the stored fingerprint is just a number compared against it. Stale
  entries for freed memory are therefore inert until evicted.
- **Concurrency:** a global `std::mutex` guards map lookup/insert/evict. Each
  entry has its OWN mutex held across `reshape + setup + run` — XNNPACK
  operator objects are mutated by those calls, and although the numpy
  runtime's per-Runtime lock serializes executes within one Runtime, separate
  Runtime instances may execute concurrently. The TSan race harness
  (`native_tests/race_harness.cpp` + `tsan_suppressions.txt`) must stay clean.
- **Bound:** LRU with a small cap (16 entries). Eviction destroys the xnn
  operator (frees packed weights). Evicting an entry whose mutex is held must
  be impossible by construction (e.g. hold a shared_ptr to the entry while
  running so eviction only drops the map reference).
- `XnnLinear::run` keeps its `pthreadpool_t` parameter; the cache is only
  about creation.

### 4.5 Threadpool

Use `executorch::extension::threadpool::get_pthreadpool()` (header
`executorch/extension/threadpool/threadpool.h` in the dist include tree) for
the batched input projection ONLY. This is the same pool the XNNPACK delegate
uses — sharing it avoids oversubscription. Do not create a private
`pthreadpool_create` pool (that was spike-only).

## 5. Evidence backing the design decisions (do not re-litigate these)

### 5.1 Where the time goes (i7, per step, H=128)

| component | current | after restructure |
|---|---|---|
| input projection | ~2µs (GEMV/step) | ~0.3µs amortized (one threaded GEMM) |
| recurrent GEMV | ~1.8µs | ~1.8µs (irreducible LSTM data dependency) |
| cell update | ~10µs (scalar libm) | ~0.46µs (Highway dynamic dispatch) |
| weight packing | 92µs/execute | ~0 (cache hit) |

Naive comparison: ~16.3µs/step + 0.03ms fixed locally. Target ≈2.6µs/step.

### 5.2 Library spike (scratchpad/cell_lib_spike/, runnable)

Baseline-flags (the wheel's real flags) per-step cell cost at H=128:
scalar 11.7µs / hand-rolled poly 4.7µs (max err 0.68 — DISQUALIFIED) /
Eigen 1.33µs (SSE2-bound) / Highway static 2.4µs / **Highway dynamic 0.46µs**
(matches -march=native ceiling; dispatched AVX3_DL on the i7; max err 3e-7).
This is why Highway + dynamic dispatch is locked in. Eigen and SLEEF were
considered and rejected (compile-time SIMD; no dispatch story, respectively).

## 6. Testing plan

New native tests (add to `native_tests/CMakeLists.txt`, same pattern as
existing ones — they whole-archive `etnp_kernels` and run the nm-guard):

1. **`lstm_cell_test.cpp`**: drive `etnp::lstm_cell_update` directly against a
   scalar double-precision reference implementation over randomized 256-step
   chains (state feeds back). Assert max abs err <= 1e-5. Cover: H in
   {13, 32, 128} (13 exercises the scalar tail), B in {1, 3}.
2. **`xnn_linear_cache_test.cpp`**: expose a small test-only introspection API
   on the cache (e.g. `struct Stats { size_t hits, misses, size; }` +
   `XnnLinearCache::stats()` — cheap, no reason to hide it behind ifdefs).
   Assert: same weights twice → second lookup is a hit; different pointer →
   miss; same pointer with mutated weight content → fingerprint mismatch
   forces repack (and produces correct new results); insert 17 distinct keys
   → size stays <= 16.
3. **`lstm_kernel_test.cpp` (existing)**: enlarge the temp `MemoryAllocator`
   arena — the kernel now allocates `T·B·4H` floats + `B·4H` floats; size the
   arena generously (e.g. 4 MiB) so future shape bumps don't hit it. The test
   currently constructs the context with a 64 KiB arena; a bare context with
   no temp allocator throws "No temp allocator provided" (known gotcha).
4. **`lstm_parity_test.cpp` (existing, golden)**: must keep passing at its
   current tolerance (1e-4). Highway-vs-libm diffs are ~3e-7 — if this test
   fails, the cell math is wrong (gate order, aliasing), not "tolerance".
5. **Race + leak harnesses (existing)**: rerun under TSan/ASan. The op cache
   and the shared threadpool are the new shared state under test.

## 7. Verification workflow (exact commands)

All commands from the repo root. Two venvs are involved — do not mix them:
- **Export venv (HAS torch):** `/home/corey/workspace/executorch/.venv` —
  used ONLY to produce `.pte` files. Naive export shells out to `flatc`:
  prepend the venv bin to PATH (`PATH="/home/corey/workspace/executorch/.venv/bin:$PATH"`).
- **Bench venv (torch-free):** `/tmp/lstm_bench_venv` — runs the bench against
  the injected-kernel wheel.

```bash
# 1. (Only if /tmp/lstm_ptes is missing .pte pairs) export fixtures:
PATH="/home/corey/workspace/executorch/.venv/bin:$PATH" \
  /home/corey/workspace/executorch/.venv/bin/python tools/export_lstm_bench_models.py /tmp/lstm_ptes
# NOTE: naive T256_H128 never finishes (known feasibility wall) — the script/
# bench already skip it. T256_H32 takes a few minutes; be patient.

# 2. Build the bench wheel WITH the kernel + Highway (see §4.3 CAUTION):
CMAKE_ARGS="-DETNP_EXTRA_KERNEL_SOURCES=$PWD/examples/custom_kernels/lstm/etnp_lstm.cpp;$PWD/examples/custom_kernels/lstm/lstm_cell.cc -DETNP_KERNELS_USE_HIGHWAY=ON" \
  uv build --wheel -o /tmp/lstm_bench_wheel

# 3. Fresh torch-free venv + install + bench:
uv venv /tmp/lstm_bench_venv --python 3.12
VIRTUAL_ENV=/tmp/lstm_bench_venv uv pip install /tmp/lstm_bench_wheel/*.whl numpy
/tmp/lstm_bench_venv/bin/python tools/bench_lstm.py /tmp/lstm_ptes

# 4. Native tests: native_tests/ is a standalone CMake project that includes
#    the same cmake/Kernels.cmake, so it takes the same injection flags:
cmake -S native_tests -B build/lstm \
  -DETNP_EXTRA_KERNEL_SOURCES="$PWD/examples/custom_kernels/lstm/etnp_lstm.cpp;$PWD/examples/custom_kernels/lstm/lstm_cell.cc" \
  -DETNP_KERNELS_USE_HIGHWAY=ON
cmake --build build/lstm -j
# then run: lstm_kernel_test, lstm_parity_test, xnn_linear_test,
# lstm_cell_test, xnn_linear_cache_test (new), and the race/leak harnesses.
```

Success = §3. Baseline to beat (measured 2026-07-10, same machine, for
reference — expect variance):

```
T16_H32_B1   naive 0.076  custom 0.037  (2.06x)   <- must not regress below 1.0x
T64_H32_B1   naive 0.204  custom 0.129  (1.58x)
T256_H32_B1  naive 1.006  custom 0.537  (1.87x)
T16_H64_B1   naive 0.069  custom 0.076  (0.90x)   <- currently loses; must flip
T64_H64_B1   naive 0.249  custom 0.289  (0.86x)   <- must flip
T16_H128_B1  naive 0.090  custom 0.191  (0.47x)   <- must flip
T64_H128_B1  naive 0.611  custom 0.709  (0.86x)   <- must flip
```

## 8. Known pitfalls (each one already cost time once — read carefully)

- **Thermal throttling:** the i7 laptop throttles under sustained load.
  Absolute ms are NOT comparable across bench runs — only ratios within one
  run. Never conclude a regression from cross-run absolute numbers; rerun the
  full bench and read the speedup column.
- **`flatc` not on PATH** → naive export dies in XNNPACK serialization. Fix:
  export-venv `bin/` on PATH (§7).
- **rtk shell proxy:** this machine rewrites some shell commands through `rtk`
  which can filter/truncate grep/cat output oddly. If a grep result looks
  impossible, retry with different tooling.
- **foreach_target include re-processing:** `lstm_cell.cc` is compiled once
  but textually re-included per SIMD target. System/library includes must sit
  before `foreach_target.h`; everything not per-target goes under
  `#if HWY_ONCE`. The spike file is the template — deviate at your peril.
- **`HWY_TARGET_INCLUDE` path resolution:** must be findable via `-I`. If you
  hit "No such file or directory" on the file's OWN name, the include dir for
  the TU's directory (or repo root) is missing.
- **nm-guard false failure:** if the guard demands `_GLOBAL__sub_I_lstm_cell.cc`,
  the §4.3(1) conditional-expectation change didn't take effect. If the guard
  PASSES with the registrar dropped, you broke the guard — test both ways.
- **Kernel context temp allocator:** direct kernel invocation without a temp
  allocator throws. The runtime provides malloc-backed; native tests provide a
  fixed arena (must be enlarged, §6.3).
- **Aliasing:** `lin_hh` must read all of `h` before the cell update writes
  `h` (and `output`'s `out_t` row may NOT alias `g_ih_all`). Keep the
  projection-then-update order within each timestep.
- **Gate order is i,f,g,o** (PyTorch). The golden/parity tests will catch a
  mixup — trust them over intuition.
- **B>1 layout:** `g_ih_all` rows are per (t,b): row index `t·B + b`, each row
  is `[i|f|g|o]` blocks of H. `bench_lstm.py` only exercises B=1;
  `lstm_cell_test` must cover B=3 (§6.1).

## 9. Documentation / follow-through (part of this work)

1. `docs/custom-kernels.md`: update the bench-wheel recipe (§4.3(4)) and add a
   short "LSTM kernel internals" note: batched input projection, Highway cell
   update (+ the FetchContent pin), op cache, shared threadpool.
2. `problem-statement.md`: append a "Post-restructure results" section with
   the new bench table (same format as the baseline) once §3.1 passes.
3. `docs/superpowers/specs/2026-07-09-upstream-lstm-productionization-handoff.md`:
   amend — (a) the ported bundle now includes `lstm_cell.cc` + the Highway
   dependency (pin + build flags) and the op cache; (b) gotcha B's "operators
   created once per execute / single-threaded" description is superseded;
   (c) the "reach for it at narrow-H only" envelope guidance must be rewritten
   from the post-restructure bench results.
4. `scratchpad/README.md`: add a line for `cell_lib_spike/` (already copied
   there) noting it is the validated Highway template for `lstm_cell.cc`.

## 10. Out of scope (do not do these even if tempting)

- Renaming the op, changing the schema, or touching the AOT/export Python.
- Multi-layer / bidirectional / non-f32 / batch_first support.
- Threading the recurrent GEMV or the cell update.
- Replacing XNNPACK FC with another GEMM (Highway is for the cell update ONLY).
- Raising the wheel's `-march` baseline.
- Ryzen (16-thread host) validation — follow-up after local success.
- Porting to `executorch-runtime-dist` — separate handoff, only amend its spec.
