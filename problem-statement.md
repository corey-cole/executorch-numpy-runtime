# LSTM Sequence Operation Performance Issue

## Problem Statement

The LSTM sequence operator `examples/custom_kernels/lstm/etnp_lstm.cpp` exhibits surprising performance results.
At smaller values for `H`, the operator is significantly faster than a naive decomposition.  As `H` grows,
the performance of the operator decreases, crossing over somewhere around H=64.

This is surprising because both the naive decomposition and the LSTM sequence operator use XNNPACK.

The second surprising result is that on a 16 core host, the naive decomposition wins every time.

Taken together these result seem contradictory.


## Local Benchmark Data

Summary below captures on local host with 4 core/8 thread CPU (11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz)


Calibrated to the MVP (single-layer, unidirectional, batch_first=False, f32, B=1):
  SIZE   custom .pte constant in T; naive grows with T.
  SPEED  custom wins at H=32 (1.3x–1.9x, widening with T); marginally LOSES at H=64
         (~0.87–0.92x) and clearly loses at H=128 (0.49–0.85x) — XNNPACK batched FC
         beats the per-timestep kernel once H is wide. Crossover is between H=32 and 64.
  FEASIB naive export of T=256,H=128 never finished in 120s; T=256,H=32 was fine.
These are heuristics from a few data points — tune the thresholds as you learn more.

The benchmark application is present in `tools/bench_lstm.py`

Full benchmark results on 8 core/16 thread host (AMD Ryzen 7 5800XT 8-Core Processor)
```
config         naive_size custom_size  naive_ms custom_ms size_ratio  speedup
-----------------------------------------------------------------------------
T16_H32_B1          98432       35712     0.058     0.137      2.76x   0.43x (max|diff|=1.19e-07)
T64_H32_B1         273536       35712     0.229     0.526      7.66x   0.44x (max|diff|=1.19e-07)
T256_H32_B1        973568       35712     0.870     2.062     27.26x   0.42x (max|diff|=1.49e-07)
T16_H64_B1         197760      135040     0.084     0.266      1.46x   0.32x (max|diff|=1.49e-07)
T64_H64_B1         372864      135040     0.275     1.315      2.76x   0.21x (max|diff|=1.19e-07)
T16_H128_B1        593024      530304     0.109     1.237      1.12x   0.09x (max|diff|=1.19e-07)
T64_H128_B1        768128      530304     0.359     2.344      1.45x   0.15x (max|diff|=1.49e-07)
T256_H128_B1   (missing .pte pair -- skipped)

rtol 0.0001 cross-check over 7 config(s): max abs diff = 1.490e-07 (PASS)
Configs where custom did NOT win:
  T16_H32_B1: speed ratio/speedup = 0.43
  T64_H32_B1: speed ratio/speedup = 0.44
  T256_H32_B1: speed ratio/speedup = 0.42
  T16_H64_B1: speed ratio/speedup = 0.32
  T64_H64_B1: speed ratio/speedup = 0.21
  T16_H128_B1: speed ratio/speedup = 0.09
  T64_H128_B1: speed ratio/speedup = 0.15
```


## Verdict (2026-07-10 investigation)

**Not intractable.** The slowdown is implementation structure, not XNNPACK. Both
paths use the same XNNPACK microkernels but with completely different structure
around them: the naive decomposition gets load-time weight packing, a hoisted
batched input GEMM, a threadpool, and vectorized activations; the custom kernel
gets per-call packing, sequential batch-1 GEMVs, a single thread, and scalar
libm activations. Three root causes were identified and quantified by direct
measurement on the local i7-1185G7 (probe benches preserved in `scratchpad/`,
see its README for build commands).

### Root cause 1: per-execute weight packing

`etnp_lstm.cpp` calls `XnnLinear::create` (→ `xnn_create_fully_connected_nc_f32`,
which repacks the weight matrices) inside the kernel body, so it runs on **every
inference**. The XNNPACK delegate packs once at model load. Measured create cost
for the two projections: 2µs at H=32, **92µs at H=128, 465µs at H=256** — grows
~H². A slope/intercept fit to the Ryzen data shows a ~0.87ms fixed cost per
execute at H=128, while naive's *entire* T16_H128 run is 0.109ms. This alone
explains the worst data point (T16_H128 = 0.09x).

### Root cause 2: input projection not batched over T

Dumping the naive model's edge graph (`scratchpad/inspect_naive_graph.py`) shows
PyTorch's LSTM decomposition **hoists `x @ W_ih^T` out of the time loop**: the
first `addmm` is `(T·B, I) @ (I, 4H)` — one GEMM for all timesteps; only the
recurrent `h @ W_hh^T` runs per step. The custom kernel instead runs T batch-1
GEMVs for the input projection. Measured: one batch=T FC run is **1.5–2.5×
faster** than T batch-1 runs single-threaded. Since `x` is fully known upfront,
there is no data dependency preventing the same hoist inside the kernel.

### Root cause 3: scalar cell update dominates (the hidden cost)

Per-component breakdown at H=128, T=256: batched input GEMM 0.07ms, recurrent
GEMVs 0.46ms, **scalar cell update 1.68ms**. The `sigmoidf`/`std::tanh` gate
loop makes 5·H scalar libm calls per timestep (~30ns per sigmoid+tanh pair,
~10µs/step at H=128 — ~60% of the real kernel's measured 17µs/step). The naive
path runs sigmoid/tanh through XNNPACK's *vectorized* microkernels. A
SIMD-friendly rational approximation measured **3.6× faster**; a production
version needs XNNPACK-grade accuracy (~1 ulp) to stay inside the bench's 1e-4
gate (the quick probe polynomial does not: 1.5e-3 at H=128).

### Threading (explains the 16-core result)

The kernel passes `tp=nullptr` everywhere. `libxnnpack_backend.a` in the pinned
dist pulls `executorch::extension::threadpool::get_pthreadpool()` — the delegate
runs on a shared pool sized to all hardware threads. With a threadpool, the
batched input GEMM gains another **2–3.4×** on 8 threads (more on 16). This is
why the Ryzen amplifies naive's advantage: naive runs ~5.2µs/step there vs
~16.3µs/step on the i7, while the single-threaded custom kernel cannot scale
with cores at all. The same `extension_threadpool` target ships in the dist, so
the kernel can join the shared pool with no new dependency.

### Why the crossover happens where it does

Naive carries constant per-step *graph overhead* (19 delegate calls plus ~176
portable ops per inference at T=16, incl. per-step slice/squeeze copies) that
dominates at H=32. Custom carries per-step *scalar-activation + unbatched-GEMV*
costs that grow with H. The curves cross between H=32 and H=64. On the local
i7 at H=128, naive and custom have nearly identical per-step slopes (16.3 vs
17µs/step); custom loses on the fixed per-execute packing cost.

### Fix path (validated by microbenchmark; not yet implemented)

1. **Batch the input projection**: one FC run with `batch=T·B` on the shared
   threadpool before the time loop — exactly naive's graph shape. Alternative:
   split the op so the input projection stays in the graph as a normal
   partitioned linear and `etnp::lstm` does only the recurrence (delegate then
   handles its packing AND threading).
2. **Vectorize the fused cell update** — the single biggest win.
3. **Amortize operator creation** — e.g. a static cache keyed by weight
   pointer/shape (constant-segment pointers are stable per loaded program; the
   per-Runtime mutex already serializes execute).

The irreducible core — T sequential recurrent GEMVs, the LSTM data dependency —
costs ~1.8µs/step at H=128 on the i7, and naive pays it too, plus dispatch.
Predicted restructured cost: **~3–4µs/step vs naive's ~16µs/step locally**, so
custom should win everywhere on the i7 and, extrapolating, modestly beat
naive's 5.2µs/step on the Ryzen (needs verification there). The custom op's
size win (27× smaller at T256, constant in T) and export-feasibility win
(naive T256_H128 never exports) are untouched by the fix.

### Measurement caveat

The local i7-1185G7 thermal-throttles under sustained load: absolute ms are
NOT comparable across bench runs, only ratios within a single run.

## Post-restructure results (2026-07-XX)

Kernel restructured per docs/superpowers/specs/2026-07-10-lstm-kernel-restructure-design.md
(batched input projection on the shared threadpool + Highway SIMD cell update
+ cached packed FC operators). Same host, same fixtures:

```
config         naive_size custom_size  naive_ms custom_ms size_ratio  speedup
-----------------------------------------------------------------------------
T16_H32_B1          98432       35712     0.061     0.015      2.76x   3.96x (max|diff|=1.19e-07)
T64_H32_B1         273536       35712     0.226     0.036      7.66x   6.24x (max|diff|=1.19e-07)
T256_H32_B1        973568       35712     0.968     0.099     27.26x   9.78x (max|diff|=1.19e-07)
T16_H64_B1         197760      135040     0.071     0.027      1.46x   2.61x (max|diff|=1.49e-07)
T64_H64_B1         372864      135040     0.260     0.071      2.76x   3.68x (max|diff|=1.19e-07)
T16_H128_B1        593024      530304     0.099     0.060      1.12x   1.66x (max|diff|=1.49e-07)
T64_H128_B1        768128      530304     0.741     0.182      1.45x   4.07x (max|diff|=1.49e-07)
T256_H128_B1   (missing .pte pair -- skipped)

rtol 0.0001 cross-check over 7 config(s): max abs diff = 1.490e-07 (PASS)
Custom won on both size and speed for every config checked.
```

All configs now >= 1.0x with the rtol 1e-4 cross-check passing.