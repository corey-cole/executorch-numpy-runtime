# Handoff: productionize `etnp::lstm.out` in `executorch-runtime-dist`

**Date:** 2026-07-09
**Audience:** an agent working **inside the `executorch-runtime-dist` repo** (the project that builds the relocatable ExecuTorch library + ships the hash-pinned tarball).
**Derives from:** [`2026-07-09-lstm-op-delivery-strategy.md`](2026-07-09-lstm-op-delivery-strategy.md) — read it first; this document is the executable version of its **Decision 1** (move the op upstream) and **Decision 3** (the `extras` bundle convention).

## Source repo (read files here — do not copy code into this handoff)

All referenced paths are absolute and live in the MVP repo on this machine:

    SRC = /home/corey/workspace/executorch-numpy-runtime

Both repos are local, so you can read `$SRC/...` directly from the `executorch-runtime-dist` checkout. **Port/adapt from these files; do not re-derive.** The MVP already paid down the hard parts — your job is to move a *correctness-stable, benchmark-validated* op into its permanent home, not to rewrite it.

## Objective / definition of done

Ship `etnp::lstm.out` in the relocatable ExecuTorch library so that every consumer of the tarball (including the numpy runtime) gets it registered at load time, **and** provide the AOT definition + a live torch round-trip test that proves export→lower→run matches `torch.nn.LSTM`. Concretely:

1. The runtime kernel is compiled into the shipped library, whole-archived, and its registrar TU is guarded so it can't be dropped at link.
2. The op is **always-on in every variant**, namespace **stays `etnp::lstm.out`** (baked into `.pte`s — do not rename). See strategy Decision 1.
3. A **live** torch test exports an `nn.LSTM`, lowers it through the custom op, runs it against the built library, and compares to torch eager at rtol/atol 1e-4. This **replaces** the MVP's committed-golden-header workaround (which only existed because the numpy repo is torch-free).
4. The op follows a documented per-op **`extras` bundle** layout (four faces below), as extra #1 / the template. **Build the convention, not the framework** (strategy Decision 3).

## The four faces of the bundle → where each comes from

| Face | Port from (`$SRC/...`) | Notes |
|---|---|---|
| **1. Runtime kernel + registration** (torch-free C++) | `examples/custom_kernels/lstm/etnp_lstm.cpp`, `lstm_cell.{h,cc}`, `xnn_linear.h`, `xnn_linear_cache.h` | The kernel + its XNNPACK-FC helper. See **Gotcha A** (registration) and **B** (XNNPACK reuse). |
| **2. Op schema / name = single source of truth** | schema strings in `tools/etnp_lstm_op.py` (`_lib.define(...)`) and the header comment in `etnp_lstm.cpp` | The functional + `.out` schemas below. Make ONE source both C++ and Python read; op-name drift is the nastiest failure (exports fine / op-not-found at runtime). |
| **3. AOT definition** (Python + torch) | `tools/etnp_lstm_op.py` | The `torch.library` registration that makes the op survive lowering. See **Gotcha C**. |
| **4. Torch round-trip test** (mandatory) | *new* — replaces `tools/gen_lstm_golden.py` + `native_tests/lstm_golden.h` + `native_tests/lstm_parity_test.cpp` | Upstream has torch, so do the live round-trip instead of the committed golden. Reuse the parity tolerance/shape logic from the MVP parity test. |

The canonical schema (keep byte-identical across faces):

    lstm(Tensor input, Tensor h0, Tensor c0, Tensor w_ih, Tensor w_hh,
         Tensor? b_ih, Tensor? b_hh) -> (Tensor, Tensor, Tensor)
    lstm.out(Tensor input, Tensor h0, Tensor c0, Tensor w_ih, Tensor w_hh,
             Tensor? b_ih, Tensor? b_hh, *,
             Tensor(a!) output, Tensor(b!) hn, Tensor(c!) cn)
             -> (Tensor(a!), Tensor(b!), Tensor(c!))

**Scope (do not widen):** single-layer, unidirectional, `batch_first=False`, float32, contiguous. `input [T,B,I]`, `h0/c0 [B,H]`, `w_ih [4H,I]`, `w_hh [4H,H]`, optional biases `[4H]`; `output [T,B,H]`, `hn/cn [B,H]`. **Gate row order i,f,g,o** (PyTorch).

## Critical gotchas — the hard-won lessons the code alone won't tell you

These are the traps the MVP already hit and solved. Each cites the file that demonstrates it.

**A. The 3-output kernel CANNOT use `EXECUTORCH_LIBRARY`.**
`$SRC/examples/custom_kernels/lstm/etnp_lstm.cpp:129-130` registers via a hand-rolled boxed trampoline:
`register_kernel(Kernel("etnp::lstm.out", lstm_boxed))` where `lstm_boxed(KernelRuntimeContext&, Span<EValue*>)` unpacks the 10-slot stack. **Why:** in the pinned 1.3.1 runtime, `extension/kernel_util/make_boxed_from_unboxed_functor.h` has `static_assert(num_nonconst_tensors == 1)`, so the auto-unboxing macro rejects our three mutable outputs (`output, hn, cn`). **⚠️ Re-verify in YOUR ExecuTorch version** — you build the runtime, so you may be on a different version. If your `make_boxed` still caps at 1 output, keep the boxed registrar. If a newer version lifts the cap, you *may* use the macro, but the boxed path is known-good and version-robust — prefer it unless you have a reason.

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

**C. The AOT "survives lowering" recipe.**
`$SRC/tools/etnp_lstm_op.py` (docstring + `_lib.define` for BOTH `lstm` and `lstm.out`). **Why it works:** register both the functional and the `.out` overload as `CompositeExplicitAutograd` + a `register_fake`; `to_edge_transform_and_lower` **without** a partitioner keeps the op opaque, and ExecuTorch's `ToOutVarPass` (run inside `.to_executorch()`) converts it to `etnp::lstm.out` automatically — no partitioner, no non-decomposable hackery needed. This was verified: the lowered `.pte` contained exactly `['etnp::lstm.out']`.

**D. The kernel uses `ctx.allocate_temp`.** Any test/harness that invokes the kernel directly must construct `KernelRuntimeContext` **with a temp allocator** (a bare context throws "No temp allocator provided"). See `$SRC/native_tests/lstm_kernel_test.cpp` (a 64 KiB `MemoryAllocator` is wired in).

**E. Naive export needs `flatc` on `PATH`.** The XNNPACK-partitioner lowering path shells out to `flatc`; run exports with the venv's `bin/` on `PATH`. (Only relevant if you reproduce the benchmark; the custom-op export path doesn't need it.)

## Build + guard machinery to port

The MVP's compile/whole-archive/guard pattern is directly reusable — generalize it into the `extras` **manifest check** (strategy Decision 3):

- `$SRC/cmake/Kernels.cmake` — how a kernel source is compiled into a static lib and how the expected registrar TUs (`_GLOBAL__sub_I_<basename>`) are computed.
- `$SRC/cmake/assert_kernels_registered.cmake` — the **nm-guard** that proves registrar TUs survived `--gc-sections`/whole-archive. Generalize: assert **one registrar TU per manifested op** survived, and **no duplicate op names**.
- `$SRC/native_tests/CMakeLists.txt` — the whole-archive link (`$<LINK_LIBRARY:WHOLE_ARCHIVE,...>`) + POST_BUILD guard invocation pattern.

Add per-op `variants:` metadata now (`[all]` for LSTM), even though everything is always-on — cheap to design in, annoying to retrofit.

## Live torch round-trip test (replaces the golden workaround)

Instead of `gen_lstm_golden.py` → committed `lstm_golden.h` → `lstm_parity_test.cpp`, write one upstream test that, in your torch env: builds an `nn.LSTM`, exports+lowers it through the custom op to a `.pte`, runs it against the built library, and compares to torch eager at **rtol/atol 1e-4**. Reuse the input/shape/tolerance logic from `$SRC/native_tests/lstm_parity_test.cpp` and the export recipe from `$SRC/tools/etnp_lstm_op.py`. Make the test **mandatory** in the bundle (refuse to build an extra without one).

## Evidence this op is worth shipping (and what to document)

From the MVP benchmark (`$SRC/docs/custom-kernels.md`, and `$SRC/tools/{export_lstm_bench_models,bench_lstm,lstm_feasibility}.py`; static advisor in `$SRC/tools/lstm_advisor.py`):

- **Size:** custom `.pte` is **constant in T**; naive decomposition grows with T. Win at every H, widening with T (2.8×→27× over T=16→256 at H=32).
- **Speed:** custom **wins at narrow H** (H=32: 1.3×–1.9×, widening with T), is **marginally slower at H=64** (~0.87–0.92×) and clearly slower at H=128 (0.49×–0.85×). Crossover is between H=32 and H=64.
- **Speed (post-restructure 2026-07):** superseded — see the post-restructure
  table in `$SRC/problem-statement.md`; the custom op now wins at every
  benchmarked (T,H) on the reference host, so the "reach for it at narrow-H
  only" guidance below is obsolete for latency (size/feasibility wins unchanged).
- **Feasibility:** naive export of **T=256, H=128** never completes within a 120 s budget; custom is trivial and constant. That's the "impossible-for-naive" existence proof.

Document this envelope in the op's upstream docs so consumers know: **reach for it at narrow-H / long-T** (wins both size and speed), and at wide H keep it only when `.pte` size or export feasibility is the binding constraint (size-only win, latency cost).

## Unknowns to resolve in the `executorch-runtime-dist` repo

1. **ExecuTorch version vs. the `make_boxed` 1-output cap** (Gotcha A) — check your version's `make_boxed_from_unboxed_functor.h`; keep the boxed registrar unless you've confirmed otherwise.
2. **How the repo already packages kernels into the tarball** — find the existing kernel/backend registration + whole-archive setup and slot the LSTM extra into it (that's where `Kernels.cmake`/nm-guard patterns map).
3. **Where the `extras` directory + manifest should live** and how the build globs it.
4. **XNNPACK availability in the upstream build** (headers + link) — the kernel depends on it.

## Do NOT

- Do not rename the op or change the schema (`.pte` permanence — strategy Decision 4 / Decision 1).
- Do not widen scope to multi-layer/bidirectional/dtypes/packed sequences (that's the "bulletproof full nn.LSTM" explicitly cut).
- Do not build a plugin framework yet — LSTM is extra #1; formalize a manifest/codegen only after ops #2–3 exist.
- Do not port the throwaway MVP bench/feasibility tooling as production code — it's *evidence*, not a deliverable. The advisor (`lstm_advisor.py`) is optional and can travel separately.
- Do not reintroduce a torch dependency into the numpy runtime repo — the numpy repo's follow-up (pin bump + torch-free consumer smoke test) is strategy Decision 2 and is **out of scope for this handoff**.

## Follow-up in the numpy runtime repo (not your job, for context)

Once your tarball ships the op: bump `cmake/RuntimePin.cmake`, keep the generic custom-kernel seam, and replace the MVP's native LSTM tests with a torch-free consumer smoke test (load a committed `.pte` using `etnp::lstm.out`, run, assert). See strategy Decision 2.
