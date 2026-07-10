# Custom kernels

This is a shared, general-purpose ExecuTorch runtime. When a PyTorch op does not
decompose efficiently to ExecuTorch core ops, you can supply a hand-written
kernel and compile it into the extension. Real kernels (e.g. `nn.LSTM`) are
**not** bundled here — they live in your build via the injection seam below. The
bundled `etnp::triple.out` is a trivial reference/exemplar kept CI-tested so the
build wiring can't rot.

## The contract

1. **Write an out-variant kernel** with the ExecuTorch calling convention:
   `Tensor& your_op(KernelRuntimeContext&, const Tensor& in, ..., Tensor& out)`.
   See `kernels/reference/etnp_reference_ops.cpp` for the minimal shape.
2. **Register it** with `EXECUTORCH_LIBRARY(ns, "op_name.out", your_op)`. This
   registers the exact string `"ns::op_name.out"`.
3. **The registered name must match the name serialized into your `.pte`.** How a
   model that uses a custom op is exported is documented in the upstream
   ExecuTorch custom-ops guide and is out of scope here.

## Building your kernel in

Pass absolute source paths at configure time — no fork required:

    cmake -S . -B build \
      -DETNP_EXTRA_KERNEL_SOURCES="/abs/path/my_kernels.cpp;/abs/path/more.cpp"

Your sources are compiled into the `etnp_kernels` archive and **whole-archived**
into the module, so their static-init registrars survive the linker. A post-link
guard (`cmake/assert_kernels_registered.cmake`) fails the **build** — not model
load — if any registrar is dropped. A worked example is in
`examples/custom_kernels/negate_op.cpp`.

To omit the bundled reference op from a lean production build, configure with
`-DETNP_BUILD_REFERENCE_KERNEL=OFF`.

## Leak-checking a kernel

If your kernel allocates (workspace, state, scratch buffers), leak-check it under
AddressSanitizer/LeakSanitizer. `native_tests/kernel_registration_test.cpp` is
built both plain (`kernel_registration_test`) and under ASan (`kernel_leak_test`);
copy that target for your op, or point it at your kernel by name, and run:

    ASAN_OPTIONS=detect_leaks=1 ./build/nt/kernel_leak_test

Prefer allocating scratch via the `KernelRuntimeContext` temp allocator over raw
`malloc`/`new`; kernels that own raw allocations and skip freeing them surface
here as leaks. (Registration itself is a one-time static allocation LSan ignores,
so only per-invocation allocations in the kernel body are flagged.)

## Benchmarking the LSTM example

The `examples/custom_kernels/lstm/` op (`etnp::lstm.out`) ships with a throwaway
MVP benchmark comparing it against ExecuTorch's naive LSTM decomposition on
`.pte` size and execution latency, using identical weights and inputs.

Because the default wheel does not bundle the LSTM op, build a benchmark-only
wheel with the kernel injected, then run the harness in a throwaway venv:

    # 1. Export both .pte's (naive nn.LSTM vs custom op) in the torch export venv.
    #    Run from the repo root so `tools` resolves as a package.
    /path/to/export-venv/bin/python tools/export_lstm_bench_models.py /tmp/lstm_ptes

    # 2. Build a wheel WITH the LSTM kernel compiled in. The kernel is two
    #    sources (op + Highway SIMD cell update) and needs the pinned Highway dep.
    CMAKE_ARGS="-DETNP_EXTRA_KERNEL_SOURCES=$PWD/examples/custom_kernels/lstm/etnp_lstm.cpp;$PWD/examples/custom_kernels/lstm/lstm_cell.cc -DETNP_KERNELS_USE_HIGHWAY=ON" \
      uv build --wheel -o /tmp/lstm_bench_wheel

    # 3. Install it into a fresh torch-free venv and run the bench.
    uv venv /tmp/lstm_bench_venv --python 3.12
    VIRTUAL_ENV=/tmp/lstm_bench_venv uv pip install /tmp/lstm_bench_wheel/*.whl numpy
    /tmp/lstm_bench_venv/bin/python tools/bench_lstm.py /tmp/lstm_ptes

`bench_lstm.py` loads both `.pte`s in this runtime, cross-checks the custom vs
naive outputs at rtol 1e-4 (the honesty gate — a benchmark of two different
computations is worthless), then prints a
`config | naive_size | custom_size | naive_ms | custom_ms | size_ratio | speedup`
table. The custom op's `.pte` size is independent of sequence length `T` (the op
is opaque, so weights are baked once), while the naive path unrolls per timestep —
so the size advantage widens sharply with `T` (2.8x at `T=16` up to 27x at `T=256`
for `H=32`), and holds at every `H`.

Latency, by contrast, depends on hidden size `H`. **Before the 2026-07 restructure** the
custom kernel ran the gate projections one timestep at a time and — because XNNPACK batches
the naive path's matmuls across timesteps — only beat naive at narrow `H` (winning at `H=32`,
losing at `H>=64`). The restructure described below (one batched input projection on the shared
threadpool + cached packed FC operators + a Highway-SIMD fused cell update) closed that gap: the
custom op now wins on **both** size and speed at every benchmarked `(T, H)` on the reference
host — see the post-restructure table in `problem-statement.md`. The size and export-feasibility
advantages (below) are largest at long `T`.

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

### Feasibility crossover (Task 6)

`tools/lstm_feasibility.py` answers a narrower question than the benchmark
above: is there a `T` at which naive decomposition stops being *practical* to
export at all, while the custom op stays trivial? It sweeps `T` at a fixed
`H=128`, running each naive and custom export attempt in a **child process**
under a hard wall-clock `subprocess.run(..., timeout=...)`. This matters
because a large-`T` naive export doesn't error — it just runs for a very long
time (an earlier manual attempt at `T=256, H=128` ran past 10 minutes at 100%
CPU before being killed) — so only an external timeout, not a try/except,
can bound it.

Declared practicality budgets (arbitrary but explicit — see the constants at
the top of the script):

- `EXPORT_TIME_BUDGET_S = 120` seconds per export attempt
- `SIZE_BUDGET_MB = 8` for the resulting `.pte`

Observed crossover: **`T* = 256` at `H=128`**. Naive export succeeds at
`T=16` (15.5s, 593KB) and `T=64` (49.6s, 768KB), then exceeds the 120s time
budget at `T=256` and `T=1024` (both killed at the timeout, no `.pte`
produced). The custom `etnp::lstm` export succeeds at every `T` in ~5-6s and
produces a constant-size `.pte` (~530KB) since the op stays opaque and the
sequence loop lives in the kernel, not in the graph. The custom `.pte` for
`T=256, H=128` — the first config naive cannot produce within budget — is the
durable artifact for this task.

"Impossible" here means *exceeds the declared time/size budget above*, not a
universal impossibility — a longer timeout or more patience might eventually
produce a naive `.pte` at `T=256`. The point is the practical crossover: past
`T*`, naive decomposition is not something you would reach for, while the
custom op is unaffected by `T` at all.
