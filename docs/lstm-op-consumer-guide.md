# Consumer Guide: `etnp::lstm.out`

The pinned ExecuTorch runtime (executorch-runtime-dist **v1.3.1-4**, all variants)
ships a first-party custom LSTM operator, `etnp::lstm.out`. The op is not pulled in
by linking the runtime alone — it ships as a separate static-init archive that must
be whole-archived (see **Linking** below). This numpy runtime's build already
performs that step, so the op is available here with no extra action on your part —
just run a `.pte` that uses it. Confirm it is registered:

    from executorch_numpy_runtime import _core
    assert "etnp::lstm.out" in _core.operator_names()

## The op
- **Name (frozen, baked into `.pte`s):** `etnp::lstm.out` (functional `etnp::lstm`).
- **Schema:** single-layer, unidirectional, `batch_first=False`, float32, contiguous.
  `input [T,B,I]`, `h0/c0 [B,H]`, `w_ih [4H,I]`, `w_hh [4H,H]`, optional biases `[4H]`;
  `output [T,B,H]`, `hn/cn [B,H]`. Gate order i,f,g,o.
- `.pte`s using this op are produced upstream (the AOT definition + a live torch
  round-trip test live in `executorch-runtime-dist`); this repo consumes them.

## Running it here
Load and run exactly like any other model (see `tests/test_lstm_smoke.py`). How a
`.pte` exposes the op is an export choice: the vendored fixture bakes the weights
in as constants, so its `forward` takes only `(input, h0, c0)` and returns the
sequence output `[T,B,H]`:

    rt = _core.load_path("lstm.pte")
    out = rt.run_method("forward", [input, h0, c0])   # out[0] = output [T,B,H]

A model that instead exposes the weights as inputs would call the op with the full
`(input, h0, c0, w_ih, w_hh, b_ih, b_hh)` signature — that is a property of the
exported `.pte`, not of the runtime.

## Linking (how this repo — and any consumer — gets the op)
The op is NOT auto-registered just by linking the runtime: it ships as a separate
static-initializer archive `libetnp_ops_lstm.a` that must be **whole-archived** at
the final link, or it is GC'd and you get "operator etnp::lstm.out not found" at
model-load time. This repo does exactly that in `CMakeLists.txt` — after
`find_package(ExecuTorch)` it includes the tarball's `ETNPExtras.cmake` and calls
`etnp_extras_whole_archive(_core)`, which applies the correct per-OS flag and pulls
`libhwy.a` transitively. Any other consumer that links the tarball into its own
binary does the same on its own target. (ExecuTorch's own archives —
`xnnpack_backend`, the portable/optimized ops libs — still need whole-archiving per
ExecuTorch's guidance; `ETNPExtras` covers only the first-party extras.)

## Performance envelope (why it exists)
Versus the naive decomposition ExecuTorch emits, the custom `.pte` is **constant in
T** (naive grows with T; 2.8×–27× smaller over T=16→256 at H=32), **faster at every
benchmarked (T,H)** (1.66×–9.78× across H∈{32,64,128}, T∈{16,64,256}), and exports
shapes the naive path cannot complete (T=256,H=128 never finishes a 120s budget).
It is the default choice for the supported LSTM shape.
