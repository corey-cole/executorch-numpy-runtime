# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **torch-free** Python runtime for ExecuTorch `.pte` files. numpy is the only required runtime dependency. It links a **prebuilt, pinned ExecuTorch 1.3.1 static-lib tarball** (this repo does **not** build ExecuTorch) and exposes it as a `cp312-abi3` nanobind extension. CPU only (XNNPACK delegate + portable/optimized/quantized kernels).

The ExecuTorch version is an **exact** compatibility contract (1.3.1). A `.pte` from an incompatible exporter fails to load and looks like a corrupt file.

## Common commands

```bash
# Editable install (build stack: scikit-build-core + nanobind)
uv pip install -e . --no-build-isolation

# Rebuild after ANY C++/CMake edit — editable install does NOT auto-recompile
rm -rf build && uv pip install -e . --no-build-isolation --reinstall

# Tests (two skip by design: test_parity needs torch; non-CPU-backend test needs a CoreML fixture)
python -m pytest tests/
python -m pytest tests/test_forward.py::test_name   # single test

# Lint
ruff check .

# Local wheel build/audit/test (ensure build/ is EMPTY first — CMake cache is path-sensitive)
uvx cibuildwheel --platform linux
```

Point at a self-unpacked runtime instead of the auto-fetched tarball with
`-DETNP_RUNTIME_PREFIX=/path/to/executorch-runtime-1.3.1-...`.

## Native QA gates (also enforced in CI, `.github/workflows/qa-gate.yml`)

These drive the C++ core **without Python**. Sanitizers can't be combined, so leak and race are separate binaries.

```bash
# Leak (ASan/LSan) — expect "leak_harness: 500 iters OK", exit 0
cmake -S native_tests -B build/leak && cmake --build build/leak
ASAN_OPTIONS=detect_leaks=1 ./build/leak/leak_harness tests/models/add.pte 500

# Race (TSan) — setarch -R is REQUIRED on 6.x kernels (ASLR entropy aborts TSan)
cmake -S native_tests -B build/race && cmake --build build/race --target race_harness
setarch "$(uname -m)" -R env TSAN_OPTIONS="suppressions=native_tests/tsan_suppressions.txt" \
  ./build/race/race_harness tests/models/add.pte 8 200
```

TSan scope caveat: the prebuilt ExecuTorch libs are **not** instrumented, so this gate catches races only in *our* code (`et_core`, binding), not inside ExecuTorch itself (`Module::methods_`, XNNPACK, pthreadpool).

## Architecture

Three layers, bottom-up:

- **`src/et_core/`** — binding-agnostic C++ core. Owns ExecuTorch `Module` lifetime, copies outputs out of the arena, and holds the **per-`Runtime` mutex**. No Python/numpy headers. Key contract in `et_core.h`: inputs are borrowed zero-copy (`InputDesc`), outputs are owned by a RAII `ForwardResult` (`OutputView` points into its storage). Errors are `EtException` carrying an `ErrorKind`.
- **`src/binding/`** — nanobind glue (`_core` module): numpy↔`EValue` marshalling, dtype table (`dtype_map.cpp`), GIL discipline, and translation of `ErrorKind` → Python exception subclasses.
- **`executorch_numpy_runtime/`** — pure-Python `Runtime`/`Program`/`Method` wrappers over `_core`, plus `runtime_info()` and the `ExecuTorchError` hierarchy. `Runtime.get()` is a process singleton; `load_program()` returns a `Program`; `Method.__call__` forces `np.ascontiguousarray` at the boundary.

**Concurrency model:** a shared `Runtime` is thread-safe but method calls serialize (per-`Runtime` lock — ExecuTorch `Module` is not thread-safe). For true parallelism, one `Runtime` per thread. XNNPACK parallelizes within a single call.

**Outputs are always fresh copies**, never views into runtime memory.

**bfloat16:** surfaced as raw `uint16` bits (numpy has no native bf16); `uint16` inputs are interpreted as bf16. The `[bf16]` extra adds `ml_dtypes` for a real dtype.

## Build-time guards (these fail the *build*, not runtime)

- **`cmake/assert_kernels_registered.cmake`** (POST_BUILD `nm` guard): fails the build if the XNNPACK/quantized/optimized kernel-registration static-init TUs were garbage-collected out of the `.so`. This is why the module is built `NOSTRIP` — a stripped symbol table would make the guard blind. Otherwise these only surface as "backend/operator not found" at model-load time.
- **`cmake/assert_usdt_probes.cmake`** (POST_BUILD guard): fails the build if the ExecuTorch USDT tracepoints don't survive the link into `_core`. Self-arms by reading `usdt=on` from the runtime prefix's `BUILDINFO` — deliberately not a platform check, so runtimes without USDT (Windows records `usdt=n/a`; pre-v1.3.1-5 runtimes have no `usdt=` key) disarm automatically. The probes come from `libetnp_ops_lstm.a` via `etnp_extras_whole_archive()`, not from ExecuTorch's core libs. See `docs/usdt-tracepoints.md`.

## Runtime fetch & provenance

`cmake/RuntimePin.cmake` pins the tarball URL + SHA256 and fetches+hash-verifies via `FetchContent`. Source: [`executorch-runtime-dist`](https://github.com/measly-java-learning/executorch-runtime-dist/releases). CI additionally verifies provenance with `gh attestation verify --repo measly-java-learning/executorch-runtime-dist` (byte-hash alone doesn't prove origin).

## Custom kernels

This is a general-purpose runtime; real custom ops (e.g. `nn.LSTM`) are **not** bundled — inject them at build time. `cmake/Kernels.cmake` gathers the bundled reference kernel (`kernels/reference/etnp_reference_ops.cpp`, `etnp::triple.out`, kept CI-tested so the wiring can't rot) plus consumer sources from `ETNP_EXTRA_KERNEL_SOURCES` into the `etnp_kernels` archive, and computes the TU symbols the nm-guard must find. See `docs/custom-kernels.md`. The registered op name (`EXECUTORCH_LIBRARY(ns, "op.out", fn)`) must match the name serialized into the `.pte`.

## `.pte` fixtures

Committed under `tests/models/`. Generated offline in a **separate** env that has `executorch==1.3.1` + torch (CPU) + `flatc` on PATH — kept separate on purpose so torch never enters the runtime env:

```bash
/path/to/et-venv/bin/python tools/export_fixtures.py tests/models
```

## Benchmarking

`tools/bench.py --backend {torch,et_pybindings,numpy_rt}` — cross-backend parity + latency (torch/et_pybindings backends require their own extra deps, not project dependencies).
