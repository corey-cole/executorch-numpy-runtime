# executorch-numpy-runtime

A torch-free Python runtime for ExecuTorch `.pte` files. **numpy is the only required dependency.** Lightweight runtime that doesn't require PyTorch/libtorch—useful when even the CPU version of torch requires more disk space than is available. Loads and runs arbitrary CPU-targeted `.pte` artifacts.

## Install

```bash
pip install executorch-numpy-runtime            # numpy only
pip install executorch-numpy-runtime[bf16]      # + ml_dtypes for real bfloat16
```

**Wheels:** `cp312-abi3`, `manylinux_2_28_x86_64`, Python 3.12+.

## Quick start

```python
import numpy as np, executorch_numpy_runtime as en

prog = en.Runtime.get().load_program("model.pte")
out = prog.load_method("forward")([np.ones(3, np.float32), np.ones(3, np.float32)])
```

## Compatibility contract

- **ExecuTorch version: 1.3.1 (exact).** A `.pte` exported by an incompatible ExecuTorch version fails to load and **looks like a corrupt file** — check the exporter version before assuming corruption. Query the build version with `en.runtime_info()`.

- **Backends: CPU only** — XNNPACK delegate + portable fallback. `.pte` files lowered for CoreML / QNN / Vulkan / Metal are unsupported and raise `BackendNotAvailable`. Note: a `.pte` lowered for a non-CPU backend may currently surface as `ProgramLoadError` instead of the more specific `BackendNotAvailable` (ExecuTorch 1.3.1 doesn't cheaply expose delegate id at load time).

- **Operators: core ATen + optimized + quantized kernels.** **Custom operators are NOT included. torchao low-bit kernels are NOT included.**

- **Custom kernels:** compile your own ExecuTorch kernels into the runtime — see [docs/custom-kernels.md](docs/custom-kernels.md).

- **Dtypes:** float32, float64, float16, int64, int32, int16, int8, uint8, bool. **BFloat16 is surfaced as raw `uint16` bits** (no native numpy bf16). Install the `[bf16]` extra for a real `ml_dtypes.bfloat16` dtype. Note: `uint16` inputs are interpreted as BFloat16. Unsupported dtypes raise.

- **Outputs are always fresh copies** — never views into runtime memory; safe to keep across subsequent calls.

## Platform support

| Platform | Wheel | Kernel libs (`kernel_libs`) | Custom ops | USDT probes |
|---|---|---|---|---|
| Linux x86_64 (manylinux_2_28) | `cp312-abi3` | portable, optimized, quantized, lstm | `etnp::lstm.out`, `etnp::triple.out` | yes |
| Linux aarch64 (manylinux_2_28) | `cp312-abi3` | portable, optimized, quantized, lstm | `etnp::lstm.out`, `etnp::triple.out` | yes |
| Windows x86_64 | `cp312-abi3` | **portable only** | **none** | no (Linux-only) |

`lstm` appears in `kernel_libs` because `libetnp_ops_lstm.a` is literally a kernel library —
the one providing the `etnp::lstm.out` custom op. The two columns describe the same artifact
from different angles: what was linked, and what op that gives you.

`etnp::triple.out` is a bundled **reference** kernel, built by default on Linux and kept
CI-tested so the custom-kernel wiring can't rot. It is deliberately **not** built on Windows
(`ETNP_BUILD_REFERENCE_KERNEL` defaults OFF there) because the upstream Windows runtime ships
no extras yet — which is what makes the Windows "none" true rather than aspirational.

**Windows is a reduced runtime.** The upstream ExecuTorch distribution ships a core-only
build for Windows: no optimized-kernel library, no quantized-kernel library, and no
`etnp::lstm.out`. A `.pte` that loads on Linux may therefore fail on Windows with an
operator-not-found error at load time. XNNPACK delegation works on all three.

Query what a given install actually has — never assume from the platform:

```python
from executorch_numpy_runtime import runtime_info
runtime_info()["kernel_libs"]   # e.g. ['portable', 'optimized', 'quantized', 'lstm']
```

This list is derived from the build's real link line, so it cannot drift from what shipped.

## Concurrency

A single `Runtime` instance is thread-safe to share across threads, but method calls serialize (per-Runtime lock). For true parallel inference, create one `Runtime` per thread. XNNPACK still parallelizes within a single call via thread pool.

## Performance

The runtime executes through the same ExecuTorch C++ core (XNNPACK delegate) as
the official ExecuTorch `pybindings`, so its steady-state speed matches upstream —
the torch-free packaging costs nothing at inference time. Against PyTorch eager it
is substantially faster, because the `.pte` avoids both the ATen kernels and the
Python/TorchScript per-call dispatch overhead.

MobileNetV2, batch 1, CPU, float32 `[1,3,224,224]` — 100 timed iterations after
warmup, identical seeded input, all backends returning byte-identical logits.
Representative figures on an Intel i7-1185G7 (4 cores / 8 threads):

| Backend | Artifact | latency (approx) | throughput | vs torch eager |
|---|---|---:|---:|---:|
| **executorch-numpy-runtime** | `.pte` | **~4.5 ms** | **~220 it/s** | **~3× faster** |
| ExecuTorch `pybindings` (official) | `.pte` | ~4.5 ms | ~220 it/s | ~3× faster |
| PyTorch eager (best: 1 thread) | `.pt` | ~14 ms | ~72 it/s | 1.0× (baseline) |

Two takeaways: this runtime is **~3× faster than PyTorch eager** on this model, and
**performs identically to official ExecuTorch pybindings** — the numpy marshalling
adds no measurable overhead (the two `.pte` runtimes share the same XNNPACK core, and
their best-case latency is indistinguishable). As a bonus, cold-start program load is
milliseconds versus seconds for a `torch`-based process (no `import torch`).

Caveats: single machine, one model, batch 1 — illustrative, not a broad claim.
**Only the ratios are stable**; absolute latencies drift ~2× on this thermally
throttling laptop (≈2 ms cold, ≈5 ms sustained), so treat the millisecond figures as
approximate. PyTorch is shown at its fastest configuration (single-threaded; it
regressed with more threads on this 4-core part). XNNPACK manages its own thread pool,
so exact thread parity across backends isn't guaranteed. Reproduce or extend with
`tools/bench.py --backend {torch,et_pybindings,numpy_rt}`.

## Errors

`ExecuTorchError` (base) → `ProgramLoadError`, `BackendNotAvailable`, `OperatorNotFound`, `ExecutionError`.

- `ProgramLoadError`: Malformed, corrupt, or version-incompatible `.pte`.
- `BackendNotAvailable`: The `.pte` was lowered for a delegate this runtime does not link (CPU-only).
- `OperatorNotFound`: An operator required by the `.pte` is not in the linked kernel set.
- `ExecutionError`: Runtime failure during execution (e.g. shape/dtype mismatch).

Honesty note: `OperatorNotFound` and `BackendNotAvailable` are best-effort under ExecuTorch 1.3.1. That version's `Module::load()`/`execute()` don't always expose which operator or delegate was missing, so some missing-operator/missing-backend cases surface as the more generic `ProgramLoadError` or `ExecutionError` instead of the specific subclass.

## Introspection

```python
en.runtime_info()
# {
#   "executorch_version": "1.3.1",
#   "backends": [...],
#   "operators": [...],
#   "kernel_libs": ["portable", "optimized", "quantized"],
#   "supported_dtypes": [...],
#   "bfloat16": "uint16-passthrough"
# }
```

## Developing

For contributors building from source.

### Prerequisites

- Python **3.12+**, CMake **≥ 3.26**, a C++17 compiler, and `nm` (binutils).
- The prebuilt, position-independent ExecuTorch **1.3.1** runtime. This project does **not** build ExecuTorch itself — it links a pinned, attested static-lib tarball from
  [`executorch-runtime-dist`](https://github.com/measly-java-learning/executorch-runtime-dist/releases).

### 1. Runtime fetch (automatic)

`cmake/RuntimePin.cmake` pins the tarball URL + SHA256 for the current ExecuTorch release and fetches + hash-verifies it via CMake's `FetchContent` — no manual download step needed for a normal build.

If you'd rather point at a runtime you've unpacked yourself (e.g. a local rebuild, or an air-gapped environment), pass `-DETNP_RUNTIME_PREFIX=/path/to/executorch-runtime-1.3.1-logging-linux-x86_64` and the fetch is skipped in favor of that prefix; the build fails early with a clear message if it doesn't contain `lib/cmake/ExecuTorch/executorch-config.cmake`.

In CI, provenance (that the tarball came from `executorch-runtime-dist`'s own release CI, not just that its bytes match the pinned hash) is verified separately with `gh attestation verify --repo measly-java-learning/executorch-runtime-dist`, mirroring the pattern in `djl-executorch-engine/native/.github/workflows/native-build-job.yml`.

### 2. Build (editable install)

```bash
uv pip install -e . --no-build-isolation      # or: pip install -e . --no-build-isolation
```

Build stack is **scikit-build-core + nanobind** → a single `cp312-abi3` extension.

> **Rebuild caveat:** the editable install does **not** auto-recompile after C++/CMake edits. After changing anything under `src/` or `cmake/`, force a rebuild:
> ```bash
> rm -rf build && uv pip install -e . --no-build-isolation --reinstall
> ```

### 3. Run the tests

```bash
python -m pytest tests/
```

Two tests skip by design: `test_parity.py` skips unless `torch` is importable (it's never a package dependency — parity is a CI/offline check), and the non-CPU-backend test skips without a CoreML fixture.

### 4. Regenerate `.pte` fixtures

The committed fixtures under `tests/models/` are generated offline in a **separate** environment that has `executorch==1.3.1` and `torch` (CPU) installed (`flatc` must be on `PATH`):

```bash
/path/to/et-venv/bin/python tools/export_fixtures.py tests/models
```

Keep this env separate from your runtime dev env — the whole point of the project is to not pull torch into the runtime.

### 5. Leak QA gate (ASan/LSan)

The memory model is gated by a Python-free harness that drives the C++ core directly under LeakSanitizer. It's the same check CI enforces on every PR:

```bash
cmake -S native_tests -B build/leak && cmake --build build/leak
ASAN_OPTIONS=detect_leaks=1 ./build/leak/leak_harness tests/models/add.pte 500
```

Expect `leak_harness: 500 iters OK` and exit 0. A leak → non-zero exit → merge blocked.

### 6. Race QA gate (TSan)

A second native harness (`native_tests/race_harness.cpp`) drives the core from many threads under ThreadSanitizer — a **separate binary**, since TSan and ASan can't be combined in one build. It runs two scenarios: many threads sharing one `Runtime` (the per-`Runtime` mutex surface) and many threads each owning their own `Runtime`.

```bash
cmake -S native_tests -B build/race && cmake --build build/race --target race_harness
setarch "$(uname -m)" -R \
  env TSAN_OPTIONS="suppressions=native_tests/tsan_suppressions.txt" \
  ./build/race/race_harness tests/models/add.pte 8 200
```

Expect `race_harness: ... OK` and exit 0; a data race → non-zero exit.

- **`setarch -R` is required** on recent (6.x) kernels: high ASLR entropy (`vm.mmap_rnd_bits`) makes TSan abort with `FATAL: unexpected memory mapping`. `setarch -R` disables ASLR for the child process — no root needed.
- **Scope — read before trusting a green run.** The prebuilt ExecuTorch libs are **not** TSan-instrumented, so this gate sees races in *our* code (`et_core`, the binding) but is **blind to races inside ExecuTorch itself** (`Module::methods_`, XNNPACK, pthreadpool). This was verified empirically: injecting an unsynchronized write into `run_method` **is** caught; removing the `method_meta`/`method_names` locks (whose race lives inside ExecuTorch's uninstrumented `methods_`) is **not**. So the gate protects synchronization of the data structures *we* own and guards against that regression class — it does **not** validate our serialization of ExecuTorch's internal state. Doing that would need a TSan-instrumented ExecuTorch build (the attested tarball isn't one).
- `native_tests/tsan_suppressions.txt` quiets known-noisy uninstrumented frames; refine it per toolchain (none were needed on the reference build).

### 7. Verify `cibuildwheel` locally

CI uses `cibuildwheel` to build, audit, and test the package.  Note that the CMake cache is sensitive to the differences in build paths between local and container builds.
Therefore, before executing this step, ensure that the `build/` directory is empty.

`uvx cibuildwheel --platform linux`

### Layout

| Path | Responsibility |
|---|---|
| `src/et_core/` | Binding-agnostic C++ core (ExecuTorch `Module` lifetime, arena copy-out, per-`Runtime` mutex). No Python/numpy headers. |
| `src/binding/` | nanobind glue: numpy↔`EValue` marshalling, dtype table, GIL discipline, exception translation. |
| `executorch_numpy_runtime/` | Pure-Python `Runtime`/`Program`/`Method`, `runtime_info()`, error hierarchy. |
| `native_tests/` | ASan/LSan leak harness + TSan race harness (link `et_core` directly, no Python) and the TSan suppressions file. |
| `tools/export_fixtures.py` | Offline `.pte` fixture generation (torch env). |
| `cmake/` | `RuntimePin.cmake` (runtime prefix), `assert_kernels_registered.cmake` (post-link guard). |

### Build-time guards & gotchas

- **Whole-archive `nm` guard:** `cmake/assert_kernels_registered.cmake` runs POST_BUILD and **fails the build** if the XNNPACK / quantized / optimized kernel-registration static-initializer TUs were garbage-collected out of the final `.so` — otherwise they'd only surface as "backend/operator not found" at model-load time.
- **`libm` path workaround:** the top-level `CMakeLists.txt` (and `native_tests/CMakeLists.txt`) rewrite the runtime config's baked-in `/usr/lib64/libm.so` to a portable `-lm`. The prebuilt tarball resolves that absolute path inside its manylinux (RHEL) build container; the rewrite is a harmless no-op where the path exists and unblocks builds on hosts where it doesn't (e.g. Debian/Ubuntu multiarch). Remove it once a corrected tarball is published upstream.