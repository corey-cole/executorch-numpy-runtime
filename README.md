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

- **Dtypes:** float32, float64, float16, int64, int32, int16, int8, uint8, bool. **BFloat16 is surfaced as raw `uint16` bits** (no native numpy bf16). Install the `[bf16]` extra for a real `ml_dtypes.bfloat16` dtype. Note: `uint16` inputs are interpreted as BFloat16. Unsupported dtypes raise.

- **Outputs are always fresh copies** — never views into runtime memory; safe to keep across subsequent calls.

## Concurrency

A single `Runtime` instance is thread-safe to share across threads, but method calls serialize (per-Runtime lock). For true parallel inference, create one `Runtime` per thread. XNNPACK still parallelizes within a single call via thread pool.

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

- Python **3.12+**, CMake **≥ 3.24**, a C++17 compiler, and `nm` (binutils).
- The prebuilt, position-independent ExecuTorch **1.3.1** runtime. This project does **not** build ExecuTorch itself — it links a pinned, attested static-lib tarball from
  [`executorch-runtime-dist`](https://github.com/measly-java-learning/executorch-runtime-dist/releases).

### 1. Fetch the runtime

Download `executorch-runtime-1.3.1-logging-linux-x86_64.tar.gz` and unpack it into `third_party/` so that this path exists:

```
third_party/executorch-runtime-1.3.1-logging-linux-x86_64/lib/cmake/ExecuTorch/executorch-config.cmake
```

`cmake/RuntimePin.cmake` points at that prefix; the build fails early with a clear message if it's missing.

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

### Layout

| Path | Responsibility |
|---|---|
| `src/et_core/` | Binding-agnostic C++ core (ExecuTorch `Module` lifetime, arena copy-out, per-`Runtime` mutex). No Python/numpy headers. |
| `src/binding/` | nanobind glue: numpy↔`EValue` marshalling, dtype table, GIL discipline, exception translation. |
| `executorch_numpy_runtime/` | Pure-Python `Runtime`/`Program`/`Method`, `runtime_info()`, error hierarchy. |
| `native_tests/` | ASan/LSan leak harness (links `et_core` directly, no Python). |
| `tools/export_fixtures.py` | Offline `.pte` fixture generation (torch env). |
| `cmake/` | `RuntimePin.cmake` (runtime prefix), `assert_kernels_registered.cmake` (post-link guard). |

### Build-time guards & gotchas

- **Whole-archive `nm` guard:** `cmake/assert_kernels_registered.cmake` runs POST_BUILD and **fails the build** if the XNNPACK / quantized / optimized kernel-registration static-initializer TUs were garbage-collected out of the final `.so` — otherwise they'd only surface as "backend/operator not found" at model-load time.
- **`libm` path workaround:** the top-level `CMakeLists.txt` (and `native_tests/CMakeLists.txt`) rewrite the runtime config's baked-in `/usr/lib64/libm.so` to a portable `-lm`. The prebuilt tarball resolves that absolute path inside its manylinux (RHEL) build container; the rewrite is a harmless no-op where the path exists and unblocks builds on hosts where it doesn't (e.g. Debian/Ubuntu multiarch). Remove it once a corrected tarball is published upstream.