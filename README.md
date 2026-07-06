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