# ExecuTorch Torch-Free numpy Runtime — Design Spec

**Date:** 2026-07-06
**Status:** Approved for planning
**Supersedes decisions in:** `executorch-numpy-runtime-plan.md` (this spec is the authoritative resolution of that plan's §12 open questions)

---

## 1. Objective

Build a lightweight Python package that loads and executes arbitrary **CPU-targeted
`.pte` artifacts** with **numpy as the only required Python dependency** — no `torch`,
no `libtorch`.

The ~10x size win (torch CPU build 100+MB → static ExecuTorch `.so` ~8–11MB) is *already
banked* by dropping torch. This project does **not** chase further size savings. It is a
**general CPU runtime**: the op set is unknown at build time.

### In scope
- A torch-free C++ core (grown from the proven JNI core) + a nanobind layer marshalling
  numpy ↔ ExecuTorch `EValue`.
- Correct memory management across the numpy ↔ ExecuTorch boundary.
- Clean Python exceptions for the "arbitrary `.pte`" failure modes.
- A documented version/coverage contract, queryable at runtime.

### Out of scope
- Producing the portable `-fPIC` `.so` with XNNPACK — **already solved**; consumed as a
  prebuilt, SHA-pinned tarball (`executorch-runtime-dist` v1.3.1).
- Selective build / op-list stripping (general runtime).
- Size micro-optimization (LTO, `.eh_frame` stripping, `EXECUTORCH_OPTIMIZE_SIZE`).
- GPU/NPU delegates (CoreML/QNN/Vulkan/Metal) — CPU-only contract.
- Bundled programs, ETDump/profiling, custom data loaders, threadpool control — present in
  upstream pybindings but deliberately excluded (see §3).

---

## 2. Locked Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Core strategy** | **Hybrid**: grow the torch-free JNI core (`et_runtime.cpp`), port 4 narrow patterns from upstream pybindings | JNI core is ~120 lines, torch-free, already leak-gated; pybindings is 1827 lines / 115 torch touch points, ~90% out-of-scope features. Porting 4 patterns beats forking + stripping. |
| Binding framework | **nanobind** | Small generated code; `nb::ndarray<>` gives buffer-protocol in + numpy out with no torch/jax dep |
| Build backend | **scikit-build-core** | Pairs with existing CMake + nanobind; single-wheel output |
| Linkage | Static-link full ExecuTorch stack into one `_core.<abi3>.so` | Self-contained wheel; numpy the only Python dep |
| Kernel libs | Portable + optimized + quantized + XNNPACK, **whole-archived by the dist config** | General runtime; whole-archive already handled by `ExecuTorchTargets.cmake` |
| Backend | XNNPACK (delegated) + portable (fallback) | CPU contract; covers any CPU-targeted `.pte` |
| Python tensor type | numpy `ndarray` | Only runtime dependency |
| API shape | **High-level only** now: `Runtime`/`Program`/`Method`, numpy-valued | Mirrors `executorch.runtime`; low-level handle deferred (see §9) |
| **BFloat16** | **uint16 passthrough** now; `ml_dtypes.bfloat16` behind a future `[bf16]` extra | Preserves "numpy is the only dep"; clean upgrade path |
| **ABI target** | **`cp312-abi3`**, Python **3.12+ floor** | nanobind stable ABI only exists for CPython 3.12+; single wheel |
| **Platform** | **`manylinux_2_28_x86_64`** | Only target the dist publishes |
| **ET version pin** | **1.3.1** | Both the ET checkout and the dist tarball are 1.3.1 |

---

## 3. The Four Ported Patterns (core strategy detail)

The foundation is the JNI core's proven RAII/memory model. It covers the happy path but
is missing exactly four capabilities the contract needs — all torch-free, all present in
upstream `extension/pybindings/pybindings.cpp`. We **port the logic** (study it), we do
**not** fork the file.

| # | Capability | JNI core today | Source pattern |
|---|---|---|---|
| 1 | Load from **buffer** (not just path) | path only | `_load_for_executorch_from_buffer` |
| 2 | **Multi-method** (`method_names`, `run_method` by name) | hardcodes `"forward"` | `PyModule::method_names` / `run_method` |
| 3 | Richer **method_meta** (`num_outputs`, `output_tensor_meta`, `nbytes`) | inputs only | `PyMethodMeta` |
| 4 | **Introspection** (`is_available`, backend names, operator names) | none | `_is_available` / `_get_registered_backend_names` / `_get_operator_names` |

Everything else in pybindings (bundled programs, ETDump/profiling, custom data loaders,
threadpool control, attribute tensors) is deliberately excluded from this scope.

---

## 4. Architecture & Layering

Three layers, each independently testable. The memory-critical logic is quarantined in a
**binding-agnostic** C++ core (`et_core`) that knows nothing about numpy or Python.

```
┌─ Python: executorch_numpy_runtime/ ────────────────────────┐
│  Runtime → Program → Method   (numpy-valued, mirrors        │
│  runtime_info(), __version__   executorch.runtime)          │
├─ nanobind glue: _core.<abi3>.so ───────────────────────────┤
│  dtype table (both directions) · numpy↔EValue marshalling   │
│  · keepalive refs · GIL-release · Error→exception mapping    │
├─ C++ core: et_core (torch-free, grown from JNI et_runtime) ─┤
│  Module load (path+buffer) · method_names/run_method        │
│  · method_meta(in+out) · ForwardResult RAII owns arena      │
│  · introspection: is_available/backends/ops                 │
└─ Prebuilt static libs (dist v1.3.1 tarball, whole-archived) ┘
```

**Boundary rule:** `et_core` speaks `InputDesc`/`OutputView` (host pointer + shape +
`ScalarType` code), exactly like the JNI core. Consequences:
- The ASan/LSan leak harness runs against `et_core` directly, with no Python — leaks are
  attributable to the binding vs the runtime.
- The future low-level escape hatch (§9) is a second thin binding over the same core, not
  a rewrite.

---

## 5. Marshalling & Memory Model (correctness core)

These are **hard requirements**, not suggestions — they are the failure modes that pass
every fp32 test and then corrupt/crash on a real user model.

### 5.1 Input path (zero-copy in)
- Accept `nb::ndarray<>`; **enforce C-contiguity** (`nb::ndarray<nb::c_contig>` cast;
  reject or copy non-contiguous — see §5.5).
- Map numpy dtype → `ScalarType` via the explicit table (§5.4); **reject unmapped dtypes
  with a clear exception** — never guess.
- `from_blob` / `make_tensor_ptr` non-owning wrap of the numpy buffer (no copy).
- **Take a keepalive reference on every input array before GIL release**, held for the
  entire call.

### 5.2 Execution (GIL discipline)
- `nb::gil_scoped_release` wraps **only** `module.execute()`.
- All input tensors built and all keepalive refs taken **before** the release.
- **Zero Python-object access inside** the released region.
- GIL re-acquired **before** allocating output numpy arrays.

### 5.3 Output path (copy out, never view) — MANDATORY
- For each output `EValue`, allocate a **fresh** numpy array of the right shape/dtype and
  `memcpy` `nbytes()` from `const_data_ptr()`.
- **Never** hand back a view into arena memory. Output tensors alias the method's planned
  memory, which is reused on the next `execute()` and freed with the `Module`.
- Enforced structurally: `ForwardResult` RAII owns the `EValue` vector; the copy happens
  before it drops.
- **No `clone_outputs=False` escape hatch.** Unlike upstream pybindings (which defaults
  `clone_outputs=True` but allows `False`), copy is unconditional here — there is no code
  path that returns an aliasing view.

### 5.4 dtype table (pinned, both directions)

| ExecuTorch `ScalarType` | numpy dtype |
|---|---|
| Float | float32 |
| Double | float64 |
| Long | int64 |
| Int | int32 |
| Short | int16 |
| Byte | uint8 |
| Char | int8 |
| Bool | bool_ |
| Half | float16 |
| **BFloat16** | **uint16** (raw bits; documented; `ml_dtypes.bfloat16` behind future `[bf16]` extra) |

Any `ScalarType` not in this table → clean exception on both the input and output path.

### 5.5 Non-contiguous inputs
Decision: **copy at the boundary** via `np.ascontiguousarray` semantics (C++ side detects
non-contiguity and makes a contiguous copy), rather than rejecting. Rationale: a copy is a
minor cost and a strictly friendlier contract than a hard error; the memory-safety
invariant (dense layout into the runtime) is preserved either way.

---

## 6. Error Handling

The runtime consumes **arbitrary, uncontrolled `.pte` files**, so these are *expected*
runtime events and MUST become clean, catchable Python exceptions — never C++ aborts.

```
ExecuTorchError (base)
├─ ProgramLoadError      — malformed/corrupt OR version-incompatible .pte.
│                          Message distinguishes "corrupt" vs "version mismatch"
│                          where detectable (verification failure heuristics).
├─ BackendNotAvailable   — .pte lowered for a delegate we don't link
│                          (CoreML/QNN/Vulkan/Metal). Raised at load.
├─ OperatorNotFound      — op not in the linked kernel set. Raised at
│                          load_method/execute, with the op name if available.
└─ ExecutionError        — shape/dtype mismatch vs method_meta, or other runtime
                           failure. Surfaces the method_meta expectation.
```

Implementation: the core carries an **error-kind enum + optional op/backend name** on its
failure path (enriching the JNI core's bare `std::runtime_error`), which the nanobind layer
maps to the right subclass. `_is_available(backend)` (pattern #4) enables a precise
pre-check so a non-CPU `.pte` raises `BackendNotAvailable` with a clear message rather than
a generic load failure.

---

## 7. Introspection & Caller Contract

### 7.1 Runtime introspection
```python
runtime_info() -> {
  "executorch_version": "1.3.1",       # pinned build version
  "backends": [...],                    # _get_registered_backend_names()
  "operators": [...],                   # _get_operator_names() (optional; large)
  "kernel_libs": ["portable", "optimized", "quantized"],  # from build manifest
  "supported_dtypes": [...],
  "bfloat16": "uint16-passthrough",
}
__version__  # package version + embedded ET version
```

### 7.2 Caller-facing contract (shipped in README)
1. **`.pte` format is version-bounded to ET 1.3.1.** A `.pte` exported by an incompatible
   ET version surfaces as a **load-time verification failure that looks like a corrupt
   file** — called out explicitly so users check version before assuming corruption.
2. **Backend contract:** CPU `.pte` only (XNNPACK delegate + portable fallback).
   CoreML/QNN/Vulkan/Metal-lowered artifacts are unsupported and raise `BackendNotAvailable`.
3. **Op coverage:** core ATen + optimized + quantized kernels. **Custom operators are NOT
   included. torchao low-bit kernels are NOT included** (not in the dist whole-archive set).
4. **numpy dtype caveats:** the §5.4 table; BFloat16 is raw `uint16` (see `[bf16]` extra).
5. **Platform/ABI:** `cp312-abi3`, `manylinux_2_28_x86_64`, Python 3.12+.

---

## 8. Packaging & Build

- **scikit-build-core** drives CMake, which consumes the dist v1.3.1 tarball via its
  `lib/cmake/ExecuTorch/executorch-config.cmake`, **SHA256-pinned** (port of the dist's
  `EtRuntimePin.cmake` pattern).
- **nanobind** + `Py_LIMITED_API` → single `cp312-abi3` wheel.
- numpy is the only runtime dependency. `[bf16]` optional extra pulls `ml_dtypes`.
- **Whole-archive linkage is inherited** from the dist config's `INTERFACE_LINK_OPTIONS`
  (`portable_ops_lib`, `optimized_ops_lib`, `optimized_native_cpu_ops_lib`,
  `quantized_ops_lib`, `xnnpack_backend`, `executorch`).
- **Post-link `nm` assertion** (port of `assert_xnnpack_registered.cmake`), extended to
  assert the quantized/optimized kernel-registration static-init TUs survived, not just
  XNNPACK. **Validated at build step 1–2**, not discovered at test time.

---

## 9. Deferred: low-level escape hatch (documented future work)

The high-level `Runtime`/`Program`/`Method` API ships first. Because `et_core` is
binding-agnostic, exposing a low-level module handle later (`set_inputs`/`execute`/
`get_outputs`, pybindings-style) is a **second thin binding over the same core** — no
core changes, no memory-model rework. This is captured as future work, not built now.

---

## 10. Testing & QA

Two tiers, mirroring the JNI project's leak-gate discipline.

### 10.1 Correctness (pytest)

| Case | Expected |
|---|---|
| fp32 model (XNNPACK-delegated) | Correct output; numpy round-trip fidelity |
| Quantized model (LLM-style) | Works — proves optimized/quantized kernels linked |
| Undelegated ops (portable fallback) | Works |
| Dynamic shapes (bounded) | Works within bounds; clean error outside |
| Multi-method program | Each method loads/executes |
| Sequential `forward()` calls | Prior outputs **not** clobbered (proves output-copy) |
| Concurrent execution across threads | Correct under GIL-release |
| Non-contiguous numpy input | Handled via copy (§5.5) |
| `.pte` lowered for non-CPU backend | Clean `BackendNotAvailable`, no crash |
| Version-mismatched `.pte` | Clean load error, distinguishable from corruption |
| Each dtype incl. bf16 | Correct round-trip or documented uint16 handling |

Plus a **numerical-parity check**: reference models where numpy-runtime output matches a
torch-runtime reference within tolerance. The torch side runs **offline / CI-only**, never
as a package dependency.

### 10.2 Leak QA (merge gate)
Port of `et_leak_harness`: model-agnostic load→forward loop over `et_core` directly (no
Python), built under ASan/LSan. Exercises load/destroy balance and per-forward allocations.
A leak → non-zero exit → **blocks merge**, matching the JNI QA bar.

---

## 11. Suggested Task Breakdown (for the plan phase)

1. **Scaffold**: scikit-build-core + nanobind + CMake wiring to the SHA-pinned dist tarball.
   Empty extension imports. **Validate whole-archive `nm` assertion here.**
2. **Port `et_core`**: lift the JNI core; add buffer-load and multi-method (patterns 1–2).
   Opaque handle, no tensors yet.
3. **dtype table** (§5.4) both directions, reject-on-unmapped.
4. **Input marshalling**: numpy → non-owning ET tensor; contiguity handling; keepalive.
5. **Output marshalling**: ET tensor → fresh numpy (mandatory copy). Wire `run_method`.
6. **GIL-release** around execute only, per §5.2 ordering.
7. **Error hierarchy** (§6): enrich core failure path; map to exception subclasses.
8. **method_meta + introspection** (patterns 3–4): richer meta; `runtime_info()`.
9. **Thin Python wrapper**: `Runtime`/`Program`/`Method`; `__version__`.
10. **Test matrix** (§10.1), including out-of-contract cases.
11. **Leak QA harness** (§10.2) wired as a merge gate.
12. **Docs**: the §7.2 caller-facing contract in the README.

---

## 12. Open Questions — Resolved

All of `executorch-numpy-runtime-plan.md` §12 is now resolved:

| Plan §12 question | Resolution |
|---|---|
| BFloat16 | uint16 passthrough now; `ml_dtypes` behind `[bf16]` extra later |
| abi3 floor | `cp312-abi3`, Python 3.12+ (nanobind stable-ABI constraint) |
| manylinux target + arch | `manylinux_2_28_x86_64` (only dist target) |
| Kernel-lib set + whole-archive | portable + optimized + quantized + XNNPACK; whole-archived by dist config; **torchao NOT included** |
| ET version pin | 1.3.1 |
| API surface | High-level `Runtime`/`Program`/`Method` only; low-level deferred (§9) |
| Runtime introspection depth | backends + operators + version via ported introspection fns (§7.1) |

---

## References
- Plan: `executorch-numpy-runtime-plan.md`
- JNI core (reuse foundation): `djl-executorch-engine/native/core/et_runtime.{h,cpp}`
- Leak harness (port target): `djl-executorch-engine/native/harness/et_leak_harness.cpp`
- Whole-archive assert (port target): `djl-executorch-engine/native/cmake/assert_xnnpack_registered.cmake`
- Upstream pybindings (pattern source): `executorch/extension/pybindings/pybindings.cpp`
- Prebuilt runtime: `executorch-runtime-dist` v1.3.1, consumed via `executorch-config.cmake`
