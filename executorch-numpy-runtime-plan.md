# ExecuTorch Torch-Free Python Runtime — Planning Document

> Bootstrap doc for a Claude Code agent session. Encodes decisions already made,
> constraints discovered, and the gotchas that must be respected. The goal of the
> session is a **numpy-based, torch-free Python binding** over an existing
> ExecuTorch C++ `Module` runtime.

---

## 1. Objective & Scope

Build a lightweight Python package that can **load and execute arbitrary CPU-targeted
`.pte` artifacts** with **numpy as the only Python dependency** — no `torch`, no
`libtorch`.

**Motivation:** the CPU build of `torch` is 100+MB. A statically-linked ExecuTorch
`.so` with the XNNPACK backend is ~8–11MB (already confirmed via a working JNI build).
That ~10x reduction is the entire point of the project and is *already banked* by
dropping torch — so this project does **not** chase further size savings.

**In scope (this session):**
- Fork ExecuTorch's own pybindings and replace the `torch::Tensor` marshalling with numpy.
- Correct memory management across the numpy ↔ ExecuTorch boundary.
- Error surfacing for the "arbitrary `.pte`" case.
- The version / coverage contract we must document to callers.

**Out of scope (already solved via JNI work — do not redo):**
- How to produce the portable, `-fPIC` `.so` with XNNPACK linked. We have this.
- The generic EValue/Module glue reasoning (conceptually reused from JNI).

**Explicitly NOT doing:**
- Selective build / op-list stripping — this is a **general runtime**, op set is unknown at build time.
- Size micro-optimizations (`EXECUTORCH_OPTIMIZE_SIZE`, `.eh_frame` stripping, LTO). Not worth it at 8–11MB.
- GPU / NPU delegates (CoreML, QNN, Vulkan, Metal). CPU-only contract (see §7).

---

## 2. Locked Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Binding framework | **nanobind** (not pybind11) | Smaller generated code; clean `abi3`/stable-ABI wheels; `nb::ndarray<>` gives buffer-protocol in + numpy out with no torch/jax dependency |
| Build backend | **scikit-build-core** | Pairs cleanly with existing CMake + nanobind; single-wheel output |
| Linkage | Static-link full ExecuTorch stack into **one `.so`** | Self-contained wheel; numpy the only Python dep |
| Kernel libs | **Full** portable + optimized + quantized, whole-archived | General runtime; cannot know op set ahead of time |
| Backend | **XNNPACK + portable fallback** | CPU contract; covers any CPU-targeted `.pte` |
| Python tensor type | **numpy `ndarray`** | Only runtime dependency |
| API shape | Mirror `executorch.runtime` (`Runtime`/`Program`/`Method`) but numpy-valued | Existing ET docs/patterns transfer; only the tensor type differs |

---

## 3. What We're Forking

Start from **`extension/pybindings/pybindings.cpp`** in the ExecuTorch source, *not*
from the JNI code. It is already a pybind11 binding over the exact `Module` runtime we
target. The **only** reason it needs torch is that its tensor marshalling converts
to/from `torch::Tensor`.

**Strategy:** keep its structure; swap the tensor bridge for numpy; port to nanobind.

- **Keep (port, don't reinvent):** EValue pack/unpack, `Module` lifetime management,
  `Error`/`Result` → exception mapping, `method_meta` exposure, program load paths
  (`_load_for_executorch` / `_load_for_executorch_from_buffer`).
- **Replace:** everything that touches `torch::Tensor` / ATen types → numpy via
  `nb::ndarray<>` (input) and freshly-allocated numpy arrays (output).
- **Lift from JNI where it fits:** the (shape, dtype, ptr) → ExecuTorch tensor
  conversion logic maps closely; `JNIEnv`/`jarray` becomes `nb::ndarray` + buffer protocol.

Reusability reality check: ~60–70% conceptual reuse from JNI, ~0% literal code reuse.
Forking upstream pybindings is likely faster than growing the JNI binding sideways.

---

## 4. The Binding: numpy Marshalling

### 4.1 Input path (zero-copy is OK)
- Accept `nb::ndarray<>` arguments.
- **Require C-contiguity.** Enforce on the Python side with `np.ascontiguousarray`,
  and/or check contiguity in C++ and reject (or copy) non-contiguous inputs.
- Wrap the numpy buffer with `from_blob` / `make_tensor_ptr` (non-owning) using the
  buffer pointer, shape, and mapped dtype. No copy needed for inputs.
- **Hold a reference to each input array** for the full duration of the call (critical
  once the GIL is released — see §5).

### 4.2 Output path (copy — never view)
- Output EValues' tensors point into the method's **planned arena memory** (see §5).
- For each output tensor: allocate a **fresh** numpy array of the right shape/dtype and
  `memcpy` `nbytes()` from `const_data_ptr()`. Do **not** hand back a view.

### 4.3 dtype mapping table (pin explicitly)

| ExecuTorch `ScalarType` | numpy dtype | Notes |
|---|---|---|
| Float | float32 | |
| Double | float64 | |
| Long | int64 | |
| Int | int32 | |
| Short | int16 | |
| Byte | uint8 | |
| Char | int8 | |
| Bool | bool_ | |
| Half | float16 | |
| **BFloat16** | *(no native numpy dtype)* | Use `ml_dtypes.bfloat16`, **or** expose as raw `uint16` and document. Decide in §12. |

Reject unmapped dtypes with a clear exception rather than guessing.

### 4.4 Illustrative skeleton (nanobind — verify against nanobind docs + upstream pybindings)

```cpp
// PATTERN ONLY — not compile-checked. Fill in against nanobind + upstream pybindings.cpp.
nb::list run_method(Module& module, const std::string& name, nb::args inputs) {
    std::vector<EValue> et_inputs;
    std::vector<nb::object> keepalive;      // refs that must outlive the call

    for (auto& obj : inputs) {
        auto arr = nb::cast<nb::ndarray<nb::c_contig>>(obj);  // enforce contiguity
        keepalive.push_back(nb::borrow(obj));                  // pin buffer
        auto st = numpy_dtype_to_scalartype(arr.dtype());      // explicit table
        std::vector<SizesType> sizes(arr.shape_ptr(), arr.shape_ptr() + arr.ndim());
        // non-owning wrap of the numpy buffer:
        et_inputs.push_back(make_tensor_ptr(sizes, arr.data(), st));
    }

    std::vector<EValue> et_outputs;
    {
        nb::gil_scoped_release nogil;   // release AFTER inputs/tensors are built
        // NB: touch NO Python objects inside this block.
        auto result = module.execute(name, et_inputs);   // map errors -> exceptions
        et_outputs = std::move(result);   // handle Error/Result before this line
    }

    nb::list out;
    for (auto& ev : et_outputs) {
        const auto& t = ev.toTensor();
        // allocate fresh numpy array, memcpy t.nbytes() from t.const_data_ptr()
        out.append(copy_tensor_to_numpy(t));   // COPY, never a view
    }
    return out;
}
```

---

## 5. Memory Management Gotchas (Requirements, not suggestions)

These are the failure modes that pass every fp32 test model and then corrupt or crash on
a real user model. Treat each as a hard requirement.

1. **Outputs must be copied, never viewed.**
   Output tensors alias the method's planned memory (`MemoryManager` arenas). That memory
   is **reused on the next `forward()`** and **freed with the `Module`**. A zero-copy view
   → next inference silently clobbers the user's result; a freed Module → dangling pointer.
   → Allocate fresh numpy + `memcpy` on the way out.

2. **`from_blob` / `make_tensor_ptr` are non-owning → input buffers must outlive the call.**
   These wrap the numpy buffer without owning it. Fine while `forward()` is synchronous and
   arg refs sit on the stack — **but** if you release the GIL, you must have taken and be
   holding refs (`keepalive`) to the input arrays *before* the release.

3. **GIL release discipline.**
   Wrap only `module.execute()` in `gil_scoped_release` (throughput + lets other Python
   threads run). Build all input tensors and take all keepalive refs **before** the release.
   **Touch no Python objects inside the released region.** Re-acquire before allocating the
   output numpy arrays.

4. **Force C-contiguity on inputs.** Non-contiguous numpy → wrong strides into a runtime
   that assumes dense layout. Enforce/copy at the boundary.

5. **`Module` lifetime owns everything downstream.** Any numpy array still aliasing internal
   memory after the owning `Module` is GC'd is a use-after-free. Rule #1 (copy outputs)
   already prevents the common case; make sure no other path leaks an internal pointer.

---

## 6. Error Surfacing Requirements

Because the runtime consumes **arbitrary, uncontrolled `.pte` files**, the following are
*expected* runtime events and must be turned from C++ aborts/`Error` codes into **clean
Python exceptions** (not crashes):

- **Backend not registered** — `.pte` was lowered for a delegate we don't link
  (CoreML/QNN/Vulkan/etc.). Fail at `load` with a clear, catchable message.
- **Operator not found / unregistered** — op not in our linked kernel set.
  Fail at `load_method`/execute with the op name if available.
- **Program verification failure** — malformed or **version-incompatible** `.pte`
  (see §8). Distinguish "corrupt" from "version mismatch" in the message if possible.
- **Shape/dtype mismatch** vs method metadata — surface `method_meta` expectations.

Define a small exception hierarchy (e.g. `ExecuTorchError` base, with
`BackendNotAvailable`, `OperatorNotFound`, `ProgramLoadError`, `ExecutionError`).

---

## 7. Op & Backend Coverage Contract

"General" here means **coverage**, not size. Two surfaces silently under-cover:

- **Kernel libraries.** Portable alone covers core ATen. Real deployed `.pte` files —
  especially **quantized** (i.e. most LLMs) — call ops that register from *separate* libs
  (optimized kernels, quantized-ops lib, torchao low-bit kernels). Link portable **+**
  optimized **+** quantized, all whole-archived, or quantized models fail at execute with
  operator-not-found even though fp32 models work.
- **Backends.** XNNPACK (delegated subgraphs) + portable (undelegated fallback) covers any
  **CPU-targeted** `.pte`. That is the chosen contract. A `.pte` lowered for a non-CPU
  delegate is out of contract → clean error (§6), not a crash.

**Linkage note (re-verify in the Python build):** XNNPACK and the kernel libs register via
**static initializers**. If the linker GCs those translation units as "unused," you get
backend/op-not-found at runtime despite a clean compile. Keep
`-Wl,--whole-archive` (`-force_load` on Apple) around the XNNPACK backend and the
portable/optimized/quantized kernel archives. You solved this for JNI; it re-emerges when a
setuptools/CMake extension links differently.

---

## 8. Version & Compatibility Documentation (Caller-Facing)

A general runtime does **not** control the export side, so compatibility must be documented,
not assumed. Ship these facts with the package (README + a queryable `__version__` /
`runtime_info()`):

1. **`.pte` format is version-bounded.** Pin and document the **exact ExecuTorch
   version(s)** the `.so` is built against. A `.pte` exported by an incompatible ET version
   surfaces as a **load-time verification failure that looks like a corrupt file** — call
   this out explicitly so users check version before assuming corruption.
2. **Backend contract:** "CPU `.pte` files only (XNNPACK delegate + portable fallback)."
   State that CoreML/QNN/Vulkan/Metal-lowered artifacts are unsupported and will raise.
3. **Op coverage:** core ATen + optimized + quantized kernels. **Custom operators are not
   included** — a `.pte` using custom ops needs those kernels linked in and is out of the
   box unsupported.
4. **numpy dtype caveats:** enumerate the supported dtype table (§4.3). Call out
   **BFloat16** handling explicitly (depends on §12 decision).
5. **Platform/ABI:** which OS/arch wheels are published; Python `abi3` floor if using
   stable ABI.

Expose enough at runtime for users to self-diagnose: build-time ET version, linked backends,
and (if cheaply available) linked kernel-lib set.

---

## 9. Packaging

- Single self-contained `.so`, numpy the only runtime dependency.
- **scikit-build-core** driving the existing CMake build.
- **nanobind** + `abi3` stable-ABI wheel → one wheel across Python versions.
- No size micro-opt passes.
- Decide `manylinux` target + minimum Python for `abi3` (see §12).

---

## 10. Testing Strategy / Acceptance

The general-runtime risk is "passes my models, fails the user's." Test matrix must include
the *out-of-contract* cases, which should raise cleanly rather than crash:

| Case | Expected |
|---|---|
| fp32 model (XNNPACK-delegated) | Correct output, numpy round-trip fidelity |
| **Quantized** model (LLM-style) | Works — proves optimized/quantized kernels are linked |
| Undelegated ops (portable fallback) | Works |
| Dynamic shapes (bounded) | Works within bounds; clean error outside |
| Multi-method program | Each method loads/executes |
| Multiple sequential `forward()` calls | Prior outputs **not** clobbered (proves output-copy) |
| Concurrent execution across threads | Correct under GIL-release |
| Non-contiguous numpy input | Handled (copied) or rejected clearly |
| `.pte` lowered for non-CPU backend | Clean `BackendNotAvailable`, no crash |
| Version-mismatched `.pte` | Clean load error distinguishable from corruption |
| Each dtype in the table (esp. bf16) | Correct round-trip or documented handling |

Add a numerical-parity check: a couple of reference models where the numpy-runtime output
matches a torch-runtime reference within tolerance (run the torch side offline / in CI only,
never as a package dependency).

---

## 11. Suggested Task Breakdown for the Agent

1. **Scaffold** the package: scikit-build-core + nanobind + CMake wiring to the existing
   ExecuTorch static libs. Get an empty extension importing.
2. **Port program-load paths** from upstream pybindings (buffer + path), returning an
   opaque module handle. No tensors yet.
3. **Implement the dtype table** (§4.3) both directions, with explicit reject-on-unmapped.
4. **Input marshalling:** numpy → non-owning ET tensor, contiguity enforcement, keepalive.
5. **Output marshalling:** ET tensor → fresh numpy (copy). Wire `run_method`.
6. **GIL-release** around execute only, with the ordering rules in §5.
7. **Error hierarchy** (§6): map Error/Result and known abort paths to exceptions.
8. **Thin Python wrapper** mirroring `Runtime`/`Program`/`Method` (numpy-valued).
9. **`runtime_info()` / `__version__`** exposing ET build version + linked backends (§8).
10. **Test matrix** (§10), including out-of-contract cases.
11. **Docs:** the §8 caller-facing contract in the README.

Whole-archive linkage (§7) should be validated at step 1–2, not discovered at step 10.

---

## 12. Open Questions to Resolve Early

- **BFloat16:** `ml_dtypes` dependency vs raw `uint16` passthrough? (Adds a dep vs pushes
  reinterpretation onto the user.) Decide before finalizing the dtype table.
- **abi3 floor:** minimum Python version for the stable-ABI wheel.
- **manylinux target** + which arches to publish.
- **Kernel-lib set confirmation:** verify the exact archive names/targets for
  optimized + quantized + torchao low-bit in the current ET version, and that whole-archive
  covers all of them.
- **ET version pin:** which release(s) to build against and support (§8.1).
- **API surface:** expose only the high-level `Runtime`/`Program`/`Method`, or also the
  low-level module handle for power users?
- **Runtime introspection depth:** can we cheaply enumerate linked kernel libs at runtime,
  or only backends?

---

_Assumes the portable `-fPIC` `.so` build (XNNPACK confirmed) is already solved from prior
JNI work. This session is the numpy binding, its memory correctness, and the caller
contract — not the C++ build._

## Appendix

JNI project mentioned above is here: https://github.com/corey-cole/djl-executorch-engine and also downloaded locally as `/home/corey/workspace/djl-executorch-engine`
The core (Java/JNI free) source code file used to orchestrate ExecuTorch integration is here: https://github.com/corey-cole/djl-executorch-engine/blob/main/native/core/et_runtime.cpp

The JNI project has specific QA gates to ensure that the native core is leak-free, an approach we'll want to replicate with this project

There's a separate GitHub project for building portable binaries.  These are attested to via GitHub with a clearly defined `.cmake` file for consumers: https://github.com/measly-java-learning/executorch-runtime-dist/releases/tag/v1.3.1-2

The full code for the runtime build repo is downloaded locally as `/home/corey/workspace/executorch-runtime-dist`

