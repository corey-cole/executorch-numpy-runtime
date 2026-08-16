# XNNPACK Workspace Metric Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the XNNPACK delegate's workspace arena size, in bytes, as `executorch_numpy_runtime.xnnpack_workspace_size_bytes()`.

**Architecture:** A thin vertical slice through the project's three existing layers. `et_core` gains a free function beside `registered_backends()` that reads the read-only `workspace_size_bytes` backend option via `executorch::ET_RUNTIME_NAMESPACE::get_option`; the nanobind layer exposes it with the GIL released; the Python package re-exports it. No new types, no state, no lock — the read is synchronized with delegate init inside ExecuTorch.

**Tech Stack:** C++17, ExecuTorch 1.3.1 backend-options API, nanobind, pytest.

**Spec:** `docs/superpowers/specs/2026-08-16-runtime-1.3.1-10-adoption-design.md` (step B, "XNNPACK workspace-size metric")

## Global Constraints

- Requires runtime **v1.3.1-10** or later (landed in PR #9). The option does not exist in v1.3.1-6.
- Backend id is exactly `"XnnpackBackend"`; option key is exactly `"workspace_size_bytes"`. **Both are hardcoded string literals** — `XNNPACKBackend.h` is not an installed header, so it cannot be included.
- The option is a **vendored patch in this distribution**, not upstream ExecuTorch. Code depending on it will not build against a stock ExecuTorch. Record that in the C++ source, not only in docs.
- Semantics that must be honored, not "fixed": the value is **0 until the first XNNPACK-delegated method loads** (lazy arena creation); it is **process-wide** (`EXECUTORCH_XNNPACK_SHARED_WORKSPACE=ON`, so never sum it across instances); it is a **high-water mark including alignment padding**, never shrunk; it is **read-only** (`set_option` returns `InvalidArgument` — never call it).
- Value is `int` bytes, **saturating at `INT_MAX`**. Report as-is; never re-derive.
- Available on **every platform this project ships**, including Windows. The test is not platform-gated.
- No new `ErrorKind`. Failures reuse the existing hierarchy so `src/binding/module.cpp:123-127`'s translation switch is untouched.
- After any C++/CMake edit: `rm -rf build && uv pip install -e . --no-build-isolation --reinstall`. The editable install does **not** auto-recompile.

---

### Task 1: Core reader and binding

**Files:**
- Modify: `src/et_core/et_core.h` (declaration beside `registered_backends()`/`operator_names()`, currently lines 93-95)
- Modify: `src/et_core/et_core.cpp` (implementation beside `operator_names()`, and two includes)
- Modify: `src/binding/module.cpp` (one `m.def` beside the existing free-function defs at lines 192-194)
- Test: `tests/test_workspace_size.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `int etnp::xnnpack_workspace_size_bytes()` in C++, exposed to Python as `_core.xnnpack_workspace_size_bytes() -> int`. Task 2 wraps this in the public package namespace.

- [ ] **Step 1: Write the failing test**

Create `tests/test_workspace_size.py`:

```python
import numpy as np
from executorch_numpy_runtime import _core
from conftest import model_or_skip


def test_workspace_size_grows_after_delegated_load():
    """A delegated load allocates the arena, so the reported size becomes non-zero.

    Only the transition is asserted, never the value: the figure is a high-water mark
    that includes allocator alignment padding, so it is not stable across runs or
    platforms. It is also process-wide, which is why this asserts ">0 after" rather
    than an exact delta -- another test in this process may already have grown the arena.
    """
    rt = _core.load_path(model_or_skip("add.pte"))
    rt.run_method("forward", [np.ones(3, np.float32), np.ones(3, np.float32)])

    assert _core.xnnpack_workspace_size_bytes() > 0
```

`add.pte` is XNNPACK-delegated (`grep -c XnnpackBackend tests/models/add.pte` → `1`) and takes two float32 tensors of shape `(3,)`; the call shape above matches `tests/test_forward.py`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_workspace_size.py -v`

Expected: FAIL with `AttributeError: module 'executorch_numpy_runtime._core' has no attribute 'xnnpack_workspace_size_bytes'`.

If it instead fails on the forward call, fix the call shape first — the `AttributeError` is the red state this task is chasing.

- [ ] **Step 3: Declare the function in the core header**

In `src/et_core/et_core.h`, add beneath the existing `operator_names()` declaration:

```cpp
// XNNPACK workspace arena size in bytes (process-wide high-water mark).
// Zero until the first XNNPACK-delegated method loads -- the arena is created
// lazily during delegate init, so a zero here is correct, not a broken read.
int xnnpack_workspace_size_bytes();
```

- [ ] **Step 4: Add the includes**

In `src/et_core/et_core.cpp`, add to the standard-library includes (which currently start at `<cstring>`):

```cpp
#include <cstdio>
#include <variant>
```

and to the ExecuTorch includes, beside the existing `<executorch/runtime/backend/interface.h>`:

```cpp
#include <executorch/runtime/backend/options.h>
```

- [ ] **Step 5: Implement the reader**

In `src/et_core/et_core.cpp`, add after `operator_names()`:

```cpp
// xnnpack_workspace_size_bytes(): reads the read-only "workspace_size_bytes"
// option from the XNNPACK delegate via the backend-options API.
//
// This option is a VENDORED PATCH in executorch-runtime-dist, not upstream
// ExecuTorch -- stock ExecuTorch/XNNPACK have no size accessor, so this
// translation unit will not build against an unpatched ExecuTorch. Upstream
// applies the patch on every platform it ships (build-runtime.sh, unguarded),
// so this needs no platform conditional.
//
// The two strings are hardcoded deliberately: XNNPACKBackend.h is not an
// installed header, so a consumer names them by string. Do not "fix" this by
// hunting for a header to include -- there isn't one.
//
// Never call set_option on this key: it returns InvalidArgument by design,
// because the surrounding option chain would otherwise swallow a write as a
// silent no-op success.
int xnnpack_workspace_size_bytes() {
  using executorch::runtime::BackendOption;
  using executorch::runtime::Span;

  BackendOption opt{};
  std::snprintf(opt.key, sizeof(opt.key), "%s", "workspace_size_bytes");
  Span<BackendOption> span(&opt, 1);

  const Error err =
      executorch::ET_RUNTIME_NAMESPACE::get_option("XnnpackBackend", span);
  if (err != Error::Ok) {
    if (err == Error::NotFound) {
      throw EtException({ErrorKind::BackendMissing,
          "XnnpackBackend is not registered, so its workspace size cannot be read",
          "XnnpackBackend"});
    }
    throw EtException({ErrorKind::Execution,
        "get_option(XnnpackBackend, workspace_size_bytes) failed with ExecuTorch error "
            + std::to_string(static_cast<int>(err)),
        "XnnpackBackend"});
  }

  // Upstream's example returns -1 here. We throw instead: this codebase's error
  // contract is EtException, and a -1 sentinel is indistinguishable from a byte
  // count to any caller that forgets to check it.
  const int* val = std::get_if<int>(&opt.value);
  if (val == nullptr) {
    throw EtException({ErrorKind::Execution,
        "XnnpackBackend workspace_size_bytes did not hold an int",
        "XnnpackBackend"});
  }
  return *val;
}
```

- [ ] **Step 6: Expose it from the binding**

In `src/binding/module.cpp`, add after `m.def("operator_names", &etnp::operator_names);`:

```cpp
  // GIL released: the read synchronizes with delegate init inside ExecuTorch.
  m.def("xnnpack_workspace_size_bytes", &etnp::xnnpack_workspace_size_bytes,
        nb::call_guard<nb::gil_scoped_release>());
```

- [ ] **Step 7: Rebuild**

```bash
rm -rf build && uv pip install -e . --no-build-isolation --reinstall
```

Expected: compiles clean. A compile error naming `get_option` or `BackendOption` means the runtime prefix predates v1.3.1-10 — check `cmake/EtRuntimePin.cmake` says `v1.3.1-10`.

- [ ] **Step 8: Run the test to verify it passes**

Run: `python -m pytest tests/test_workspace_size.py -v`

Expected: PASS.

If it fails with `assert 0 > 0`, the fixture did not actually delegate. Confirm with `grep -c XnnpackBackend tests/models/add.pte` (expected: `1`) before assuming the reader is wrong.

- [ ] **Step 9: Commit**

```bash
git add src/et_core/et_core.h src/et_core/et_core.cpp src/binding/module.cpp tests/test_workspace_size.py
git commit -m "feat: read the XNNPACK workspace arena size

Exposes the read-only workspace_size_bytes backend option through
_core. The backend id and option key are hardcoded because
XNNPACKBackend.h is not an installed header.

Failures raise EtException rather than returning upstream's -1 sentinel,
which is indistinguishable from a byte count to a caller that forgets to
check it. Reuses BackendMissing/Execution so the binding's error
translation switch is untouched."
```

---

### Task 2: Python surface and the zero-before-load guarantee

**Files:**
- Modify: `executorch_numpy_runtime/info.py`
- Modify: `executorch_numpy_runtime/__init__.py`
- Test: `tests/test_workspace_size.py` (extend)

**Interfaces:**
- Consumes: `_core.xnnpack_workspace_size_bytes() -> int` from Task 1.
- Produces: `executorch_numpy_runtime.xnnpack_workspace_size_bytes() -> int`, re-exported in `__all__`.

- [ ] **Step 1: Write the failing subprocess test**

The "reads 0 before any delegated load" property is destroyed by any other test in the same process — the arena is process-wide and monotonic. It therefore must be asserted in a fresh interpreter.

Append to `tests/test_workspace_size.py`:

```python
import subprocess
import sys
import textwrap


def test_workspace_size_is_zero_before_any_delegated_load():
    """Zero before the first delegated load is CORRECT, not a broken accessor.

    Runs in a fresh interpreter on purpose: the arena is process-wide and is never
    shrunk, so any earlier delegated load in this pytest process would make the
    'before' read non-zero and silently invalidate the assertion.
    """
    script = textwrap.dedent(
        """
        import executorch_numpy_runtime as en
        assert en.xnnpack_workspace_size_bytes() == 0, "expected a cold arena"
        print("ok")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_workspace_size.py::test_workspace_size_is_zero_before_any_delegated_load -v`

Expected: FAIL — the subprocess exits non-zero with `AttributeError: module 'executorch_numpy_runtime' has no attribute 'xnnpack_workspace_size_bytes'`, surfaced through the `proc.stderr` assertion message.

- [ ] **Step 3: Add the public function**

In `executorch_numpy_runtime/info.py`, add after `runtime_info()`:

```python
def xnnpack_workspace_size_bytes() -> int:
    """Size in bytes of the XNNPACK delegate's workspace arena.

    Deliberately NOT a key in :func:`runtime_info`. That function reports static
    capability description; this is mutable process state, and a ``0`` sitting beside
    ``supported_dtypes`` would read as broken capability reporting rather than as
    correct lazy initialization.

    Returns ``0`` until the first XNNPACK-delegated method loads -- the arena is created
    lazily during delegate init. The figure is process-wide (all live XNNPACK delegates
    share one arena), a high-water mark that is never shrunk, and includes allocator
    alignment padding, so it slightly over-estimates live tensor bytes. It saturates at
    ``INT_MAX`` rather than wrapping. Report it as-is; do not sum it across instances.
    """
    return _core.xnnpack_workspace_size_bytes()
```

- [ ] **Step 4: Export it**

In `executorch_numpy_runtime/__init__.py`, change the info import to:

```python
from .info import runtime_info, xnnpack_workspace_size_bytes
```

and add `"xnnpack_workspace_size_bytes",` to `__all__`, directly after `"runtime_info",`.

- [ ] **Step 5: Run the whole file to verify both tests pass**

Run: `python -m pytest tests/test_workspace_size.py -v`

Expected: both tests PASS.

- [ ] **Step 6: Prove the subprocess isolation is actually load-bearing**

A test that would pass without its own process is a test that will rot. Verify the isolation matters:

```bash
python -c "
import numpy as np, executorch_numpy_runtime as en
print('cold:', en.xnnpack_workspace_size_bytes())
p = en.Runtime.get().load_program('tests/models/add.pte')
p.forward(np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32))
print('warm:', en.xnnpack_workspace_size_bytes())
"
```

Expected: `cold: 0` then `warm: <non-zero>`. If `cold` is non-zero, something loads a delegated model at import time and the spec's semantics need revisiting. Adjust the `Runtime`/`load_program`/`forward` call shape to match `tests/test_api.py` if the above does not match the real API.

- [ ] **Step 7: Run the full suite and lint**

```bash
python -m pytest tests/ -q
ruff check .
```

Expected: passes with the two by-design skips (`test_parity` needs torch; the non-CPU-backend test needs a CoreML fixture). Confirm the new tests are not among the skips.

- [ ] **Step 8: Commit**

```bash
git add executorch_numpy_runtime/info.py executorch_numpy_runtime/__init__.py tests/test_workspace_size.py
git commit -m "feat: expose xnnpack_workspace_size_bytes() from the package

Kept out of runtime_info(): that reports static capability, and a mutable
high-water mark reading 0 before the first delegated load would look like
broken capability reporting rather than correct lazy init.

The zero-before-load test runs in a fresh interpreter because the arena is
process-wide and never shrunk -- any earlier delegated load in the same
pytest process would invalidate the assertion silently."
```

---

### Task 3: Documentation and gate sweep

**Files:**
- Modify: `CLAUDE.md` (new subsection under Architecture)
- Modify: `README.md` (API surface listing)

**Interfaces:**
- Consumes: everything from Tasks 1-2.
- Produces: no new interfaces.

- [ ] **Step 1: Document the metric in CLAUDE.md**

Add after the "Outputs are always fresh copies" line in the Architecture section:

```markdown
**XNNPACK workspace size:** `xnnpack_workspace_size_bytes()` reports the delegate's arena size for host-side native-memory accounting. It reads a **read-only** backend option (`XnnpackBackend` / `workspace_size_bytes`) that is a **vendored patch in executorch-runtime-dist**, not upstream ExecuTorch — this will not build against a stock ExecuTorch, and the feature disappears if this project ever stops consuming these tarballs. The value is `0` until the first XNNPACK-delegated method loads (lazy arena creation), process-wide (`EXECUTORCH_XNNPACK_SHARED_WORKSPACE=ON`, so never sum it), and a high-water mark including alignment padding. Requires runtime v1.3.1-10+.
```

- [ ] **Step 2: Document it in README.md**

Find the section listing the public API (near where `runtime_info()` is described) and add `xnnpack_workspace_size_bytes()` with a one-line description: *"Size in bytes of the XNNPACK workspace arena; 0 before the first delegated load, process-wide, high-water mark."*

```bash
grep -n "runtime_info" README.md
```

- [ ] **Step 3: Verify the docs test still passes**

This repo has `tests/test_docs.py`, which may assert documented symbols exist:

```bash
python -m pytest tests/test_docs.py -v
```

Expected: PASS. If it enumerates the public API, add the new symbol there too.

- [ ] **Step 4: Run the native QA gates**

The new code lives in `et_core.cpp`, which the native harnesses compile directly (`native_tests/CMakeLists.txt` builds `../src/et_core/et_core.cpp` into each harness), so both gates must be re-run.

```bash
rm -rf build/leak && cmake -S native_tests -B build/leak && cmake --build build/leak
ASAN_OPTIONS=detect_leaks=1 ./build/leak/leak_harness tests/models/add.pte 500
```

Expected: `leak_harness: 500 iters OK`, exit 0.

```bash
rm -rf build/race && cmake -S native_tests -B build/race && cmake --build build/race --target race_harness
setarch "$(uname -m)" -R env TSAN_OPTIONS="suppressions=native_tests/tsan_suppressions.txt" \
  ./build/race/race_harness tests/models/add.pte 8 200
```

Expected: exit 0, no TSan reports. (`setarch -R` is required on 6.x kernels — ASLR entropy aborts TSan.)

- [ ] **Step 5: Full suite, guard, and lint**

```bash
rm -rf build && uv pip install -e . --no-build-isolation --reinstall
python -m pytest tests/ -q
./scripts/check-pin-rows.sh
ruff check .
```

Expected: all clean, two by-design skips.

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: describe the XNNPACK workspace metric and its caveats

Records that the option is a vendored patch rather than upstream
ExecuTorch, and the three semantics most likely to be misread: zero
before the first delegated load is correct, the figure is process-wide
and must not be summed, and it is a padded high-water mark."
```

- [ ] **Step 7: Open the PR**

```bash
git push -u origin feature/xnnpack-workspace-metric
gh pr create --fill
```

- [ ] **Step 8: Confirm the Windows job is green**

The metric is expected to work on Windows — upstream applies the XNNPACK patch unconditionally in `build-runtime.sh`, and `build-runtime.ps1` only shims that same script. The Windows wheel job running this test suite is what confirms it.

```bash
gh pr checks --watch
```

Expected: all jobs green, including the Windows wheel build and its test run. If the new tests fail only on Windows, do **not** platform-gate them without first checking whether the Windows tarball really lacks the patched delegate — the expectation is that it has it.

---

## Notes for the executor

- **Do not add a mutex.** The read is synchronized with delegate init inside ExecuTorch; the per-`Runtime` lock guards `Module`, which this function does not touch.
- **Do not call `set_option` on this key**, even to "test that it fails." It returns `InvalidArgument` by design, and exercising it adds no coverage of anything this project owns.
- **Do not assert on the absolute byte value** anywhere. It varies with allocator alignment and is a high-water mark.
- **Do not add `openvino_backend` to any `target_link_libraries`.** That is step C, which still has five open questions in its scope boundary.

## Branch

Work on `feature/xnnpack-workspace-metric` off `main` (which now carries PR #9). Repo convention is `feature/*` branches merged via PR — see `main`'s history.
