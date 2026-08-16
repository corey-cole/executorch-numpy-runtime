# OpenVINO Delegate Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the OpenVINO delegate a supported feature on `linux-x86_64` — installable extra, automatic `OPENVINO_LIB_PATH` resolution, committed fixture, and CI that executes a delegated model.

**Architecture:** Detection is lazy and metadata-driven: `MethodMeta::uses_backend("OpenvinoBackend")` answers whether a method needs OpenVINO, and it does so before the method loads. A Python resolver sets `OPENVINO_LIB_PATH` from the installed wheel at program-load time; a C++ guard refuses the load outright when the backend is unregistered or the path is unusable. Both run before ExecuTorch is entered, so a misconfiguration raises a catchable exception instead of poisoning the interpreter.

**Tech Stack:** C++17, ExecuTorch 1.3.1 backend/MethodMeta APIs, nanobind, numpy, pytest, OpenVINO 2025.4.1 (runtime only — no torch, ever).

**Spec:** `docs/superpowers/specs/2026-08-16-openvino-linux-x86_64-design.md`

## Global Constraints

- Backend id is exactly **`OpenvinoBackend`** (lowercase `v`). Used as a string in C++, Python, and CMake nm-guard rows.
- OpenVINO pin is **`2025.4.1`, exact**, and must agree across three files: `pyproject.toml`, `cmake/EtRuntimePin.cmake` (`ET_RUNTIME_OPENVINO_VERSION`), and `tests/models/openvino/MANIFEST`.
- Delegate ships on **`linux-x86_64` only**. All platform variance is expressed as `if(TARGET openvino_backend)` — capability, never a platform name.
- **`CMakeLists.txt:35`'s required-target loop must NOT gain `openvino_backend`.** That loop catches `find_package` early-returning; a target that legitimately doesn't exist on 2 of 3 platforms would turn a real guard into a conditional.
- Parity tolerance is **`atol=1e-2`** and must not be tightened. OpenVINO picks bf16 or f32 from the CPU it lands on at import time; bf16 lands ~2.5e-3 from the f32 eager golden, f32 ~6e-8. Both correct. A tighter bound asserts which CI runner we got.
- **No new `ErrorKind`.** Reuse `BackendMissing`, which maps to `BackendNotAvailable` at `src/binding/module.cpp:124`.
- Never use `os.environ.setdefault` for `OPENVINO_LIB_PATH` — its eager default raises when the wheel is absent even if the variable is already correct. Use an explicit `not in os.environ` check.
- Nothing in this work may import torch.
- After any C++/CMake edit: `rm -rf build && uv pip install -e . --no-build-isolation --reinstall`.

## Facts already verified (do not re-derive)

- Fixture asset: `https://github.com/measly-java-learning/executorch-runtime-dist/releases/download/v1.3.1-10/etnp-openvino-fixtures-1.3.1-2025.4.1.tar.gz`
- Its SHA256: `ea2306a98354a6831f7d00c0581ad477fa7160e68a836d79bcdeabede435fb0f`
- Members: `openvino_tiny.pte` (4.7K, contains `OpenvinoBackend`), `in.bin` (32B), `out.bin` (32B), `shape` (contents: `OV_IN=8` / `OV_OUT=8`)
- **`in.bin`/`out.bin` are float32**: 32 bytes / 8 values = 4 bytes each. Tensor shape is `(1, 8)` — read it from `method_meta` rather than hardcoding.
- `openvino==2025.4.1` publishes `cp312-cp312-manylinux2014_x86_64` (glibc 2.17), comfortably under our `manylinux_2_28` floor.
- `Module::method_meta` does **not** load the method: measured at ~8 µs leaving the XNNPACK workspace at 0, versus 537 µs for run #1 and 19 µs for run #2.
- `method_meta` dict shape: `{"num_inputs", "num_outputs", "inputs": [{"scalar_type", "shape"}], "outputs": [...]}`.

---

### Task 1: `method_uses_backend` and the detection-timing guard

**Files:**
- Modify: `src/et_core/et_core.h` (declare beside `method_meta`, line ~85)
- Modify: `src/et_core/et_core.cpp` (implement beside `method_meta`)
- Modify: `src/binding/module.cpp` (one `.def` on `_Runtime`, beside `method_meta` at line 153)
- Test: `tests/test_backend_detection.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `bool etnp::Runtime::method_uses_backend(const std::string& name, const std::string& backend) const`, exposed as `_core._Runtime.method_uses_backend(name, backend) -> bool`. Tasks 4 and 5 both call it.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_backend_detection.py`:

```python
from executorch_numpy_runtime import _core
from conftest import model_or_skip


def test_method_uses_backend_true_for_delegated_fixture():
    rt = _core.load_path(model_or_skip("conv.pte"))
    assert rt.method_uses_backend("forward", "XnnpackBackend") is True


def test_method_uses_backend_false_for_absent_backend():
    rt = _core.load_path(model_or_skip("conv.pte"))
    assert rt.method_uses_backend("forward", "CoreMLBackend") is False


def test_method_meta_does_not_initialize_delegates():
    """Detection must run BEFORE delegate init, or OPENVINO_LIB_PATH is set too late.

    Module::method_meta not loading the method is an implementation detail of the Module
    extension, not a documented contract -- the header says only "Loads the program if
    needed". If a runtime bump makes it eager, OpenVINO detection silently becomes too
    late and fails at dlopen with no obvious connection to the change. The XNNPACK
    workspace is a direct probe: it is grown during delegate init, so it staying at 0
    across method_meta proves no delegate was initialized.

    Runs in a fresh interpreter because the workspace is process-wide and never shrunk.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        from executorch_numpy_runtime import _core
        rt = _core.load_path("tests/models/conv.pte")
        assert _core.xnnpack_workspace_size_bytes() == 0, "load_path initialized a delegate"
        rt.method_meta("forward")
        rt.method_uses_backend("forward", "XnnpackBackend")
        assert _core.xnnpack_workspace_size_bytes() == 0, "metadata access initialized a delegate"
        print("ok")
        """
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_backend_detection.py -v`

Expected: all three FAIL with `AttributeError: '_Runtime' object has no attribute 'method_uses_backend'`.

- [ ] **Step 3: Declare it in the core header**

In `src/et_core/et_core.h`, inside `class Runtime`, add after the `method_meta` declaration:

```cpp
  // True if `name` is lowered to `backend`. Reads MethodMeta only -- does NOT load
  // the method, which is what lets a caller configure a delegate (e.g. set
  // OPENVINO_LIB_PATH) before delegate init runs. Guarded by
  // tests/test_backend_detection.py::test_method_meta_does_not_initialize_delegates.
  bool method_uses_backend(const std::string& name, const std::string& backend) const;
```

- [ ] **Step 4: Implement it**

In `src/et_core/et_core.cpp`, add immediately after `Runtime::method_meta`:

```cpp
// Uses MethodMeta::uses_backend (runtime/executor/method_meta.h:257). Same locking
// rationale as method_meta(): Module::method_meta mutates internal state, so hold
// exec_mutex to avoid racing execute().
bool Runtime::method_uses_backend(const std::string& name,
                                  const std::string& backend) const {
  std::lock_guard<std::mutex> guard(state_->exec_mutex);
  auto meta = state_->module->method_meta(name);
  if (!meta.ok()) {
    throw EtException({ErrorKind::Load,
        "Could not read metadata for method '" + name + "'", name});
  }
  return meta->uses_backend(backend.c_str());
}
```

- [ ] **Step 5: Expose it from the binding**

In `src/binding/module.cpp`, add to the `nb::class_<Runtime>(m, "_Runtime")` chain, after the `method_meta` def (note the chain currently ends with `;` after `method_meta` — move the semicolon):

```cpp
      .def("method_uses_backend",
           [](Runtime& self, const std::string& n, const std::string& b) {
             return self.method_uses_backend(n, b);
           });
```

- [ ] **Step 6: Rebuild**

```bash
rm -rf build && uv pip install -e . --no-build-isolation --reinstall
```

- [ ] **Step 7: Run to verify they pass**

Run: `python -m pytest tests/test_backend_detection.py -v`

Expected: all three PASS.

If `test_method_meta_does_not_initialize_delegates` fails, **stop** — the whole detection strategy in this plan is invalid and the spec needs revisiting, not the test.

- [ ] **Step 8: Commit**

```bash
git add src/et_core/et_core.h src/et_core/et_core.cpp src/binding/module.cpp tests/test_backend_detection.py
git commit -m "feat: expose per-method backend detection via MethodMeta

method_uses_backend reads MethodMeta without loading the method, which is
what allows a delegate to be configured before its init runs. That timing
is an implementation detail of the Module extension rather than a
documented contract, so it is pinned by a test that probes the XNNPACK
workspace -- grown during delegate init -- across a metadata read."
```

---

### Task 2: Link the OpenVINO delegate

**Files:**
- Modify: `CMakeLists.txt` (link block after line 49; nm-guard expectations near line 140)
- Test: `tests/test_backend_detection.py` (extend)

**Interfaces:**
- Consumes: nothing from Task 1 (independent, but sequenced after so the detection test suite already exists).
- Produces: `OpenvinoBackend` present in `_core.registered_backends()` on `linux-x86_64`. Tasks 3, 5, 6 depend on this.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_backend_detection.py`:

```python
import sys
import platform
import pytest

_IS_LINUX_X86_64 = sys.platform == "linux" and platform.machine() == "x86_64"


@pytest.mark.skipif(not _IS_LINUX_X86_64, reason="OpenVINO delegate ships on linux-x86_64 only")
def test_openvino_backend_is_registered_on_linux_x86_64():
    """The delegate ships in the linux-x86_64 tarball; linking it must register it.

    Platform-conditional rather than capability-conditional on purpose: this is the one
    test that must FAIL if the link is dropped. A requires_backend marker would skip
    instead, turning a regression into a silent pass.
    """
    assert "OpenvinoBackend" in _core.registered_backends()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_backend_detection.py -k openvino_backend_is_registered -v`

Expected: FAIL — `assert 'OpenvinoBackend' in [...]`, listing the backends without it.

- [ ] **Step 3: Link the delegate**

In `CMakeLists.txt`, after the existing `target_link_libraries(_core PRIVATE executorch xnnpack_backend ...)` block, add:

```cmake
# OpenVINO delegate: present in the linux-x86_64 tarball only. Keyed on the TARGET rather
# than on _ETNP_PLATFORM so a platform that later gains the delegate works with no edit --
# the same capability-driven philosophy conftest.py documents for requires_kernel_lib.
#
# Deliberately NOT added to the required-target loop above: that loop catches
# find_package(ExecuTorch) early-returning and producing an empty link line. A target that
# legitimately does not exist on two of three platforms would convert that guard into a
# platform conditional and weaken it.
#
# The delegate dlopens the OpenVINO C API at first use via OPENVINO_LIB_PATH; nothing
# links against OpenVINO here, so the wheel gains only this static archive.
if(TARGET openvino_backend)
  target_link_libraries(_core PRIVATE openvino_backend)
  list(APPEND ETNP_KERNEL_EXPECT_TUS "_GLOBAL__sub_I_OpenvinoBackend.cpp")
  message(STATUS "etnp: OpenVINO delegate linked")
else()
  message(STATUS "etnp: OpenVINO delegate not available in this runtime prefix")
endif()
```

Place it **before** the `add_custom_command(TARGET _core POST_BUILD ...)` block that consumes `ETNP_KERNEL_EXPECT_TUS` (around line 138), or the nm guard will not see the new row.

- [ ] **Step 4: Rebuild**

```bash
rm -rf build && uv pip install -e . --no-build-isolation --reinstall
```

Expected: configure prints `etnp: OpenVINO delegate linked`, and the POST_BUILD nm guard passes with the new TU row.

- [ ] **Step 5: Run to verify it passes**

```bash
python -m pytest tests/test_backend_detection.py -v
python -c "from executorch_numpy_runtime import runtime_info; print(runtime_info()['backends'])"
```

Expected: tests pass; the backend list includes `OpenvinoBackend`.

- [ ] **Step 6: Confirm the nm guard actually covers it**

A guard that silently found nothing is worse than no guard. Verify the symbol is really in the `.so`:

```bash
nm -C $(python -c "import executorch_numpy_runtime._core as c; print(c.__file__)") \
  | grep _GLOBAL__sub_I_OpenvinoBackend
```

Expected: one match. If empty while the build passed, the `list(APPEND ...)` landed after the POST_BUILD command and is being ignored — move it earlier.

- [ ] **Step 7: Confirm nothing else regressed**

```bash
python -m pytest tests/ -q
```

Expected: passes with the two by-design skips. `test_meta_info.py`'s `"XnnpackBackend" in info["backends"]` is a membership test and is unaffected by the new entry.

- [ ] **Step 8: Commit**

```bash
git add CMakeLists.txt tests/test_backend_detection.py
git commit -m "feat: link the OpenVINO delegate where the runtime provides it

Keyed on if(TARGET openvino_backend) rather than on the platform slug, so
a platform that later gains the delegate needs no edit. Deliberately kept
out of the required-target loop, which exists to catch find_package
early-returning and would be weakened by a legitimately-absent target."
```

---

### Task 3: Fixture, extra, and the version-coupling guard

**Files:**
- Create: `tests/models/openvino/{openvino_tiny.pte,in.bin,out.bin,shape,MANIFEST}`
- Modify: `pyproject.toml`
- Modify: `scripts/check-pin-rows.sh`
- Modify: `tests/models/README.md`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: the committed fixture at `tests/models/openvino/openvino_tiny.pte` (used by Tasks 4, 5, 6) and the `openvino` extra.

- [ ] **Step 1: Vendor the fixture**

```bash
mkdir -p tests/models/openvino
cd /tmp && gh release download v1.3.1-10 \
  --repo measly-java-learning/executorch-runtime-dist \
  --pattern 'etnp-openvino-fixtures-*' --dir /tmp/ovfx --clobber
cd /tmp/ovfx && sha256sum -c etnp-openvino-fixtures-1.3.1-2025.4.1.tar.gz.sha256
tar -xzf etnp-openvino-fixtures-1.3.1-2025.4.1.tar.gz
cp openvino_tiny.pte in.bin out.bin shape "$OLDPWD/tests/models/openvino/"
```

Expected: `sha256sum -c` prints `OK`. The asset's SHA256 is
`ea2306a98354a6831f7d00c0581ad477fa7160e68a836d79bcdeabede435fb0f`.

- [ ] **Step 2: Write the manifest**

Create `tests/models/openvino/MANIFEST`:

```
# Provenance for the vendored OpenVINO fixture. Flat key=value on purpose:
# scripts/check-pin-rows.sh reads openvino_version from here in bash, and a format
# needing a parser would split the pin guards across two languages.
tarball_url=https://github.com/measly-java-learning/executorch-runtime-dist/releases/download/v1.3.1-10/etnp-openvino-fixtures-1.3.1-2025.4.1.tar.gz
tarball_sha256=ea2306a98354a6831f7d00c0581ad477fa7160e68a836d79bcdeabede435fb0f
openvino_version=2025.4.1
executorch_version=1.3.1
```

- [ ] **Step 3: Declare the extra**

In `pyproject.toml`, under `[project.optional-dependencies]`, add beside the existing `bf16` entry:

```toml
# Exact pin, never a range: a .pte embeds a precompiled OpenVINO blob, so a floating
# version lets a rebuild import a blob-incompatible runtime -- which surfaces at model
# load, not at install. Must equal ET_RUNTIME_OPENVINO_VERSION in cmake/EtRuntimePin.cmake
# and openvino_version in tests/models/openvino/MANIFEST; scripts/check-pin-rows.sh asserts it.
#
# The environment marker means the extra EXISTS everywhere but installs nothing where the
# delegate is absent, so `pip install ...[openvino]` on aarch64 succeeds as a no-op instead
# of failing resolution. The user then meets our typed platform error at load time.
openvino = ["openvino==2025.4.1; sys_platform=='linux' and platform_machine=='x86_64'"]
```

- [ ] **Step 4: Extend the pin guard**

Append to `scripts/check-pin-rows.sh`, before its final `echo`:

```bash
# The OpenVINO version now lives in three files. They must agree, or a rebuild can install
# a runtime that cannot import the fixture's precompiled blob.
ov_pin="$(grep -oE 'set\(ET_RUNTIME_OPENVINO_VERSION "[^"]+"\)' "$PIN" \
  | grep -oE '"[^"]+"' | tr -d '"')"
ov_manifest="$(grep -oE '^openvino_version=.*' tests/models/openvino/MANIFEST | cut -d= -f2)"
ov_pyproject="$(grep -oE 'openvino==[0-9][^";]*' pyproject.toml | head -1 | cut -d= -f3)"

[ -n "$ov_pin" ]       || fail "no ET_RUNTIME_OPENVINO_VERSION in $PIN"
[ -n "$ov_manifest" ]  || fail "no openvino_version in tests/models/openvino/MANIFEST"
[ -n "$ov_pyproject" ] || fail "no openvino== pin in pyproject.toml"

[ "$ov_pin" = "$ov_manifest" ] && [ "$ov_pin" = "$ov_pyproject" ] || fail \
  "OpenVINO version disagreement: pin=$ov_pin manifest=$ov_manifest pyproject=$ov_pyproject"

echo "ok: openvino ${ov_pin} agrees across pin, manifest, and pyproject"
```

- [ ] **Step 5: Run the guard to verify it passes**

```bash
./scripts/check-pin-rows.sh
```

Expected: the existing platform lines, then `ok: openvino 2025.4.1 agrees across pin, manifest, and pyproject`.

- [ ] **Step 6: Prove the guard actually bites**

A guard that cannot fail is decoration. Break it deliberately:

```bash
sed -i 's/openvino==2025.4.1/openvino==2025.3.0/' pyproject.toml
./scripts/check-pin-rows.sh; echo "exit=$?"
git checkout pyproject.toml
```

Expected: FAILS with the disagreement message and a non-zero exit, then the checkout restores it.

- [ ] **Step 7: Document the fixture**

Add to `tests/models/README.md`: *"`openvino/` — vendored from the upstream release asset (see `openvino/MANIFEST`). `Linear(8,8)+ReLU` fully delegated to OpenVINO, `CompileSpec(device=CPU)`. `in.bin`/`out.bin` are float32, 8 values each, tensor shape `(1,8)`; `out.bin` is the **eager** golden, so comparisons are tolerance-based (see the spec: `atol=1e-2`)."*

- [ ] **Step 8: Commit**

```bash
git add tests/models/openvino pyproject.toml scripts/check-pin-rows.sh tests/models/README.md
git commit -m "feat: vendor the OpenVINO fixture and declare the [openvino] extra

The OpenVINO version is now pinned in three files, so check-pin-rows.sh
asserts they agree -- a disagreement would install a runtime that cannot
import the fixture's precompiled blob, failing at model load rather than
at install.

The extra carries an environment marker so it exists on every platform but
installs nothing where the delegate is absent."
```

---

### Task 4: C++ guard — refuse misconfigured OpenVINO loads

**Files:**
- Modify: `src/et_core/et_core.cpp` (guard helper + call it from the method-load path)
- Test: `tests/test_openvino_errors.py`

**Interfaces:**
- Consumes: `Runtime::method_uses_backend` (Task 1), the linked delegate (Task 2), the fixture (Task 3).
- Produces: `run_method` on an OpenVINO method raises `BackendNotAvailable` when the backend is unregistered or `OPENVINO_LIB_PATH` is unusable. Task 5's Python resolver relies on this as the backstop.

This guard exists because `_core` is importable and bypasses the Python layer — our own test suite uses it directly. Without it, a `_core` user who misconfigures OpenVINO burns the process's `call_once` and cannot recover without restarting.

- [ ] **Step 1: Write the failing test**

Create `tests/test_openvino_errors.py`:

```python
import subprocess
import sys
import textwrap

import pytest

from executorch_numpy_runtime import _core

OPENVINO_LINKED = "OpenvinoBackend" in _core.registered_backends()


def _run(script: str) -> subprocess.CompletedProcess:
    """Each case runs fresh: OPENVINO_LIB_PATH is process env, and the delegate's
    dlopen is std::call_once with no retry, so cases must not share an interpreter."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)], capture_output=True, text=True
    )


@pytest.mark.skipif(OPENVINO_LINKED, reason="this asserts the UNLINKED platform's message")
def test_unlinked_platform_names_the_platform():
    proc = _run(
        """
        from executorch_numpy_runtime import _core
        from executorch_numpy_runtime.errors import BackendNotAvailable
        import numpy as np
        rt = _core.load_path("tests/models/openvino/openvino_tiny.pte")
        try:
            rt.run_method("forward", [np.zeros((1, 8), np.float32)])
        except BackendNotAvailable as e:
            assert "OpenvinoBackend" in str(e), str(e)
            assert "linux-x86_64" in str(e), str(e)
            print("ok")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


@pytest.mark.skipif(not OPENVINO_LINKED, reason="requires the linked delegate")
def test_lib_path_pointing_at_a_directory_is_refused():
    """A directory is upstream's documented top mistake -- the underlying error message
    mentions LD_LIBRARY_PATH, which reads like it wants a directory. It does not."""
    proc = _run(
        """
        import os
        os.environ["OPENVINO_LIB_PATH"] = "/tmp"
        from executorch_numpy_runtime import _core
        from executorch_numpy_runtime.errors import BackendNotAvailable
        import numpy as np
        rt = _core.load_path("tests/models/openvino/openvino_tiny.pte")
        try:
            rt.run_method("forward", [np.zeros((1, 8), np.float32)])
        except BackendNotAvailable as e:
            assert "OPENVINO_LIB_PATH" in str(e), str(e)
            print("ok")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_openvino_errors.py -v`

Expected: on linux-x86_64, `test_lib_path_pointing_at_a_directory_is_refused` FAILS — no `BackendNotAvailable` is raised; the misconfigured `dlopen` is reached and the delegate reports its own failure. On other platforms, `test_unlinked_platform_names_the_platform` FAILS with `ProgramLoadError` instead of `BackendNotAvailable`, which is exactly the misleading message this task replaces.

- [ ] **Step 3: Add the guard helper**

In `src/et_core/et_core.cpp`, add before `Runtime::run_method`:

```cpp
// Refuses an OpenVINO-delegated method that cannot possibly succeed, BEFORE ExecuTorch is
// entered. This matters more than a typical precondition check: the delegate resolves the
// OpenVINO C API with dlopen under std::call_once and never retries, so a failure that
// reaches it leaves the process permanently broken. Raising here keeps the error ordinary
// and the interpreter usable.
//
// Duplicated by the Python layer (_api.Program), deliberately: _core is importable and
// bypasses it, and our own test suite uses _core directly.
static void guard_openvino(const Runtime& rt, const std::string& method) {
  if (!rt.method_uses_backend(method, "OpenvinoBackend")) return;

  if (!backend_available("OpenvinoBackend")) {
    throw EtException({ErrorKind::BackendMissing,
        "This .pte uses the OpenvinoBackend delegate, which this build does not provide. "
        "The OpenVINO delegate ships on linux-x86_64 only. "
        "Re-export without the OpenVINO partitioner to run here.",
        "OpenvinoBackend"});
  }

  const char* lib = std::getenv("OPENVINO_LIB_PATH");
  if (lib == nullptr || *lib == '\0') {
    throw EtException({ErrorKind::BackendMissing,
        "This .pte uses the OpenvinoBackend delegate, but OPENVINO_LIB_PATH is not set. "
        "Set it to the FULL PATH OF THE .so FILE (not a directory) before the first "
        "inference, or install the 'openvino' extra and load through "
        "executorch_numpy_runtime.Runtime, which resolves it for you.",
        "OpenvinoBackend"});
  }

  struct stat st{};
  if (::stat(lib, &st) != 0 || !S_ISREG(st.st_mode)) {
    throw EtException({ErrorKind::BackendMissing,
        std::string("OPENVINO_LIB_PATH does not name a regular file: '") + lib +
        "'. It must be the full path to libopenvino_c.so, not a directory.",
        "OpenvinoBackend"});
  }
}
```

Add the includes this needs, beside the existing standard-library includes:

```cpp
#include <cstdlib>
#include <sys/stat.h>
```

> `<sys/stat.h>` is POSIX. This guard only compiles where the delegate can exist, but the
> function is compiled unconditionally, so wrap the `stat` branch in `#ifndef _WIN32` and
> skip the file check on Windows — the backend is never registered there, so the first
> branch already returns.

- [ ] **Step 4: Call the guard**

In `Runtime::run_method`, as the first statement **before** acquiring `exec_mutex` (the guard takes the lock itself via `method_uses_backend`, so calling it under the lock would deadlock on a non-recursive mutex):

```cpp
  guard_openvino(*this, name);
```

- [ ] **Step 5: Rebuild and run**

```bash
rm -rf build && uv pip install -e . --no-build-isolation --reinstall
python -m pytest tests/test_openvino_errors.py -v
```

Expected: the applicable test for this platform PASSES; the other SKIPS.

- [ ] **Step 6: Verify no deadlock and no regression**

```bash
python -m pytest tests/ -q
rm -rf build/leak && cmake -S native_tests -B build/leak && cmake --build build/leak
ASAN_OPTIONS=detect_leaks=1 ./build/leak/leak_harness tests/models/add.pte 500
```

Expected: suite passes; `leak_harness: 500 iters OK`. The leak harness exercises `run_method` in a tight loop, so a mutex mistake in Step 4 shows up here as a hang.

- [ ] **Step 7: Commit**

```bash
git add src/et_core/et_core.cpp tests/test_openvino_errors.py
git commit -m "feat: refuse misconfigured OpenVINO loads before entering ExecuTorch

The delegate's dlopen runs under std::call_once with no retry, so a
misconfiguration that reaches it breaks the process until restart. Checking
first keeps the failure an ordinary catchable exception.

Also replaces the coarse ProgramLoadError on platforms without the delegate
-- that message blames the .pte, when the file is fine and the platform
simply cannot run it."
```

---

### Task 5: Python resolver and the capability marker

**Files:**
- Create: `executorch_numpy_runtime/_openvino.py`
- Modify: `executorch_numpy_runtime/_api.py` (`Program.__init__`)
- Modify: `tests/conftest.py`
- Test: `tests/test_openvino_resolution.py`

**Interfaces:**
- Consumes: `_core._Runtime.method_uses_backend` (Task 1), the fixture (Task 3), the C++ guard as backstop (Task 4).
- Produces: `executorch_numpy_runtime._openvino.ensure_openvino_lib_path(runtime) -> None`, called from `Program.__init__`; and a `requires_backend(name)` pytest marker.

- [ ] **Step 1: Add the capability marker**

In `tests/conftest.py`, add beside the existing `requires_kernel_lib` registration:

```python
    config.addinivalue_line(
        "markers",
        "requires_backend(name): skip unless <name> is in runtime_info()['backends']. "
        "Capability-driven, not platform-driven: a platform that later gains the backend "
        "starts running these tests with no edit.",
    )
```

and in `pytest_collection_modifyitems`, after the existing kernel-lib loop:

```python
    backends = set(runtime_info()["backends"])
    for item in items:
        for marker in item.iter_markers(name="requires_backend"):
            required = marker.args[0]
            if required not in backends:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"backend {required!r} not linked in this build "
                        f"(linked: {sorted(backends)})"
                    )
                )
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_openvino_resolution.py`:

```python
import subprocess
import sys
import textwrap

import pytest

FIXTURE = "tests/models/openvino/openvino_tiny.pte"


def _run(script: str) -> subprocess.CompletedProcess:
    """Fresh interpreter per case: OPENVINO_LIB_PATH is process env and the delegate's
    dlopen is std::call_once with no retry."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)], capture_output=True, text=True
    )


@pytest.mark.requires_backend("OpenvinoBackend")
def test_resolver_sets_lib_path_to_an_existing_file():
    proc = _run(
        f"""
        import os
        os.environ.pop("OPENVINO_LIB_PATH", None)
        import executorch_numpy_runtime as en
        en.Runtime.get().load_program({FIXTURE!r})
        p = os.environ["OPENVINO_LIB_PATH"]
        assert os.path.isfile(p), p
        assert "libopenvino_c.so" in os.path.basename(p), p
        print("ok")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


@pytest.mark.requires_backend("OpenvinoBackend")
def test_operator_provided_path_is_not_overwritten():
    """An operator override must win -- they may be pointing at the upstream pinned
    bundle rather than the pip wheel."""
    proc = _run(
        f"""
        import os, sys, glob
        from importlib.util import find_spec
        libs = os.path.join(list(find_spec("openvino").submodule_search_locations)[0], "libs")
        real = sorted(glob.glob(os.path.join(libs, "libopenvino_c.so*")))[0]
        os.environ["OPENVINO_LIB_PATH"] = real
        import executorch_numpy_runtime as en
        en.Runtime.get().load_program({FIXTURE!r})
        assert os.environ["OPENVINO_LIB_PATH"] == real
        print("ok")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


@pytest.mark.requires_backend("OpenvinoBackend")
def test_directory_valued_override_raises_before_execution():
    proc = _run(
        f"""
        import os
        os.environ["OPENVINO_LIB_PATH"] = "/tmp"
        import executorch_numpy_runtime as en
        from executorch_numpy_runtime.errors import BackendNotAvailable
        try:
            en.Runtime.get().load_program({FIXTURE!r})
        except BackendNotAvailable as e:
            assert "OPENVINO_LIB_PATH" in str(e)
            print("ok")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_non_openvino_program_never_touches_the_env():
    """Users who never load an OpenVINO model must not pay for this feature -- no
    openvino import, no global env mutation."""
    proc = _run(
        """
        import os, sys
        os.environ.pop("OPENVINO_LIB_PATH", None)
        import numpy as np
        import executorch_numpy_runtime as en
        prog = en.Runtime.get().load_program("tests/models/conv.pte")
        prog.load_method("forward")([np.ones((1, 3, 16, 16), np.float32)])
        assert "OPENVINO_LIB_PATH" not in os.environ
        assert "openvino" not in sys.modules
        print("ok")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
```

- [ ] **Step 3: Run to verify they fail**

Run: `python -m pytest tests/test_openvino_resolution.py -v`

Expected: on linux-x86_64 the three marked tests FAIL (`KeyError: 'OPENVINO_LIB_PATH'` for the first — nothing sets it). `test_non_openvino_program_never_touches_the_env` PASSES already; that is correct — it is a regression guard for Step 4, not a red test.

- [ ] **Step 4: Write the resolver module**

Create `executorch_numpy_runtime/_openvino.py`:

```python
"""OPENVINO_LIB_PATH resolution.

The OpenVINO delegate resolves the OpenVINO C API with dlopen at first use, ONCE, under
std::call_once, with no retry. If that first attempt fails the process stays broken until
it restarts. Everything here therefore runs before ExecuTorch is entered, so a
misconfiguration raises an ordinary exception and the interpreter survives.

The pip wheel ships only the SONAME-versioned libopenvino_c.so.<n>, never the unversioned
name the delegate looks for by default, which is why setting this variable is mandatory
rather than a convenience.
"""

from __future__ import annotations

import glob
import os
from importlib.util import find_spec

_BACKEND = "OpenvinoBackend"
_ENV = "OPENVINO_LIB_PATH"


def _wheel_lib_path() -> str:
    """Absolute path to libopenvino_c.so.* inside the installed openvino wheel."""
    from .errors import BackendNotAvailable

    spec = find_spec("openvino")
    if spec is None or not spec.submodule_search_locations:
        raise BackendNotAvailable(
            "This .pte uses the OpenvinoBackend delegate, but the 'openvino' package is "
            "not installed. Install it with: "
            'pip install "executorch-numpy-runtime[openvino]"'
        )
    libs = os.path.join(list(spec.submodule_search_locations)[0], "libs")
    matches = sorted(glob.glob(os.path.join(libs, "libopenvino_c.so*")))
    if not matches:
        raise BackendNotAvailable(
            f"The installed openvino package has no libopenvino_c.so* under {libs}. "
            f"Set {_ENV} manually to the full path of that file."
        )
    return matches[0]


def ensure_openvino_lib_path(runtime) -> None:
    """Configure OPENVINO_LIB_PATH if this program needs OpenVINO. No-op otherwise.

    Called from Program.__init__, i.e. after the program is loaded but before any method
    is. Reads MethodMeta only, so no delegate has been initialized yet.
    """
    from .errors import BackendNotAvailable

    if not any(
        runtime.method_uses_backend(name, _BACKEND) for name in runtime.method_names()
    ):
        return

    # Explicit membership test, never os.environ.setdefault: setdefault evaluates its
    # default eagerly, so it would resolve the wheel -- and raise when absent -- even when
    # the variable is already set correctly (e.g. by an operator pointing at the upstream
    # pinned bundle).
    existing = os.environ.get(_ENV)
    if existing is not None:
        if not os.path.isfile(existing):
            raise BackendNotAvailable(
                f"{_ENV} is set to {existing!r}, which is not a file. It must be the full "
                f"path to libopenvino_c.so, not a directory. (The underlying error message "
                f"mentions LD_LIBRARY_PATH, which reads like it wants a directory; it does "
                f"not.)"
            )
        return

    os.environ[_ENV] = _wheel_lib_path()
```

- [ ] **Step 5: Call it from `Program`**

In `executorch_numpy_runtime/_api.py`, change `Program.__init__`:

```python
class Program:
    def __init__(self, runtime: "_core._Runtime"):
        self._rt = runtime
        # Must run before any load_method: the OpenVINO delegate's dlopen is once-only.
        # Cheap and import-free for programs that do not use OpenVINO.
        from ._openvino import ensure_openvino_lib_path

        ensure_openvino_lib_path(runtime)
```

- [ ] **Step 6: Run to verify they pass**

Run: `python -m pytest tests/test_openvino_resolution.py -v`

Expected: on linux-x86_64 all four PASS. Elsewhere, three SKIP with the `requires_backend` reason and `test_non_openvino_program_never_touches_the_env` PASSES.

- [ ] **Step 7: Commit**

```bash
git add executorch_numpy_runtime/_openvino.py executorch_numpy_runtime/_api.py tests/conftest.py tests/test_openvino_resolution.py
git commit -m "feat: resolve OPENVINO_LIB_PATH lazily at program load

Detection reads MethodMeta, so it runs before any delegate init and only
for programs that actually use OpenVINO -- users who never touch it pay
nothing and see no global env mutation.

An operator-set path always wins but is validated first, converting a
directory-valued override from a permanent call_once poisoning into an
ordinary exception."
```

---

### Task 6: Parity test, CI, and documentation

**Files:**
- Test: `tests/test_openvino_parity.py`
- Modify: `.github/workflows/qa-gate.yml`
- Modify: `pyproject.toml` (`test-extras`)
- Modify: `CLAUDE.md`, `README.md`

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces: no new interfaces.

- [ ] **Step 1: Write the parity test**

Create `tests/test_openvino_parity.py`:

```python
from pathlib import Path

import numpy as np
import pytest

import executorch_numpy_runtime as en

FIXTURES = Path(__file__).parent / "models" / "openvino"


@pytest.mark.requires_backend("OpenvinoBackend")
def test_openvino_fixture_matches_eager_golden(capsys):
    """Compare the delegate's output against the EAGER golden shipped with the fixture.

    Tolerance is atol=1e-2 and MUST NOT be tightened. OpenVINO selects inference precision
    from the CPU it lands on, at import time rather than at blob-compile time: on a
    bf16-capable runner (avx512_bf16/AMX) it computes in bf16 and lands ~2.5e-3 from the
    f32 golden; elsewhere ~6e-8. Both are correct OpenVINO results, so any tolerance drawn
    between them asserts which machine CI happened to allocate -- a property this project
    does not control. Upstream hit exactly this (executorch-runtime-dist ea393da) with a
    green run and a red run on identical artifacts.

    np.testing.assert_allclose's default rtol=1e-7 would reproduce that flake precisely.

    The loose bound still catches everything it exists to catch: a delegate returning
    zeros, garbage, or the wrong model is orders of magnitude out.
    """
    prog = en.Runtime.get().load_program(str(FIXTURES / "openvino_tiny.pte"))
    method = prog.load_method("forward")

    shape = tuple(method.metadata["inputs"][0]["shape"])
    x = np.frombuffer((FIXTURES / "in.bin").read_bytes(), dtype=np.float32).reshape(shape)
    golden = np.frombuffer((FIXTURES / "out.bin").read_bytes(), dtype=np.float32)

    out = method([x])[0].reshape(-1)

    # Report the precision actually used. atol=1e-2 alone cannot distinguish "correct in
    # bf16" from "quietly degraded", so record which one this run saw. Querying the pip
    # wheel is correct here (unlike upstream, where the wheel is a DIFFERENT OpenVINO than
    # the bundle under test) because the wheel IS the runtime we dlopen.
    try:
        import openvino

        precision = openvino.Core().get_property("CPU", "INFERENCE_PRECISION_HINT")
    except Exception as exc:  # noqa: BLE001 - diagnostic only, never fails the test
        precision = f"(unavailable: {exc})"

    with capsys.disabled():
        print(f"\nov PRECISION {precision}")
        print(f"openvino fixture: max abs diff {np.max(np.abs(out - golden)):.3e}, tol 1e-2")

    np.testing.assert_allclose(out, golden, atol=1e-2, rtol=0)
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_openvino_parity.py -v -s`

Expected on linux-x86_64 with the extra installed: PASS, printing `ov PRECISION f32` (or `bf16`) and a max abs diff well under 1e-2. Elsewhere: SKIP.

If `openvino` is not installed locally: `uv pip install "openvino==2025.4.1"` first.

- [ ] **Step 3: Verify the tolerance is not vacuous**

Confirm the test can actually fail, so a broken delegate would be caught:

```bash
python -c "
import numpy as np
from pathlib import Path
F = Path('tests/models/openvino')
g = np.frombuffer((F/'out.bin').read_bytes(), dtype=np.float32)
print('golden magnitude:', np.max(np.abs(g)))
print('zeros would differ by:', np.max(np.abs(np.zeros_like(g) - g)))
"
```

Expected: the zeros case differs by far more than 1e-2, confirming the bound still catches a dead delegate.

- [ ] **Step 4: Wire CI**

In `.github/workflows/qa-gate.yml`, add after the existing test steps:

```yaml
      - name: OpenVINO delegate (linux-x86_64 only)
        if: matrix.platform == 'linux-x86_64'
        run: |
          set -euo pipefail
          export PATH=/opt/python/cp312-cp312/bin:$PATH
          python -m pip install --quiet "openvino==2025.4.1"
          python -m pytest tests/test_openvino_parity.py tests/test_openvino_resolution.py \
            tests/test_openvino_errors.py -v -s

      - name: OpenVINO absence is reported correctly (non-x86_64)
        if: matrix.platform != 'linux-x86_64'
        run: |
          set -euo pipefail
          export PATH=/opt/python/cp312-cp312/bin:$PATH
          python -m pytest tests/test_openvino_errors.py -v
```

Both matrix legs now assert something real: x86_64 executes a delegated model, aarch64 asserts the platform error rather than merely skipping.

- [ ] **Step 5: Wire the wheel test leg**

In `pyproject.toml`, under `[tool.cibuildwheel]`, beside `test-requires`:

```toml
# The extra's environment marker makes this correctly inert on aarch64 and Windows, so one
# setting covers every leg with no conditionals -- the payoff for putting the marker in the
# extra rather than branching here.
test-extras = ["openvino"]
```

> Place this **before** the `[tool.cibuildwheel.windows]` sub-table. Every key after that
> header silently becomes Windows-only (TOML sub-table scoping) — the file already warns
> about this.

- [ ] **Step 6: Measure the wheel-size delta**

```bash
rm -rf build && uvx cibuildwheel --platform linux
ls -l wheelhouse/*.whl
```

Compare against the pre-change wheel size (build one from `main` if you don't have it). Expected: an increase consistent with one static archive, not with a vendored OpenVINO runtime. If the wheel grew by tens of MB, something is being bundled that should be `dlopen`ed — investigate before merging.

- [ ] **Step 7: Document it**

Add to `CLAUDE.md` after the XNNPACK workspace paragraph:

```markdown
**OpenVINO delegate:** linked on `linux-x86_64` only (`if(TARGET openvino_backend)` — capability, not platform slug). The delegate `dlopen`s the OpenVINO C API at first use via `OPENVINO_LIB_PATH`, **once**, under `std::call_once` with no retry — so a failed first attempt breaks the process until restart. `Program.__init__` therefore resolves that path lazily (only for programs whose `MethodMeta` reports `OpenvinoBackend`) and a C++ guard in `run_method` refuses an unusable configuration before ExecuTorch is entered. Install with the `[openvino]` extra, pinned to `openvino==2025.4.1` exactly — a `.pte` embeds a precompiled blob. Fixture parity uses `atol=1e-2` because OpenVINO picks bf16 or f32 from the host CPU; see `docs/superpowers/specs/2026-08-16-openvino-linux-x86_64-design.md`.
```

Add `openvino` to the extras list in `README.md` beside `bf16`, with a one-line description and the linux-x86_64 caveat.

- [ ] **Step 8: Full sweep**

```bash
python -m pytest tests/ -q
./scripts/check-pin-rows.sh
ruff check .
rm -rf build/leak && cmake -S native_tests -B build/leak && cmake --build build/leak
ASAN_OPTIONS=detect_leaks=1 ./build/leak/leak_harness tests/models/add.pte 500
```

Expected: all clean, two by-design skips.

- [ ] **Step 9: Commit and open the PR**

```bash
git add tests/test_openvino_parity.py .github/workflows/qa-gate.yml pyproject.toml CLAUDE.md README.md
git commit -m "feat: OpenVINO fixture parity, CI coverage, and docs

The gate executes a delegated model on linux-x86_64 and reports the
inference precision OpenVINO chose, which is what keeps the deliberately
loose atol=1e-2 honest. The aarch64 leg asserts the platform error rather
than skipping, so both legs test something real."
git push -u origin feature/openvino-linux-x86_64
gh pr create --fill
```

- [ ] **Step 10: Confirm CI**

```bash
gh pr checks --watch
```

Expected: all green. Read the x86_64 log for the `ov PRECISION` line — if it says `bf16`, the max abs diff should be ~2.5e-3 and still passing. That is the scenario the tolerance exists for, and seeing it pass is worth more than seeing f32 pass.

---

## Notes for the executor

- **Never tighten `atol=1e-2`.** It will look absurdly loose next to a `5.96e-08` measured diff. That measurement is from an f32 runner; a bf16 runner produces `2.5e-3`, and both are correct. See the test's docstring.
- **Every OpenVINO test runs in a subprocess.** `OPENVINO_LIB_PATH` is process env and the delegate's `dlopen` is once-only, so cases that share an interpreter contaminate each other in ways that look like flakes.
- **Do not add `openvino_backend` to `CMakeLists.txt:35`'s required-target loop.** It is a guard against `find_package` early-returning, not an inventory.
- **Do not import torch anywhere**, including in tests. Export is permanently out of scope.
- If Task 1's `test_method_meta_does_not_initialize_delegates` ever fails, stop and escalate — the detection strategy, not the test, is what broke.
