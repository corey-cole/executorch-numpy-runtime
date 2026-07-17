# Windows Wheels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `cp312-win_amd64` wheels with a reduced, explicitly declared kernel set (XNNPACK + portable only), and a C++ runtime that actually loads on a machine without Visual Studio.

**Architecture:** The Windows ExecuTorch tarball is core-only, so the link line becomes conditional and the surviving kernel set is published through a CMake compile define rather than a hardcoded Python list. Tests skip by *capability*, not by platform, so they start running for free if upstream ever ships parity. CI pins `windows-2022` to match the image that produced the `.lib`s, and `delvewheel` vendors the CRT that CPython doesn't ship.

**Tech Stack:** CMake ≥3.24, scikit-build-core + nanobind, pytest, cibuildwheel + delvewheel, GitHub Actions, MSVC 2022.

**Spec:** `docs/superpowers/specs/2026-07-16-usdt-and-windows-wheels-design.md` (§5.2–5.10)

## Global Constraints

- ExecuTorch version is an **exact** contract: `1.3.1`. Runtime pin: **`v1.3.1-6`**.
- `cmake/RuntimePin.cmake` URL rows MUST stay **fully-resolved literals**, never `${VAR}` templates — CI greps this file's raw text before invoking CMake (see its comment at lines 10-14).
- windows-x86_64 SHA256: `d2bc1859429fe33940adfd110f75d81bb5bedf3163c919d93d8c63531a967a2e` (verified against the published `.sha256` and `djl-executorch-engine`'s landed row).
- Windows ships the **`logging` variant only** — there is no bare/devtools Windows build upstream.
- **Link the `xnnpack_backend` CMake target, never a raw `.lib` path.** Upstream's `ExecuTorchTargets.cmake:160` carries `/WHOLEARCHIVE:` as `INTERFACE_LINK_OPTIONS`; a raw path loses it and silently breaks backend registration, with **no `nm` guard on Windows to catch it**.
- **Never** strip the wheel: no `--strip`, no `CMAKE_INSTALL_DO_STRIP`, no linker `-s`. `NOSTRIP` stays.
- Do **not** set scikit-build-core's `python_hints = false` — it force-pins `Python_EXECUTABLE` and sets `Python_FIND_REGISTRY=NEVER`, which is what keeps CMake off the wrong interpreter.
- Do **not** touch `scripts/check-usdt-notes.sh` — verbatim vendored copy, must stay byte-identical to upstream.
- Editable installs do not auto-recompile: `rm -rf build && uv pip install -e . --no-build-isolation --reinstall`.
- `python` is NOT on PATH locally — use `.venv/bin/python`.
- Lint with `ruff check .` before any commit touching Python. (5 pre-existing issues in `tests/test_lstm_smoke.py` and `export_fixtures.py` are NOT yours — leave them.)

---

## Established facts — do not re-litigate

From the completed spike (`docs/superpowers/notes/2026-07-16-windows-spike-findings.md`):

- `xnnpack_backend` links under MSVC and `XnnpackBackend` **self-registers**, verified in a `LoadLibrary`'d DLL. `/OPT:REF` does not drop the static-init TU.
- CMake selects the Visual Studio generator and finds `cl.exe` **without** `Launch-VsDevShell`. (Contingent: forcing `-G Ninja` reinstates the activation requirement.)
- `/WHOLEARCHIVE:` comes free from upstream's `INTERFACE_LINK_OPTIONS` when linking the target.
- The Windows tarball lacks `optimized_native_cpu_ops_lib`, `quantized_ops_lib`, and ETNPExtras/`etnp_ops_lstm` entirely.
- `_core`'s DLL closure (measured, `dumpbin /dependents`): needs `MSVCP140.dll` and `VCRUNTIME140_THREADS.dll` (neither ships with CPython); `VCRUNTIME140.dll`/`VCRUNTIME140_1.dll` do ship; **no `vcomp140.dll`** — no OpenMP.
- `find_package` on Windows prints a flood of `<X> library is not found` lines. **Normal for a core-only build.** Configure still completes. Not an error.

**Two hosts that cannot answer clean-machine questions:** winbox AND the GitHub `windows-2022` runner both have Visual Studio installed, so both have the CRT in `System32`. An `import` succeeding on either proves **nothing** about a user's machine. See §5.9.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `cmake/RuntimePin.cmake` | add windows row; make platform detection OS-aware | 1 |
| `CMakeLists.txt` | conditional link line; publish `ETNP_KERNEL_LIBS` | 2 |
| `src/binding/module.cpp` | expose `__kernel_libs__` | 2 |
| `executorch_numpy_runtime/info.py` | read the define instead of hardcoding | 2 |
| `tests/conftest.py` | capability-marker skip hook | 3 |
| `tests/test_kernel_libs.py` | the contract's own tests | 3 |
| `tests/test_meta_info.py` | XnnpackBackend assertion (Windows's only registration net) | 3 |
| `.github/workflows/build-wheels.yml` | windows-2022 leg, delvewheel, wheel-content assertion | 4 |
| `pyproject.toml` | delvewheel dep for the repair command | 4 |
| `README.md` | platform support table | 5 |

---

## Task 1: Windows pin row + OS-aware platform detection

**Files:**
- Modify: `cmake/RuntimePin.cmake:23-34` (detection), and the pin rows below it

**Interfaces:**
- Consumes: nothing.
- Produces: `_ETNP_PLATFORM` resolving to `windows-x86_64` on Windows; a `logging_windows-x86_64` URL/SHA row. Task 2 depends on the prefix resolving on Windows.

- [ ] **Step 1: Make platform detection OS-aware**

`cmake/RuntimePin.cmake:23-34` currently maps the processor to `linux-*` unconditionally. Replace that block with:

```cmake
if(NOT _ETNP_PLATFORM)
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _etnp_arch)
  # CMAKE_SYSTEM_PROCESSOR is "AMD64" on Windows, "x86_64" on Linux.
  if(_etnp_arch MATCHES "^(x86_64|amd64)$")
    set(_etnp_machine "x86_64")
  elseif(_etnp_arch MATCHES "^(aarch64|arm64)$")
    set(_etnp_machine "aarch64")
  else()
    message(FATAL_ERROR
      "Unsupported target architecture '${CMAKE_SYSTEM_PROCESSOR}' for the ExecuTorch runtime pin; "
      "expected x86_64 or aarch64. Set _ETNP_PLATFORM explicitly to override.")
  endif()

  if(WIN32)
    set(_etnp_os "windows")
  elseif(UNIX AND NOT APPLE)
    set(_etnp_os "linux")
  else()
    message(FATAL_ERROR
      "Unsupported target OS for the ExecuTorch runtime pin (Windows and Linux only). "
      "Set _ETNP_PLATFORM explicitly to override.")
  endif()

  set(_ETNP_PLATFORM "${_etnp_os}-${_etnp_machine}")
endif()
```

- [ ] **Step 2: Add the windows-x86_64 pin row**

Add below the existing linux rows, keeping the fully-resolved-literal style:

```cmake
# windows-x86_64 ships the `logging` variant ONLY -- there is no bare/devtools Windows build
# upstream. This tarball is also CORE-ONLY: no optimized/quantized ops libs, no ETNPExtras
# (and therefore no USDT probes; its BUILDINFO records usdt=n/a).
set(ETNP_RUNTIME_URL_logging_windows-x86_64
  "https://github.com/measly-java-learning/executorch-runtime-dist/releases/download/v1.3.1-6/executorch-runtime-1.3.1-logging-windows-x86_64.tar.gz")
set(ETNP_RUNTIME_SHA256_logging_windows-x86_64 "d2bc1859429fe33940adfd110f75d81bb5bedf3163c919d93d8c63531a967a2e")
```

- [ ] **Step 3: Verify Linux still resolves correctly (regression check)**

```bash
rm -rf build/pintest && cmake -S . -B build/pintest -DPython_EXECUTABLE=$PWD/.venv/bin/python 2>&1 | grep -iE "runtime|prefix" | head -3
```
Expected: configures successfully, resolving the linux-x86_64 row exactly as before. This is the only locally-testable half of this task — Windows detection is proven in Task 4's CI run.

- [ ] **Step 4: Verify the windows row would resolve (without a Windows host)**

```bash
cmake -S . -B build/wintest -D_ETNP_PLATFORM=windows-x86_64 -DPython_EXECUTABLE=$PWD/.venv/bin/python 2>&1 | tail -20
```
Expected: FetchContent downloads and **hash-verifies** the Windows tarball (the `_ETNP_PLATFORM` override exists exactly for this). Configure will then likely FAIL at `find_package`/link because the Windows `.lib`s aren't linkable on Linux — that is fine and expected. What you are proving here is only that the URL and SHA256 row resolve and the hash matches. Report exactly how far it got.

- [ ] **Step 5: Commit**

```bash
git add cmake/RuntimePin.cmake
git commit -m "feat: Add the windows-x86_64 runtime pin row

Platform detection was linux-only; it now derives OS and machine separately
(CMAKE_SYSTEM_PROCESSOR is AMD64 on Windows). Windows ships the logging variant
only, and the tarball is core-only -- no optimized/quantized ops libs, no
ETNPExtras, no USDT."
```

---

## Task 2: The capability contract

**Files:**
- Modify: `CMakeLists.txt:33-39` (link line) and near `:45` (the compile define)
- Modify: `src/binding/module.cpp` (near line 98, where `__et_version__` is exposed)
- Modify: `executorch_numpy_runtime/info.py:23`

**Interfaces:**
- Consumes: Task 1's prefix.
- Produces: `_core.__kernel_libs__` — a **semicolon-free, comma-separated** string, e.g. `"portable,optimized,quantized"` on Linux and `"portable"` on Windows. `runtime_info()["kernel_libs"]` returns it as a `list[str]`. Task 3 keys its skips off this.

**Why:** `info.py:23` currently hardcodes `["portable", "optimized", "quantized"]`. On Windows that is simply false — `optimized_native_cpu_ops_lib` and `quantized_ops_lib` do not exist in the tarball. The link line must stay the single source of truth; Python must never sniff `sys.platform`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kernel_libs.py`:

```python
"""The declared kernel-lib set must match what was actually linked.

kernel_libs is derived from a CMake compile define computed from the real link
line, so these assertions hold on any platform without sniffing sys.platform.
"""

import executorch_numpy_runtime as en
from executorch_numpy_runtime import _core


def test_kernel_libs_is_a_nonempty_list_of_strings():
    libs = en.runtime_info()["kernel_libs"]
    assert isinstance(libs, list)
    assert libs, "kernel_libs must never be empty: portable is always linked"
    assert all(isinstance(x, str) and x for x in libs)


def test_portable_is_always_present():
    # portable_ops_lib exists in every runtime tarball on every platform.
    assert "portable" in en.runtime_info()["kernel_libs"]


def test_kernel_libs_matches_the_compile_define():
    # The Python view must be exactly the C++ define, not an independent guess.
    expected = [s for s in _core.__kernel_libs__.split(",") if s]
    assert en.runtime_info()["kernel_libs"] == expected


def test_kernel_libs_has_no_duplicates():
    libs = en.runtime_info()["kernel_libs"]
    assert len(libs) == len(set(libs))


def test_quantized_implies_the_op_is_registered():
    # Guards the contract against lying in the direction that matters: claiming a
    # kernel lib we did not actually link would make a model fail at load time
    # with "operator not found" instead of a clear capability error.
    info = en.runtime_info()
    if "quantized" in info["kernel_libs"]:
        assert any("quantized" in op for op in info["operators"])
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_kernel_libs.py -v`
Expected: `test_kernel_libs_matches_the_compile_define` FAILS with `AttributeError: __kernel_libs__`. Others may pass against the hardcoded list — that's expected; the define is what's missing.

- [ ] **Step 3: Make the link line conditional and compute the define**

In `CMakeLists.txt`, replace the `target_link_libraries(_core PRIVATE executorch optimized_native_cpu_ops_lib xnnpack_backend quantized_ops_lib ...)` block (lines 33-35) with:

```cmake
# The Windows runtime tarball is core-only: optimized_native_cpu_ops_lib and quantized_ops_lib
# do not exist there. Link what the prefix actually provides, and let that link line be the
# single source of truth for the advertised kernel set (ETNP_KERNEL_LIBS below).
# NOTE: link the TARGETS, never raw .lib paths -- upstream's ExecuTorchTargets.cmake carries
# /WHOLEARCHIVE: as INTERFACE_LINK_OPTIONS, and a raw path silently loses backend registration.
target_link_libraries(_core PRIVATE
  executorch xnnpack_backend
  extension_module_static extension_data_loader extension_tensor)

# portable is unconditional: portable_ops_lib ships in every tarball on every platform.
set(_etnp_kernel_libs "portable")

if(TARGET optimized_native_cpu_ops_lib)
  target_link_libraries(_core PRIVATE optimized_native_cpu_ops_lib)
  list(APPEND _etnp_kernel_libs "optimized")
endif()

if(TARGET quantized_ops_lib)
  target_link_libraries(_core PRIVATE quantized_ops_lib)
  list(APPEND _etnp_kernel_libs "quantized")
endif()

# Comma-separated, NOT semicolon: a CMake list would expand into separate compile-define
# arguments. This string is consumed verbatim by info.py.
list(JOIN _etnp_kernel_libs "," _etnp_kernel_libs_str)
message(STATUS "etnp: kernel libs linked = ${_etnp_kernel_libs_str}")
```

Then, alongside the existing `ETNP_ET_VERSION` define (line 45):

```cmake
target_compile_definitions(_core PRIVATE ETNP_KERNEL_LIBS="${_etnp_kernel_libs_str}")
```

**Do not** link `portable_ops_lib` explicitly — it arrives transitively via `executorch`, exactly as it does today. This task must not change what Linux links; only how it is *described*.

- [ ] **Step 4: Expose it through the binding**

In `src/binding/module.cpp`, beside line 98's `m.attr("__et_version__") = ETNP_ET_VERSION;`:

```cpp
m.attr("__kernel_libs__") = ETNP_KERNEL_LIBS;
```

- [ ] **Step 5: Read the define in info.py**

In `executorch_numpy_runtime/info.py`, replace the hardcoded line 23:

```python
        "kernel_libs": [s for s in _core.__kernel_libs__.split(",") if s],
```

- [ ] **Step 6: Rebuild and run the tests**

```bash
rm -rf build && uv pip install -e . --no-build-isolation --reinstall 2>&1 | grep -i "kernel libs"
.venv/bin/python -m pytest tests/test_kernel_libs.py tests/test_meta_info.py -v
```
Expected: the build prints `etnp: kernel libs linked = portable,optimized,quantized`, and all tests pass.

- [ ] **Step 7: Prove Linux's advertised set did not regress**

```bash
.venv/bin/python -c "import executorch_numpy_runtime as en; print(en.runtime_info()['kernel_libs'])"
```
Expected exactly: `['portable', 'optimized', 'quantized']` — identical to the old hardcoded list. **If this differs on Linux, the conditional link line broke something; stop and report.**

- [ ] **Step 8: Lint and commit**

```bash
ruff check .
git add CMakeLists.txt src/binding/module.cpp executorch_numpy_runtime/info.py tests/test_kernel_libs.py
git commit -m "feat: Derive kernel_libs from the link line

info.py hardcoded [portable, optimized, quantized], which is false on Windows:
the core-only tarball has no optimized_native_cpu_ops_lib and no
quantized_ops_lib. The link line is now conditional on target existence and
publishes the surviving set as the ETNP_KERNEL_LIBS compile define, so the
advertised capabilities cannot drift from what was linked. Python reads the
define rather than sniffing sys.platform."
```

---

## Task 3: Capability-driven test skips

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tests/test_meta_info.py`
- Modify: whichever tests need quantized/LSTM kernels (see Step 3)

**Interfaces:**
- Consumes: `runtime_info()["kernel_libs"]` from Task 2.
- Produces: a `@pytest.mark.requires_kernel_lib("<name>")` marker that skips when the lib isn't linked.

**Why capability-driven and not `sys.platform == "win32"`:** if upstream later ships Windows parity, these tests start running with no edit. A platform check would keep skipping and quietly under-test the new capability.

**Note on conftest:** a skip hook is a legitimate `conftest.py` resident — `pytest_configure`/`pytest_collection_modifyitems` are pytest hooks. This is not "helper functions parked in conftest."

- [ ] **Step 1: Write the failing test**

Add to `tests/test_kernel_libs.py`:

```python
import pytest


@pytest.mark.requires_kernel_lib("portable")
def test_marker_runs_when_lib_present():
    # portable is always linked, so this must actually execute, not skip.
    assert "portable" in en.runtime_info()["kernel_libs"]


@pytest.mark.requires_kernel_lib("definitely-not-a-real-kernel-lib")
def test_marker_skips_when_lib_absent():
    raise AssertionError("must have been skipped by the requires_kernel_lib marker")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_kernel_libs.py -v 2>&1 | tail -20`
Expected: `test_marker_skips_when_lib_absent` FAILS with the AssertionError (the marker does nothing yet), plus a `PytestUnknownMarkWarning`.

- [ ] **Step 3: Implement the hook**

Add to `tests/conftest.py`:

```python
import pytest

from executorch_numpy_runtime import runtime_info


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_kernel_lib(name): skip unless <name> is in runtime_info()['kernel_libs']. "
        "Capability-driven, not platform-driven: if a platform later gains the kernel lib, "
        "the test starts running with no edit.",
    )


def pytest_collection_modifyitems(config, items):
    linked = set(runtime_info()["kernel_libs"])
    for item in items:
        for marker in item.iter_markers(name="requires_kernel_lib"):
            required = marker.args[0]
            if required not in linked:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"kernel lib {required!r} not linked in this build "
                        f"(linked: {sorted(linked)})"
                    )
                )
```

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_kernel_libs.py -v 2>&1 | tail -12`
Expected: `test_marker_runs_when_lib_present` PASSES, `test_marker_skips_when_lib_absent` SKIPS with the reason naming the linked libs. No unknown-mark warning.

- [ ] **Step 5: Mark the capability-dependent tests**

Find the tests that need kernels Windows lacks:

```bash
grep -rln "quantized" tests/
grep -rln "lstm" tests/
```

Add `@pytest.mark.requires_kernel_lib("quantized")` to tests using `tests/models/quantized.pte`, and `@pytest.mark.requires_kernel_lib("lstm")` to the LSTM tests in `tests/test_lstm_smoke.py`.

**Important:** `"lstm"` is NOT currently a member of `kernel_libs` — Task 2 only emits portable/optimized/quantized. Add it in `CMakeLists.txt` alongside the others so the marker has something to key on:

```cmake
if(COMMAND etnp_extras_whole_archive)
  list(APPEND _etnp_kernel_libs "lstm")
endif()
```
Place this next to the other `list(APPEND ...)` calls, before the `list(JOIN ...)`. It must sit inside the existing `if(COMMAND etnp_extras_whole_archive)` block or duplicate its condition — that command only exists when the prefix ships ETNPExtras, which is exactly when `etnp::lstm.out` is linked. Windows has no ETNPExtras, so `"lstm"` is absent there.

Rebuild after this CMake edit (`rm -rf build && uv pip install -e . --no-build-isolation --reinstall`).

- [ ] **Step 6: Verify nothing regressed on Linux**

```bash
.venv/bin/python -c "import executorch_numpy_runtime as en; print(en.runtime_info()['kernel_libs'])"
.venv/bin/python -m pytest tests/ -q 2>&1 | tail -2
```
Expected: `['portable', 'optimized', 'quantized', 'lstm']`, and the suite passes with the **same 2 by-design skips as before** (`test_parity` needs torch; the non-CPU-backend test needs a CoreML fixture). **If the skip count went up on Linux, a marker is wrong** — on Linux every kernel lib is present, so no `requires_kernel_lib` test should skip. Report the count.

- [ ] **Step 7: Strengthen the Windows registration net**

Windows ships **no symbol guard** (spec D5), so `test_meta_info.py`'s backend assertion is the only thing standing between a `/OPT:REF`-dropped registrar and a silently broken wheel. Confirm `tests/test_meta_info.py` already contains:

```python
def test_runtime_info_reports_version_and_backends():
    info = en.runtime_info()
    assert "XnnpackBackend" in info["backends"]
```

It does. Add a comment above it recording *why* it is load-bearing on Windows:

```python
# Load-bearing on Windows: there is no nm/symbol guard there (see the spec's D5), so this
# runtime assertion is the only check that XNNPACK's static-init registrar survived the
# link. cibuildwheel runs this suite against the built wheel, which is what makes it a gate.
```

- [ ] **Step 8: Lint and commit**

```bash
ruff check .
git add tests/conftest.py tests/test_kernel_libs.py tests/test_meta_info.py tests/test_lstm_smoke.py CMakeLists.txt
git commit -m "test: Skip by capability rather than by platform

Tests needing quantized or LSTM kernels now skip via a requires_kernel_lib
marker keyed on runtime_info()['kernel_libs'], so they skip themselves on the
core-only Windows build and start running for free if upstream ships parity.
A platform check would keep skipping and under-test the new capability.

Also records why the XnnpackBackend assertion is load-bearing on Windows: with
no symbol guard there, it is the only net under the static-init registrar."
```

---

## Task 4: CI — Windows leg + delvewheel

**Files:**
- Modify: `.github/workflows/build-wheels.yml`
- Modify: `pyproject.toml` (cibuildwheel config)

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: a repaired `cp312-win_amd64` wheel containing the vendored CRT.

**The two facts driving this task:**
1. **cibuildwheel does not repair Windows wheels by default.** There is no `auditwheel` on Windows. Without an explicit `delvewheel` command, nothing is vendored.
2. `_core` imports **`MSVCP140.dll`** and **`VCRUNTIME140_THREADS.dll`**, and CPython ships neither. `delvewheel` skips `System32`-resolved DLLs as "system" **by default**, so they must be named explicitly — naming them is the entire point of `--add-dll`.

- [ ] **Step 1: Add the Windows matrix entry**

In `.github/workflows/build-wheels.yml`, the `build_wheels` job's matrix becomes:

```yaml
    strategy:
      # A Windows failure must not cancel the Linux legs (and vice versa).
      fail-fast: false
      matrix:
        include:
          - os: ubuntu-latest
          - os: ubuntu-24.04-arm
          # Pinned, NOT windows-latest: executorch-runtime-dist's build-windows job pins
          # windows-2022 to produce the .libs we link, so this keeps consumer and producer
          # toolchains aligned. The MSVC version also measurably changes _core's DLL closure
          # (VCRUNTIME140_THREADS.dll appears only with newer MSVC), which the delvewheel
          # --add-dll list below depends on.
          - os: windows-2022
```

- [ ] **Step 2: Configure the Windows repair**

In `pyproject.toml`'s `[tool.cibuildwheel]` section, add:

```toml
# cibuildwheel does NOT repair Windows wheels by default -- there is no auditwheel there.
# _core is C++ and imports MSVCP140.dll + VCRUNTIME140_THREADS.dll, neither of which ships
# with CPython, so a machine without the VC++ Redistributable fails to import with an opaque
# DLL-load error. delvewheel treats System32-resolved DLLs as system libraries and skips them
# unless named, which is why --add-dll is required rather than optional.
# Static CRT is not an option: the prebuilt ExecuTorch .libs are /MD.
before-build-windows = "pip install delvewheel"
repair-wheel-command-windows = "delvewheel repair -w {dest_dir} {wheel} --add-dll msvcp140.dll;vcruntime140_threads.dll"
build-verbosity = 1
```

`build-verbosity = 1` matters: without it the CMake log is swallowed and the POST_BUILD guards' STATUS lines never appear in CI.

Do **not** touch `repair-wheel-command` for Linux — stock `auditwheel repair` must stay, and it must not gain `--strip`.

- [ ] **Step 3: Assert the CRT was actually vendored**

The existing USDT post-repair step is `if: runner.os == 'Linux'`. Add its Windows counterpart after it:

```yaml
      - name: Assert the C++ runtime was vendored into the Windows wheel
        # The ONLY authoritative check that delvewheel did its job. An `import` on the runner
        # proves nothing: the windows-2022 image ships Visual Studio, so the CRT is in its
        # System32 and any wheel imports fine there -- the same blind spot winbox has.
        # delvewheel mangles vendored DLL names (msvcp140-<hash>.dll), so match by prefix.
        if: runner.os == 'Windows'
        shell: bash
        run: |
          set -euo pipefail
          shopt -s nullglob
          wheels=(wheelhouse/*.whl)
          test "${#wheels[@]}" -gt 0 || { echo "no wheels in wheelhouse/"; exit 1; }
          for whl in "${wheels[@]}"; do
            echo "== $whl"
            contents="$(unzip -Z1 "$whl")"
            echo "$contents" | grep -iE 'msvcp140[^/]*\.dll' \
              || { echo "FAIL: msvcp140 not vendored into $whl"; echo "$contents"; exit 1; }
            echo "$contents" | grep -iE 'vcruntime140_threads[^/]*\.dll' \
              || { echo "FAIL: vcruntime140_threads not vendored into $whl"; echo "$contents"; exit 1; }
          done
```

This fails closed: no wheels, or a wheel missing either DLL, fails the job.

- [ ] **Step 4: Add a manual trigger**

`build-wheels.yml` currently triggers only on `push: branches: [main]`, `pull_request`, and `release` — so pushing this branch fires **no run at all**, and there is otherwise no way to exercise the Windows leg without opening a PR. `qa-gate.yml` already has `workflow_dispatch`; this workflow lacking it is an inconsistency. Add it:

```yaml
on:
  push:
    branches: [main]
  pull_request:
  release:
    types: [published]
  workflow_dispatch:
```

- [ ] **Step 5: Push, trigger, and read the real CI run**

This task **cannot be verified locally** — there is no Windows host in the loop, and winbox cannot answer the question anyway (it has Visual Studio; see the facts section). CI is the only evidence.

```bash
git push -u origin feature/windows-wheels
gh workflow run "Build and publish wheels" --ref feature/windows-wheels
sleep 10
gh run watch "$(gh run list --workflow='Build and publish wheels' --branch feature/windows-wheels --limit 1 --json databaseId -q '.[0].databaseId')"
```

If the run fails, read the actual log (`gh run view <id> --log-failed`) and report what it says. Do **not** infer the cause from the plan's expectations — this is the first time cibuildwheel has ever run on Windows for this project, and first-run friction is expected.

Then report, from the real log:
1. Did the Windows leg configure? Look for `Check for working CXX compiler` naming `cl.exe`. Confirms the spike's Q3 on the *actual* runner (winbox was VS 18.8; this is VS 2022 — a major version apart).
2. What did `etnp: kernel libs linked = ...` print on Windows? Expect `portable` only.
3. Did `assert_kernels_registered` run? It shells to `nm`, which MSVC lacks. **If it fails the Windows build, that is a real finding** — report it; the fix is to make that guard Linux-only, mirroring how `assert_usdt_probes` self-disarms. Do not delete the guard.
4. Did `assert_usdt_probes` disarm cleanly on Windows (`usdt=n/a`)? It should print the "records no USDT" STATUS and not fail.
5. Did the test suite pass, and did the quantized/LSTM tests **skip** rather than fail?
6. Did delvewheel vendor both DLLs (Step 3's assertion)?

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/build-wheels.yml pyproject.toml
git commit -m "ci: Build and repair cp312-win_amd64 wheels

cibuildwheel does not repair Windows wheels by default and there is no
auditwheel there, so delvewheel vendors the C++ runtime _core needs: MSVCP140
and VCRUNTIME140_THREADS, neither of which ships with CPython. They must be
named explicitly because delvewheel skips System32-resolved DLLs as system
libraries. Static CRT is unavailable -- the prebuilt ExecuTorch .libs are /MD.

windows-2022 is pinned to match the image executorch-runtime-dist uses to
produce those .libs; the MSVC version also changes the DLL closure.

The wheel-content assertion is the only authoritative check that vendoring
happened: the runner ships Visual Studio, so an import there would succeed
regardless."
```

---

## Task 5: Document the platform contract

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 2's `kernel_libs`.
- Produces: nothing code depends on.

**Constraint:** `tests/test_docs.py` asserts `README.md` contains `1.3.1`, `manylinux_2_28_x86_64`, `cp312-abi3`, and the phrases `CPU`, `custom operator`, `torchao`, `BFloat16`, `uint16`. Do **not** remove any of them — run that test.

- [ ] **Step 1: Add the platform support table**

Add to `README.md`, near the existing backend/platform prose:

```markdown
## Platform support

| Platform | Wheel | Kernel libs (`kernel_libs`) | Custom ops | USDT probes |
|---|---|---|---|---|
| Linux x86_64 (manylinux_2_28) | `cp312-abi3` | portable, optimized, quantized, lstm | `etnp::lstm.out` | yes |
| Linux aarch64 (manylinux_2_28) | `cp312-abi3` | portable, optimized, quantized, lstm | `etnp::lstm.out` | yes |
| Windows x86_64 | `cp312-abi3` | **portable only** | **none** | no (Linux-only) |

`lstm` appears in `kernel_libs` because `libetnp_ops_lstm.a` is literally a kernel library —
the one that provides the `etnp::lstm.out` custom op. The two columns describe the same
artifact from different angles: what was linked, and what op that gives you.

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
```

- [ ] **Step 2: Verify the doc's claim against reality**

The table says Linux has all four. Confirm:

```bash
.venv/bin/python -c "from executorch_numpy_runtime import runtime_info; print(runtime_info()['kernel_libs'])"
```
Expected: `['portable', 'optimized', 'quantized', 'lstm']`, matching the table's Linux rows. A doc that contradicts the code is a defect.

- [ ] **Step 3: Run the docs test**

Run: `.venv/bin/python -m pytest tests/test_docs.py -v`
Expected: 2 passed. If it fails, you removed a required phrase — restore it.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: State the Windows kernel gap in the README

The Windows wheel is a reduced runtime: XNNPACK + portable kernels only, no
optimized or quantized ops, no etnp::lstm.out. A .pte that loads on Linux can
fail on Windows with operator-not-found, so the gap is declared rather than
left to surface at load time. Points readers at runtime_info()['kernel_libs'],
which is derived from the link line and cannot drift from what shipped."
```

---

## Verification checklist

Before considering this branch done:

- [ ] `.venv/bin/python -m pytest tests/` passes on Linux with **exactly the same 2 by-design skips as before** — no new skips on Linux.
- [ ] `runtime_info()["kernel_libs"]` on Linux is `['portable', 'optimized', 'quantized', 'lstm']` — unchanged capability, newly derived.
- [ ] `ruff check .` shows only the 5 pre-existing issues.
- [ ] The Windows CI leg builds, and `etnp: kernel libs linked = portable`.
- [ ] Windows tests pass, with quantized/LSTM **skipped** (not failed, not silently absent).
- [ ] `XnnpackBackend` is asserted present on Windows — the only registration net there.
- [ ] `assert_usdt_probes` **disarms** on Windows rather than failing.
- [ ] The repaired Windows wheel **contains** `msvcp140*.dll` and `vcruntime140_threads*.dll`.
- [ ] Linux wheels are unaffected: still repaired by stock `auditwheel`, still carrying USDT probes.

## Known limits (state these; do not paper over them)

- **A true clean-machine import is unproven.** Both winbox and the `windows-2022` runner ship Visual Studio, so the CRT is in `System32` on each and an import succeeds there whether or not vendoring worked. The wheel-content assertion proves vendoring happened; it does not prove the wheel loads without the redistributable. A container without the redist would close this (scikit-learn's approach) and is out of scope here.
- **`assert_kernels_registered` on Windows is an open question** (Task 4 Step 4, item 3). It shells to `nm`. If it fails the Windows build, mirror `assert_usdt_probes`'s self-disarm rather than deleting the guard.
- **cibuildwheel-on-Windows has never run for this project.** winbox's nuget CPython provisioning failed, so the first real exercise is this CI run. Expect first-run friction.
