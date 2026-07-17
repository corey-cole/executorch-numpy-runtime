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
| `CMakeLists.txt` | floor assertion; conditional link line; publish `ETNP_KERNEL_LIBS`; gate the nm-guard | 2, 3 |
| `src/binding/module.cpp` | expose `__kernel_libs__` | 2 |
| `executorch_numpy_runtime/info.py` | read the define instead of hardcoding | 2 |
| `cmake/Kernels.cmake` | default the reference kernel OFF on Windows; document GNU-only symbols | 3 |
| `cmake/assert_kernels_registered.cmake` | derive the codegen count from the link line | 3 |
| `tests/conftest.py` | capability-marker skip hook | 4 |
| `tests/test_kernel_libs.py` | the contract's own tests | 4 |
| `tests/test_meta_info.py` | XnnpackBackend assertion (Windows's only registration net) | 4 |
| `.github/workflows/build-wheels.yml` | windows-2022 leg, workflow_dispatch, wheel-content assertion | 5 |
| `pyproject.toml` | pinned delvewheel + the Windows repair command | 5 |
| `README.md` | platform support table | 6 |

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
Expected: configures successfully, resolving the linux-x86_64 row exactly as before. This is the only locally-testable half of this task — Windows detection is proven in Task 5's CI run.

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
- Produces: `_core.__kernel_libs__` — a **semicolon-free, comma-separated** string, e.g. `"portable,optimized,quantized"` on Linux and `"portable"` on Windows. `runtime_info()["kernel_libs"]` returns it as a `list[str]`. Task 4 keys its skips off this.

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

**Read this first — it is why Step 3 has a guard in it.** `executorch-config.cmake` loops
`find_library` over a required-lib list and, on the **first miss**, does:

```cmake
  if(NOT ${lib_var})
    set(EXECUTORCH_FOUND OFF)
    return()          # returns BEFORE include(ExecuTorchTargets.cmake)
  endif()
```

It sets `EXECUTORCH_FOUND` (all-caps), but CMake's `REQUIRED` checks `ExecuTorch_FOUND`
(the find_package name's casing). **Different variable.** So
`find_package(ExecuTorch CONFIG REQUIRED)` can **silently succeed while defining zero
targets**. Every `if(TARGET ...)` below would then be false — and the kernel set would
advertise `portable` for entirely the wrong reason. Verified: this is exactly what happens
when configuring the Windows prefix from Linux (`find_library` looks for `libexecutorch.a`,
the tarball has `executorch.lib`).

So the conditional link line needs a floor assertion.

**This step gives you the ONE authoritative layout for `CMakeLists.txt` lines 31-45.** Do not
treat it as a patch against the current text — write the region to look exactly like this.
Ordering is load-bearing: **every `list(APPEND _etnp_kernel_libs ...)` must precede the
`list(JOIN ...)`**, or the appended entry is silently dropped from the define.

```cmake
# find_package(ExecuTorch CONFIG REQUIRED) can succeed while defining NO targets: its config
# early-returns on the first find_library miss, setting EXECUTORCH_FOUND (all-caps) -- which is
# NOT the ExecuTorch_FOUND that REQUIRED checks. Without this floor, a config that bailed would
# silently produce an empty link line and advertise kernel_libs=portable for the wrong reason.
foreach(_req executorch xnnpack_backend)
  if(NOT TARGET ${_req})
    message(FATAL_ERROR
      "find_package(ExecuTorch) did not define the '${_req}' target. Its config early-returns on "
      "the first find_library miss WITHOUT failing find_package (it sets EXECUTORCH_FOUND, not "
      "ExecuTorch_FOUND). Prefix: ${ETNP_RUNTIME_PREFIX} -- check it matches this platform and "
      "that its lib/ contains the expected library naming for this toolchain.")
  endif()
endforeach()

# Link what the prefix actually provides: the Windows tarball is core-only (no
# optimized_native_cpu_ops_lib, no quantized_ops_lib). The runtime config self-whole-archives
# the kernel/backend archives via INTERFACE_LINK_OPTIONS, so linking the TARGETS is enough --
# never raw .lib paths, which lose /WHOLEARCHIVE: and silently break backend registration.
target_link_libraries(_core PRIVATE
  executorch xnnpack_backend
  extension_module_static extension_data_loader extension_tensor)

# _etnp_kernel_libs is the SINGLE SOURCE OF TRUTH for the advertised kernel set. Every APPEND
# below must come before the JOIN at the bottom of this block.
# portable is unconditional: portable_ops_lib ships in every tarball on every platform, and
# arrives transitively via `executorch` -- do NOT link it explicitly.
set(_etnp_kernel_libs "portable")

if(TARGET optimized_native_cpu_ops_lib)
  target_link_libraries(_core PRIVATE optimized_native_cpu_ops_lib)
  list(APPEND _etnp_kernel_libs "optimized")
endif()

if(TARGET quantized_ops_lib)
  target_link_libraries(_core PRIVATE quantized_ops_lib)
  list(APPEND _etnp_kernel_libs "quantized")
endif()

# PRESERVED from the current file (do not drop): the custom-kernel seam's archive. On Windows
# this target does not exist -- Task 3 defaults ETNP_BUILD_REFERENCE_KERNEL OFF there, so
# etnp_kernels has no sources and is never created.
if(TARGET etnp_kernels)
  target_link_libraries(_core PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,etnp_kernels>")
endif()

# PRESERVED from the current file (do not drop), with one line added: the "lstm" append lives
# HERE, inside this block and before the JOIN. etnp_extras_whole_archive only exists when the
# prefix ships ETNPExtras, which is exactly when etnp::lstm.out is linked. Windows has no
# ETNPExtras, so "lstm" is absent there.
if(COMMAND etnp_extras_whole_archive)
  etnp_extras_whole_archive(_core)
  list(APPEND _etnp_kernel_libs "lstm")
endif()

# Comma-separated, NOT semicolon: a CMake list would expand into separate compile-define
# arguments. This string is consumed verbatim by info.py. MUST come after every APPEND above.
list(JOIN _etnp_kernel_libs "," _etnp_kernel_libs_str)
message(STATUS "etnp: kernel libs linked = ${_etnp_kernel_libs_str}")

target_compile_definitions(_core PRIVATE ETNP_ET_VERSION="${ETNP_ET_VERSION}")
target_compile_definitions(_core PRIVATE ETNP_KERNEL_LIBS="${_etnp_kernel_libs_str}")
```

**What this replaces, and what it deliberately keeps:**

| Current lines | Fate |
|---|---|
| 31-32 (`# Link the whole ExecuTorch stack...` comment) | **rewritten** — "linking these targets is enough" stops being true once the link is conditional |
| 33-35 (`target_link_libraries(... optimized_native_cpu_ops_lib ... quantized_ops_lib ...)`) | **replaced** by the conditional form |
| 37-39 (`if(TARGET etnp_kernels)`) | **kept**, moved above the JOIN |
| 41-43 (`if(COMMAND etnp_extras_whole_archive)`) | **kept**, moved above the JOIN, plus the `"lstm"` append |
| 45 (`ETNP_ET_VERSION` define) | **kept**, now joined by the `ETNP_KERNEL_LIBS` define |
| 47-53 (`assert_kernels_registered` POST_BUILD) | **untouched by this task** — Task 3 gates it |

This task must not change **what** Linux links — only how it is *described*. Step 7 proves that.

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

## Task 3: Windows-proof the build guards and the reference kernel

**Files:**
- Modify: `cmake/Kernels.cmake` (reference-kernel default; document the symbol convention)
- Modify: `CMakeLists.txt:47-53` (gate the `assert_kernels_registered` POST_BUILD invocation)
- Modify: `cmake/assert_kernels_registered.cmake` (derive expectations from the linked set)

**Interfaces:**
- Consumes: `_etnp_kernel_libs` from Task 2.
- Produces: a build whose guards are correct on both toolchains. Task 5's CI run depends on this; without it the Windows leg fails at POST_BUILD.

**Why this task exists — three certainties, not speculations.** `assert_kernels_registered`
**will** fail on Windows, and we know it from reading the code rather than from a CI round-trip:

1. **No `nm`.** `cmake/assert_kernels_registered.cmake:11-15` does
   `execute_process(COMMAND "${NM}" "${SO}" ...)` and FATAL_ERRORs on non-zero. MSVC ships no `nm`.
2. **Its symbols cannot exist under MSVC even if `nm` did.** It requires
   `_GLOBAL__sub_I_XNNPACKBackend` — a GNU-style static-init symbol. MSVC emits `??__E`-mangled
   dynamic initializers instead.
3. **Two of its three checks reference libraries Windows doesn't have.** The
   `_codegen_count >= 2` check counts TUs contributed by `quantized_ops_lib` +
   `optimized_native_cpu_ops_lib`. Neither is in the Windows tarball, so the count is
   structurally 0.

Separately, `cmake/Kernels.cmake` defaults `ETNP_BUILD_REFERENCE_KERNEL` **ON**, so the Windows
wheel would ship `etnp::triple.out` — which nobody intended, and which the README table
(Task 6) claims is absent.

- [ ] **Step 1: Default the reference kernel OFF on Windows**

In `cmake/Kernels.cmake`, the option is currently a bare `ON`:

```cmake
option(ETNP_BUILD_REFERENCE_KERNEL
  "Compile the bundled reference custom kernel (etnp::triple.out)" ON)
```

Make the default platform-dependent, keeping it an overridable option:

```cmake
# Default OFF on Windows: the upstream Windows runtime distribution ships no extras yet, so
# Windows is a core-only runtime with no custom ops (see the README's platform table). Linux
# keeps this ON, which is what keeps the custom-kernel seam CI-tested via
# native_tests/kernel_registration_test.cpp -- the seam cannot rot just because Windows skips it.
# Flip back on for Windows once upstream extras land there (and see the note on
# ETNP_KERNEL_EXPECT_TUS below: the nm-guard's symbol names are GNU-only).
if(WIN32)
  set(_etnp_ref_kernel_default OFF)
else()
  set(_etnp_ref_kernel_default ON)
endif()
option(ETNP_BUILD_REFERENCE_KERNEL
  "Compile the bundled reference custom kernel (etnp::triple.out)" ${_etnp_ref_kernel_default})
```

Consequence, which is the point: on Windows `_etnp_kernel_sources` is empty → the
`etnp_kernels` target is never created → `if(TARGET etnp_kernels)` is false → nothing custom is
linked → `ETNP_KERNEL_EXPECT_TUS` stays empty.

- [ ] **Step 2: Document that the expected-TU symbols are GNU-only**

Still in `cmake/Kernels.cmake`, above the `ETNP_KERNEL_EXPECT_TUS` loop, record the constraint
so the next reader doesn't have to rediscover it:

```cmake
  # NOTE: these are GNU-style static-init symbol names. MSVC never emits _GLOBAL__sub_I_*
  # (it emits ??__E-mangled dynamic initializers), so this list is only meaningful for a
  # GNU/Clang link -- which is why CMakeLists.txt gates the nm-guard that consumes it.
  foreach(_src IN LISTS _etnp_kernel_sources)
```

- [ ] **Step 3: Gate the `nm` guard on a toolchain that has one**

In `CMakeLists.txt`, wrap the existing `assert_kernels_registered` POST_BUILD command (lines
47-53). Keep the command itself byte-identical apart from the added `-DEXPECT_CODEGEN` argument
in Step 4:

```cmake
# Post-link kernel-registration guard (fail the BUILD, not runtime). The custom kernel seam
# appends its expected registrar TUs via ETNP_KERNEL_EXPECT_TUS.
#
# GNU/Clang only, deliberately. Under MSVC this guard cannot work for three independent
# reasons: there is no `nm`; MSVC emits ??__E-mangled initializers rather than the
# _GLOBAL__sub_I_* symbols the guard matches; and the codegen TUs it counts come from
# quantized_ops_lib + optimized_native_cpu_ops_lib, which the Windows tarball does not ship.
# Windows substitutes a RUNTIME assertion instead -- XnnpackBackend must appear in
# registered_backends() (see the spec's D5 and tests/test_meta_info.py). This is a known,
# accepted asymmetry, not an oversight.
if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
  add_custom_command(TARGET _core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -DSO=$<TARGET_FILE:_core> -DNM=nm
            "-DEXTRA_TUS=${ETNP_KERNEL_EXPECT_TUS}"
            "-DEXPECT_CODEGEN=${_etnp_expect_codegen}"
            -P ${CMAKE_SOURCE_DIR}/cmake/assert_kernels_registered.cmake
    VERBATIM)
else()
  message(STATUS
    "etnp: assert_kernels_registered SKIPPED -- no nm/GNU symbols under '${CMAKE_CXX_COMPILER_ID}'. "
    "Backend registration is asserted at runtime instead (see tests/test_meta_info.py).")
endif()
```

The `else()` branch is not decoration: a guard that vanishes silently is exactly the failure
mode this repo's guards exist to prevent. Say so in the log.

- [ ] **Step 4: Derive the codegen expectation from what was actually linked**

`cmake/assert_kernels_registered.cmake:31-40` hardcodes `_codegen_count LESS 2`. That "2" is
only correct by coincidence of `quantized_ops_lib` and `optimized_native_cpu_ops_lib` both
always being present — a latent fragility on Linux, and simply wrong anywhere they aren't.
Task 2 already computes the linked set, so derive it.

In `CMakeLists.txt`, immediately after Task 2's `list(JOIN ...)` block, compute the expectation:

```cmake
# The nm-guard counts one codegen registrar TU per codegen'd ops lib actually linked. Derive it
# from the link line rather than hardcoding 2, which was only right while both libs were always
# present.
set(_etnp_expect_codegen 0)
if(TARGET optimized_native_cpu_ops_lib)
  math(EXPR _etnp_expect_codegen "${_etnp_expect_codegen} + 1")
endif()
if(TARGET quantized_ops_lib)
  math(EXPR _etnp_expect_codegen "${_etnp_expect_codegen} + 1")
endif()
```

Then in `cmake/assert_kernels_registered.cmake`, replace the hardcoded check:

```cmake
# EXPECT_CODEGEN is supplied by the caller from the real link line (quantized_ops_lib and
# optimized_native_cpu_ops_lib each codegen their own registration TU from the same upstream
# template, so both land under the IDENTICAL local-symbol name -- count, don't string(FIND)).
if(NOT DEFINED EXPECT_CODEGEN)
  set(EXPECT_CODEGEN 2)  # back-compat for callers predating the derived count
endif()
string(REGEX MATCHALL "_GLOBAL__sub_I_RegisterCodegenUnboxedKernelsEverything\\.cpp"
       _codegen_matches "${_syms}")
list(LENGTH _codegen_matches _codegen_count)
if(_codegen_count LESS ${EXPECT_CODEGEN})
  message(FATAL_ERROR
    "Expected ${EXPECT_CODEGEN} kernel-registration TU(s) named "
    "'_GLOBAL__sub_I_RegisterCodegenUnboxedKernelsEverything.cpp' in ${SO}, found "
    "${_codegen_count}: whole-archive regressed at final link for one of the kernel libs -> "
    "op not found at model-load time.")
endif()
```

Leave the `_GLOBAL__sub_I_XNNPACKBackend` check and the `EXTRA_TUS` loop untouched.

**`native_tests/` also invokes this guard.** Check whether it passes `EXPECT_CODEGEN`; the
`if(NOT DEFINED ...)` default above keeps it working either way, which is why the default
exists. Confirm rather than assume:

```bash
grep -rn "assert_kernels_registered\|EXTRA_TUS" native_tests/CMakeLists.txt
```

- [ ] **Step 5: Verify Linux is completely unchanged**

This task must be a no-op on Linux. Rebuild and prove it:

```bash
rm -rf build && uv pip install -e . --no-build-isolation --reinstall 2>&1 | grep -iE "kernel libs|assert_kernels|SKIPPED"
.venv/bin/python -c "from executorch_numpy_runtime import runtime_info; print(sorted(o for o in runtime_info()['operators'] if o.startswith('etnp')))"
.venv/bin/python -m pytest tests/ -q 2>&1 | tail -2
```
Expected: the build still prints `etnp: kernel libs linked = portable,optimized,quantized,lstm`;
`assert_kernels_registered` still **runs** (no SKIPPED line — this is GCC); etnp ops are still
`['etnp::lstm.out', 'etnp::triple.out']`; suite passes with the same 2 by-design skips.

**If the etnp ops list lost `triple.out` on Linux, Step 1's `if(WIN32)` is wrong** — stop and
report.

- [ ] **Step 6: Prove the guard still fails when it should (do not skip this)**

Gating a guard is exactly when to re-verify it still bites on the platform where it runs:

```bash
REAL=$(.venv/bin/python -c "import executorch_numpy_runtime._core as c; print(c.__file__)")
cp "$REAL" /tmp/nostrip-test.so && strip --strip-all /tmp/nostrip-test.so
cmake -DSO=/tmp/nostrip-test.so -DNM=nm -DEXPECT_CODEGEN=2 \
      -P cmake/assert_kernels_registered.cmake; echo "rc=$?"
```
Expected: FATAL_ERROR naming the missing registration, `rc=1`.

- [ ] **Step 7: Commit**

```bash
git add cmake/Kernels.cmake CMakeLists.txt cmake/assert_kernels_registered.cmake
git commit -m "feat: Make the build guards and reference kernel Windows-correct

assert_kernels_registered cannot work under MSVC for three independent reasons:
there is no nm; MSVC emits ??__E-mangled initializers rather than the
_GLOBAL__sub_I_* symbols it matches; and the codegen TUs it counts come from
quantized_ops_lib + optimized_native_cpu_ops_lib, which the Windows tarball does
not ship. It is now gated to GNU/Clang and says so in the log rather than
vanishing silently. Windows substitutes the runtime XnnpackBackend assertion.

The codegen count is derived from the real link line instead of a hardcoded 2,
which was only correct while both ops libs were always present.

ETNP_BUILD_REFERENCE_KERNEL now defaults OFF on Windows: upstream ships no extras
there, so Windows is core-only with no custom ops. Linux keeps it ON, which is
what keeps the custom-kernel seam CI-tested."
```

---

## Task 4: Capability-driven test skips

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

**No CMake edit is needed here.** `"lstm"` is already appended to `_etnp_kernel_libs` by Task 2's layout, inside the `if(COMMAND etnp_extras_whole_archive)` block and before the `list(JOIN ...)`. That ordering is load-bearing — an append after the JOIN is silently dropped — which is why the layout owns it rather than this step. Verify it is there before marking anything:

```bash
.venv/bin/python -c "from executorch_numpy_runtime import runtime_info; print(runtime_info()['kernel_libs'])"
```
Expected: `['portable', 'optimized', 'quantized', 'lstm']`. If `lstm` is missing, Task 2's layout was mis-applied (append landed after the JOIN) — fix that rather than adding a second append here.

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

## Task 5: CI — Windows leg + delvewheel

**Files:**
- Modify: `.github/workflows/build-wheels.yml`
- Modify: `pyproject.toml` (cibuildwheel config)

**Interfaces:**
- Consumes: Tasks 1-4.
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
#
# build-verbosity is GLOBAL on purpose: without it the CMake log is swallowed and the
# POST_BUILD guards' STATUS lines never appear in CI -- wanted on every leg, not just Windows.
build-verbosity = 1
```

Then, as a **nested sub-table** — not flat suffixed keys:

```toml
# cibuildwheel does NOT repair Windows wheels by default -- there is no auditwheel there.
# _core is C++ and imports MSVCP140.dll + VCRUNTIME140_THREADS.dll, neither of which ships
# with CPython, so a machine without the VC++ Redistributable fails to import with an opaque
# DLL-load error. delvewheel treats System32-resolved DLLs as system libraries and skips them
# unless named, which is why --add-dll is required rather than optional.
# Static CRT is not an option: the prebuilt ExecuTorch .libs are /MD.
#
# delvewheel is PINNED on purpose. It decides which DLLs land in the shipped wheel, so an
# unpinned upgrade could silently change the wheel's contents -- in a project that pins its
# runtime tarball by SHA256 and attests it in CI, an unpinned wheel-repair tool would be the
# loosest link in the chain. Bump procedure: raise the pin deliberately, then re-run the
# Step 4 assertion below to confirm both CRT DLLs are still vendored under the new version.
[tool.cibuildwheel.windows]
before-build = "pip install delvewheel==1.13.0"
repair-wheel-command = "delvewheel repair -w {dest_dir} {wheel} --add-dll msvcp140.dll;vcruntime140_threads.dll"
```

**The nested-table form is mandatory — do not use flat `before-build-windows` keys.** Verified
against the pinned `cibuildwheel==4.1.0`'s own `resources/cibuildwheel.schema.json`: its
top-level `properties` expose flat OS-suffixed keys **only** for the
`manylinux-*-image` / `musllinux-*-image` family. `before-build-windows` is not a key at all.
Per-platform overrides live in the `windows` / `linux` / `macos` sub-tables, each
`additionalProperties: false`.

This matters more than a syntax nit: cibuildwheel validates `[tool.cibuildwheel]` **once,
regardless of runner OS**, so an invalid key here fails **every** leg — including the Linux
ones that were previously green. A first attempt at this task did exactly that.

Leave the Linux `repair-wheel-command` alone — there isn't one, and that is correct: it relies
on cibuildwheel's stock `auditwheel repair` default, which must not gain `--strip`.

`1.13.0` is the current release (2026-05-28). If a newer one exists when you implement this, use it and say so in the report — the point is that the version is *chosen and recorded*, not that it is frozen forever.

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

- [ ] **Step 5: Validate the config with a LOCAL Linux build — BEFORE pushing**

**Do not skip this and do not push first.** cibuildwheel validates `[tool.cibuildwheel]` **once,
regardless of platform**, so a structural config error fails every leg — and a local Linux run
catches it in seconds instead of burning a CI round-trip. A first attempt at this task pushed a
malformed key straight to CI and took down the Windows leg *and* both previously-green Linux
legs; the local run below would have caught it before the push.

`build/` must be EMPTY first — the CMake cache is path-sensitive:

```bash
rm -rf build wheelhouse
uvx cibuildwheel --platform linux 2>&1 | tail -30
```

What this proves, and what it cannot:

| Proves | Cannot prove |
|---|---|
| The TOML parses and every key is valid **for this pinned cibuildwheel version** | Anything Windows-specific |
| The Windows sub-table doesn't break the Linux legs | That delvewheel vendors correctly |
| Linux wheels still build and their tests pass | That MSVC resolves |

Expected: Linux wheels build and pass. **Any `Option '...' not supported in a config file` error
means the config is structurally wrong — fix it here, not in CI.**

Then confirm the Linux wheel is still repaired and traceable (the USDT probes must survive, as before):

```bash
for whl in wheelhouse/*.whl; do
  tmp="$(mktemp -d)"; unzip -q "$whl" -d "$tmp"
  so="$(find "$tmp" -name '_core*.so' -print -quit)"
  echo "== $whl"; bash scripts/check-usdt-notes.sh --expect on "$so"
done
```
Expected: `USDT probe contract OK` per wheel. If this regressed, your `pyproject.toml` change
disturbed the Linux repair path — fix before pushing.

- [ ] **Step 6: Push, trigger, and read the real CI run**

Only after Step 5 is green. The **Windows** half of this task cannot be verified locally — there is no Windows host in the loop, and winbox cannot answer the question anyway (it has Visual Studio; see the facts section). CI is the only evidence for Windows.

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
3. Did `assert_kernels_registered` **skip**, loudly? Task 3 gated it to GNU/Clang, so the Windows log must contain `etnp: assert_kernels_registered SKIPPED -- no nm/GNU symbols under 'MSVC'`. Two failure modes to look for: if it **ran** and failed, the gate's compiler-ID match is wrong; if **no line appears at all**, the gate is silently swallowing it, which is the failure mode this repo's guards exist to prevent. Report which you saw.
4. Did `assert_usdt_probes` disarm cleanly on Windows (`usdt=n/a`)? It should print the "records no USDT" STATUS and not fail.
5. Did the test suite pass, and did the quantized/LSTM tests **skip** rather than fail?
6. Did delvewheel vendor both DLLs (Step 3's assertion)?

- [ ] **Step 7: Commit**

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

## Task 6: Document the platform contract

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
```

- [ ] **Step 2: Verify EVERY column of the table against the real build**

An earlier draft of this table was wrong on **both** Linux rows and the Windows row, because it
was written from intent rather than from the build. Do not repeat that: read the values out and
match them.

```bash
.venv/bin/python - <<'PY'
from executorch_numpy_runtime import runtime_info
info = runtime_info()
print("kernel_libs :", info["kernel_libs"])
print("etnp ops    :", sorted(o for o in info["operators"] if o.startswith("etnp")))
PY
```
Expected on Linux, and the table's Linux rows must match **exactly**:
```
kernel_libs : ['portable', 'optimized', 'quantized', 'lstm']
etnp ops    : ['etnp::lstm.out', 'etnp::triple.out']
```

Both columns matter. The custom-ops column is the one that was previously wrong — it listed
only `lstm.out` while the wheel also ships `triple.out`. A doc that contradicts the code is a
defect, and this table makes two independent claims, so check two.

The Windows row cannot be verified locally; it follows from Task 3 Step 1 (reference kernel OFF
→ no custom ops) and Task 5's CI run, which prints `etnp: kernel libs linked = portable`.

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
- **cibuildwheel-on-Windows has never run for this project.** winbox's nuget CPython provisioning failed, so the first real exercise is this CI run. Expect first-run friction.
- **The custom-kernel seam is unproven under MSVC** — and stays that way by choice. Task 3 defaults `ETNP_BUILD_REFERENCE_KERNEL` OFF on Windows, so nothing custom is linked there and there is nothing to prove yet. Linux keeps the seam CI-tested, so it cannot rot. When upstream ships extras for windows-x86_64, this reopens: `EXECUTORCH_LIBRARY`'s registrar is **not** the same code path as XNNPACK's own, and `/OPT:REF` is the hazard. The spike proved XNNPACK's registrar survives; ours is untested on MSVC.
- **No symbol-level registration guard exists on Windows, by design** (spec D5). `assert_kernels_registered` is gated to GNU/Clang in Task 3 for three reasons that cannot be engineered around here: no `nm`, MSVC's `??__E` initializer mangling, and the codegen TUs coming from ops libs the Windows tarball doesn't ship. The runtime `XnnpackBackend` assertion is the whole net. If it ever regresses, Windows loses its only registration check.
