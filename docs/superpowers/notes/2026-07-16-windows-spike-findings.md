# Windows MSVC spike findings (2026-07-16)

Investigation gating the Windows-wheels work. **No production code was written.**
Answers three questions: does `xnnpack_backend` link under MSVC (Q1), does
`XnnpackBackend` self-register (Q2), and does cibuildwheel/scikit-build-core need
explicit VS dev-shell activation (Q3).

## Verdict

| | Question | Answer |
|---|---|---|
| **Q1** | Does `xnnpack_backend` link into a SHARED target under MSVC? | **PASS** — all five needed targets exist; DLL + EXE link clean. |
| **Q2** | Does `XnnpackBackend` self-register at runtime? | **PASS** — registers in both an EXE and a `LoadLibrary`'d DLL. |
| **Q3** | Does the build need explicit VS activation? | **No.** CMake discovers MSVC on its own; activation changes nothing. |

**Q1+Q2 pass → the Windows work is unblocked.** No upstream parity request is needed
for XNNPACK.

**Was `/WHOLEARCHIVE:` needed?** Mechanically yes — but **no manual wrapping is
required**. Upstream's `ExecuTorchTargets.cmake` already ships it as an
`INTERFACE_LINK_OPTIONS` on the imported target, so it propagates automatically.
See Q2 for the constraint this places on the future link line.

## Environment

- Host: `winbox`, Windows 10.0.26200.8875.
- **Visual Studio: `18.8.0`, `Visual Studio Community 2026`**, at
  `C:\Program Files\Microsoft Visual Studio\18\Community`.
  - Compiler: **MSVC 19.51.36248.0**, toolset `14.51.36231`, Hostx64/x64.
- **Parity caveat:** winbox is **VS 18 Community / VS "2026"**; the CI runner is
  **VS 2022 / 17 Enterprise**. This spike is necessary but **NOT sufficient** —
  **CI is the acceptance gate.** The gap here is wider than the plan assumed: winbox
  is a whole major version *ahead* of CI, not merely a different edition.
- Artifact: release `v1.3.1-6`, `executorch-runtime-1.3.1-logging-windows-x86_64.tar.gz`,
  SHA256 `d2bc1859429fe33940adfd110f75d81bb5bedf3163c919d93d8c63531a967a2e` —
  **verified on Linux and re-verified on winbox** (`Get-FileHash` matched).
- Tarball `BUILDINFO`: `platform=windows-x86_64`, `variant=logging`,
  `toolchain=msvc-2022`, `usdt=n/a`, and crucially `-DEXECUTORCH_BUILD_XNNPACK=ON`.

## Q1 — Does `xnnpack_backend` link into a SHARED target under MSVC?

### Libraries actually shipped

All five targets our `_core` needs are present as `.lib` static archives:

```
executorch.lib          extension_data_loader.lib   xnnpack_backend.lib
executorch_core.lib     extension_module_static.lib extension_tensor.lib
XNNPACK.lib  xnnpack-microkernels-prod.lib  cpuinfo.lib  pthreadpool.lib
portable_kernels.lib  portable_ops_lib.lib  flatccrt.lib ...
```

Confirming the known core-only gaps: **no** `optimized_native_cpu_ops_lib`, **no**
`quantized_ops_lib`, no ETNPExtras/LSTM. Expected; the Windows plan handles them.

### Probe

`CMakeLists.txt` (modelled on upstream `test/consumer/`, but linking what we need)
reports target presence, then builds both a SHARED lib and an executable:

```cmake
find_package(ExecuTorch CONFIG REQUIRED)
set(ETNP_PROBE_LIBS executorch xnnpack_backend extension_module_static
    extension_data_loader extension_tensor)
add_library(xnn_probe SHARED probe.cpp)
target_link_libraries(xnn_probe PRIVATE ${ETNP_PROBE_LIBS})
add_executable(probe_main probe.cpp)
target_link_libraries(probe_main PRIVATE ${ETNP_PROBE_LIBS})
```

### Commands

```bash
# VS activation -> Git-Bash handoff (per windows-jni-handoff.md)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=C:/Users/cored/etnp-spike/executorch-runtime-1.3.1-logging-windows-x86_64
cmake --build build
```

### Output (trimmed)

The expected `<X> library is not found` flood appeared (`optimized_kernels`,
`quantized_ops_lib`, `vulkan_backend`, `coreml*`, `qnn*`, `custom_ops`, `etdump`, …)
— **normal for a core-only build**, configure still completed:

```
-- ETNP_TARGET_PRESENT: executorch
-- ETNP_TARGET_PRESENT: xnnpack_backend
-- ETNP_TARGET_PRESENT: extension_module_static
-- ETNP_TARGET_PRESENT: extension_data_loader
-- ETNP_TARGET_PRESENT: extension_tensor
-- Configuring done (1.8s)
-- Generating done (0.0s)

[3/4] Linking CXX executable probe_main.exe
[4/4] Linking CXX shared library xnn_probe.dll
```

Both artifacts produced (`xnn_probe.dll` 1,882,624 bytes; `probe_main.exe` 1,883,648).

**Q1 = PASS.** All five targets exist and link into a SHARED target under MSVC.

## Q2 — Does `XnnpackBackend` actually self-register? (the important one)

Linking is not registering. XNNPACK registers from a static-init TU, and MSVC's
`/OPT:REF` can drop unreferenced objects much as `--gc-sections` does. Per spec D5 we
are **not** porting the Linux `nm` guard (`assert_kernels_registered.cmake`) to MSVC,
so **this runtime check is the only thing standing in for it.**

The probe uses `get_num_registered_backends()` / `get_backend_name(size_t)` — the
exact pair `src/et_core/et_core.cpp:232-241` uses — so a pass here means the real
`registered_backends()` sees it too. Expected name `"XnnpackBackend"` matches
`tests/test_meta_info.py:17`.

### Result A — executable

```
$ ./probe_main.exe
registered backends: 1
  XnnpackBackend
SPIKE PASS: XnnpackBackend registered
PROBE_EXIT=0
```

### Result B — DLL (added beyond the brief)

The brief specified an EXE. But `_core` is a **DLL loaded by Python**, and static-init
timing differs (DllMain load-time vs. EXE startup), so an EXE-only pass would not have
proven the shape we actually ship. A second probe exports an entry point from a SHARED
lib and calls it via `LoadLibrary`/`GetProcAddress`, mirroring Python's import:

```
$ dll_loader.exe
[dll] registered backends: 1
[dll]   XnnpackBackend
SPIKE PASS: XnnpackBackend registered inside DLL
DLL_EXIT=0
```

**Q2 = PASS**, in both the EXE and the DLL shape. `/OPT:REF` did **not** drop the
registration TU.

### Why it survived — and the constraint this imposes

Registration is *not* luck, and not something MSVC does for us. Static libs normally
only pull objects that resolve an undefined symbol, and the registration TU resolves
none — it should have been dropped. It survived because **upstream's config already
requests whole-archive**:

```cmake
# lib/cmake/ExecuTorch/ExecuTorchTargets.cmake:160
set_target_properties(xnnpack_backend PROPERTIES
  INTERFACE_LINK_OPTIONS "SHELL:LINKER:/WHOLEARCHIVE:\$<TARGET_FILE:xnnpack_backend>")
```

(also present for `executorch:133` and `portable_ops_lib:122`.)

Verified reaching the real link line in the generated `build.ninja` — for **both** the
DLL and the EXE:

```
LINK_FLAGS = /machine:x64 /INCREMENTAL:NO
  /WHOLEARCHIVE:.../lib/executorch.lib
  /WHOLEARCHIVE:.../lib/xnnpack_backend.lib
```

> **Constraint for the Windows link line.** `/WHOLEARCHIVE:` is required, but comes
> for free **only if you link the `xnnpack_backend` CMake target**. If future work
> bypasses the imported target and links raw `.lib` paths (or strips
> `INTERFACE_LINK_OPTIONS`), the flags vanish, registration silently breaks, and —
> with no `nm` guard on Windows — it will surface only as "backend not found" at
> model-load time. Keep a runtime registration assertion in the Windows test suite as
> the standing substitute.

## Q3 — Does cibuildwheel/scikit-build-core need explicit VS activation?

### cibuildwheel end-to-end: NOT exercised (host limitation)

```
$ py -m cibuildwheel --platform windows
Building cp312-win_amd64 wheel
Installing Python cp312...
+ Download https://dist.nuget.org/win-x86-commandline/latest/nuget.exe ...
+ nuget.exe install python -Version 3.12.10 ...
The system cannot find the path specified.
cibuildwheel: error: Command [...nuget.exe, install, python, ...] failed with code 1.
```

cibuildwheel always provisions its **own** CPython via `nuget`, ignoring system
interpreters — so this failed identically whether driven by 3.13 or by a real 3.12.
It **never reached CMake**, so it is not a Q3 answer.

**Recorded explicitly: cibuildwheel was NOT exercised end-to-end on winbox. CI remains
the acceptance gate for that specific claim.**

Fallback: drive the same `scikit-build-core → CMake` path that cibuildwheel wraps,
via `python -m build`, both ways.

### Test A — plain shell (no VS activation)

`cl` confirmed absent from PATH first (`where cl` → `CL_NOT_ON_PATH`).

```
$ python3.12 -m build --wheel -C build.verbose=true
*** scikit-build-core 1.0.3 using CMake 4.4.0 (wheel)
*** Configuring CMake...
-- Building for: Visual Studio 18 2026
-- Selecting Windows SDK version 10.0.26100.0 to target Windows 10.0.26200.
-- The CXX compiler identification is MSVC 19.51.36248.0
-- Check for working CXX compiler: C:/Program Files/Microsoft Visual Studio/18/
   Community/VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64/cl.exe - skipped
-- Detecting CXX compile features - done
```

### Test B — activated VS dev shell

Via `Launch-VsDevShell.ps1 -Arch amd64` (`cl on PATH: True`): **byte-for-byte the same**
generator, compiler identification, and `cl.exe` path.

### Answer

**Q3 = No activation required.** CMake selects the `Visual Studio 18 2026` generator and
locates `cl.exe` through the VS installer/registry, independently of PATH. Activation
made **no difference**. The prior art's `Launch-VsDevShell` requirement was an artifact
of driving **Ninja** directly — Ninja needs `cl.exe` on PATH; the VS generator does not.

> Practical note: this holds for the default VS generator. If the Windows plan forces
> `-G Ninja` (as this spike's Q1/Q2 probe did), activation **is** required again.

### Expected post-detection failure (NOT a Q3 answer)

Both runs then failed identically, *after* compiler detection:

```
CMake Error at .../_deps/etnp_runtime-src/lib/cmake/tokenizers/tokenizers-config.cmake:43:
  Could not find sentencepiece library at .../lib/sentencepiece.lib
Call Stack:
  .../lib/cmake/ExecuTorch/executorch-config.cmake:29 (find_package)
  CMakeLists.txt:9 (find_package)
```

**Root cause, useful to the Windows plan:** `cmake/RuntimePin.cmake` derives its slug as
`linux-x86_64`/`linux-aarch64` only, so on Windows it fetched the **linux-x86_64**
tarball (confirmed: fetched `BUILDINFO` says `platform=linux-x86_64`). The pin needs a
`windows-x86_64` row. This is the known missing-pin/missing-extras class of gap
(spec §5.3) and is what the Windows plan will fix.

## Incidental findings worth carrying forward

### Enumerating Pythons on Windows is unreliable — this is the real lesson

I concluded "winbox has only 3.13 and 3.10" and was about to request a software install
on the strength of it. **3.12.10 was already installed.** The official enumerator lied:

```
$ py -0p
 -3.13-64       python3.13.exe
 -3.10-64       C:\Users\cored\...\Python310\python.exe   # 3.12 absent!

$ python3.12 -c "import sys; print(sys.executable)"
C:\Users\cored\AppData\Local\Microsoft\WindowsApps\
  PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe   # it was there
```

The Microsoft Store install is invisible to `py -0p`. **This is precisely why the plan
insists on pinning `-DPython_EXECUTABLE` explicitly and never trusting a bare `python`:
if you cannot even reliably *enumerate* the interpreters on a Windows host, you
certainly cannot let `find_package(Python 3.12 ...)` — a floor, not a pin — choose one
for you.** That lesson outranks the Q3 answer.

Also: `WindowsApps\python3.12.exe` is a Store **alias shim**, not the interpreter.
Always derive the real path from `sys.executable` and pass **that**.

### The floor-not-pin hazard does NOT apply on the scikit-build-core path

Configure died at `CMakeLists.txt:9` *before* the `find_package(Python 3.12 ...)` at
line 16, so CMake's own resolution was never observed. But it does not get the chance
to: scikit-build-core FORCE-pins the interpreter in its initial cache
(`build/cp312-abi3-win_amd64/CMakeInit.txt`):

```cmake
set(Python_EXECUTABLE [===[.../build-env-.../Scripts/python.exe]===] CACHE PATH "" FORCE)
set(Python_ROOT_DIR   [===[.../PythonSoftwareFoundation.Python.3.12_...]===] CACHE PATH "" FORCE)
set(Python_FIND_REGISTRY [===[NEVER]===] CACHE STRING "" FORCE)
```

So the newest-wins hazard is real for **raw `cmake`** invocations (e.g. the
`native_tests/` harnesses) but is already neutralised for wheel builds —
`Python_FIND_REGISTRY=NEVER` additionally blocks registry scanning.

### Driving winbox from Linux

`& $bash -c '...'` from PowerShell mangles embedded double quotes — PowerShell's native
argument re-quoting word-splits the string, so `bash -c` silently receives a truncated
script and the rest becomes positional args (output appears to vanish). Prefer running
the target `.exe` directly over `ssh` (default shell is `cmd`), or keep the bash
one-liner free of embedded double quotes. Also note `cmd`'s `%ERRORLEVEL%` expands at
parse time, so `cmd & echo %ERRORLEVEL%` reports a stale exit code — log to a file and
inspect it instead.

## Reproduction

Spike tree on winbox: `C:\Users\cored\etnp-spike\` (probe sources, tarball, `repo/`).
Probe sources are not committed — this task produces no production code.
