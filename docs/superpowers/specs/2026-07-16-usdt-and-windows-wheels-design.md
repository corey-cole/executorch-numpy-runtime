# USDT Tracepoint Guard + Windows Wheels — Design Spec

**Date:** 2026-07-16
**Status:** Approved for planning
**Supersedes:** `docs/usdt-preparation-checklist.md` (this spec resolves that checklist; the
checklist is rewritten as `docs/usdt-tracepoints.md` in PR2)

---

## 1. Objective

Land two features unlocked by upstream `executorch-runtime-dist` release **v1.3.1-6**:

1. **USDT tracepoint guard** — the runtime now ships systemtap USDT probes on Linux. Guard
   the build so a stripped or GC'd probe table fails CI rather than silently producing an
   un-traceable wheel.
2. **Windows amd64 wheels** — v1.3.1-6 ships a `logging` windows-x86_64 tarball, enough to
   build and publish a Windows wheel with a **reduced, explicitly declared** kernel set.

### Findings that shaped this spec

Both features were investigated against the actual v1.3.1-6 artifacts rather than the
release notes. Three findings materially changed the design:

- **USDT is already shipping, and not where the checklist assumed.** The linux-x86_64
  logging tarball records `usdt=on` in `BUILDINFO`. Exactly one archive carries probes:
  `libetnp_ops_lstm.a` — provider `etnp`, probes `lstm_xnn_cache__{hit,miss,evict}`.
  **Nothing in the ExecuTorch core libs.** The probes therefore ride in via the ETNPExtras
  whole-archive path (`etnp_extras_whole_archive`), not the ExecuTorch stack. The
  checklist's reasoning holds; its target archive was wrong.

- **The Windows tarball is not at parity with Linux.** Comparing exported CMake targets:

  | Target (needed by our link line) | linux-x86_64 | windows-x86_64 |
  |---|---|---|
  | `xnnpack_backend`, `portable_ops_lib`, `extension_*` | yes | yes |
  | `optimized_native_cpu_ops_lib` | yes | **absent** |
  | `quantized_ops_lib` | yes | **absent** |
  | ETNPExtras / `etnp_ops_lstm` | yes | **absent** |

  Windows `BUILDINFO` records `usdt=n/a`, `toolchain=msvc-2022`. `CMakeLists.txt:33-35`
  hard-links two targets that do not exist on Windows.

- **Upstream already owns a tested USDT checker.** `executorch-runtime-dist`'s
  `scripts/check-usdt-notes.sh` asserts provider + probe names + per-probe argument arity,
  with a `USDT_READELF_TEXT` injection seam that `test/usdt_notes.test.sh` unit-tests
  against canned `readelf` output (including partial-loss cases).

### In scope

- Pin bump to v1.3.1-6.
- A Linux-only, self-arming USDT build guard + post-`auditwheel` check.
- Windows wheels with a declared per-platform kernel-lib contract.

### Out of scope

- macOS wheels (tracked separately; see the multiplatform roadmap).
- Windows `bare`/`devtools` variants — upstream ships **logging only** for Windows.
- Restoring optimized/quantized/LSTM on Windows. That is an upstream artifact change; this
  spec declares the gap rather than closing it.
- Extending `qa-gate.yml` to Windows. Those harnesses are ASan/TSan, Linux-only.

---

## 2. Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | Windows wheel ships **reduced and declared**: XNNPACK + portable kernels only | Shipping something real beats waiting on upstream parity; declaring it beats an opaque "operator not found" at model-load time |
| D2 | Three PRs: pin → USDT → Windows | Each PR has one reviewable concern; the risky one lands last against a known-good base |
| D3 | USDT guard **vendors** upstream's `check-usdt-notes.sh` | Gets provider + names + arity and a tested seam for free; upstream owns the probe contract, so drift is upstream's to manage |
| D4 | Guard arms itself from `BUILDINFO`, not a platform check | `usdt=on` / `usdt=n/a` is self-describing; Windows disarms for free with no `if(LINUX)` anywhere |
| D5 | No MSVC symbol guard; Windows substitutes a runtime backend assertion | Extracting symbols on Windows was a known problem for the runtime project too. Minimum effort: assert `XnnpackBackend` in `registered_backends()` |
| D6 | Kernel-lib set derives from a CMake compile define | Keeps the link line the single source of truth; Python never sniffs `sys.platform` |
| D7 | PR3 is gated by a `winbox` spike before any CI work | Prior art never linked `xnnpack_backend` under MSVC (see §5.1). If it fails, PR3 becomes an upstream request. **SATISFIED 2026-07-16 — spike PASSED; PR3 is unblocked** |

---

## 3. PR1 — Pin to v1.3.1-6

Pure supply-chain change; no behavior change.

**`cmake/RuntimePin.cmake`**
- `ETNP_RUNTIME_VERSION` → `1.3.1-6`.
- Update both Linux URL literals + SHA256 rows:
  - linux-x86_64 → `e1c29f4fe7d0e108bfc3a4dc6f0bfb98eb5af97a175b5bae95da61446d8542cd`
  - linux-aarch64 → `feea21ea4d18673601bc7ce231ede25e19a48a1a0ba67d0b02dd490f6ce11eb5`
- Keep URLs as **fully-resolved literals**, per the existing comment at lines 10-14: CI
  greps this file's raw text before invoking CMake, so `${VAR}` templating would break the
  scrape.

Rows are sourced from the release's generated **`EtRuntimePin.cmake` asset** rather than
hand-transcribed. Verification status of the hashes quoted here:

- **linux-x86_64** — recomputed from the downloaded tarball bytes, matches the published
  `.sha256` asset, and matches `djl-executorch-engine`'s landed row.
- **linux-aarch64** — matches the published `.sha256` asset and `djl-executorch-engine`'s
  landed row; **not** recomputed from tarball bytes.
- **windows-x86_64** (§5.2) — recomputed from the downloaded tarball bytes, matches the
  published `.sha256` asset, and matches `djl-executorch-engine`'s landed row.

These are a convenience cross-check, not the gate. The gate is CMake's `URL_HASH`
re-verifying on every fetch, plus CI's `gh attestation verify` for provenance.

**`CLAUDE.md`** — delete the libm path-rewrite bullet from "Build-time guards". The
workaround was removed in `cd4e3d3` when upstream fixed it in v1.3.1-3 (confirmed via
`git log -S"usr/lib64"`); the doc still describes it as live.

**Verification:** existing `qa-gate.yml` and `build-wheels.yml` stay green;
`gh attestation verify` passes on the new tarballs.

---

## 4. PR2 — USDT tracepoint guard

### 4.1 Vendored checker

Copy from `executorch-runtime-dist` into this repo:
- `scripts/check-usdt-notes.sh` — `--expect on|off <binary>`; asserts provider `etnp`,
  probes `lstm_xnn_cache__hit` (arity 4), `lstm_xnn_cache__miss` (arity 4),
  `lstm_xnn_cache__evict` (arity 7).
- Its unit test, driving the `USDT_READELF_TEXT` seam (no ELF needed).

The vendored copy records its upstream origin and the release it was taken from.

### 4.2 Build guard

`cmake/assert_usdt_probes.cmake`, invoked POST_BUILD on `_core`, shaped like the existing
`assert_kernels_registered.cmake` (same `-DSO=` invocation style, same fail-the-build-not-
runtime philosophy).

Arming: read `${ETNP_RUNTIME_PREFIX}/BUILDINFO`; run the checker only when it records
`usdt=on`. This is the load-bearing choice — Windows (`usdt=n/a`) disarms itself, so the
guard needs no platform conditional. bash is available wherever the guard arms, since
`usdt=on` implies Linux.

### 4.3 Post-repair check

A second checker invocation in `build-wheels.yml` **after `auditwheel repair`**, against
the `.so` unzipped from the repaired wheel. The build-time and shipped `.so` differ, and
`patchelf` — which auditwheel runs, and which has historically shuffled note sections — is
the specific risk this catches.

### 4.4 Documentation

`docs/usdt-preparation-checklist.md` → `docs/usdt-tracepoints.md`, corrected to reality:
- Probes are **shipped**, not upcoming.
- They come from `libetnp_ops_lstm.a` via `etnp_extras_whole_archive`, **not** the
  ExecuTorch core.
- Provider `etnp`; the three probe names and arities.
- Surviving rules from the checklist: never add `--strip` to a `CIBW_REPAIR_WHEEL_COMMAND`
  override, never set `CMAKE_INSTALL_DO_STRIP`, never pass `-s` to the linker. If stripping
  ever becomes necessary, pass `--keep-section=.note.stapsdt`.
- A `bpftrace` one-liner, so the probes are actually usable.

---

## 5. PR3 — Windows wheels

### 5.1 Spike — **DONE 2026-07-16: PASSED, PR3 is unblocked**

Full record: `docs/superpowers/notes/2026-07-16-windows-spike-findings.md`. Results, with
the parts that change PR3's design:

| Question | Verdict |
|---|---|
| Q1 — does `xnnpack_backend` link into a SHARED target under MSVC? | **PASS** — `xnn_probe.dll` + `probe_main.exe` link clean |
| Q2 — does `XnnpackBackend` self-register? | **PASS** — verified in an EXE *and* in a `LoadLibrary`'d DLL (the shape `_core` actually is). `/OPT:REF` did not drop the static-init TU |
| Q3 — is `Launch-VsDevShell` activation needed? | **No** — CMake selects the `Visual Studio 18 2026` generator and finds `cl.exe` via installer/registry; plain vs. activated shell identical |

Q2 was the gate: this project deliberately ships **no** symbol guard on Windows (D5), so a
runtime registration check is the only stand-in for the Linux `nm` guard. It holds.

**Three findings that bind §5.3 and §5.6 — do not lose them:**

1. **`/WHOLEARCHIVE:` is required but free — conditionally.** Upstream's
   `ExecuTorchTargets.cmake:160` already carries it as `INTERFACE_LINK_OPTIONS`, confirmed
   on the real link line, so PR3 must **not** hand-wrap `xnnpack_backend`. **But this holds
   only while linking the `xnnpack_backend` CMake target.** Linking a raw `.lib` path
   instead would silently break registration — and with no `nm` guard on Windows, nothing
   would catch it until a model failed to load.
2. **Q3's answer is generator-dependent.** Forcing `-G Ninja` reinstates the activation
   requirement (Ninja needs `cl.exe` on `PATH`). The "no activation needed" answer is
   contingent on letting CMake pick the Visual Studio generator.
3. **The VS parity gap runs the opposite way from what this spec assumed.** winbox is
   **VS 18.8.0 "Community 2026" / MSVC 19.51** — a major version *ahead* of CI's VS 2022/17
   Enterprise, not behind it. So the spike proves a **newer** toolchain works, which does
   not establish that the older CI toolchain does. CI remains the acceptance gate, and for
   a sharper reason than "winbox is a dev box".

**Known gap:** cibuildwheel was **not** exercised end-to-end on winbox — its `nuget`-based
CPython provisioning fails there (for 3.13 and for the real 3.12 alike), never reaching
CMake. Q3 was answered via `python -m build`, which drives the identical
scikit-build-core → CMake path that cibuildwheel wraps. The claim "cibuildwheel works on
Windows" is therefore **untested**; CI is the first place it will be proven.

### 5.2 Pin

Add the `windows-x86_64` row:
`d2bc1859429fe33940adfd110f75d81bb5bedf3163c919d93d8c63531a967a2e` (verified against the
published `.sha256` and against `djl-executorch-engine`'s landed row).

Platform detection at `RuntimePin.cmake:23-34` currently maps processor → `linux-*`
unconditionally; make it OS-aware. `CMAKE_SYSTEM_PROCESSOR` is `AMD64` on Windows.

Note for the row comment: Windows ships **logging only** — there is no bare/devtools
Windows build upstream.

### 5.3 Capability contract

- `CMakeLists.txt:33-35`: make `optimized_native_cpu_ops_lib` and `quantized_ops_lib`
  `if(TARGET …)`-conditional.
- Accumulate the surviving set into an `ETNP_KERNEL_LIBS` compile define, exactly as
  `ETNP_ET_VERSION` already works at line 45.
- Expose it through `_core`.
- `executorch_numpy_runtime/info.py:23` currently hardcodes
  `["portable", "optimized", "quantized"]` — false on Windows. Read the define instead.

The link line stays the single source of truth (D6).

Expect a flood of `<X> library is not found` lines from `find_package` on Windows
(`optimized_kernels`, `quantized_kernels`, every non-XNNPACK backend, …). Per the handoff
doc this is **normal for a core-only build** — configure still completes. Do not treat it
as an error.

### 5.4 Tests

Capability-driven skips, **not** platform-driven: a marker resolved by a `conftest.py` hook
against `runtime_info()["kernel_libs"]`. `tests/models/quantized.pte` and the LSTM tests
skip where their kernels aren't linked.

Capability-driven is the point: if upstream later ships Windows parity, the tests start
running with no edit. (A skip hook is a legitimate `conftest.py` resident — this is a
pytest hook, not helper functions parked in conftest.)

### 5.5 Kernel-registration guard

`assert_kernels_registered.cmake` shells to `nm` and greps `_GLOBAL__sub_I_*`; MSVC has
neither. Keep the symbol guard **Linux-only**. Windows substitutes a runtime assertion —
`XnnpackBackend` present in `registered_backends()`, as a test. Document as a deliberate
asymmetry, with the reason: symbol extraction on Windows was a known problem for the
runtime project too, and this is the minimum-effort check that still catches the failure
that matters.

### 5.6 CI

`build-wheels.yml` matrix gains **`windows-2022`** — pinned, matching the recipe, not
`windows-latest`.

Carry over two hard-won details from the recipe's `build-windows` job:
- Discover VS **edition-agnostically** via `vswhere` (`-latest -products *`), so a runner
  image edition change doesn't silently break the build.
- If Git-Bash is invoked, invoke it **by explicit path**
  (`${env:ProgramFiles}\Git\bin\bash.exe`) — bare `bash` on a Windows runner can resolve to
  WSL's `System32\bash.exe` and run with no MSVC toolchain.

The recipe's **MAX_PATH/`core.longpaths`** and **`core.symlinks`** blind spots applied to
*building* ExecuTorch from a recursive checkout. This project consumes a prebuilt tarball
and does no recursive ET checkout, so those two are not expected to apply; don't port that
ceremony without evidence it's needed.

**The multi-Python lesson does apply — see §5.8.**

`qa-gate.yml` stays Linux-only.

### 5.7 Documentation

README gains a platform support table stating the Windows kernel gap plainly.

### 5.8 Pin the Python interpreter on Windows

**Carried directly from the runtime project's release experience:** the recipe passed a
path to 3.12, but a later **bare `python` invocation** resolved to the newest interpreter
on the runner (3.14.x), producing `ModuleNotFoundError: yaml` deep into the build. A
Windows runner ships several Pythons; a warm dev host typically has one, so this failure
mode is invisible until CI runs it.

**Why this repo is exposed.** `CMakeLists.txt:16` is:

```cmake
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
```

`3.12` here is a **floor, not a pin**. FindPython's default `Python_FIND_STRATEGY` is
`VERSION` — *newest wins*. With 3.12 and 3.14 both present, this resolves 3.14. Line 17
then runs:

```cmake
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir ...)
```

nanobind is installed in the **3.12** build env, not 3.14 — so this fails with the same
class of `ModuleNotFoundError` the recipe hit.

**Why this is latent on Linux.** *Not* because manylinux has one Python — it ships many
(`/opt/python/cp3*-cp3*/bin`). Two independent mechanisms happen to cover the two
different paths, and neither is a general guarantee:

1. **`find_package(Python)` never runs outside scikit-build-core.** `native_tests/
   CMakeLists.txt` calls only `find_package(Threads)` and `find_package(ExecuTorch)`, so
   the qa-gate CMake builds resolve no interpreter at all. The only thing that reaches
   `CMakeLists.txt:16` is the skbuild-driven build (wheel + editable install), and
   scikit-build-core pins `Python_EXECUTABLE` via `python_hints`, default `True`
   (`scikit_build_core/settings/skbuild_model.py:261`).
2. **Bare `python` calls are covered by an explicit PATH export.**
   `qa-gate.yml:55` prepends `/opt/python/cp312-cp312/bin` to `PATH` inside the *Generate
   add.pte* step, which is what makes its `python -m pip install` and
   `python tools/export_fixtures.py` resolve 3.12. This is the same lesson the runtime
   project learned, already applied here.

So the protection is circumstantial, not structural — it holds because of where the calls
happen to live today.

**Where it bites on Windows** (refined by the §5.1 spike — the hazard is **narrower** than
first written here, but real):

- **cibuildwheel / wheel build — protected, more strongly than mechanism (1) suggested.**
  The spike found scikit-build-core **force-pins** `Python_EXECUTABLE` *and* sets
  `Python_FIND_REGISTRY=NEVER`, so FindPython cannot wander off to a registry-installed
  interpreter. **Do not set `python_hints = false`**, and don't assume this extends beyond
  the skbuild-driven path.
- **Raw `cmake` — exposed.** This is where the floor-vs-pin hazard actually lives: anything
  not driven by scikit-build-core, including `native_tests/` and any ad-hoc spike build.
  Pass `-DPython_EXECUTABLE=` explicitly and never rely on bare `python`.
- **Any added Windows CI step invoking `python` bare — exposed**, and mechanism (2) does
  not port: there is no `/opt/python/cp312-cp312/bin` equivalent to prepend. Use an
  explicit interpreter path rather than a PATH prepend.

**A cautionary note from the spike, worth more than the rule it illustrates.** The spike
agent enumerated winbox's interpreters, concluded 3.12 was absent, and was about to have it
installed — on a host where `python3.12` (3.12.10, 64-bit) was present the whole time.
`py -0p` does not list Microsoft Store installs. Two lessons:

- If a careful agent with shell access cannot reliably enumerate the interpreters on a
  Windows host, **`find_package(Python 3.12)` cannot either** — which is the entire case
  for pinning `Python_EXECUTABLE` rather than letting CMake choose.
- On Windows, `…\WindowsApps\python3.12.exe` is a Store **alias shim**, not the real
  interpreter (`sys.executable` resolves to a
  `PythonSoftwareFoundation.Python.3.12_*\python.exe` path). Pass the **resolved**
  `sys.executable`, not the shim, to `-DPython_EXECUTABLE`.

**Action:** treat this as an acceptance criterion for PR3, not a footnote. During the
spike, confirm which interpreter CMake actually resolved (echo `${Python_EXECUTABLE}`
at configure time) rather than assuming. If the spike shows the floor-vs-pin behavior
biting outside the skbuild path, consider tightening line 16 to a version range
(`3.12...<3.13`) so the failure is a clear configure-time error instead of a
`ModuleNotFoundError` later in the build.

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| ~~`xnnpack_backend` doesn't link/register under MSVC~~ | **RETIRED 2026-07-16** — spike proved it links and self-registers, including in a `LoadLibrary`'d DLL (§5.1) |
| PR3 links `xnnpack_backend` by raw `.lib` path instead of by CMake target, losing upstream's `/WHOLEARCHIVE:` and silently breaking registration | Link the **target**, never a path (§5.1 finding 1). No `nm` guard exists on Windows to catch it; the runtime `XnnpackBackend` assertion (D5) is the only net |
| Spike passed on VS 18.8.0/MSVC 19.51 — a major version **ahead** of CI's VS 2022/17 Enterprise, so it proves the newer toolchain, not the older | CI is the acceptance gate. If CI's MSVC fails where winbox passed, suspect the version skew first |
| cibuildwheel on Windows is unproven — its nuget CPython provisioning failed on winbox, so it was never exercised end-to-end | Q3 was answered via `python -m build` (same skbuild→CMake path). Expect first-run friction in CI and budget for it |
| PR3 does five things at once | If review feels heavy, split §5.3 (capability contract) ahead of §5.6 (CI matrix) — it's independently useful and Linux-testable |
| Vendored checker drifts from upstream | Vendored copy records its origin + release. Consider proposing upstream publish it as a release asset alongside `EtRuntimePin.cmake` |
| A Windows user hits a model needing quantized/optimized ops | Declared in `runtime_info()["kernel_libs"]` and the README table (D1) |
| CMake resolves the wrong Python on a multi-Python Windows runner (`find_package(Python 3.12)` is a floor; newest wins) — cost the runtime project a build failure at ~92% | §5.8: pin `-DPython_EXECUTABLE` in the spike, never bare `python`, keep `python_hints` on. Linux's current immunity is circumstantial, not structural |

---

## 7. References

- `~/workspace/windows-jni-handoff.md` — Windows artifact scope + winbox access/toolchain
- `executorch-runtime-dist`: `.github/workflows/release.yml` (`build-windows` job),
  `scripts/check-usdt-notes.sh`, `test/usdt_notes.test.sh`,
  `test/relocatability-windows.sh`, `test/consumer/`
- `djl-executorch-engine`: `native/cmake/EtRuntimePin.cmake` — landed windows-x86_64 pin row
- Release: https://github.com/measly-java-learning/executorch-runtime-dist/releases/tag/v1.3.1-6
