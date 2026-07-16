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
| D7 | PR3 is gated by a `winbox` spike before any CI work | Prior art never linked `xnnpack_backend` under MSVC (see §5.1). If it fails, PR3 becomes an upstream request |

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

### 5.1 Spike first (gating)

**Before any CI work**, on `winbox` (`ssh winbox`; VS activation → Git-Bash `bash -c`
non-login handoff, per `~/workspace/windows-jni-handoff.md`):

1. Link a `_core`-shaped **SHARED** target against `xnnpack_backend` under MSVC.
2. Assert `XnnpackBackend` appears in `registered_backends()` at runtime.
3. Determine whether cibuildwheel/scikit-build-core needs explicit `Launch-VsDevShell`
   activation, or whether CMake's own MSVC discovery suffices.

**Why this gates the PR:** no prior art covers it. The handoff doc scopes the Windows
artifact as "core-only" and says *"Only rely on `executorch`"*;
`executorch-runtime-dist`'s `test/consumer/CMakeLists.txt` links exactly that one target.
The `.lib` and its `/WHOLEARCHIVE:` link options are present in the tarball, so this is
expected to work — but it is unproven. **If the spike fails, PR3 stops and becomes an
upstream parity request.**

**Parity caveat:** winbox is **VS 18 Community**; the GitHub runner is **VS 2022/17
Enterprise**. Per the handoff doc, winbox is for iterating and CI is the acceptance gate —
a green spike is necessary, not sufficient.

Item 3 is genuinely open: the recipe needed explicit activation, but it drove Ninja
directly rather than going through scikit-build-core. Resolve by spiking, not by guessing.

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

The recipe's warm-host blind spots (MAX_PATH/`core.longpaths`, `core.symlinks`,
multi-Python codegen mismatch) applied to *building* ExecuTorch from a recursive checkout.
This project **consumes a prebuilt tarball** and does no ET codegen, so they are not
expected to apply. Do not port that ceremony without evidence it's needed.

`qa-gate.yml` stays Linux-only.

### 5.7 Documentation

README gains a platform support table stating the Windows kernel gap plainly.

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| `xnnpack_backend` doesn't link/register under MSVC | §5.1 spike gates the PR; failure converts PR3 into an upstream request |
| winbox (VS 18 Community) diverges from the runner (VS 2022/17 Enterprise) | Spike proves feasibility; the runner is the acceptance gate |
| PR3 does five things at once | If review feels heavy, split §5.3 (capability contract) ahead of §5.6 (CI matrix) — it's independently useful and Linux-testable |
| Vendored checker drifts from upstream | Vendored copy records its origin + release. Consider proposing upstream publish it as a release asset alongside `EtRuntimePin.cmake` |
| A Windows user hits a model needing quantized/optimized ops | Declared in `runtime_info()["kernel_libs"]` and the README table (D1) |

---

## 7. References

- `~/workspace/windows-jni-handoff.md` — Windows artifact scope + winbox access/toolchain
- `executorch-runtime-dist`: `.github/workflows/release.yml` (`build-windows` job),
  `scripts/check-usdt-notes.sh`, `test/usdt_notes.test.sh`,
  `test/relocatability-windows.sh`, `test/consumer/`
- `djl-executorch-engine`: `native/cmake/EtRuntimePin.cmake` — landed windows-x86_64 pin row
- Release: https://github.com/measly-java-learning/executorch-runtime-dist/releases/tag/v1.3.1-6
