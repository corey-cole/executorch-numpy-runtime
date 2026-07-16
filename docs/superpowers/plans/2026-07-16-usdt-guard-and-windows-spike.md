# USDT Guard + Windows Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bump the runtime pin to v1.3.1-6, guard the ExecuTorch USDT tracepoints so a stripped or GC'd probe table fails the build, and run the Windows spike that gates the Windows-wheels work.

**Architecture:** Three shippable PRs. PR1 is a pure pin bump. PR2 vendors upstream's tested USDT checker and drives it from a self-arming `cmake -P` guard (POST_BUILD) plus a post-`auditwheel` CI check; the guard arms itself off the runtime tarball's `BUILDINFO` (`usdt=on`) rather than a platform conditional, so Windows disarms for free. PR3-spike is an investigation on a remote MSVC host whose findings become a separate plan.

**Tech Stack:** CMake ≥3.24 (`cmake -P` guards), scikit-build-core + nanobind, pytest, bash, GitHub Actions + cibuildwheel, ExecuTorch 1.3.1 prebuilt static libs.

**Spec:** `docs/superpowers/specs/2026-07-16-usdt-and-windows-wheels-design.md`

## Global Constraints

- ExecuTorch version is an **exact** compatibility contract: `1.3.1`. Never change it.
- Runtime pin target for this plan: **`v1.3.1-6`** of `measly-java-learning/executorch-runtime-dist`.
- `cmake/RuntimePin.cmake` URL rows MUST stay **fully-resolved literals**, never `${VAR}` templates: CI greps the file's raw text before invoking CMake (see that file's comment at lines 10-14).
- Verified SHA256s (do not re-derive):
  - linux-x86_64: `e1c29f4fe7d0e108bfc3a4dc6f0bfb98eb5af97a175b5bae95da61446d8542cd`
  - linux-aarch64: `feea21ea4d18673601bc7ce231ede25e19a48a1a0ba67d0b02dd490f6ce11eb5`
- USDT probe contract (owned upstream): provider `etnp`; probes `lstm_xnn_cache__hit` (arity 4), `lstm_xnn_cache__miss` (arity 4), `lstm_xnn_cache__evict` (arity 7).
- Guards fail the **build**, not runtime. Follow the house style of `cmake/assert_kernels_registered.cmake`.
- **Never** strip the wheel: no `--strip` in a `CIBW_REPAIR_WHEEL_COMMAND` override, no `CMAKE_INSTALL_DO_STRIP`, no linker `-s`. `nanobind_add_module(... NOSTRIP)` must stay.
- Editable installs do **not** auto-recompile. After any C++/CMake edit: `rm -rf build && uv pip install -e . --no-build-isolation --reinstall`.
- Lint with `ruff check .` before every commit that touches Python.
- Do not set scikit-build-core's `python_hints = false` — it is what pins `Python_EXECUTABLE`.

---

## Scope note (read before starting)

This plan covers **PR1 (pin)**, **PR2 (USDT guard)**, and **the PR3 spike only**.

The Windows-wheels implementation is deliberately **not** planned here. Its tasks depend on facts only the spike can establish (does `xnnpack_backend` link and self-register under MSVC; does cibuildwheel need explicit VS activation). Planning them now would require placeholders, which is a plan failure. Task 6 produces a findings document; the Windows plan gets written from it.

**Already verified during planning — do not re-litigate:**
- All three probes survive a `--whole-archive` link into a `.so` **even with `--gc-sections`**, and upstream's checker passes against that real linked `.so` with correct arity. PR2 is a guard against regression, not a fix for a live break.
- Releases before v1.3.1-5 carry **no `usdt=` key at all** in BUILDINFO (verified on the cached v1.3.1-3 prefix), so arming on `usdt=on` is backward-safe.

---

## Task 1: Bump the runtime pin to v1.3.1-6

**Files:**
- Modify: `cmake/RuntimePin.cmake:16` (version), `:36-42` (URL + SHA256 rows)
- Modify: `CLAUDE.md:68` (delete the stale libm bullet)

**Interfaces:**
- Consumes: nothing.
- Produces: an `ETNP_RUNTIME_PREFIX` whose `BUILDINFO` records `usdt=on` and which ships `libetnp_ops_lstm.a` with probes. Tasks 3 and 4 depend on this.

- [ ] **Step 1: Update the version variable**

In `cmake/RuntimePin.cmake`, change line 16:

```cmake
set(ETNP_RUNTIME_VERSION "1.3.1-6" CACHE STRING "Pinned executorch-runtime-dist package revision")
```

- [ ] **Step 2: Update both Linux pin rows**

Replace lines 36-42 of `cmake/RuntimePin.cmake` with:

```cmake
set(ETNP_RUNTIME_URL_logging_linux-x86_64
  "https://github.com/measly-java-learning/executorch-runtime-dist/releases/download/v1.3.1-6/executorch-runtime-1.3.1-logging-linux-x86_64.tar.gz")
set(ETNP_RUNTIME_SHA256_logging_linux-x86_64 "e1c29f4fe7d0e108bfc3a4dc6f0bfb98eb5af97a175b5bae95da61446d8542cd")

set(ETNP_RUNTIME_URL_logging_linux-aarch64
  "https://github.com/measly-java-learning/executorch-runtime-dist/releases/download/v1.3.1-6/executorch-runtime-1.3.1-logging-linux-aarch64.tar.gz")
set(ETNP_RUNTIME_SHA256_logging_linux-aarch64 "feea21ea4d18673601bc7ce231ede25e19a48a1a0ba67d0b02dd490f6ce11eb5")
```

- [ ] **Step 3: Delete the stale libm bullet from CLAUDE.md**

Delete this line (`CLAUDE.md:68`) entirely:

```markdown
- **`libm` path rewrite** in top-level and `native_tests/CMakeLists.txt`: rewrites the tarball's baked-in absolute `/usr/lib64/libm.so` to portable `-lm`. Harmless where the path exists (manylinux/RHEL); unblocks Debian/Ubuntu. Remove once a corrected upstream tarball ships.
```

The workaround it describes was removed in `cd4e3d3` when upstream fixed it in v1.3.1-3. The bullet above it (`assert_kernels_registered.cmake`) stays. Because that leaves one bullet under "Build-time guards", change the section's plural heading if it reads awkwardly — keep it as `## Build-time guards (these fail the *build*, not runtime)`; a single-item list is fine.

- [ ] **Step 4: Rebuild against the new pin**

Run:
```bash
rm -rf build && uv pip install -e . --no-build-isolation --reinstall
```
Expected: build succeeds. The existing `assert_kernels_registered` POST_BUILD guard prints its STATUS line.

- [ ] **Step 5: Verify the fetched runtime is v1.3.1-6 and records usdt=on**

Run:
```bash
grep -E 'usdt|package_tag|platform' build/*/_deps/etnp_runtime-src/BUILDINFO
```
Expected output includes:
```
platform=linux-x86_64
usdt=on
package_tag=v1.3.1-6
```
If `usdt=on` is absent, STOP — the wrong tarball was fetched and Tasks 3-4 cannot work.

- [ ] **Step 6: Run the test suite**

Run: `python -m pytest tests/`
Expected: PASS, with the two by-design skips (`test_parity` needs torch; the non-CPU-backend test needs a CoreML fixture).

- [ ] **Step 7: Commit**

```bash
git add cmake/RuntimePin.cmake CLAUDE.md
git commit -m "chore: Bump ExecuTorch runtime pin to v1.3.1-6

Unlocks the USDT probes (usdt=on) and the windows-x86_64 tarball. Hashes
match the published .sha256 assets and djl-executorch-engine's landed rows.

Also drops the CLAUDE.md libm bullet: that workaround was removed in cd4e3d3
when upstream fixed the abspath leak in v1.3.1-3."
```

---

## Task 2: Vendor the USDT checker and unit-test it

**Files:**
- Create: `scripts/check-usdt-notes.sh`
- Create: `tests/test_usdt_notes.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `scripts/check-usdt-notes.sh`, invoked as
  `bash scripts/check-usdt-notes.sh --expect <on|off> <binary>`; exit 0 = contract holds, 1 = violated, 2 = usage error. Honors `USDT_READELF_TEXT` to bypass `readelf`. Tasks 3 and 4 both shell out to it.

**Why `scripts/` and not `tools/`:** `tools/` holds this repo's Python dev tools. This file is a verbatim vendored copy, and mirroring upstream's `scripts/` path keeps `diff` against upstream trivial.

- [ ] **Step 1: Write the failing test**

Create `tests/test_usdt_notes.py`:

```python
"""Unit tests for the vendored USDT probe checker (scripts/check-usdt-notes.sh).

Drives the checker's USDT_READELF_TEXT seam, so these need no ELF binary, no
readelf, and no USDT-enabled runtime. Ported from executorch-runtime-dist's
test/usdt_notes.test.sh so the vendored copy keeps its upstream coverage.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CHECKER = Path(__file__).parent.parent / "scripts" / "check-usdt-notes.sh"

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or shutil.which("bash") is None,
    reason="USDT is Linux-only and the checker is a bash script",
)

# Canned `readelf --notes` output: all three probes at correct arity (4/4/7).
GOOD_NOTES = """\
Displaying notes found in: .note.stapsdt
  Owner                Data size \tDescription
  stapsdt              0x0000004c\tNT_STAPSDT (SystemTap probe descriptors)
    Provider: etnp
    Name: lstm_xnn_cache__hit
    Location: 0x1234, Base: 0x2000, Semaphore: 0x0
    Arguments: -4@%edi -4@%esi -4@%edx -4@%ecx
  stapsdt              0x0000004e\tNT_STAPSDT (SystemTap probe descriptors)
    Provider: etnp
    Name: lstm_xnn_cache__miss
    Location: 0x1240, Base: 0x2000, Semaphore: 0x0
    Arguments: -4@%edi -4@%esi -4@%edx -4@%ecx
  stapsdt              0x0000005a\tNT_STAPSDT (SystemTap probe descriptors)
    Provider: etnp
    Name: lstm_xnn_cache__evict
    Location: 0x1250, Base: 0x2000, Semaphore: 0x0
    Arguments: -4@%edi -4@%esi -4@%edx -4@%ecx -4@%r8d -4@%r9d -8@%rax
"""


def run_checker(notes: str, expect: str) -> int:
    """Run the checker against canned readelf text; return its exit code."""
    return subprocess.run(
        ["bash", str(CHECKER), "--expect", expect, "/nonexistent"],
        env={**os.environ, "USDT_READELF_TEXT": notes},
        capture_output=True,
        text=True,
    ).returncode


def test_all_probes_present_passes():
    assert run_checker(GOOD_NOTES, "on") == 0


def test_missing_probe_fails():
    # Total loss of one probe is the regression the guard exists to catch.
    notes = "\n".join(
        line for line in GOOD_NOTES.splitlines() if "lstm_xnn_cache__evict" not in line
    )
    assert run_checker(notes, "on") == 1


def test_wrong_arity_fails():
    # Drop one argument from __hit (4 -> 3). The trailing stapsdt line anchors
    # the replacement to __hit's Arguments rather than __miss's identical one.
    notes = GOOD_NOTES.replace(
        "    Arguments: -4@%edi -4@%esi -4@%edx -4@%ecx\n"
        "  stapsdt              0x0000004e",
        "    Arguments: -4@%edi -4@%esi -4@%edx\n"
        "  stapsdt              0x0000004e",
    )
    assert run_checker(notes, "on") == 1


def test_expect_off_with_no_probes_passes():
    assert run_checker("no notes here", "off") == 0


def test_expect_off_but_probes_present_fails():
    assert run_checker(GOOD_NOTES, "off") == 1


def test_expect_on_but_no_provider_fails():
    assert run_checker("no notes here", "on") == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_usdt_notes.py -v`
Expected: all 6 FAIL — the checker does not exist yet, so `bash` exits 127.

- [ ] **Step 3: Vendor the checker**

Create `scripts/check-usdt-notes.sh` with **exactly** this content (verbatim from `executorch-runtime-dist` @ v1.3.1-6, plus a two-line provenance header — do not "improve" the body, or `diff` against upstream stops being meaningful):

```bash
#!/usr/bin/env bash
# VENDORED VERBATIM from measly-java-learning/executorch-runtime-dist @ v1.3.1-6
# (scripts/check-usdt-notes.sh). Upstream owns this probe contract; re-vendor on pin bump.
# Assert the committed etnp USDT probe contract in a linked binary (provider +
# probe names + per-probe arity), or assert probes are ABSENT when disabled.
#   check-usdt-notes.sh --expect <on|off> <binary>
# Test hook: if USDT_READELF_TEXT is set, it is used instead of running readelf.
# Arity is the number of '@' in a probe's "Arguments:" line (each arg is size@loc).
set -euo pipefail
EXPECT=""; BIN=""
while [ $# -gt 0 ]; do
  case "$1" in
    --expect) EXPECT="${2:-}"; shift 2 ;;
    -*) echo "unknown arg: $1" >&2; exit 2 ;;
    *) BIN="$1"; shift ;;
  esac
done
case "$EXPECT" in on|off) ;; *) echo "usage: check-usdt-notes.sh --expect <on|off> <binary>" >&2; exit 2 ;; esac

if [ -n "${USDT_READELF_TEXT+x}" ]; then
  notes="$USDT_READELF_TEXT"
else
  [ -n "$BIN" ] && [ -f "$BIN" ] || { echo "check-usdt-notes: binary not found: '$BIN'" >&2; exit 2; }
  notes="$(readelf --notes "$BIN")"
fi

has_stapsdt=0
if printf '%s\n' "$notes" | grep -q 'NT_STAPSDT'; then has_stapsdt=1; fi

if [ "$EXPECT" = "off" ]; then
  if [ "$has_stapsdt" -eq 1 ]; then
    echo "FAIL: --expect off but NT_STAPSDT notes are present" >&2; exit 1
  fi
  echo "ok: no stapsdt notes (USDT disabled)"; exit 0
fi

# --expect on
if ! printf '%s\n' "$notes" | grep -q 'Provider: etnp'; then
  echo "FAIL: --expect on but provider 'etnp' absent" >&2; exit 1
fi

fails=0
check_probe() { # <name> <expected-argc>
  local name="$1" want="$2" args got
  # The Arguments line that follows the matching "Name:" line for this probe.
  args="$(printf '%s\n' "$notes" | awk -v n="$name" '
    /^[[:space:]]*Name:[[:space:]]/     { cur=$2 }
    /^[[:space:]]*Arguments:/ { if (cur==n) { sub(/^[[:space:]]*Arguments:[[:space:]]*/,""); print; exit } }')"
  if [ -z "$args" ]; then
    echo "FAIL: probe '$name' not found (or no Arguments line)" >&2; fails=$((fails+1)); return
  fi
  got="$(printf '%s' "$args" | tr -cd '@' | wc -c | tr -d ' ')"
  if [ "$got" -ne "$want" ]; then
    echo "FAIL: probe '$name' arity $got != expected $want (args: $args)" >&2; fails=$((fails+1)); return
  fi
  echo "ok: probe '$name' present with arity $want"
}
check_probe lstm_xnn_cache__hit   4
check_probe lstm_xnn_cache__miss  4
check_probe lstm_xnn_cache__evict 7
[ "$fails" -eq 0 ] || { echo "$fails USDT probe check(s) failed" >&2; exit 1; }
echo "USDT probe contract OK"
```

Make it executable: `chmod +x scripts/check-usdt-notes.sh`

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_usdt_notes.py -v`
Expected: 6 passed.

- [ ] **Step 5: Verify against a real linked .so (not just canned text)**

Run:
```bash
bash scripts/check-usdt-notes.sh --expect on executorch_numpy_runtime/_core*.so
```
Expected:
```
ok: probe 'lstm_xnn_cache__hit' present with arity 4
ok: probe 'lstm_xnn_cache__miss' present with arity 4
ok: probe 'lstm_xnn_cache__evict' present with arity 7
USDT probe contract OK
```
This requires Task 1 to be done and the tree rebuilt against v1.3.1-6. If it fails with "provider 'etnp' absent", the `.so` is stale — re-run Task 1 Step 4.

- [ ] **Step 6: Lint and commit**

```bash
ruff check .
git add scripts/check-usdt-notes.sh tests/test_usdt_notes.py
git commit -m "test: Vendor upstream's USDT probe checker with unit tests

Verbatim copy of executorch-runtime-dist's scripts/check-usdt-notes.sh:
asserts provider etnp plus per-probe names and argument arity. Its unit test
is ported to pytest (this repo's only test harness) and drives the
USDT_READELF_TEXT seam, so it needs no ELF, no readelf, and no USDT runtime."
```

---

## Task 3: Self-arming POST_BUILD guard

**Files:**
- Create: `cmake/assert_usdt_probes.cmake`
- Modify: `CMakeLists.txt` (append a second POST_BUILD custom command after the existing one at lines 49-53)

**Interfaces:**
- Consumes: `scripts/check-usdt-notes.sh` from Task 2; `ETNP_RUNTIME_PREFIX` from `cmake/RuntimePin.cmake`.
- Produces: a build that fails when the probe contract breaks. Invoked as
  `cmake -DSO=<lib> -DPREFIX=<runtime-prefix> -DCHECKER=<script> -P cmake/assert_usdt_probes.cmake`.

- [ ] **Step 1: Write the guard**

Create `cmake/assert_usdt_probes.cmake`:

```cmake
# Post-link guard: prove the etnp USDT probe contract survived the final .so link.
# Sibling of assert_kernels_registered.cmake -- same philosophy: fail the BUILD, not runtime.
# Probes live in libetnp_ops_lstm.a (an ETNPExtras archive), NOT in ExecuTorch's own libs, and
# reach _core only because etnp_extras_whole_archive() whole-archives it. A stripped wheel or a
# regressed whole-archive would otherwise produce a silently un-traceable .so.
#
# SELF-ARMING: reads ${PREFIX}/BUILDINFO and enforces only when the runtime tarball records
# `usdt=on`. windows-x86_64 records `usdt=n/a`, and releases before v1.3.1-5 carry no usdt key
# at all -- both disarm with no platform conditional here.
#
# Invoked: cmake -DSO=<lib> -DPREFIX=<runtime-prefix> -DCHECKER=<script> -P assert_usdt_probes.cmake
if(NOT SO OR NOT EXISTS "${SO}")
  message(FATAL_ERROR "assert_usdt_probes: SO not found: '${SO}'")
endif()
if(NOT CHECKER OR NOT EXISTS "${CHECKER}")
  message(FATAL_ERROR "assert_usdt_probes: CHECKER not found: '${CHECKER}'")
endif()

set(_buildinfo "${PREFIX}/BUILDINFO")
if(NOT EXISTS "${_buildinfo}")
  message(STATUS "assert_usdt_probes: no BUILDINFO at '${_buildinfo}' -- USDT guard disarmed")
  return()
endif()

file(READ "${_buildinfo}" _bi)
if(NOT _bi MATCHES "(^|\n)usdt=on([\r\n]|$)")
  message(STATUS "assert_usdt_probes: runtime does not record usdt=on -- USDT guard disarmed")
  return()
endif()

find_program(_bash bash REQUIRED)
execute_process(COMMAND "${_bash}" "${CHECKER}" --expect on "${SO}"
                RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR
    "USDT probe contract broken in ${SO} (rc=${_rc}).\n${_out}${_err}\n"
    "The pinned runtime records usdt=on, so probes MUST reach the linked .so. Likely causes: "
    "the link or wheel was stripped (NOSTRIP removed, CMAKE_INSTALL_DO_STRIP set, linker -s), "
    "or etnp_ops_lstm stopped being whole-archived.")
endif()
message(STATUS "assert_usdt_probes: etnp USDT probe contract present in ${SO}")
```

- [ ] **Step 2: Wire it into the build**

In `CMakeLists.txt`, immediately after the existing `add_custom_command(TARGET _core POST_BUILD ...)` block (lines 49-53) and before the `install(TARGETS ...)` line, add:

```cmake
# Post-link USDT guard (fail the BUILD, not runtime). Self-arms off the runtime's BUILDINFO,
# so this is a no-op on runtimes built without USDT (e.g. windows-x86_64: usdt=n/a).
add_custom_command(TARGET _core POST_BUILD
  COMMAND ${CMAKE_COMMAND} -DSO=$<TARGET_FILE:_core>
          -DPREFIX=${ETNP_RUNTIME_PREFIX}
          -DCHECKER=${CMAKE_SOURCE_DIR}/scripts/check-usdt-notes.sh
          -P ${CMAKE_SOURCE_DIR}/cmake/assert_usdt_probes.cmake
  VERBATIM)
```

- [ ] **Step 3: Rebuild and verify the guard arms and passes**

Run:
```bash
rm -rf build && uv pip install -e . --no-build-isolation --reinstall 2>&1 | grep -i usdt
```
Expected: `assert_usdt_probes: etnp USDT probe contract present in ...`

If it instead prints `USDT guard disarmed`, Task 1 did not land — the pin is not v1.3.1-6.

- [ ] **Step 4: Prove the guard actually fails when the contract breaks**

A guard never seen failing is not a guard. Strip a copy and confirm it is caught:

```bash
cp executorch_numpy_runtime/_core*.so /tmp/probe-test.so
strip --remove-section=.note.stapsdt /tmp/probe-test.so
cmake -DSO=/tmp/probe-test.so \
      -DPREFIX="$(dirname build/*/_deps/etnp_runtime-src/BUILDINFO)" \
      -DCHECKER="$PWD/scripts/check-usdt-notes.sh" \
      -P cmake/assert_usdt_probes.cmake
echo "rc=$?"
```
Expected: `FATAL_ERROR` naming the broken contract, `rc=1`.

- [ ] **Step 5: Prove the disarm path works**

```bash
mkdir -p /tmp/fake-prefix && printf 'platform=windows-x86_64\nusdt=n/a\n' > /tmp/fake-prefix/BUILDINFO
# Reuse the stripped .so from Step 4 on purpose: it has NO probes, so if the disarm
# path is broken this fails loudly instead of passing for the wrong reason.
cmake -DSO=/tmp/probe-test.so \
      -DPREFIX=/tmp/fake-prefix \
      -DCHECKER="$PWD/scripts/check-usdt-notes.sh" \
      -P cmake/assert_usdt_probes.cmake
echo "rc=$?"
```
Expected: `assert_usdt_probes: runtime does not record usdt=on -- USDT guard disarmed`, `rc=0`. This is the Windows path; it must not fail.

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest tests/`
Expected: PASS (with the two by-design skips).

- [ ] **Step 7: Commit**

```bash
git add cmake/assert_usdt_probes.cmake CMakeLists.txt
git commit -m "feat: Fail the build if USDT probes don't survive the link

POST_BUILD guard in the style of assert_kernels_registered.cmake. Arms itself
off the runtime tarball's BUILDINFO (usdt=on) rather than a platform check, so
windows-x86_64 (usdt=n/a) and pre-1.3.1-5 runtimes disarm with no conditional.

Probes reach _core only via etnp_extras_whole_archive of libetnp_ops_lstm.a;
a stripped wheel or regressed whole-archive would otherwise ship a silently
un-traceable .so."
```

---

## Task 4: Assert probes survive `auditwheel repair`

**Files:**
- Modify: `.github/workflows/build-wheels.yml` (insert a step in `build_wheels`, after the cibuildwheel step at lines 28-29, before the upload at lines 31-34)

**Interfaces:**
- Consumes: `scripts/check-usdt-notes.sh` from Task 2.
- Produces: CI failure if patchelf/auditwheel drops the notes.

**Why this exists separately from Task 3:** the build-time `.so` and the *shipped* `.so` are different bytes. `auditwheel repair` runs `patchelf`, which rewrites soname/rpath/dynamic sections and has historically shuffled or duplicated note sections. Task 3 cannot see that.

- [ ] **Step 1: Add the post-repair check**

In `.github/workflows/build-wheels.yml`, between the `Build wheels` step and the `upload-artifact` step, insert:

```yaml
      - name: Assert USDT probes survived auditwheel repair
        # The build-time .so and the repaired .so are different bytes: auditwheel runs
        # patchelf, which rewrites soname/rpath/dynamic sections and has historically
        # shuffled note sections. cmake/assert_usdt_probes.cmake only sees the former.
        # Linux-only: USDT does not exist on other platforms (BUILDINFO usdt=n/a).
        if: runner.os == 'Linux'
        shell: bash
        run: |
          set -euo pipefail
          shopt -s nullglob
          wheels=(wheelhouse/*.whl)
          test "${#wheels[@]}" -gt 0 || { echo "no wheels in wheelhouse/"; exit 1; }
          for whl in "${wheels[@]}"; do
            tmp="$(mktemp -d)"
            unzip -q "$whl" -d "$tmp"
            so="$(find "$tmp" -name '_core*.so' -print -quit)"
            test -n "$so" || { echo "no _core*.so inside $whl"; exit 1; }
            echo "== $whl"
            bash scripts/check-usdt-notes.sh --expect on "$so"
          done
```

The `if: runner.os == 'Linux'` is redundant today (both matrix entries are Linux) but is required the moment the Windows matrix entry lands, and costs nothing now.

- [ ] **Step 2: Verify the check logic locally against a real repaired wheel**

Ensure `build/` is EMPTY first — the CMake cache is path-sensitive:

```bash
rm -rf build
uvx cibuildwheel --platform linux
```
Then run the same assertion the CI step runs:
```bash
for whl in wheelhouse/*.whl; do
  tmp="$(mktemp -d)"; unzip -q "$whl" -d "$tmp"
  so="$(find "$tmp" -name '_core*.so' -print -quit)"
  echo "== $whl"; bash scripts/check-usdt-notes.sh --expect on "$so"
done
```
Expected: `USDT probe contract OK` for each wheel.

This is the step that proves auditwheel/patchelf does not eat the notes. If it fails, STOP and report — that is a real finding, not a plan defect, and it means the shipped wheel is untraceable today.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/build-wheels.yml
git commit -m "ci: Assert USDT probes survive auditwheel repair

The POST_BUILD guard only sees the build-time .so. auditwheel runs patchelf,
which rewrites note-adjacent sections, so the shipped wheel needs its own
check against the .so unzipped from the repaired artifact."
```

---

## Task 5: Replace the preparation checklist with real documentation

**Files:**
- Create: `docs/usdt-tracepoints.md`
- Delete: `docs/usdt-preparation-checklist.md`

**Interfaces:**
- Consumes: the behavior built in Tasks 2-4.
- Produces: nothing code depends on.

**Note:** `docs/usdt-preparation-checklist.md` is **untracked** — it was never committed. Use `rm`, not `git rm`. It will not appear in the commit diff; that is expected.

- [ ] **Step 1: Write the replacement doc**

Create `docs/usdt-tracepoints.md`:

```markdown
# USDT tracepoints

The pinned ExecuTorch runtime ships **systemtap USDT tracepoints** (`.note.stapsdt`
probes) on Linux. This documents what ships, how it is guarded, and how to trace it.

## What ships

Provider **`etnp`**, three probes, all from the LSTM XNNPACK weight cache:

| Probe | Arity |
|---|---|
| `lstm_xnn_cache__hit` | 4 |
| `lstm_xnn_cache__miss` | 4 |
| `lstm_xnn_cache__evict` | 7 |

The probes live in **`libetnp_ops_lstm.a`** — an ETNPExtras archive — **not** in
ExecuTorch's own libraries. They reach `_core` only because `etnp_extras_whole_archive()`
whole-archives that library. Whole-archiving is what carries a static library's probe
notes into the final `.so`.

Linux only. The runtime tarball's `BUILDINFO` records `usdt=on`; the windows-x86_64
tarball records `usdt=n/a` and has no probes (and no LSTM op).

## How it is guarded

- **`cmake/assert_usdt_probes.cmake`** — POST_BUILD on `_core`. Self-arms by reading
  `usdt=on` from the runtime prefix's `BUILDINFO`, so it is a no-op on runtimes built
  without USDT rather than needing a platform conditional. Fails the *build*.
- **`build-wheels.yml`** — re-checks the `.so` unzipped from the wheel *after*
  `auditwheel repair`, because patchelf rewrites note-adjacent sections and the shipped
  bytes differ from the build-time bytes.
- Both shell out to **`scripts/check-usdt-notes.sh`**, vendored verbatim from
  `executorch-runtime-dist`. It asserts provider + probe names + per-probe arity.
  Re-vendor it when bumping the runtime pin.

## Do not strip the wheel

This is the real risk. Probes are notes; stripping is what removes them.

- `nanobind_add_module(... NOSTRIP)` keeps the *link* from stripping. Keep it.
- `auditwheel` does **not** strip by default (`--strip` is opt-in). Keep it that way.
- **Never**: add `--strip` to a `CIBW_REPAIR_WHEEL_COMMAND` override, set
  `CMAKE_INSTALL_DO_STRIP`, or pass `-s` to the linker.
- Plain `strip` preserves `.note.stapsdt`, but if stripping ever becomes necessary, pass
  `--keep-section=.note.stapsdt`.

`--gc-sections` is **not** a risk: a probe note references its site via a relocation, and
the probed code is reachable. This has been verified directly — all three probes survive a
`--whole-archive` link with `--gc-sections` active.

## Tracing

Requires `bpftrace` and root:

```sh
# Count cache hits/misses/evictions by probe name
sudo bpftrace -e 'usdt:/path/to/_core.abi3.so:etnp:* { @[probe] = count(); }'

# Watch the cache decisions live
sudo bpftrace -e 'usdt:/path/to/_core.abi3.so:etnp:lstm_xnn_cache__miss { printf("miss\n"); }'
```

Find the `.so` with:

```sh
python -c "import executorch_numpy_runtime._core as c; print(c.__file__)"
```

List the probes an installed wheel actually carries:

```sh
readelf -n "$(python -c 'import executorch_numpy_runtime._core as c; print(c.__file__)')" | grep -A3 stapsdt
```
```

- [ ] **Step 2: Remove the superseded checklist**

```bash
rm docs/usdt-preparation-checklist.md
```

- [ ] **Step 3: Verify the doc's claims against reality**

The doc tells readers to run two commands. Run both yourself; a doc that ships a broken command is a defect.

```bash
python -c "import executorch_numpy_runtime._core as c; print(c.__file__)"
readelf -n "$(python -c 'import executorch_numpy_runtime._core as c; print(c.__file__)')" | grep -A3 stapsdt
```
Expected: the second prints all three `etnp` probes.

- [ ] **Step 4: Run the docs test**

Run: `python -m pytest tests/test_docs.py -v`
Expected: PASS. (`test_docs.py` asserts on `README.md`, not this file, so it should be unaffected — confirm rather than assume.)

- [ ] **Step 5: Commit**

```bash
git add docs/usdt-tracepoints.md
git commit -m "docs: Document the shipped USDT tracepoints

Replaces the preparation checklist, which predated the probes and assumed they
would live in ExecuTorch's core libs. They actually ship in libetnp_ops_lstm.a
(provider etnp, three cache probes) and arrive via etnp_extras_whole_archive.

Records how the two guards work, the never-strip rules, and bpftrace usage."
```

---

## Task 6: Windows spike (gates the Windows-wheels plan)

**Files:**
- Create: `docs/superpowers/notes/2026-07-16-windows-spike-findings.md`

**Interfaces:**
- Consumes: the `windows-x86_64` tarball from v1.3.1-6, SHA256 `d2bc1859429fe33940adfd110f75d81bb5bedf3163c919d93d8c63531a967a2e`.
- Produces: the findings that the Windows-wheels plan will be written from. **No production code.**

**This is an investigation, not a feature.** It has no TDD cycle. Its deliverable is answers.

**Read first:** `~/workspace/windows-jni-handoff.md` — it documents `winbox` access, the VS-activation → Git-Bash handoff, and the file-transfer gotchas. Do not rediscover them.

**Parity caveat:** winbox is **VS 18 Community**; the CI runner is **VS 2022/17 Enterprise**. A green spike is necessary, not sufficient — CI is the acceptance gate.

**Critical, per spec §5.8:** never invoke bare `python` on winbox, and pass `-DPython_EXECUTABLE=` explicitly to any raw `cmake`. `find_package(Python 3.12)` is a **floor, not a pin** — FindPython's default `VERSION` strategy takes the *newest* interpreter, which is exactly how the runtime project lost a build to 3.14.x.

- [ ] **Step 1: Answer Q1 — does `xnnpack_backend` link into a SHARED target under MSVC?**

This is the question the whole Windows effort rests on, and **no prior art covers it**: the handoff doc scopes the artifact as "core-only" and says *"Only rely on `executorch`"*, and `executorch-runtime-dist`'s `test/consumer/CMakeLists.txt` links exactly that one target. Our `_core` needs more.

Get the tarball onto winbox (per the handoff doc: `gh release download` on Linux, then `scp` to an **absolute forward-slash** path, with the destination dir pre-created via PowerShell `New-Item`).

Build a probe modelled on `test/consumer/`, but linking what we actually need:

```cmake
cmake_minimum_required(VERSION 3.24)
project(xnn_probe LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(ExecuTorch CONFIG REQUIRED)
add_library(xnn_probe SHARED probe.cpp)
target_link_libraries(xnn_probe PRIVATE
  executorch xnnpack_backend extension_module_static extension_data_loader extension_tensor)
```

Expect a flood of `<X> library is not found` lines from `find_package` — per the handoff doc that is **normal for a core-only build**. Configure still completes. Do not treat it as failure.

Record: does it configure? Does it link? Which of those five targets exist?

- [ ] **Step 2: Answer Q2 — does XnnpackBackend actually self-register under MSVC?**

Linking is not registering. XNNPACK registers from a pure static-init TU, and MSVC's linker (`/OPT:REF`) can drop unreferenced objects much as `--gc-sections` does. This is precisely the failure `assert_kernels_registered.cmake` exists to catch on Linux — and per spec D5 we are **not** porting that symbol guard to MSVC, so this runtime check is the only thing standing in for it.

Use an executable rather than a `.so` + Python, so the spike stays native. Extend the
Step 1 `CMakeLists.txt` with an executable target that links the same libraries:

```cmake
add_executable(probe_main probe.cpp)
target_link_libraries(probe_main PRIVATE
  executorch xnnpack_backend extension_module_static extension_data_loader extension_tensor)
```

`probe.cpp`:

```cpp
#include <cstdio>
#include <string>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/platform/runtime.h>

int main() {
  executorch::runtime::runtime_init();
  size_t n = executorch::runtime::get_num_registered_backends();
  std::printf("registered backends: %zu\n", n);
  bool found_xnn = false;
  for (size_t i = 0; i < n; ++i) {
    auto r = executorch::runtime::get_backend_name(i);
    if (r.ok()) {
      std::printf("  %s\n", *r);
      if (std::string(*r) == "XnnpackBackend") found_xnn = true;
    }
  }
  std::printf("%s\n", found_xnn ? "SPIKE PASS: XnnpackBackend registered"
                                : "SPIKE FAIL: XnnpackBackend ABSENT");
  return found_xnn ? 0 : 1;
}
```

`get_num_registered_backends()` / `get_backend_name(size_t)` are the exact pair `src/et_core/et_core.cpp:232-241` uses, so a pass here means our real `registered_backends()` will see it too. The expected name string is `"XnnpackBackend"` (asserted in `tests/test_meta_info.py:17`).

Build it as an executable (add `add_executable(probe_main probe.cpp)` linking the same libraries) and run it. If XnnpackBackend is absent, try wrapping `xnnpack_backend` in `/WHOLEARCHIVE:` and record whether that fixes it — that answer directly shapes the Windows link line.

- [ ] **Step 3: Answer Q3 — does cibuildwheel/scikit-build-core need explicit VS activation?**

Unknown, and not guessable: the recipe needed `Launch-VsDevShell` activation, but it drove Ninja directly rather than going through scikit-build-core, whose CMake invocation can discover MSVC on its own.

Attempt the real thing on winbox with `build/` empty:

```
python -m pip install cibuildwheel
python -m cibuildwheel --platform windows
```

Run it **both** ways: from a plain shell, and from an activated VS dev shell. Record which works. Use an explicit interpreter path for `python` here — see the note above.

Do not treat a failure caused by the missing `optimized_native_cpu_ops_lib` / `quantized_ops_lib` targets as a Q3 answer; that is expected (spec §5.3) and is what the Windows plan will fix. Q3 is only about whether the toolchain is found.

- [ ] **Step 4: Write the findings**

Create `docs/superpowers/notes/2026-07-16-windows-spike-findings.md` recording, for each of Q1/Q2/Q3: the exact commands run, the verbatim output (trimmed to what matters), and the answer. Include the VS version winbox actually reported, so the Enterprise-vs-Community gap stays visible to whoever reads this next.

State a clear verdict:
- **Q1+Q2 pass** → the Windows plan is unblocked; record whether `/WHOLEARCHIVE:` was needed for `xnnpack_backend`.
- **Q1 or Q2 fails** → **STOP.** Per spec D7/§5.1 the Windows work converts into an upstream parity request against `executorch-runtime-dist`. Write up what fails and why; do not attempt to work around it in this repo.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/notes/2026-07-16-windows-spike-findings.md
git commit -m "docs: Record Windows MSVC spike findings

Answers the three questions gating the Windows-wheels work: whether
xnnpack_backend links under MSVC, whether XnnpackBackend self-registers
(the substitute for the Linux-only nm guard), and whether cibuildwheel
needs explicit VS dev-shell activation."
```

- [ ] **Step 6: Hand off**

Report the verdict. If green, the next step is a **new plan** for the Windows wheels (spec §5.2-5.8), written against these findings. Do not begin Windows implementation from this plan.

---

## Verification checklist

Before considering PR1+PR2 done:

- [ ] `python -m pytest tests/` passes (two by-design skips).
- [ ] `ruff check .` clean.
- [ ] A fresh `rm -rf build && uv pip install -e . --no-build-isolation --reinstall` prints `assert_usdt_probes: etnp USDT probe contract present`.
- [ ] The guard has been **seen failing** on a stripped `.so` (Task 3 Step 4).
- [ ] The guard has been **seen disarming** on a `usdt=n/a` BUILDINFO (Task 3 Step 5).
- [ ] A locally built, auditwheel-repaired wheel passes the probe check (Task 4 Step 2).
- [ ] `readelf -n` on the installed `.so` shows all three `etnp` probes.
