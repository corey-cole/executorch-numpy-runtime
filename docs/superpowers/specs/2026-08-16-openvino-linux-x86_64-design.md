# OpenVINO delegate support (linux-x86_64)

**Date:** 2026-08-16
**Status:** approved design
**Implements:** step C of `2026-08-16-runtime-1.3.1-10-adoption-design.md`

Makes the OpenVINO delegate — already compiled into the `linux-x86_64` runtime tarball as
`lib/libopenvino_backend.a` — a supported feature of this package: an installable extra, automatic
`OPENVINO_LIB_PATH` resolution, a committed fixture, and CI that actually executes a delegated
model rather than merely proving the archive linked.

Nothing here imports torch. Export — the partitioner and the quantizer — stays out of scope
permanently.

## The constraint everything else is shaped around

The delegate resolves the OpenVINO C API with `dlopen` at first use, reading `OPENVINO_LIB_PATH`.
That load happens **once**, under `std::call_once`, **with no retry**. If the first attempt fails,
the process stays broken until it restarts.

So the design's organising principle is: **every failure we can detect must be raised before
ExecuTorch is entered.** A misconfiguration that reaches `call_once` costs the user their
interpreter; the same misconfiguration caught one frame earlier costs them a stack trace they can
act on. This is why detection is lazy-on-load rather than at import, and why the C++ guard exists
at all.

## Architecture

Three touch points, no new components.

```
Program.__init__  (Python)   ── uses_backend? ──▶ resolve/validate OPENVINO_LIB_PATH
      │
      ▼
Runtime::load_method (C++)   ── guard ──▶ EtException if backend unregistered
      │                                   or OPENVINO_LIB_PATH unusable
      ▼
ExecuTorch OpenvinoBackend::init ── call_once ──▶ dlopen   ← never reached misconfigured
```

**Detection uses metadata, not a byte scan.** `MethodMeta::uses_backend("OpenvinoBackend")`
(`runtime/executor/method_meta.h:257`) answers exactly the right question. Measured on this repo:
`Module::method_meta` costs ~8 µs and leaves the XNNPACK workspace at 0, while the first
`run_method` costs 537 µs against 19 µs for the second — so metadata is available well before
delegate init. A new `_core.method_uses_backend(method, backend) -> bool` wraps it; both the Python
resolver and the C++ guard read through it.

**Python side (`_api.py`).** `Program.__init__` asks each method whether it uses `OpenvinoBackend`.
If any does:

- `OPENVINO_LIB_PATH` set and `isfile()` → honor it untouched (operator override wins).
- Set but not a file → raise, naming the value. Upstream's troubleshooting table lists "set to a
  directory instead of a file" as a common mistake, and the error message it otherwise produces
  mentions `LD_LIBRARY_PATH`, which reads like it wants a directory. It does not.
- Unset → resolve `libopenvino_c.so*` from inside the installed `openvino` wheel and set it. If
  `openvino` is not installed, raise with the install hint.

The guard is an explicit `if "OPENVINO_LIB_PATH" not in os.environ`, never `os.environ.setdefault`,
whose eager default would resolve — and raise when the wheel is absent — even when the variable is
already correct.

**C++ side (`et_core`).** A guard on the method-load path. If the method uses `OpenvinoBackend`
and:

- the backend is not registered → `ErrorKind::BackendMissing` naming the platform; or
- `OPENVINO_LIB_PATH` is unset or not a file → `ErrorKind::BackendMissing` with the
  set-it-first-and-it-is-a-file text.

This is what makes `_core` safe to use directly — our own tests do — and demotes the Python layer
from load-bearing to convenient. No new `ErrorKind`: `BackendMissing` already maps to
`BackendNotAvailable` at `src/binding/module.cpp:124`.

### Rejected alternatives

**Import-time resolution.** Cannot let `import executorch_numpy_runtime` fail because an optional
dependency is missing, so it must swallow the not-installed case — which silently leaves the
variable unset and hands the first OpenVINO load to `call_once` unconfigured. It also imports
`openvino` and mutates global process env for every user of the package, most of whom never touch
OpenVINO.

**Explicit `enable_openvino()`.** Forgetting it produces the same unrecoverable `call_once`
failure, reported from deep inside ExecuTorch rather than from the call that was skipped.

**Byte-scanning the `.pte` for `b"OpenvinoBackend"`.** Works, and upstream's own emitter does it as
a post-export check — but it is a heuristic over a flatbuffer, and it forces reading an entire model
file that ExecuTorch is about to read again. `uses_backend` is exact and free.

**All-C++.** Resolving a path inside a pip wheel means `importlib`. Reimplementing that against
site-packages layout in C++ is worse in every dimension.

## Off-platform behaviour

Loading an OpenVINO `.pte` where the delegate is not linked raises `BackendNotAvailable`:

```
This .pte uses the OpenvinoBackend delegate, which this build does not provide.
The OpenVINO delegate ships on linux-x86_64 only (this build: linux-aarch64).
Re-export without the OpenVINO partitioner to run here.
```

Without this guard the case falls through `classify_load_failure` and surfaces as
`ProgramLoadError` — "corrupt, or exported by an ExecuTorch version other than 1.3.1" — which
actively misdirects. The `.pte` is fine; the platform cannot run it.

## Packaging

```toml
[project.optional-dependencies]
openvino = ["openvino==2025.4.1; sys_platform=='linux' and platform_machine=='x86_64'"]
```

The environment marker means the extra exists everywhere but installs nothing where the delegate is
absent, so `pip install executorch-numpy-runtime[openvino]` on aarch64 succeeds as a no-op rather
than failing resolution. The user then meets the typed platform error at load time — the message
that actually explains their situation.

**Exact pin, not a range.** `2025.4.1` matches `ET_RUNTIME_OPENVINO_VERSION` in the generated pin
and the fixture's coupled version (`etnp-openvino-fixtures-1.3.1-2025.4.1`). A `.pte` embeds a
precompiled OpenVINO blob, so a floating range lets a rebuild import a blob-incompatible runtime,
surfacing at model load rather than at install.

**The version now lives in three places** — `pyproject.toml`, `cmake/EtRuntimePin.cmake`, and the
fixture manifest. `scripts/check-pin-rows.sh` is extended to assert all three agree. That script
exists for exactly this class of bug, so this is a few lines rather than a new mechanism.

**Linking is capability-driven, not platform-driven:**

```cmake
if(TARGET openvino_backend)
  target_link_libraries(_core PRIVATE openvino_backend)
  list(APPEND _etnp_expect_tus "_GLOBAL__sub_I_OpenvinoBackend.cpp")
endif()
```

Keyed on the target existing rather than on `_ETNP_PLATFORM`, matching the philosophy `conftest.py`
already documents for `requires_kernel_lib`: if a platform later gains the delegate, it works with
no edit. The nm-guard row is conditional for the same reason; upstream verified that symbol
survives into a consumer `.so`.

The wheel does **not** vendor OpenVINO. `libopenvino_backend.a` is static and `dlopen`s at runtime,
so there is no `DT_NEEDED` for auditwheel to chase.

### `registered_backends()` becomes platform-dependent

This is the first time the reported backend set differs across the wheels we ship, and two existing
assertions have to absorb that:

- `tests/test_meta_info.py` asserts `"XnnpackBackend" in info["backends"]` — a membership test, so
  it keeps passing unchanged. Nothing there enumerates the full set, and nothing new should: an
  exact-set assertion would have to be platform-conditional and would break on the next backend.
- `CMakeLists.txt:35`'s required-target loop (`executorch xnnpack_backend`) must **not** gain
  `openvino_backend`. That loop exists to catch `find_package(ExecuTorch)` early-returning and
  silently producing an empty link line; adding a target that legitimately does not exist on two of
  three platforms would turn a real guard into a platform conditional.

The `if(TARGET openvino_backend)` link block is therefore the only place platform variance is
expressed, and it expresses it as capability rather than as a platform name.

## Fixture

Vendored from the upstream release asset
`etnp-openvino-fixtures-1.3.1-2025.4.1.tar.gz`: `openvino_tiny.pte`, `in.bin`, `out.bin`, `shape`.
The model is `Linear(8,8) + relu`, fully delegated, `CompileSpec("device", b"CPU")`.

The four members are unpacked into `tests/models/openvino/` and **committed**, matching this repo's
existing fixture convention and keeping tests offline.

Beside them sits `tests/models/openvino/MANIFEST`, a flat `key=value` file:

```
tarball_url=https://github.com/measly-java-learning/executorch-runtime-dist/releases/download/v1.3.1-10/etnp-openvino-fixtures-1.3.1-2025.4.1.tar.gz
tarball_sha256=<sha of the asset>
openvino_version=2025.4.1
executorch_version=1.3.1
```

Flat `key=value` rather than TOML or JSON deliberately: `scripts/check-pin-rows.sh` is bash and
must read `openvino_version` to compare it against `pyproject.toml` and
`ET_RUNTIME_OPENVINO_VERSION`. A format that needs a parser on the bash side would push that check
into Python and split the pin guards across two languages for no gain.

**`out.bin` is the eager golden, not the delegate's output**, so parity is a tolerance check.

## Testing

**A new capability marker.** `conftest.py` gains `requires_backend(name)` beside the existing
`requires_kernel_lib`, driven by `runtime_info()["backends"]`. Every OpenVINO test carries it, so
aarch64 and Windows skip automatically and a platform that later gains the delegate starts running
these tests with no edit.

| Test | Asserts |
|---|---|
| fixture parity | outputs match `out.bin` at `atol=1e-2`; the chosen precision is printed |
| off-platform error | `BackendNotAvailable` naming the platform — runs only where the backend is *absent* |
| `OPENVINO_LIB_PATH` validation | pointing it at a directory raises our error, in a subprocess |
| resolution from the wheel | with the var unset, `Program` sets it to an existing file in the wheel |
| `method_meta` timing guard | `xnnpack_workspace_size_bytes() == 0` after `method_meta` on `conv.pte` |

### On the tolerance

**`atol=1e-2`, and it must not be tightened.** OpenVINO selects inference precision from the CPU it
lands on, at import time rather than at blob-compile time. On a bf16-capable runner
(avx512_bf16/AMX) the delegate computes in bf16 and lands ~2.5e-3 from the f32 eager golden;
elsewhere it lands at ~6e-8. Both are correct OpenVINO results. A tolerance drawn between them
asserts which runner CI happened to allocate — a property this project does not own — and fails at
random. Upstream hit exactly this (`ea393da` in executorch-runtime-dist) after a green run and a red
run on identical artifacts.

`np.testing.assert_allclose`'s default `rtol=1e-7` would reproduce that flake precisely. The
tolerance carries a comment citing this reason, or a future reader will "fix" it.

The loose bound still catches everything it exists to catch: a delegate returning zeros, garbage,
or the wrong model is orders of magnitude out.

### On the `method_meta` guard

The entire detection strategy rests on `Module::method_meta` not loading the method. That is an
implementation detail of the `Module` extension, measured rather than contractual — the header says
only "Loads the program if needed." If a runtime bump makes it eager, detection silently becomes too
late and OpenVINO fails at `dlopen` with no obvious connection to the change. The workspace metric
from step B detects it directly, which is a stronger guard than a comment.

## CI

**`qa-gate.yml`** gains one stage on `matrix.platform == 'linux-x86_64'`: install
`openvino==2025.4.1`, run the OpenVINO-marked tests, and print `INFERENCE_PRECISION_HINT` from
`openvino.Core().get_property("CPU", ...)`.

Reporting the precision is what keeps the loose tolerance honest: `1e-2` alone cannot distinguish
"correct in bf16" from "quietly degraded", so the gate records which one it saw. Upstream
deliberately avoided the python `openvino` wheel for this probe because it is a *different*
OpenVINO than the bundle they gate; for us the pip wheel **is** the OpenVINO we run, so querying it
is the correct source rather than a shortcut.

The aarch64 leg runs the **inverse** test — that a `.pte` using an unlinked delegate produces our
platform error — so both matrix legs assert something real instead of one of them merely skipping.

**`build-wheels.yml`**: `test-extras = ["openvino"]` under `[tool.cibuildwheel]`. The environment
marker makes this correctly inert on aarch64 and Windows, so one setting covers every leg with no
conditionals. That is the payoff for putting the marker in the extra rather than branching in CI.

## Verify during implementation, do not assume

1. **`out.bin` dtype and shape.** Read from the emitter's contract, not from the unpacked tarball.
   Confirm it is float32 of the shape in the `shape` file before writing the comparison.
2. **An `openvino==2025.4.1` cp312 wheel exists for `manylinux_2_28` x86_64** — our glibc floor.
3. **Wheel-size delta** from the static archive, measured against the pre-change wheel.
4. **`Module::method_meta` does not load the method.** Measured here; the guard test makes it
   permanent.

## Out of scope, permanently

Everything AOT: export, `OpenvinoPartitioner`, `OpenVINOQuantizer` — all require torch, the one
dependency this package exists to avoid. Also out: non-CPU OpenVINO devices (the fixture is
`CompileSpec("device", b"CPU")` and CPU is the only plugin upstream ships), and vendoring the
OpenVINO runtime into the wheel.

## References

- `docs/openvino-python-consumer.md` (upstream) — dlopen contract, the unversioned-`.so` trap,
  version-compatibility matrix
- `ea393da` (upstream) — why the tolerance is 1e-2 and why the precision is reported
- `scripts/emit-openvino-fixtures.py` (upstream) — what the fixture contains and how it was made
- `runtime/executor/method_meta.h:252-273` — `uses_backend` / `num_backends` / `get_backend_name`
- `src/binding/module.cpp:123-127` — the `ErrorKind` → Python exception translation this reuses
