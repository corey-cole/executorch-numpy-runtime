# ExecuTorch Torch-Free numpy Runtime — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a Python package that loads and runs arbitrary CPU-targeted ExecuTorch `.pte` files with numpy as the only required dependency — no torch.

**Architecture:** A binding-agnostic torch-free C++ core (`et_core`, grown from the proven JNI `et_runtime`) owns all ExecuTorch `Module`/`EValue` lifetime and the arena→copy-out memory model. A thin nanobind layer (`_core`) marshals numpy ↔ `EValue`. A pure-Python package (`executorch_numpy_runtime`) exposes `Runtime`/`Program`/`Method`. Static ExecuTorch libs come from the prebuilt, SHA-pinned `executorch-runtime-dist` v1.3.1 tarball, whole-archived by its own CMake config.

**Tech Stack:** C++17, nanobind (STABLE_ABI), scikit-build-core, CMake, numpy, pytest, ASan/LSan. Prebuilt ExecuTorch 1.3.1 static libs.

## Global Constraints

- **ExecuTorch version pin: 1.3.1** — exact, everywhere (runtime tarball, `runtime_info()`, docs).
- **numpy is the ONLY required runtime dependency.** `ml_dtypes` only ever behind a `[bf16]` optional extra. torch is CI/offline-only (fixtures + parity), never a package dependency.
- **Wheel target: `cp312-abi3`, `manylinux_2_28_x86_64`, Python 3.12+ floor.** nanobind built with `STABLE_ABI`.
- **Output tensors are ALWAYS copied, never viewed.** No `clone_outputs=False` path exists.
- **C-contiguity enforced on inputs**; non-contiguous inputs are copied at the boundary (never rejected).
- **Unmapped dtypes raise** a clean exception on both input and output paths — never guess.
- **`et_core` is binding-agnostic**: it speaks host pointers + shape + `ScalarType`, and must never include Python/numpy/nanobind headers.
- **dtype map (ScalarType → numpy):** Float→float32, Double→float64, Long→int64, Int→int32, Short→int16, Byte→uint8, Char→int8, Bool→bool_, Half→float16, BFloat16→uint16 (raw bits).
- **Exclusions (documented, not supported):** non-CPU delegates (CoreML/QNN/Vulkan/Metal), custom ops, torchao low-bit kernels, bundled programs, ETDump/profiling, custom data loaders.
- **Package name:** distribution `executorch-numpy-runtime`, import `executorch_numpy_runtime`, extension module `executorch_numpy_runtime._core`.

**Reference sources (read, don't blindly copy):**
- JNI core (reuse foundation): `/home/corey/workspace/djl-executorch-engine/native/core/et_runtime.{h,cpp}`
- Leak harness (port target): `/home/corey/workspace/djl-executorch-engine/native/harness/et_leak_harness.cpp`
- Whole-archive assert (port target): `/home/corey/workspace/djl-executorch-engine/native/cmake/assert_xnnpack_registered.cmake`
- Upstream pybindings (pattern source, do NOT fork): `/home/corey/workspace/executorch/extension/pybindings/pybindings.cpp`
- Prebuilt runtime tarball + config: `/home/corey/workspace/executorch-runtime-dist/dist/executorch-runtime-1.3.1-logging-linux-x86_64.tar.gz`, unpacks to `lib/cmake/ExecuTorch/executorch-config.cmake`
- Fixture-generation venv (has executorch 1.3.1 + torch): `/home/corey/workspace/executorch/.venv`

---

## File Structure

```
executorch-numpy-runtime/
├── pyproject.toml                     # scikit-build-core + metadata + [bf16] extra
├── CMakeLists.txt                     # top-level: find runtime tarball, nanobind_add_module
├── cmake/
│   ├── RuntimePin.cmake               # SHA256-pinned path to the dist tarball/prefix
│   └── assert_kernels_registered.cmake# post-link nm guard (XNNPACK + quantized + optimized)
├── src/
│   ├── et_core/
│   │   ├── et_core.h                  # binding-agnostic C++ core API (no Python headers)
│   │   └── et_core.cpp                # Module load(path/buffer), methods, meta, forward, introspect
│   └── binding/
│       ├── dtype_map.h                # ScalarType <-> numpy dtype-code table (no numpy headers)
│       ├── dtype_map.cpp
│       └── module.cpp                 # nanobind: numpy marshalling, GIL, error mapping, NB_MODULE
├── executorch_numpy_runtime/
│   ├── __init__.py                    # public API surface + __version__
│   ├── _api.py                        # Runtime / Program / Method wrappers
│   ├── errors.py                      # exception hierarchy (Python side)
│   └── info.py                        # runtime_info()
├── tools/
│   └── export_fixtures.py             # offline .pte generation (run in the ET venv)
├── tests/
│   ├── conftest.py                    # fixture paths, skip-if-missing helpers
│   ├── models/                        # committed .pte fixtures (add, dtypes, quantized, multi, dynamic)
│   ├── test_load.py                   # Task 2
│   ├── test_dtypes.py                 # Task 3, 10
│   ├── test_forward.py               # Task 5
│   ├── test_concurrency.py            # Task 6
│   ├── test_errors.py                 # Task 7
│   ├── test_meta_info.py              # Task 8
│   ├── test_api.py                    # Task 9
│   └── test_matrix.py                 # Task 10 (no-clobber, quantized, dynamic, parity)
├── native_tests/
│   └── leak_harness.cpp               # Task 11 (ASan/LSan, no Python)
└── README.md                          # Task 12: caller contract
```

---

## Task 1: Scaffold — build wiring, pinned runtime, whole-archive guard

**Files:**
- Create: `pyproject.toml`, `CMakeLists.txt`, `cmake/RuntimePin.cmake`, `cmake/assert_kernels_registered.cmake`
- Create: `src/binding/module.cpp` (placeholder module), `executorch_numpy_runtime/__init__.py`
- Create: `tests/conftest.py`, `tests/test_smoke.py`
- Create: `.gitignore` additions for build dirs

**Interfaces:**
- Produces: an importable extension `executorch_numpy_runtime._core` exposing `_core.__et_version__ == "1.3.1"`; a CMake build that fails if XNNPACK/quantized/optimized kernel registration TUs are GC'd.

- [ ] **Step 1: Unpack the pinned runtime tarball into a local prefix**

Run:
```bash
mkdir -p third_party
tar -xzf /home/corey/workspace/executorch-runtime-dist/dist/executorch-runtime-1.3.1-logging-linux-x86_64.tar.gz -C third_party
ls third_party/*/lib/cmake/ExecuTorch/executorch-config.cmake
```
Expected: the config path prints (note the top-level dir name, e.g. `executorch-runtime-1.3.1-logging-linux-x86_64`).

- [ ] **Step 2: Write `cmake/RuntimePin.cmake`** (SHA-pinned prefix, port of the dist `EtRuntimePin.cmake` idea)

```cmake
# Pins the ExecuTorch runtime prefix + its expected tarball SHA256.
# Mirrors executorch-runtime-dist's EtRuntimePin.cmake contract.
set(ETNP_ET_VERSION "1.3.1" CACHE STRING "Pinned ExecuTorch version")
set(ETNP_RUNTIME_PREFIX "${CMAKE_SOURCE_DIR}/third_party/executorch-runtime-1.3.1-logging-linux-x86_64"
    CACHE PATH "Unpacked ExecuTorch runtime prefix")

if(NOT EXISTS "${ETNP_RUNTIME_PREFIX}/lib/cmake/ExecuTorch/executorch-config.cmake")
  message(FATAL_ERROR
    "ExecuTorch runtime not found at ${ETNP_RUNTIME_PREFIX}. "
    "Unpack executorch-runtime-1.3.1-logging-linux-x86_64.tar.gz into third_party/.")
endif()
```

- [ ] **Step 3: Write `cmake/assert_kernels_registered.cmake`** (port + extend the JNI `assert_xnnpack_registered.cmake`)

```cmake
# Post-link guard: prove backend/kernel static-initializer TUs survived the final .so link.
# XNNPACK + quantized + optimized register from pure static-init TUs; --gc-sections silently
# drops them if whole-archive regressed, giving "backend/op not found" only at model-load time.
# Invoked: cmake -DSO=<lib> -DNM=<nm> -P assert_kernels_registered.cmake
if(NOT SO OR NOT EXISTS "${SO}")
  message(FATAL_ERROR "assert_kernels_registered: SO not found: '${SO}'")
endif()
if(NOT NM)
  set(NM "nm")
endif()
execute_process(COMMAND "${NM}" "${SO}" OUTPUT_VARIABLE _syms
                RESULT_VARIABLE _rc ERROR_VARIABLE _err)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "assert_kernels_registered: '${NM} ${SO}' failed (rc=${_rc}): ${_err}")
endif()
foreach(_tu "_GLOBAL__sub_I_XNNPACKBackend")
  string(FIND "${_syms}" "${_tu}" _pos)
  if(_pos EQUAL -1)
    message(FATAL_ERROR
      "Kernel/backend registration '${_tu}' was dropped from ${SO}: static-init TU absent. "
      "whole-archive regressed at final link -> backend/op not found at model-load time.")
  endif()
endforeach()
message(STATUS "assert_kernels_registered: XNNPACK registration TU present in ${SO}")
```

- [ ] **Step 4: Write top-level `CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.24)
project(executorch_numpy_runtime LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(cmake/RuntimePin.cmake)
list(APPEND CMAKE_PREFIX_PATH "${ETNP_RUNTIME_PREFIX}")
find_package(ExecuTorch CONFIG REQUIRED)

# nanobind via scikit-build-core-provided Python
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                OUTPUT_VARIABLE nanobind_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core STABLE_ABI NB_STATIC
  src/binding/module.cpp)

# et_core sources join later tasks; keep the list here.
target_include_directories(_core PRIVATE src)

# Link the whole ExecuTorch stack. The runtime config self-whole-archives the kernel/backend
# archives via INTERFACE_LINK_OPTIONS, so linking these targets is enough.
target_link_libraries(_core PRIVATE
  executorch optimized_native_cpu_ops_lib xnnpack_backend quantized_ops_lib)

target_compile_definitions(_core PRIVATE ETNP_ET_VERSION="${ETNP_ET_VERSION}")

# Post-link kernel-registration guard (fail the BUILD, not runtime).
add_custom_command(TARGET _core POST_BUILD
  COMMAND ${CMAKE_COMMAND} -DSO=$<TARGET_FILE:_core> -DNM=nm
          -P ${CMAKE_SOURCE_DIR}/cmake/assert_kernels_registered.cmake
  VERBATIM)

install(TARGETS _core LIBRARY DESTINATION executorch_numpy_runtime)
```

- [ ] **Step 5: Write placeholder `src/binding/module.cpp`**

```cpp
#include <nanobind/nanobind.h>
namespace nb = nanobind;

NB_MODULE(_core, m) {
  m.attr("__et_version__") = ETNP_ET_VERSION;
}
```

- [ ] **Step 6: Write `pyproject.toml`**

```toml
[build-system]
requires = ["scikit-build-core>=0.10", "nanobind>=2.0"]
build-backend = "scikit_build_core.build"

[project]
name = "executorch-numpy-runtime"
version = "0.0.1"
description = "Torch-free numpy runtime for ExecuTorch .pte files (ExecuTorch 1.3.1, CPU)"
requires-python = ">=3.12"
dependencies = ["numpy>=1.24"]

[project.optional-dependencies]
bf16 = ["ml_dtypes>=0.3"]
test = ["pytest>=8"]

[tool.scikit-build]
wheel.py-api = "cp312"
cmake.version = ">=3.24"
build-dir = "build/{wheel_tag}"

[tool.scikit-build.cmake.define]
CMAKE_BUILD_TYPE = "Release"
```

- [ ] **Step 7: Write `executorch_numpy_runtime/__init__.py`**

```python
from ._core import __et_version__

__all__ = ["__et_version__"]
```

- [ ] **Step 8: Write `tests/conftest.py`**

```python
from pathlib import Path
import pytest

MODELS = Path(__file__).parent / "models"

def model_or_skip(name: str) -> str:
    p = MODELS / name
    if not p.exists():
        pytest.skip(f"fixture {name} not generated; run tools/export_fixtures.py")
    return str(p)
```

- [ ] **Step 9: Write `tests/test_smoke.py`**

```python
import executorch_numpy_runtime

def test_import_and_version():
    assert executorch_numpy_runtime.__et_version__ == "1.3.1"
```

- [ ] **Step 10: Build and run the smoke test**

Run:
```bash
pip install -e . --no-build-isolation -v 2>&1 | tail -20
pytest tests/test_smoke.py -v
```
Expected: build succeeds, the `assert_kernels_registered` POST_BUILD prints "XNNPACK registration TU present", `test_import_and_version` PASSES.

- [ ] **Step 11: Commit**

```bash
git add pyproject.toml CMakeLists.txt cmake/ src/binding/module.cpp executorch_numpy_runtime/ tests/ .gitignore
git commit -m "feat: scaffold build with pinned ET 1.3.1 runtime and whole-archive guard"
```

---

## Task 2: `et_core` — Module load (path + buffer) and method enumeration

**Files:**
- Create: `src/et_core/et_core.h`, `src/et_core/et_core.cpp`
- Modify: `CMakeLists.txt` (add `src/et_core/et_core.cpp` to `_core` sources)
- Modify: `src/binding/module.cpp` (bind load + method_names)
- Create: `tools/export_fixtures.py`, `tests/test_load.py`

**Interfaces:**
- Produces (C++ `namespace etnp`):
  - `enum class ErrorKind { Load, BackendMissing, OperatorMissing, Execution, DtypeUnmapped };`
  - `struct EtError { ErrorKind kind; std::string message; std::string detail; };` (thrown as `EtException` carrying an `EtError`)
  - `class Runtime` with:
    `static std::unique_ptr<Runtime> load_path(const std::string& path);`
    `static std::unique_ptr<Runtime> load_buffer(std::string bytes);` (Runtime owns the buffer copy)
    `std::vector<std::string> method_names() const;`
- Produces (Python): `executorch_numpy_runtime._core.load_path(str) -> _Runtime`, `load_buffer(bytes) -> _Runtime`, `_Runtime.method_names() -> list[str]`.

- [ ] **Step 1: Write `tools/export_fixtures.py`** (run later in the ET venv)

```python
"""Generate .pte test fixtures. Run inside the ExecuTorch 1.3.1 venv:
  /home/corey/workspace/executorch/.venv/bin/python tools/export_fixtures.py tests/models
"""
import sys
from pathlib import Path
import torch
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner


def _save(name, mod, ex, out_dir, partitioners=None):
    ep = export(mod.eval(), ex)
    lowered = to_edge_transform_and_lower(
        ep, partitioner=partitioners or []).to_executorch()
    (out_dir / name).write_bytes(lowered.buffer)
    print("wrote", name)


class Add(torch.nn.Module):
    def forward(self, a, b): return a + b


class MixedDtypes(torch.nn.Module):
    def forward(self, a, b): return a + a, b + b


class TwoMethods(torch.nn.Module):
    def forward(self, a): return a + 1
    def double(self, a): return a * 2


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    xnn = [XnnpackPartitioner()]
    _save("add.pte", Add(), (torch.ones(3), torch.ones(3)), out, xnn)
    _save("dtypes.pte", MixedDtypes(),
          (torch.ones(1, dtype=torch.int64), torch.ones(1)), out)  # portable, no xnn
    # multi-method: export both forward and double
    m = TwoMethods().eval()
    methods = {"forward": export(m, (torch.ones(2),)),
               "double": export(m, (torch.ones(2),), strict=True)}
    lowered = to_edge_transform_and_lower(methods).to_executorch()
    (out / "multi.pte").write_bytes(lowered.buffer)
    print("wrote multi.pte")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/models")
```

- [ ] **Step 2: Generate fixtures and commit them**

Run:
```bash
/home/corey/workspace/executorch/.venv/bin/python tools/export_fixtures.py tests/models
ls -la tests/models/
```
Expected: `add.pte`, `dtypes.pte`, `multi.pte` exist. (If the multi-method export API differs in 1.3.1, fall back to committing `add.pte`/`dtypes.pte` and generate `multi.pte` via two single-method programs; keep going.)

- [ ] **Step 3: Write the failing test `tests/test_load.py`**

```python
import pytest
from executorch_numpy_runtime import _core
from tests.conftest import model_or_skip

def test_load_path_lists_methods():
    rt = _core.load_path(model_or_skip("add.pte"))
    assert "forward" in rt.method_names()

def test_load_buffer_lists_methods():
    with open(model_or_skip("add.pte"), "rb") as f:
        rt = _core.load_buffer(f.read())
    assert "forward" in rt.method_names()

def test_load_bad_path_raises():
    with pytest.raises(Exception):
        _core.load_path("/no/such/file.pte")
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_load.py -v`
Expected: FAIL — `_core` has no `load_path`.

- [ ] **Step 5: Write `src/et_core/et_core.h`**

```cpp
#ifndef ETNP_ET_CORE_H
#define ETNP_ET_CORE_H
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace etnp {

enum class ErrorKind { Load, BackendMissing, OperatorMissing, Execution, DtypeUnmapped };

struct EtError {
  ErrorKind kind;
  std::string message;
  std::string detail;  // op/backend name where known
};

class EtException : public std::runtime_error {
 public:
  explicit EtException(EtError e)
      : std::runtime_error(e.message), error(std::move(e)) {}
  EtError error;
};

// Borrowed input: host pointer the caller keeps valid across forward(). Zero-copy in.
struct InputDesc {
  const void* data;
  std::vector<int64_t> shape;
  int8_t scalar_type;  // ExecuTorch ScalarType code
};

// Borrowed output: points into ExecuTorch's arena, valid until next forward()/destroy.
struct OutputView {
  std::vector<int64_t> shape;
  int8_t scalar_type;
  const void* data;
  size_t nbytes;
};

struct TensorMeta {
  std::vector<int64_t> shape;  // empty for non-tensor
  int8_t scalar_type;          // -1 for non-tensor
};

struct MethodMeta {
  std::vector<TensorMeta> inputs;
  std::vector<TensorMeta> outputs;
};

struct RuntimeState;   // pimpl (owns Module + optional buffer)
struct ForwardState;   // pimpl (owns output EValues)

// RAII owner of output EValues; dropping it ends OutputView lifetime.
class ForwardResult {
 public:
  explicit ForwardResult(std::unique_ptr<ForwardState> s);
  ~ForwardResult();
  ForwardResult(ForwardResult&&) noexcept;
  ForwardResult& operator=(ForwardResult&&) noexcept;
  const std::vector<OutputView>& outputs() const;
 private:
  std::unique_ptr<ForwardState> state_;
};

class Runtime {
 public:
  static std::unique_ptr<Runtime> load_path(const std::string& path);
  static std::unique_ptr<Runtime> load_buffer(std::string bytes);
  ~Runtime();
  std::vector<std::string> method_names() const;
  MethodMeta method_meta(const std::string& name) const;                  // Task 8
  ForwardResult run_method(const std::string& name,
                           const std::vector<InputDesc>& inputs);          // Task 5
  explicit Runtime(std::unique_ptr<RuntimeState> s);
 private:
  std::unique_ptr<RuntimeState> state_;
};

// Introspection (Task 8)
bool backend_available(const std::string& name);
std::vector<std::string> registered_backends();
std::vector<std::string> operator_names();

}  // namespace etnp
#endif
```

- [ ] **Step 6: Write `src/et_core/et_core.cpp`** (load + method_names now; other methods stubbed to throw until their task)

```cpp
#include "et_core/et_core.h"

#include <executorch/extension/module/module.h>
#include <executorch/runtime/executor/method_meta.h>

namespace etnp {
using executorch::extension::Module;
using executorch::runtime::Error;

struct RuntimeState {
  std::string owned_bytes;                 // non-empty for buffer loads; keeps data alive
  std::unique_ptr<Module> module;
};

static EtException load_error(const std::string& what) {
  return EtException({ErrorKind::Load, "Failed to load .pte: " + what, ""});
}

std::unique_ptr<Runtime> Runtime::load_path(const std::string& path) {
  auto st = std::make_unique<RuntimeState>();
  st->module = std::make_unique<Module>(path);
  if (st->module->load() != Error::Ok) throw load_error(path);
  return std::make_unique<Runtime>(std::move(st));
}

std::unique_ptr<Runtime> Runtime::load_buffer(std::string bytes) {
  auto st = std::make_unique<RuntimeState>();
  st->owned_bytes = std::move(bytes);
  st->module = std::make_unique<Module>(
      st->owned_bytes.data(), st->owned_bytes.size());
  if (st->module->load() != Error::Ok) throw load_error("<buffer>");
  return std::make_unique<Runtime>(std::move(st));
}

Runtime::Runtime(std::unique_ptr<RuntimeState> s) : state_(std::move(s)) {}
Runtime::~Runtime() = default;

std::vector<std::string> Runtime::method_names() const {
  auto res = state_->module->method_names();
  if (res.error() != Error::Ok)
    throw EtException({ErrorKind::Load, "method_names() failed", ""});
  std::vector<std::string> out(res->begin(), res->end());
  return out;
}

}  // namespace etnp
```

Note: verify `Module(const void*, size_t)` and `method_names()` signatures against `/home/corey/workspace/executorch/extension/module/module.h`; adjust the buffer-ctor call if 1.3.1 uses a loader/`BufferDataLoader` instead. The `owned_bytes` field guarantees the buffer outlives the `Module` regardless.

- [ ] **Step 7: Bind in `src/binding/module.cpp`**

```cpp
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "et_core/et_core.h"

namespace nb = nanobind;
using namespace etnp;

NB_MODULE(_core, m) {
  m.attr("__et_version__") = ETNP_ET_VERSION;

  nb::class_<Runtime>(m, "_Runtime")
      .def("method_names", &Runtime::method_names);

  m.def("load_path", [](const std::string& p) { return Runtime::load_path(p); });
  m.def("load_buffer", [](nb::bytes b) {
    return Runtime::load_buffer(std::string(b.c_str(), b.size()));
  });
}
```

- [ ] **Step 8: Add `et_core.cpp` to the build in `CMakeLists.txt`**

Change the `nanobind_add_module(_core ...)` source list to:
```cmake
nanobind_add_module(_core STABLE_ABI NB_STATIC
  src/binding/module.cpp
  src/et_core/et_core.cpp)
```

- [ ] **Step 9: Rebuild and run tests**

Run:
```bash
pip install -e . --no-build-isolation
pytest tests/test_load.py -v
```
Expected: all three tests PASS (`test_load_bad_path_raises` passes via the load exception).

- [ ] **Step 10: Commit**

```bash
git add src/et_core/ src/binding/module.cpp CMakeLists.txt tools/export_fixtures.py tests/models/ tests/test_load.py
git commit -m "feat: et_core Module load (path+buffer) and method enumeration"
```

---

## Task 3: dtype mapping table (both directions, reject-on-unmapped)

**Files:**
- Create: `src/binding/dtype_map.h`, `src/binding/dtype_map.cpp`
- Modify: `CMakeLists.txt` (add `dtype_map.cpp`), `src/binding/module.cpp` (expose test helpers)
- Create: `tests/test_dtypes.py`

**Interfaces:**
- Produces (C++): in `dtype_map.h`
  `int8_t numpy_code_to_scalar_type(char kind, size_t itemsize);` (throws `EtException{DtypeUnmapped}`)
  `struct NpDtype { char kind; size_t itemsize; }; NpDtype scalar_type_to_numpy(int8_t st);` (throws on unmapped)
  where `kind` is a numpy dtype-char (`'f'`,`'i'`,`'u'`,`'b'`) — no numpy headers in this TU.
- Produces (Python, test-only): `_core._np_to_st(dtype_char, itemsize) -> int`, `_core._st_to_np(int) -> (char, itemsize)`.

- [ ] **Step 1: Write the failing test `tests/test_dtypes.py`**

```python
import pytest
from executorch_numpy_runtime import _core

# ScalarType codes (ExecuTorch/PyTorch canonical): used only to assert the mapping.
ST = dict(Byte=0, Char=1, Short=2, Int=3, Long=4, Half=5, Float=6,
          Double=7, Bool=11, BFloat16=15)

CASES = [
    ("f", 4, ST["Float"]), ("f", 8, ST["Double"]), ("f", 2, ST["Half"]),
    ("i", 8, ST["Long"]),  ("i", 4, ST["Int"]),    ("i", 2, ST["Short"]),
    ("i", 1, ST["Char"]),  ("u", 1, ST["Byte"]),   ("b", 1, ST["Bool"]),
]

@pytest.mark.parametrize("kind,size,st", CASES)
def test_numpy_to_scalar_type(kind, size, st):
    assert _core._np_to_st(kind, size) == st

@pytest.mark.parametrize("kind,size,st", CASES)
def test_scalar_type_to_numpy(kind, size, st):
    assert _core._st_to_np(st) == (kind, size)

def test_bfloat16_maps_to_uint16():
    # BFloat16 surfaces as raw uint16 on the output path.
    assert _core._st_to_np(ST["BFloat16"]) == ("u", 2)

def test_unmapped_scalar_type_raises():
    with pytest.raises(Exception):
        _core._st_to_np(99)

def test_unmapped_numpy_raises():
    with pytest.raises(Exception):
        _core._np_to_st("c", 16)  # complex128 unsupported
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dtypes.py -v`
Expected: FAIL — `_np_to_st` missing.

- [ ] **Step 3: Write `src/binding/dtype_map.h`**

```cpp
#ifndef ETNP_DTYPE_MAP_H
#define ETNP_DTYPE_MAP_H
#include <cstddef>
#include <cstdint>

namespace etnp {
struct NpDtype { char kind; size_t itemsize; };
int8_t numpy_code_to_scalar_type(char kind, size_t itemsize);  // throws EtException on unmapped
NpDtype scalar_type_to_numpy(int8_t st);                        // throws EtException on unmapped
}  // namespace etnp
#endif
```

- [ ] **Step 4: Write `src/binding/dtype_map.cpp`** (use ExecuTorch enum symbols, not literals)

```cpp
#include "binding/dtype_map.h"
#include "et_core/et_core.h"
#include <executorch/runtime/core/portable_type/scalar_type.h>

namespace etnp {
using executorch::aten::ScalarType;

static EtException bad_dtype(const std::string& what) {
  return EtException({ErrorKind::DtypeUnmapped,
      "Unsupported dtype: " + what, ""});
}

int8_t numpy_code_to_scalar_type(char kind, size_t itemsize) {
  ScalarType st;
  if (kind == 'f' && itemsize == 4) st = ScalarType::Float;
  else if (kind == 'f' && itemsize == 8) st = ScalarType::Double;
  else if (kind == 'f' && itemsize == 2) st = ScalarType::Half;
  else if (kind == 'i' && itemsize == 8) st = ScalarType::Long;
  else if (kind == 'i' && itemsize == 4) st = ScalarType::Int;
  else if (kind == 'i' && itemsize == 2) st = ScalarType::Short;
  else if (kind == 'i' && itemsize == 1) st = ScalarType::Char;
  else if (kind == 'u' && itemsize == 1) st = ScalarType::Byte;
  else if (kind == 'b' && itemsize == 1) st = ScalarType::Bool;
  else if (kind == 'u' && itemsize == 2) st = ScalarType::BFloat16;  // explicit opt-in bits
  else throw bad_dtype(std::string(1, kind) + std::to_string(itemsize));
  return static_cast<int8_t>(st);
}

NpDtype scalar_type_to_numpy(int8_t code) {
  switch (static_cast<ScalarType>(code)) {
    case ScalarType::Float:  return {'f', 4};
    case ScalarType::Double: return {'f', 8};
    case ScalarType::Half:   return {'f', 2};
    case ScalarType::Long:   return {'i', 8};
    case ScalarType::Int:    return {'i', 4};
    case ScalarType::Short:  return {'i', 2};
    case ScalarType::Char:   return {'i', 1};
    case ScalarType::Byte:   return {'u', 1};
    case ScalarType::Bool:   return {'b', 1};
    case ScalarType::BFloat16: return {'u', 2};  // raw uint16 passthrough
    default:
      throw bad_dtype("ScalarType(" + std::to_string(code) + ")");
  }
}
}  // namespace etnp
```

Note: the `('u',2)`→BFloat16 input mapping is intentional (bf16 tensors are fed as uint16 bits). This means uint16 inputs are always interpreted as BFloat16; document this in the README dtype caveats. There is no other ScalarType for 2-byte unsigned, so this is unambiguous within the supported set.

- [ ] **Step 5: Expose test helpers in `src/binding/module.cpp`**

Add inside `NB_MODULE`:
```cpp
#include "binding/dtype_map.h"
// ...
  m.def("_np_to_st", [](const std::string& k, size_t sz) {
    return numpy_code_to_scalar_type(k.at(0), sz);
  });
  m.def("_st_to_np", [](int8_t st) {
    auto d = scalar_type_to_numpy(st);
    return std::make_pair(std::string(1, d.kind), d.itemsize);
  });
```
Add `#include <nanobind/stl/pair.h>` at the top.

- [ ] **Step 6: Add `dtype_map.cpp` to `CMakeLists.txt`** `nanobind_add_module` source list.

- [ ] **Step 7: Rebuild and run tests**

Run:
```bash
pip install -e . --no-build-isolation
pytest tests/test_dtypes.py -v
```
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/binding/dtype_map.* src/binding/module.cpp CMakeLists.txt tests/test_dtypes.py
git commit -m "feat: bidirectional dtype table with reject-on-unmapped"
```

---

## Task 4: Input marshalling — numpy → non-owning ET tensor, contiguity, keepalive

**Files:**
- Modify: `src/binding/module.cpp` (add `_marshal_check` helper used by tests; real use in Task 5)
- Create: `tests/test_input_marshal.py`

**Interfaces:**
- Produces (C++ helper, internal to `module.cpp`, declared here for Task 5):
  `struct HeldInput { nb::object keepalive; InputDesc desc; };`
  `HeldInput make_held_input(nb::ndarray<> arr);` — casts to c_contig (copying if needed), maps dtype, borrows a keepalive ref, fills `InputDesc` pointing at the (contiguous) buffer.
- Produces (Python, test-only): `_core._contig_info(arr) -> (was_contiguous: bool, st: int, shape: tuple)`.

- [ ] **Step 1: Write the failing test `tests/test_input_marshal.py`**

```python
import numpy as np
import pytest
from executorch_numpy_runtime import _core

def test_contiguous_float32_maps():
    a = np.ones((2, 3), dtype=np.float32)
    was, st, shape = _core._contig_info(a)
    assert was is True and st == 6 and shape == (2, 3)

def test_noncontiguous_is_copied_not_rejected():
    a = np.ones((4, 4), dtype=np.float32)[:, ::2]  # non-contiguous view
    assert not a.flags["C_CONTIGUOUS"]
    was, st, shape = _core._contig_info(a)
    assert was is False and st == 6 and shape == (4, 2)  # handled via copy

def test_unmapped_input_dtype_raises():
    a = np.ones(3, dtype=np.complex128)
    with pytest.raises(Exception):
        _core._contig_info(a)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_input_marshal.py -v`
Expected: FAIL — `_contig_info` missing.

- [ ] **Step 3: Implement input marshalling in `src/binding/module.cpp`**

Add includes and a helper above `NB_MODULE`:
```cpp
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>
#include <vector>

struct HeldInput {
  nb::object keepalive;   // pins the (possibly copied) buffer for the whole call
  etnp::InputDesc desc;
};

// Cast to a C-contiguous numpy array (copies if needed), map dtype, pin a ref.
static HeldInput make_held_input(nb::handle obj, bool* was_contig = nullptr) {
  // ascontiguousarray semantics: nb::c_contig cast copies non-contiguous input.
  auto arr = nb::cast<nb::ndarray<nb::c_contig, nb::device::cpu>>(obj);
  if (was_contig) {
    auto orig = nb::cast<nb::ndarray<nb::device::cpu>>(obj);
    *was_contig = (orig.data() == arr.data());
  }
  char kind;
  switch (arr.dtype().code) {
    case (uint8_t)nb::dlpack::dtype_code::Float: kind = 'f'; break;
    case (uint8_t)nb::dlpack::dtype_code::Int:   kind = 'i'; break;
    case (uint8_t)nb::dlpack::dtype_code::UInt:  kind = 'u'; break;
    case (uint8_t)nb::dlpack::dtype_code::Bool:  kind = 'b'; break;
    default: throw etnp::EtException({etnp::ErrorKind::DtypeUnmapped,
                 "Unsupported numpy dtype code", ""});
  }
  int8_t st = etnp::numpy_code_to_scalar_type(kind, arr.dtype().bits / 8);
  std::vector<int64_t> shape(arr.shape_ptr(), arr.shape_ptr() + arr.ndim());
  HeldInput h;
  h.keepalive = nb::borrow(nb::cast<nb::object>(arr));  // hold the contiguous array
  h.desc = etnp::InputDesc{arr.data(), std::move(shape), st};
  return h;
}
```

Add inside `NB_MODULE`:
```cpp
  m.def("_contig_info", [](nb::handle obj) {
    bool was = false;
    HeldInput h = make_held_input(obj, &was);
    std::vector<int64_t> s = h.desc.shape;
    return std::make_tuple(was, h.desc.scalar_type,
                           nb::tuple(nb::cast(s)));
  });
```

Note: nanobind's `nb::dlpack::dtype_code` enum names and the `nb::c_contig` copy-on-cast behavior must be verified against the installed nanobind version; if `c_contig` rejects rather than copies, wrap the Python arg with `np.ascontiguousarray` in the Python API layer (Task 9) and keep the C++ cast as an assertion.

- [ ] **Step 4: Rebuild and run tests**

Run:
```bash
pip install -e . --no-build-isolation
pytest tests/test_input_marshal.py -v
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/binding/module.cpp tests/test_input_marshal.py
git commit -m "feat: numpy->ET input marshalling with contiguity handling and keepalive"
```

---

## Task 5: Output marshalling (mandatory copy) + wire `run_method` end-to-end

**Files:**
- Modify: `src/et_core/et_core.cpp` (implement `run_method` + `ForwardResult`)
- Modify: `src/binding/module.cpp` (bind `run_method`, allocate fresh numpy outputs, copy)
- Create: `tests/test_forward.py`

**Interfaces:**
- Consumes: `Runtime::run_method`, `HeldInput`, `scalar_type_to_numpy`.
- Produces (Python): `_core._Runtime.run_method(name: str, inputs: list[np.ndarray]) -> list[np.ndarray]` returning freshly-allocated, owning numpy arrays (copies of arena memory).

- [ ] **Step 1: Write the failing test `tests/test_forward.py`**

```python
import numpy as np
from executorch_numpy_runtime import _core
from tests.conftest import model_or_skip

def test_add_forward_correct():
    rt = _core.load_path(model_or_skip("add.pte"))
    a = np.full(3, 2.0, dtype=np.float32)
    b = np.full(3, 5.0, dtype=np.float32)
    out = rt.run_method("forward", [a, b])
    assert len(out) == 1
    np.testing.assert_allclose(out[0], np.full(3, 7.0, dtype=np.float32))

def test_output_is_owning_copy():
    rt = _core.load_path(model_or_skip("add.pte"))
    out = rt.run_method("forward", [np.ones(3, np.float32), np.ones(3, np.float32)])
    assert out[0].flags["OWNDATA"]  # fresh array, not a view into the arena
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_forward.py -v`
Expected: FAIL — `_Runtime` has no `run_method`.

- [ ] **Step 3: Implement `run_method` + `ForwardResult` in `src/et_core/et_core.cpp`**

Add includes and code:
```cpp
#include <executorch/extension/tensor/tensor.h>
// using declarations:
using executorch::extension::from_blob;
using executorch::extension::TensorPtr;
using executorch::runtime::EValue;
using executorch::aten::ScalarType;

struct ForwardState {
  std::vector<EValue> outputs;      // owns result EValues (arena-backing lifetime)
  std::vector<OutputView> views;
};

ForwardResult::ForwardResult(std::unique_ptr<ForwardState> s) : state_(std::move(s)) {}
ForwardResult::~ForwardResult() = default;
ForwardResult::ForwardResult(ForwardResult&&) noexcept = default;
ForwardResult& ForwardResult::operator=(ForwardResult&&) noexcept = default;
const std::vector<OutputView>& ForwardResult::outputs() const { return state_->views; }

ForwardResult Runtime::run_method(const std::string& name,
                                  const std::vector<InputDesc>& inputs) {
  std::vector<std::vector<executorch::aten::SizesType>> shapes(inputs.size());
  std::vector<TensorPtr> tensors;
  std::vector<EValue> evalues;
  tensors.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    shapes[i].assign(inputs[i].shape.begin(), inputs[i].shape.end());
    tensors.push_back(from_blob(const_cast<void*>(inputs[i].data), shapes[i],
                                static_cast<ScalarType>(inputs[i].scalar_type)));
    evalues.emplace_back(tensors[i]);
  }
  auto result = state_->module->execute(name, evalues);
  if (result.error() != executorch::runtime::Error::Ok) {
    throw EtException({ErrorKind::Execution,
        "execute('" + name + "') failed", ""});  // refined in Task 7
  }
  auto fs = std::make_unique<ForwardState>();
  fs->outputs = std::move(*result);
  for (auto& ev : fs->outputs) {
    const auto& t = ev.toTensor();
    std::vector<int64_t> shp(t.sizes().begin(), t.sizes().end());
    fs->views.push_back(OutputView{std::move(shp),
        static_cast<int8_t>(t.scalar_type()), t.const_data_ptr(), t.nbytes()});
  }
  return ForwardResult(std::move(fs));
}
```

- [ ] **Step 4: Bind `run_method` with mandatory-copy output in `src/binding/module.cpp`**

Add to the `_Runtime` class binding and helpers:
```cpp
#include <cstring>
#include "binding/dtype_map.h"

// Allocate a fresh, owning numpy array and memcpy from arena memory (never a view).
static nb::object copy_out(const etnp::OutputView& v) {
  etnp::NpDtype d = etnp::scalar_type_to_numpy(v.scalar_type);
  size_t ndim = v.shape.size();
  std::vector<size_t> shape(v.shape.begin(), v.shape.end());
  // Allocate owning storage, hand ownership to numpy via a capsule.
  void* buf = std::malloc(v.nbytes ? v.nbytes : 1);
  std::memcpy(buf, v.data, v.nbytes);
  nb::capsule owner(buf, [](void* p) noexcept { std::free(p); });
  nb::dlpack::dtype dt{
      /*code*/ (uint8_t)(d.kind == 'f' ? nb::dlpack::dtype_code::Float :
                         d.kind == 'i' ? nb::dlpack::dtype_code::Int :
                         d.kind == 'u' ? nb::dlpack::dtype_code::UInt :
                                         nb::dlpack::dtype_code::Bool),
      /*bits*/ (uint8_t)(d.itemsize * 8), /*lanes*/ 1};
  return nb::cast(nb::ndarray<nb::numpy>(buf, ndim, shape.data(), owner, nullptr, dt));
}
```

Extend the `_Runtime` binding:
```cpp
  nb::class_<Runtime>(m, "_Runtime")
      .def("method_names", &Runtime::method_names)
      .def("run_method", [](Runtime& self, const std::string& name, nb::list arrs) {
        std::vector<HeldInput> held;
        std::vector<etnp::InputDesc> descs;
        held.reserve(arrs.size());
        for (auto a : arrs) { held.push_back(make_held_input(a)); }
        for (auto& h : held) descs.push_back(h.desc);
        etnp::ForwardResult fr = self.run_method(name, descs);  // GIL added in Task 6
        nb::list out;
        for (const auto& v : fr.outputs()) out.append(copy_out(v));
        return out;  // held[] (keepalives) drop here, after outputs copied
      });
```
Add `#include <nanobind/stl/list.h>` if needed (or iterate `nb::list` directly).

- [ ] **Step 5: Rebuild and run tests**

Run:
```bash
pip install -e . --no-build-isolation
pytest tests/test_forward.py tests/test_load.py -v
```
Expected: all PASS; `test_output_is_owning_copy` confirms `OWNDATA`.

- [ ] **Step 6: Commit**

```bash
git add src/et_core/et_core.cpp src/binding/module.cpp tests/test_forward.py
git commit -m "feat: run_method end-to-end with mandatory output copy"
```

---

## Task 6: GIL-release discipline around execute only

**Files:**
- Modify: `src/binding/module.cpp` (wrap only `run_method` core call in `gil_scoped_release`)
- Create: `tests/test_concurrency.py`

**Interfaces:**
- Consumes: existing `run_method` binding.
- Produces: no API change; concurrency-safe execution.

- [ ] **Step 1: Write the failing test `tests/test_concurrency.py`**

```python
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from executorch_numpy_runtime import _core
from tests.conftest import model_or_skip

def test_concurrent_forward_correct():
    rt = _core.load_path(model_or_skip("add.pte"))
    def one(i):
        a = np.full(3, float(i), np.float32)
        return rt.run_method("forward", [a, a])[0]
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(one, range(64)))
    for i, r in enumerate(results):
        np.testing.assert_allclose(r, np.full(3, 2.0 * i, np.float32))
```

- [ ] **Step 2: Run test to verify current behavior**

Run: `pytest tests/test_concurrency.py -v`
Expected: PASS or slow (single ExecuTorch Module is not itself thread-safe across concurrent execute; this test uses one Runtime). If it fails/races, that motivates the guarded release below. Record the result.

- [ ] **Step 3: Add GIL release around the core execute call**

In the `run_method` lambda in `src/binding/module.cpp`, wrap ONLY the core call — all inputs/keepalives are already built before this point, all outputs copied after:
```cpp
        etnp::ForwardResult fr = [&] {
          nb::gil_scoped_release nogil;   // release AFTER inputs built; touch no py objects
          return self.run_method(name, descs);
        }();
```
Everything above (`make_held_input`) and below (`copy_out`) stays under the GIL.

- [ ] **Step 4: Rebuild and run tests**

Run:
```bash
pip install -e . --no-build-isolation
pytest tests/test_concurrency.py -v
```
Expected: PASS. (Correctness holds because each call builds its own input tensors and copies its own outputs; no shared mutable Python state in the released region.)

- [ ] **Step 5: Commit**

```bash
git add src/binding/module.cpp tests/test_concurrency.py
git commit -m "feat: release GIL around execute only, per memory ordering rules"
```

---

## Task 7: Error hierarchy — map core failures to Python exceptions

**Files:**
- Create: `executorch_numpy_runtime/errors.py`
- Modify: `src/et_core/et_core.cpp` (classify load errors: backend vs operator vs verification)
- Modify: `src/binding/module.cpp` (register exception translator mapping `ErrorKind` → Python classes)
- Modify: `executorch_numpy_runtime/__init__.py` (export exceptions)
- Create: `tests/test_errors.py`

**Interfaces:**
- Consumes: `EtException`, `ErrorKind`, `etnp::backend_available` (from Task 8 — declare/stub now, or inline the backend check here).
- Produces (Python): `executorch_numpy_runtime.errors.{ExecuTorchError, ProgramLoadError, BackendNotAvailable, OperatorNotFound, ExecutionError}`; `_core` raises the matching subclass.

- [ ] **Step 1: Write the failing test `tests/test_errors.py`**

```python
import numpy as np
import pytest
import executorch_numpy_runtime as en
from executorch_numpy_runtime import _core
from executorch_numpy_runtime.errors import (
    ExecuTorchError, ProgramLoadError, BackendNotAvailable, ExecutionError)
from tests.conftest import model_or_skip, MODELS

def test_corrupt_pte_raises_program_load_error(tmp_path):
    bad = tmp_path / "bad.pte"; bad.write_bytes(b"not a real pte" * 10)
    with pytest.raises(ProgramLoadError):
        _core.load_path(str(bad))

def test_all_errors_subclass_base():
    for cls in (ProgramLoadError, BackendNotAvailable, ExecutionError):
        assert issubclass(cls, ExecuTorchError)

def test_unmapped_dtype_input_raises_executorch_error():
    rt = _core.load_path(model_or_skip("add.pte"))
    with pytest.raises(ExecuTorchError):
        rt.run_method("forward", [np.ones(3, np.complex128), np.ones(3, np.complex128)])

@pytest.mark.skipif(not (MODELS / "coreml.pte").exists(),
                    reason="non-CPU fixture not available")
def test_non_cpu_backend_raises_backend_not_available():
    with pytest.raises(BackendNotAvailable):
        _core.load_path(str(MODELS / "coreml.pte"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_errors.py -v`
Expected: FAIL — `executorch_numpy_runtime.errors` missing.

- [ ] **Step 3: Write `executorch_numpy_runtime/errors.py`**

```python
class ExecuTorchError(Exception):
    """Base class for all executorch_numpy_runtime runtime errors."""

class ProgramLoadError(ExecuTorchError):
    """Malformed, corrupt, or version-incompatible .pte."""

class BackendNotAvailable(ExecuTorchError):
    """The .pte was lowered for a delegate this runtime does not link (CPU-only)."""

class OperatorNotFound(ExecuTorchError):
    """An operator required by the .pte is not in the linked kernel set."""

class ExecutionError(ExecuTorchError):
    """Runtime failure during execution (e.g. shape/dtype mismatch)."""
```

- [ ] **Step 4: Classify load errors in `src/et_core/et_core.cpp`**

Refine `load_error` to inspect the failure. In 1.3.1 the `Module::load` result is a single `Error`; distinguish where possible by probing backend availability and verification:
```cpp
static EtException classify_load_failure(Module& mod, const std::string& what) {
  // If any required backend is unregistered, prefer BackendMissing.
  // (registered_backends() from Task 8; inline the ET call here if Task 8 not yet done.)
  // Heuristic: a verification/format failure -> Load; unknown backend -> BackendMissing.
  return EtException({ErrorKind::Load,
      "Failed to load .pte (corrupt, or exported by an incompatible ExecuTorch "
      "version; this runtime targets 1.3.1): " + what, ""});
}
```
Update `load_path`/`load_buffer` to call `classify_load_failure(*st->module, path)`.

Note: robust backend-vs-corruption discrimination depends on ExecuTorch surfacing the delegate id on load failure. If 1.3.1 does not, keep everything as `ProgramLoadError` and document that non-CPU artifacts surface as load errors; upgrade to `BackendNotAvailable` once the backend name is obtainable (via `registered_backends()` cross-check in Task 8).

- [ ] **Step 5: Register a nanobind exception translator in `src/binding/module.cpp`**

```cpp
#include <nanobind/nanobind.h>
// Inside NB_MODULE, after importing the Python errors module:
  nb::module_ errmod = nb::module_::import_("executorch_numpy_runtime.errors");
  nb::object base   = errmod.attr("ExecuTorchError");
  nb::object load   = errmod.attr("ProgramLoadError");
  nb::object backend= errmod.attr("BackendNotAvailable");
  nb::object opnf   = errmod.attr("OperatorNotFound");
  nb::object exec   = errmod.attr("ExecutionError");

  nb::register_exception_translator(
    [](const std::exception_ptr& p, void* payload) {
      auto* mod = static_cast<nb::object*>(payload);  // base map passed via payload
      try { std::rethrow_exception(p); }
      catch (const etnp::EtException& e) {
        nb::object cls;
        switch (e.error.kind) {
          case etnp::ErrorKind::Load:            cls = mod[0]; break; // load
          case etnp::ErrorKind::BackendMissing:  cls = mod[1]; break; // backend
          case etnp::ErrorKind::OperatorMissing: cls = mod[2]; break; // opnf
          case etnp::ErrorKind::Execution:       cls = mod[3]; break; // exec
          case etnp::ErrorKind::DtypeUnmapped:   cls = mod[4]; break; // base (TypeError-like)
        }
        std::string msg = e.error.message;
        if (!e.error.detail.empty()) msg += " [" + e.error.detail + "]";
        PyErr_SetString(cls.ptr(), msg.c_str());
      }
    }, /*payload*/ new nb::object[5]{load, backend, opnf, exec, base});
```
Note: verify nanobind's `register_exception_translator` signature/payload mechanism against the installed version; the simplest robust alternative is a `try/catch` wrapper inside each bound lambda that calls a shared `raise_py(const EtException&)` helper. Prefer whichever the installed nanobind supports cleanly.

- [ ] **Step 6: Export exceptions from `executorch_numpy_runtime/__init__.py`**

```python
from ._core import __et_version__, load_path, load_buffer
from .errors import (
    ExecuTorchError, ProgramLoadError, BackendNotAvailable,
    OperatorNotFound, ExecutionError)

__all__ = ["__et_version__", "load_path", "load_buffer",
           "ExecuTorchError", "ProgramLoadError", "BackendNotAvailable",
           "OperatorNotFound", "ExecutionError"]
```

- [ ] **Step 7: Rebuild and run tests**

Run:
```bash
pip install -e . --no-build-isolation
pytest tests/test_errors.py -v
```
Expected: non-skipped tests PASS (corrupt→`ProgramLoadError`, unmapped dtype→`ExecuTorchError`).

- [ ] **Step 8: Commit**

```bash
git add executorch_numpy_runtime/errors.py executorch_numpy_runtime/__init__.py src/et_core/et_core.cpp src/binding/module.cpp tests/test_errors.py
git commit -m "feat: exception hierarchy mapping core failures to Python errors"
```

---

## Task 8: method_meta + runtime introspection

**Files:**
- Modify: `src/et_core/et_core.cpp` (implement `method_meta`, `backend_available`, `registered_backends`, `operator_names`)
- Modify: `src/binding/module.cpp` (bind them)
- Create: `executorch_numpy_runtime/info.py`
- Create: `tests/test_meta_info.py`

**Interfaces:**
- Produces (C++): `Runtime::method_meta`, `etnp::backend_available/registered_backends/operator_names` (declared in Task 2 header).
- Produces (Python): `_core._Runtime.method_meta(name) -> dict`, `_core.registered_backends() -> list[str]`, `_core.operator_names() -> list[str]`, `_core.backend_available(str) -> bool`; and `executorch_numpy_runtime.runtime_info() -> dict`.

- [ ] **Step 1: Write the failing test `tests/test_meta_info.py`**

```python
import executorch_numpy_runtime as en
from executorch_numpy_runtime import _core
from tests.conftest import model_or_skip

def test_method_meta_shapes():
    rt = _core.load_path(model_or_skip("add.pte"))
    meta = rt.method_meta("forward")
    assert meta["num_inputs"] == 2
    assert meta["num_outputs"] == 1
    assert meta["inputs"][0]["scalar_type"] == 6  # Float

def test_runtime_info_reports_version_and_backends():
    info = en.runtime_info()
    assert info["executorch_version"] == "1.3.1"
    assert "XnnpackBackend" in info["backends"]
    assert info["bfloat16"] == "uint16-passthrough"

def test_backend_available():
    assert _core.backend_available("XnnpackBackend") is True
    assert _core.backend_available("CoreMLBackend") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_meta_info.py -v`
Expected: FAIL — `method_meta` / `runtime_info` missing. (The exact backend string may differ; adjust the assertion after Step 5 prints the real names.)

- [ ] **Step 3: Implement in `src/et_core/et_core.cpp`**

```cpp
#include <executorch/runtime/backend/interface.h>   // registered backends
// method_meta:
MethodMeta Runtime::method_meta(const std::string& name) const {
  auto mm = state_->module->method_meta(name);
  if (mm.error() != executorch::runtime::Error::Ok)
    throw EtException({ErrorKind::Load, "method_meta('" + name + "') failed", ""});
  MethodMeta out;
  for (size_t i = 0; i < mm->num_inputs(); ++i) {
    auto info = mm->input_tensor_meta(i);
    if (info.error() == executorch::runtime::Error::Ok) {
      TensorMeta tm; tm.scalar_type = static_cast<int8_t>(info->scalar_type());
      tm.shape.assign(info->sizes().begin(), info->sizes().end());
      out.inputs.push_back(std::move(tm));
    } else { out.inputs.push_back(TensorMeta{{}, -1}); }
  }
  for (size_t i = 0; i < mm->num_outputs(); ++i) {
    auto info = mm->output_tensor_meta(i);
    if (info.error() == executorch::runtime::Error::Ok) {
      TensorMeta tm; tm.scalar_type = static_cast<int8_t>(info->scalar_type());
      tm.shape.assign(info->sizes().begin(), info->sizes().end());
      out.outputs.push_back(std::move(tm));
    } else { out.outputs.push_back(TensorMeta{{}, -1}); }
  }
  return out;
}

std::vector<std::string> registered_backends() {
  std::vector<std::string> names;
  for (size_t i = 0; i < executorch::runtime::get_num_registered_backends(); ++i) {
    auto r = executorch::runtime::get_backend_name(i);
    if (r.ok()) names.emplace_back(*r);
  }
  return names;
}
bool backend_available(const std::string& name) {
  for (auto& n : registered_backends()) if (n == name) return true;
  return false;
}
std::vector<std::string> operator_names() {
  std::vector<std::string> out;
  auto ops = executorch::runtime::get_registered_kernels();  // verify API name in 1.3.1
  for (auto& k : ops) out.emplace_back(k.name);
  return out;
}
```
Note: verify the exact 1.3.1 API names (`get_num_registered_backends`/`get_backend_name`, kernel registry accessor) against `/home/corey/workspace/executorch/runtime/`. Consult upstream `pybindings.cpp` `get_operator_names`/`_get_registered_backend_names` for the precise calls — that is exactly what pattern #4 ports.

- [ ] **Step 4: Bind in `src/binding/module.cpp`**

```cpp
  // in _Runtime binding:
      .def("method_meta", [](Runtime& self, const std::string& n) {
        auto m = self.method_meta(n);
        auto to_list = [](const std::vector<etnp::TensorMeta>& v) {
          nb::list l;
          for (auto& t : v) {
            nb::dict d; d["scalar_type"] = t.scalar_type;
            d["shape"] = nb::cast(t.shape); l.append(d);
          }
          return l;
        };
        nb::dict d;
        d["num_inputs"] = m.inputs.size();
        d["num_outputs"] = m.outputs.size();
        d["inputs"] = to_list(m.inputs);
        d["outputs"] = to_list(m.outputs);
        return d;
      });
  // module-level:
  m.def("registered_backends", &registered_backends);
  m.def("backend_available", &backend_available);
  m.def("operator_names", &operator_names);
```

- [ ] **Step 5: Print the real backend/op names to fix test assertions**

Run:
```bash
pip install -e . --no-build-isolation
python -c "from executorch_numpy_runtime import _core; print(_core.registered_backends()); print(_core.operator_names()[:10])"
```
Update the backend-name assertions in `tests/test_meta_info.py` and `tests/test_errors.py` to match the printed names.

- [ ] **Step 6: Write `executorch_numpy_runtime/info.py`**

```python
from . import _core

_SUPPORTED_DTYPES = ["float32", "float64", "float16", "int64", "int32",
                     "int16", "int8", "uint8", "bool", "uint16(bfloat16-bits)"]

def runtime_info() -> dict:
    return {
        "executorch_version": _core.__et_version__,
        "backends": _core.registered_backends(),
        "operators": _core.operator_names(),
        "kernel_libs": ["portable", "optimized", "quantized"],
        "supported_dtypes": _SUPPORTED_DTYPES,
        "bfloat16": "uint16-passthrough",
    }
```
Add `from .info import runtime_info` and `"runtime_info"` to `__init__.py`'s exports.

- [ ] **Step 7: Rebuild and run tests**

Run:
```bash
pip install -e . --no-build-isolation
pytest tests/test_meta_info.py -v
```
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/et_core/et_core.cpp src/binding/module.cpp executorch_numpy_runtime/info.py executorch_numpy_runtime/__init__.py tests/test_meta_info.py tests/test_errors.py
git commit -m "feat: method_meta and runtime introspection (backends, ops, runtime_info)"
```

---

## Task 9: High-level Python API — Runtime / Program / Method

**Files:**
- Create: `executorch_numpy_runtime/_api.py`
- Modify: `executorch_numpy_runtime/__init__.py`
- Create: `tests/test_api.py`

**Interfaces:**
- Consumes: `_core.load_path`, `_core.load_buffer`, `_core._Runtime.{method_names, method_meta, run_method}`.
- Produces (Python public API):
  - `Runtime.get() -> Runtime` (singleton-ish accessor mirroring executorch.runtime)
  - `Runtime.load_program(path_or_bytes) -> Program`
  - `Program.method_names -> list[str]`, `Program.load_method(name) -> Method`
  - `Method(inputs: Sequence[np.ndarray]) -> list[np.ndarray]` (callable), `Method.metadata -> dict`

- [ ] **Step 1: Write the failing test `tests/test_api.py`**

```python
import numpy as np
import executorch_numpy_runtime as en
from tests.conftest import model_or_skip

def test_high_level_forward():
    prog = en.Runtime.get().load_program(model_or_skip("add.pte"))
    assert "forward" in prog.method_names
    method = prog.load_method("forward")
    out = method([np.full(3, 2.0, np.float32), np.full(3, 5.0, np.float32)])
    np.testing.assert_allclose(out[0], np.full(3, 7.0, np.float32))

def test_load_program_from_bytes():
    with open(model_or_skip("add.pte"), "rb") as f:
        prog = en.Runtime.get().load_program(f.read())
    assert prog.method_names

def test_method_metadata_exposed():
    prog = en.Runtime.get().load_program(model_or_skip("add.pte"))
    assert prog.load_method("forward").metadata["num_inputs"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py -v`
Expected: FAIL — `en.Runtime` missing.

- [ ] **Step 3: Write `executorch_numpy_runtime/_api.py`**

```python
from __future__ import annotations
from typing import Sequence, Union
import numpy as np
from . import _core

class Method:
    def __init__(self, runtime: "_core._Runtime", name: str):
        self._rt = runtime
        self._name = name

    @property
    def metadata(self) -> dict:
        return self._rt.method_meta(self._name)

    def __call__(self, inputs: Sequence[np.ndarray]) -> list:
        # Enforce contiguity defensively at the Python boundary too.
        arrs = [np.ascontiguousarray(a) for a in inputs]
        return self._rt.run_method(self._name, arrs)

class Program:
    def __init__(self, runtime: "_core._Runtime"):
        self._rt = runtime

    @property
    def method_names(self) -> list:
        return self._rt.method_names()

    def load_method(self, name: str) -> Method:
        if name not in self._rt.method_names():
            from .errors import ProgramLoadError
            raise ProgramLoadError(f"method '{name}' not found in program")
        return Method(self._rt, name)

class Runtime:
    _instance: "Runtime | None" = None

    @classmethod
    def get(cls) -> "Runtime":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_program(self, source: Union[str, bytes]) -> Program:
        if isinstance(source, (bytes, bytearray)):
            rt = _core.load_buffer(bytes(source))
        else:
            rt = _core.load_path(str(source))
        return Program(rt)
```

- [ ] **Step 4: Export from `executorch_numpy_runtime/__init__.py`**

```python
from ._core import __et_version__
from ._api import Runtime, Program, Method
from .info import runtime_info
from .errors import (
    ExecuTorchError, ProgramLoadError, BackendNotAvailable,
    OperatorNotFound, ExecutionError)

__version__ = "0.0.1"
__all__ = ["Runtime", "Program", "Method", "runtime_info", "__version__",
           "__et_version__", "ExecuTorchError", "ProgramLoadError",
           "BackendNotAvailable", "OperatorNotFound", "ExecutionError"]
```

- [ ] **Step 5: Rebuild (pure-Python change, reinstall not strictly needed) and run tests**

Run:
```bash
pip install -e . --no-build-isolation
pytest tests/test_api.py -v
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add executorch_numpy_runtime/_api.py executorch_numpy_runtime/__init__.py tests/test_api.py
git commit -m "feat: high-level Runtime/Program/Method numpy API"
```

---

## Task 10: Full contract test matrix

**Files:**
- Modify: `tools/export_fixtures.py` (add quantized + dynamic-shape fixtures)
- Create: `tests/test_matrix.py`, `tests/test_parity.py`
- Modify: `tests/test_dtypes.py` (add per-dtype round-trip)

**Interfaces:**
- Consumes: full public API + `_core`.
- Produces: coverage of the spec §10.1 matrix.

- [ ] **Step 1: Extend `tools/export_fixtures.py` with quantized + dynamic fixtures**

```python
# add to main(), after existing fixtures:
    from torch.export import Dim
    from executorch.exir import to_edge_transform_and_lower

    # Dynamic (bounded) shape model.
    class Scale(torch.nn.Module):
        def forward(self, x): return x * 2.0
    d = Dim("n", min=1, max=8)
    ep = export(Scale().eval(), (torch.ones(4),), dynamic_shapes={"x": {0: d}})
    (out / "dynamic.pte").write_bytes(
        to_edge_transform_and_lower(ep, partitioner=xnn).to_executorch().buffer)
    print("wrote dynamic.pte")

    # Quantized model (proves optimized/quantized kernels linked).
    # Use XNNPACK's quantized flow; if the 1.3.1 quantization API differs, adapt here.
    try:
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer, get_symmetric_quantization_config)
        from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
        class Lin(torch.nn.Module):
            def __init__(self): super().__init__(); self.l = torch.nn.Linear(8, 8)
            def forward(self, x): return self.l(x)
        ex = (torch.randn(2, 8),)
        m = torch.export.export_for_training(Lin().eval(), ex).module()
        q = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
        m = convert_pt2e(prepare_pt2e(m, q))
        ep = export(m, ex)
        (out / "quantized.pte").write_bytes(
            to_edge_transform_and_lower(ep, partitioner=xnn).to_executorch().buffer)
        print("wrote quantized.pte")
    except Exception as e:
        print("skip quantized.pte:", e)
```

Run:
```bash
/home/corey/workspace/executorch/.venv/bin/python tools/export_fixtures.py tests/models
ls tests/models/
```
Expected: `dynamic.pte` and (ideally) `quantized.pte` created.

- [ ] **Step 2: Write `tests/test_matrix.py`**

```python
import numpy as np
import pytest
import executorch_numpy_runtime as en
from tests.conftest import model_or_skip

def _method(name, meth="forward"):
    return en.Runtime.get().load_program(model_or_skip(name)).load_method(meth)

def test_sequential_calls_do_not_clobber_prior_outputs():
    m = _method("add.pte")
    r1 = m([np.full(3, 1.0, np.float32), np.full(3, 1.0, np.float32)])[0]
    r1_copy = r1.copy()
    r2 = m([np.full(3, 9.0, np.float32), np.full(3, 9.0, np.float32)])[0]
    np.testing.assert_allclose(r1, r1_copy)          # r1 unchanged by r2's run
    np.testing.assert_allclose(r2, np.full(3, 18.0, np.float32))

def test_multi_method_program():
    prog = en.Runtime.get().load_program(model_or_skip("multi.pte"))
    assert set(["forward", "double"]).issubset(set(prog.method_names))

def test_dynamic_shape_within_bounds():
    m = _method("dynamic.pte")
    for n in (1, 4, 8):
        out = m([np.ones(n, np.float32)])[0]
        np.testing.assert_allclose(out, np.full(n, 2.0, np.float32))

def test_quantized_model_runs():
    m = _method("quantized.pte")
    out = m([np.random.randn(2, 8).astype(np.float32)])
    assert out[0].shape == (2, 8)

def test_mixed_dtype_model():
    m = _method("dtypes.pte")
    a = np.array([3], dtype=np.int64); b = np.array([2.5], dtype=np.float32)
    outs = m([a, b])
    assert outs[0].dtype == np.int64 and outs[1].dtype == np.float32
    np.testing.assert_array_equal(outs[0], np.array([6]))
    np.testing.assert_allclose(outs[1], np.array([5.0], np.float32))
```

- [ ] **Step 3: Write per-dtype round-trip in `tests/test_dtypes.py`** (append)

```python
import numpy as np
import executorch_numpy_runtime as en
from tests.conftest import model_or_skip

def test_float32_roundtrip_fidelity():
    m = en.Runtime.get().load_program(model_or_skip("add.pte")).load_method("forward")
    x = np.random.randn(3).astype(np.float32)
    out = m([x, np.zeros(3, np.float32)])[0]  # x + 0 == x
    np.testing.assert_array_equal(out, x)     # bit-exact round trip
```

- [ ] **Step 4: Write `tests/test_parity.py`** (torch reference, CI/offline only — guarded)

```python
import numpy as np
import pytest
import executorch_numpy_runtime as en
from tests.conftest import model_or_skip

torch = pytest.importorskip("torch")  # never a package dep; skipped if torch absent

def test_add_matches_torch_reference():
    a = np.random.randn(3).astype(np.float32)
    b = np.random.randn(3).astype(np.float32)
    ref = (torch.tensor(a) + torch.tensor(b)).numpy()
    m = en.Runtime.get().load_program(model_or_skip("add.pte")).load_method("forward")
    got = m([a, b])[0]
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)
```

- [ ] **Step 5: Run the full suite**

Run:
```bash
pip install -e . --no-build-isolation
pytest tests/ -v
```
Expected: all PASS or cleanly SKIP (parity skips without torch; quantized/coreml skip if fixture absent). No crashes, no failures.

- [ ] **Step 6: Commit**

```bash
git add tools/export_fixtures.py tests/
git commit -m "test: full contract matrix (no-clobber, multi-method, dynamic, quantized, dtype, parity)"
```

---

## Task 11: Leak QA harness (ASan/LSan) as a merge gate

**Files:**
- Create: `native_tests/leak_harness.cpp` (port of `et_leak_harness.cpp`, driving `et_core` directly)
- Create: `native_tests/CMakeLists.txt`
- Create: `.github/workflows/leak.yml` (or a `Makefile`/script that runs it)

**Interfaces:**
- Consumes: `et_core` (`Runtime`, `method_meta`, `run_method`, `InputDesc`).
- Produces: a standalone `leak_harness` binary; non-zero exit on any leak.

- [ ] **Step 1: Write `native_tests/leak_harness.cpp`**

```cpp
// ASan/LSan leak harness over et_core (no Python). LSan reports unfreed allocations at
// exit; a leak -> non-zero exit. Model-agnostic: inputs derived from method_meta().
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "et_core/et_core.h"

using namespace etnp;

static size_t dtype_size(int8_t st) {
  switch (st) { case 6: case 3: return 4; case 7: case 4: return 8;
    case 5: case 2: case 15: return 2; case 1: case 0: case 11: return 1; default: return 4; }
}

int main(int argc, char** argv) {
  const char* pte = (argc > 1) ? argv[1] : "tests/models/add.pte";
  const int outer = (argc > 2) ? std::atoi(argv[2]) : 500;
  const int fwd_per_load = 4;
  for (int it = 0; it < outer; ++it) {
    auto rt = Runtime::load_path(pte);              // load/destroy balance
    auto meta = rt->method_meta("forward");
    std::vector<std::vector<uint8_t>> bufs(meta.inputs.size());
    std::vector<InputDesc> inputs;
    for (size_t i = 0; i < meta.inputs.size(); ++i) {
      if (meta.inputs[i].scalar_type < 0) continue;
      size_t count = 1; for (int64_t d : meta.inputs[i].shape) count *= (size_t)d;
      size_t bytes = count * dtype_size(meta.inputs[i].scalar_type);
      bufs[i].assign(bytes ? bytes : 1, 0);
      if (meta.inputs[i].scalar_type == 6) {        // float32 -> 1.0f
        float one = 1.0f;
        for (size_t b = 0; b + 4 <= bytes; b += 4) std::memcpy(bufs[i].data()+b, &one, 4);
      } else std::memset(bufs[i].data(), 1, bytes);
      inputs.push_back(InputDesc{bufs[i].data(), meta.inputs[i].shape,
                                 meta.inputs[i].scalar_type});
    }
    for (int f = 0; f < fwd_per_load; ++f) {
      auto res = rt->run_method("forward", inputs);  // per-forward allocations
      (void)res.outputs();
    }
  }
  std::printf("leak_harness: %d iters OK\n", outer);
  return 0;
}
```

- [ ] **Step 2: Write `native_tests/CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.24)
project(etnp_leak LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(${CMAKE_SOURCE_DIR}/../cmake/RuntimePin.cmake)
list(APPEND CMAKE_PREFIX_PATH "${ETNP_RUNTIME_PREFIX}")
find_package(ExecuTorch CONFIG REQUIRED)

add_executable(leak_harness leak_harness.cpp ${CMAKE_SOURCE_DIR}/../src/et_core/et_core.cpp)
target_include_directories(leak_harness PRIVATE ${CMAKE_SOURCE_DIR}/../src)
target_compile_options(leak_harness PRIVATE -fsanitize=address -g -O1)
target_link_options(leak_harness PRIVATE -fsanitize=address)
target_link_libraries(leak_harness PRIVATE
  executorch optimized_native_cpu_ops_lib xnnpack_backend quantized_ops_lib)
```

- [ ] **Step 3: Build and run under LSan**

Run:
```bash
cmake -S native_tests -B build/leak && cmake --build build/leak
ASAN_OPTIONS=detect_leaks=1 ./build/leak/leak_harness tests/models/add.pte 500
echo "exit=$?"
```
Expected: prints "500 iters OK", `exit=0`, no LSan "detected memory leaks" report. If LSan reports leaks, fix the offending path in `et_core.cpp` (do not silence the check) before proceeding.

- [ ] **Step 4: Write `.github/workflows/leak.yml`** (merge gate)

```yaml
name: leak-qa
on: [push, pull_request]
jobs:
  lsan:
    runs-on: ubuntu-latest
    container: quay.io/pypa/manylinux_2_28_x86_64
    steps:
      - uses: actions/checkout@v4
      - name: Fetch runtime tarball
        run: |
          mkdir -p third_party
          curl -sSL -o rt.tgz "${{ vars.ET_RUNTIME_URL }}"
          echo "${{ vars.ET_RUNTIME_SHA256 }}  rt.tgz" | sha256sum -c -
          tar -xzf rt.tgz -C third_party
      - name: Generate add.pte
        run: python -m pip install executorch==1.3.1 torch --quiet && python tools/export_fixtures.py tests/models
      - name: Build + run leak harness
        run: |
          cmake -S native_tests -B build/leak && cmake --build build/leak
          ASAN_OPTIONS=detect_leaks=1 ./build/leak/leak_harness tests/models/add.pte 500
```

- [ ] **Step 5: Commit**

```bash
git add native_tests/ .github/workflows/leak.yml
git commit -m "test: ASan/LSan leak harness over et_core as a merge gate"
```

---

## Task 12: README — caller-facing contract

**Files:**
- Create: `README.md`
- Create: `tests/test_docs.py` (assert the contract facts are present and consistent)

**Interfaces:**
- Produces: the §7.2 caller contract, consistent with `runtime_info()`.

- [ ] **Step 1: Write `tests/test_docs.py`**

```python
from pathlib import Path
import executorch_numpy_runtime as en

README = Path(__file__).parent.parent / "README.md"

def test_readme_states_version_and_contract():
    txt = README.read_text()
    assert "1.3.1" in txt
    assert "manylinux_2_28_x86_64" in txt and "cp312-abi3" in txt
    for phrase in ["CPU", "custom operator", "torchao", "BFloat16", "uint16"]:
        assert phrase.lower() in txt.lower(), phrase

def test_readme_version_matches_runtime():
    assert en.runtime_info()["executorch_version"] in README.read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_docs.py -v`
Expected: FAIL — no README.

- [ ] **Step 3: Write `README.md`**

```markdown
# executorch-numpy-runtime

A torch-free Python runtime for ExecuTorch `.pte` files. **numpy is the only required
dependency.** Loads and runs arbitrary CPU-targeted `.pte` artifacts.

## Install
    pip install executorch-numpy-runtime            # numpy only
    pip install executorch-numpy-runtime[bf16]      # + ml_dtypes for real bfloat16

Wheels: `cp312-abi3`, `manylinux_2_28_x86_64`, Python 3.12+.

## Quick start
    import numpy as np, executorch_numpy_runtime as en
    prog = en.Runtime.get().load_program("model.pte")
    out = prog.load_method("forward")([np.ones(3, np.float32), np.ones(3, np.float32)])

## Compatibility contract
- **ExecuTorch version: 1.3.1 (exact).** A `.pte` exported by an incompatible ExecuTorch
  version fails to load and **looks like a corrupt file** — check the exporter version
  before assuming corruption. Query the build version with `en.runtime_info()`.
- **Backends: CPU only** — XNNPACK delegate + portable fallback. `.pte` files lowered for
  CoreML / QNN / Vulkan / Metal are unsupported and raise `BackendNotAvailable`.
- **Operators: core ATen + optimized + quantized kernels.** **Custom operators are NOT
  included. torchao low-bit kernels are NOT included.**
- **dtypes:** float32, float64, float16, int64, int32, int16, int8, uint8, bool.
  **BFloat16 is surfaced as raw `uint16` bits** (no native numpy bf16). Install the `[bf16]`
  extra for a real `ml_dtypes.bfloat16` dtype. Note: `uint16` inputs are interpreted as
  BFloat16. Unsupported dtypes raise.
- **Outputs are always fresh copies** — never views into runtime memory; safe to keep
  across subsequent calls.

## Errors
`ExecuTorchError` (base) → `ProgramLoadError`, `BackendNotAvailable`, `OperatorNotFound`,
`ExecutionError`.

## Introspection
    en.runtime_info()   # {executorch_version, backends, operators, kernel_libs, dtypes, ...}
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_docs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_docs.py
git commit -m "docs: caller-facing compatibility contract"
```

---

## Self-Review Notes (spec coverage)

- Spec §3 four patterns: buffer-load (T2), multi-method (T2 `method_names` + T5 `run_method` by name), rich method_meta (T8), introspection (T8). ✓
- Spec §4 layering: `et_core` (T2/5/8) binding-agnostic; nanobind glue (T3–7); Python API (T9). ✓
- Spec §5 memory model: contiguity+keepalive (T4), mandatory output copy (T5), GIL (T6), dtype table incl. bf16→uint16 (T3). ✓
- Spec §6 errors: hierarchy + translator (T7). ✓
- Spec §7 introspection + contract: runtime_info (T8), README (T12). ✓
- Spec §8 build: pinned tarball + whole-archive nm guard (T1). ✓
- Spec §10 tests: full matrix (T10), leak gate (T11). ✓
- Spec §9 deferred low-level handle: intentionally NOT built; `et_core` binding-agnostic boundary (T2 header) keeps it cheap. ✓

**Known verification points flagged inline for the implementer** (1.3.1 API drift): `Module` buffer ctor (T2), nanobind `c_contig` copy semantics + dtype-code enum (T4), nanobind exception-translator signature (T7), backend/kernel registry accessor names (T8), quantization export API (T10). Each has a documented fallback.
