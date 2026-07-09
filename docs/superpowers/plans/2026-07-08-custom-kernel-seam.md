# Custom Kernel Seam Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give this shared runtime a CI-tested seam for compiling custom ExecuTorch kernels into the extension — a bundled trivial reference op plus a build-time injection point (`ETNP_EXTRA_KERNEL_SOURCES`) — without bundling any real production kernel here.

**Architecture:** A custom kernel is a C++ out-variant function registered into ExecuTorch's process-global operator registry via the shipped `EXECUTORCH_LIBRARY` macro. Registrars live in static-init translation units that the linker's `--gc-sections` silently drops unless the archive is whole-archived — the exact failure the existing `cmake/assert_kernels_registered.cmake` nm-guard exists to catch. We factor all kernel sources (the repo's reference op + any consumer-injected `.cpp`) into one static lib `etnp_kernels`, whole-archive it into the module and the native test via CMake's portable `$<LINK_LIBRARY:WHOLE_ARCHIVE,...>` generator expression, and extend the nm-guard to require every kernel source's registrar TU. A native test proves resolve+dispatch of the reference op through the real registry API — with no dependency on the ExecuTorch export toolchain.

**Tech Stack:** C++17, CMake ≥3.24, nanobind (extension), ExecuTorch 1.3.1 prebuilt runtime (`third_party/executorch-runtime-.../`), the project's existing nm-based post-link guard.

## Global Constraints

- **CMake ≥ 3.24** — `$<LINK_LIBRARY:WHOLE_ARCHIVE,...>` requires 3.24; already the project floor (`CMakeLists.txt:1`). This generator expression maps to `--whole-archive` (GNU/lld) or `-force_load` (Apple) per-platform, which keeps the seam correct for the arm64/macOS roadmap.
- **C++17**, `CMAKE_POSITION_INDEPENDENT_CODE ON` — match existing target settings.
- **Registration name must match the serialized op name.** `EXECUTORCH_LIBRARY(etnp, "triple.out", fn)` registers exactly `"etnp::triple.out"`. A consumer's `.pte` must reference that same string or the runtime raises operator-not-found at load.
- **Kernels are pure C++ linked into the module.** They do not cross the Python ABI, so the `STABLE_ABI`/abi3 packaging is unaffected.
- **Registration is static-init (import time), before any `Runtime` exists** — it does not interact with the per-`Runtime` execution mutex.
- **The nm-guard's TU symbol convention is `_GLOBAL__sub_I_<source-basename>`** — the toolchain names each dynamic-init TU after its source file, matching the existing guard's assumptions (`cmake/assert_kernels_registered.cmake`).
- **Do NOT bundle a real production kernel (e.g. `nn.LSTM`) in this repo.** This is the shared substrate; real kernels live in consumer builds via the injection seam. The bundled op is deliberately trivial.

---

### Task 1: Kernel seam CMake module + reference kernel + build-time guard

**Files:**
- Create: `kernels/reference/etnp_reference_ops.cpp`
- Create: `cmake/Kernels.cmake`
- Modify: `CMakeLists.txt` (include the module; whole-archive `etnp_kernels` into `_core`; feed guard the expected TUs)
- Modify: `cmake/assert_kernels_registered.cmake` (accept caller-supplied `EXTRA_TUS`)

**Interfaces:**
- Produces: CMake target `etnp_kernels` (STATIC) and cache/parent-scope var `ETNP_KERNEL_EXPECT_TUS` (list of `_GLOBAL__sub_I_<basename>` strings), both set by including `cmake/Kernels.cmake` after `find_package(ExecuTorch)`. Cache option `ETNP_BUILD_REFERENCE_KERNEL` (BOOL, default ON) and cache var `ETNP_EXTRA_KERNEL_SOURCES` (STRING; semicolon-separated absolute `.cpp` paths).
- Produces: registered operator `etnp::triple.out` with schema `triple.out(Tensor input, *, Tensor(a!) out) -> Tensor(a!)`, computing `out = 3 * input` elementwise for float32.
- Consumes: the ExecuTorch imported target `executorch` (headers + core lib) from `find_package(ExecuTorch CONFIG REQUIRED)`.

- [ ] **Step 1: Extend the guard to require caller-supplied TUs (the failing test)**

Edit `cmake/assert_kernels_registered.cmake`. After the existing `_codegen_count` block and before the final `message(STATUS ...)`, insert:

```cmake
# Caller-supplied extra static-init TUs that must also survive the final link
# (custom kernels compiled into etnp_kernels and whole-archived into the target).
# EXTRA_TUS is a CMake list of "_GLOBAL__sub_I_<source-basename>" symbol names.
foreach(_tu IN LISTS EXTRA_TUS)
  if(_tu STREQUAL "")
    continue()
  endif()
  string(FIND "${_syms}" "${_tu}" _pos)
  if(_pos EQUAL -1)
    message(FATAL_ERROR
      "Custom-kernel registration '${_tu}' was dropped from ${SO}: static-init TU absent. "
      "whole-archive regressed for etnp_kernels -> custom op not found at model-load time.")
  endif()
endforeach()
```

- [ ] **Step 2: Wire the guard into `_core` with a reference-op expectation, and run to see it fail**

In `CMakeLists.txt`, replace the existing `add_custom_command(TARGET _core POST_BUILD ...)` block (currently `CMakeLists.txt:36-39`) with:

```cmake
# Post-link kernel-registration guard (fail the BUILD, not runtime). The custom
# kernel seam appends its expected registrar TUs via ETNP_KERNEL_EXPECT_TUS.
add_custom_command(TARGET _core POST_BUILD
  COMMAND ${CMAKE_COMMAND} -DSO=$<TARGET_FILE:_core> -DNM=nm
          "-DEXTRA_TUS=${ETNP_KERNEL_EXPECT_TUS}"
          -P ${CMAKE_SOURCE_DIR}/cmake/assert_kernels_registered.cmake
  VERBATIM)
```

Set `ETNP_KERNEL_EXPECT_TUS` to the reference TU *before* the seam exists so the guard fails. Temporarily add, right above that command:

```cmake
set(ETNP_KERNEL_EXPECT_TUS "_GLOBAL__sub_I_etnp_reference_ops.cpp")
```

Run:
```bash
cmake -S . -B build/t1 >/dev/null && cmake --build build/t1 --target _core
```
Expected: build FAILS at the POST_BUILD step with `Custom-kernel registration '_GLOBAL__sub_I_etnp_reference_ops.cpp' was dropped ... static-init TU absent` (the op isn't compiled/linked yet).

- [ ] **Step 3: Write the reference kernel**

Create `kernels/reference/etnp_reference_ops.cpp`:

```cpp
// Reference custom kernel + CI-tested proof of the kernel-registration seam.
//
// Registers etnp::triple.out, a deliberately trivial elementwise op whose only
// job is to keep the compile + EXECUTORCH_LIBRARY registration + whole-archive
// link path continuously exercised. Consumers copy this file's shape to add
// their own kernels (see docs/custom-kernels.md); real kernels (e.g. nn.LSTM)
// live in the consumer's build via ETNP_EXTRA_KERNEL_SOURCES, not here.
#include <cstddef>

#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace etnp {
namespace {

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::KernelRuntimeContext;

// out = 3 * input, elementwise, float32. Trivial by design.
Tensor& triple_out(KernelRuntimeContext& ctx, const Tensor& input, Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::resize_tensor(out, input.sizes()) == Error::Ok,
      InvalidArgument,
      out);
  ET_KERNEL_CHECK(
      ctx, input.scalar_type() == ScalarType::Float, InvalidArgument, out);
  ET_KERNEL_CHECK(
      ctx, out.scalar_type() == ScalarType::Float, InvalidArgument, out);

  const float* in_data = input.const_data_ptr<float>();
  float* out_data = out.mutable_data_ptr<float>();
  for (size_t i = 0; i < static_cast<size_t>(input.numel()); ++i) {
    out_data[i] = in_data[i] * 3.0f;
  }
  return out;
}

} // namespace
} // namespace etnp

EXECUTORCH_LIBRARY(etnp, "triple.out", etnp::triple_out);
```

- [ ] **Step 4: Write the shared kernel-seam CMake module**

Create `cmake/Kernels.cmake`:

```cmake
# Defines the etnp_kernels static library: the repo's bundled reference kernel
# plus any consumer-injected sources, gathered into one archive that callers
# whole-archive into their final binary. Also computes ETNP_KERNEL_EXPECT_TUS,
# the list of static-init TU symbols the nm-guard must find post-link.
#
# Include AFTER find_package(ExecuTorch). Sets etnp_kernels + ETNP_KERNEL_EXPECT_TUS
# in the including scope. Paths resolve relative to THIS file (repo cmake/), so the
# top-level build and native_tests' standalone build agree (mirrors RuntimePin.cmake).
option(ETNP_BUILD_REFERENCE_KERNEL
  "Compile the bundled reference custom kernel (etnp::triple.out)" ON)
set(ETNP_EXTRA_KERNEL_SOURCES "" CACHE STRING
  "Semicolon-separated absolute paths to additional custom-kernel .cpp sources to \
compile and register into the module (see docs/custom-kernels.md)")

set(_etnp_kernel_sources "")
if(ETNP_BUILD_REFERENCE_KERNEL)
  list(APPEND _etnp_kernel_sources
    "${CMAKE_CURRENT_LIST_DIR}/../kernels/reference/etnp_reference_ops.cpp")
endif()
list(APPEND _etnp_kernel_sources ${ETNP_EXTRA_KERNEL_SOURCES})

set(ETNP_KERNEL_EXPECT_TUS "")
if(_etnp_kernel_sources)
  add_library(etnp_kernels STATIC ${_etnp_kernel_sources})
  set_property(TARGET etnp_kernels PROPERTY POSITION_INDEPENDENT_CODE ON)
  # PUBLIC so consumers of etnp_kernels inherit ExecuTorch's include dirs; the
  # core lib carries the kernel-registration API headers.
  target_link_libraries(etnp_kernels PUBLIC executorch)

  foreach(_src IN LISTS _etnp_kernel_sources)
    get_filename_component(_base "${_src}" NAME)
    list(APPEND ETNP_KERNEL_EXPECT_TUS "_GLOBAL__sub_I_${_base}")
  endforeach()
endif()
```

- [ ] **Step 5: Include the module and whole-archive `etnp_kernels` into `_core`**

In `CMakeLists.txt`, remove the temporary `set(ETNP_KERNEL_EXPECT_TUS ...)` line from Step 2. Add, immediately after `find_package(ExecuTorch CONFIG REQUIRED)` (`CMakeLists.txt:9`):

```cmake
include(cmake/Kernels.cmake)
```

Then in the `target_link_libraries(_core PRIVATE ...)` call (`CMakeLists.txt:29-31`), append the whole-archived kernel lib. Change it to:

```cmake
target_link_libraries(_core PRIVATE
  executorch optimized_native_cpu_ops_lib xnnpack_backend quantized_ops_lib
  extension_module_static extension_data_loader extension_tensor)

if(TARGET etnp_kernels)
  target_link_libraries(_core PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,etnp_kernels>")
endif()
```

- [ ] **Step 6: Build and verify the guard now passes**

Run:
```bash
rm -rf build/t1 && cmake -S . -B build/t1 >/dev/null && cmake --build build/t1 --target _core 2>&1 | tail -3
```
Expected: build SUCCEEDS; the guard prints `assert_kernels_registered: XNNPACK + quantized + optimized registration TUs present` and does not fatal-error on the reference TU.

Confirm the registrar symbol is actually present:
```bash
nm "$(find build/t1 -name '_core*.so' | head -1)" | grep -c '_GLOBAL__sub_I_etnp_reference_ops.cpp'
```
Expected: `1`.

- [ ] **Step 7: Commit**

```bash
git add kernels/reference/etnp_reference_ops.cpp cmake/Kernels.cmake cmake/assert_kernels_registered.cmake CMakeLists.txt
git commit -m "feat: custom-kernel seam with bundled reference op (etnp::triple.out)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Native registration + dispatch test

**Files:**
- Create: `native_tests/kernel_registration_test.cpp`
- Modify: `native_tests/CMakeLists.txt` (include `Kernels.cmake`; add the plain test target + an ASan leak variant; whole-archive `etnp_kernels`; run the same nm-guard as POST_BUILD)
- Modify: `.github/workflows/qa-gate.yml` (build + run the test, and the leak variant under LSan)

**Interfaces:**
- Consumes: target `etnp_kernels` and var `ETNP_KERNEL_EXPECT_TUS` from `cmake/Kernels.cmake` (Task 1); registered op `etnp::triple.out` (Task 1).
- Consumes: ExecuTorch registry API from `executorch/runtime/kernel/operator_registry.h` — `registry_has_op_function(const char*)`, `get_op_function_from_registry(const char*, Span<...>)` returning `Result<OpFunction>` where `OpFunction = void(*)(KernelRuntimeContext&, Span<EValue*>)`; and `executorch::extension::make_tensor_ptr` from `executorch/extension/tensor/tensor_ptr.h`.
- Produces: executable `kernel_registration_test` (registration/dispatch gate) returning `0` on success, non-zero on any mismatch; and `kernel_leak_test` — the same source built under AddressSanitizer/LeakSanitizer as the exemplar for leak-checking an injected kernel's dispatch + body allocations.

**Note on leak coverage scope:** the reference op does not allocate, so `kernel_leak_test` catches nothing *for it* — its value is as the copyable ASan target a consumer points at their own allocating kernel (e.g. an LSTM cell's workspace). Kernel *registration* is a static-init append to the global registry (a still-reachable allocation LSan ignores by design), so only per-invocation allocations in the kernel body surface. This is deliberately the direct-invoke test rather than the `.pte`-driven `leak_harness`: exercising a custom op through `leak_harness` would require a custom-op `.pte`, i.e. the de-scoped export path.

- [ ] **Step 1: Write the failing test (native)**

Create `native_tests/kernel_registration_test.cpp`:

```cpp
// Proves the bundled reference kernel is registered in the operator registry and
// dispatches correctly — the runtime half of the custom-kernel seam. No ExecuTorch
// export toolchain involved: we look the op up by name and invoke it directly.
#include <cstdio>
#include <vector>

#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::extension::make_tensor_ptr;
using executorch::runtime::EValue;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::OpFunction;
using executorch::runtime::Span;

int main() {
  executorch::runtime::runtime_init();

  static constexpr char kOp[] = "etnp::triple.out";

  if (!executorch::runtime::registry_has_op_function(kOp)) {
    std::fprintf(stderr, "FAIL: %s not present in operator registry\n", kOp);
    return 1;
  }

  // Empty kernel-key meta selects the single unspecialized kernel we registered.
  auto fn_result = executorch::runtime::get_op_function_from_registry(kOp, {});
  if (!fn_result.ok()) {
    std::fprintf(stderr, "FAIL: could not fetch %s from registry\n", kOp);
    return 1;
  }
  OpFunction fn = fn_result.get();

  std::vector<float> in_data = {1.0f, 2.0f, 3.0f};
  std::vector<float> out_data = {0.0f, 0.0f, 0.0f};
  auto in = make_tensor_ptr({3}, in_data.data());
  auto out = make_tensor_ptr({3}, out_data.data());

  EValue in_ev(*in);
  EValue out_ev(*out);
  EValue* args[] = {&in_ev, &out_ev};
  Span<EValue*> span(args, 2);

  KernelRuntimeContext ctx;
  fn(ctx, span);

  for (size_t i = 0; i < out_data.size(); ++i) {
    const float expected = in_data[i] * 3.0f;
    if (out_data[i] != expected) {
      std::fprintf(
          stderr, "FAIL: out[%zu]=%f expected %f\n", i, out_data[i], expected);
      return 1;
    }
  }
  std::printf("OK: etnp::triple.out registered and computed 3x\n");
  return 0;
}
```

- [ ] **Step 2: Add the test target to native_tests**

In `native_tests/CMakeLists.txt`, after `find_package(ExecuTorch CONFIG REQUIRED)` (`native_tests/CMakeLists.txt:12`), add:

```cmake
include(${CMAKE_SOURCE_DIR}/../cmake/Kernels.cmake)
```

At the end of the file, add:

```cmake
# Registration + dispatch gate for the custom-kernel seam. Whole-archives
# etnp_kernels so the registrar TUs survive, then the shared nm-guard asserts
# they did (covers any ETNP_EXTRA_KERNEL_SOURCES injected into this build too).
add_executable(kernel_registration_test kernel_registration_test.cpp)
target_link_libraries(kernel_registration_test PRIVATE
  ${ETNP_HARNESS_LIBS} "$<LINK_LIBRARY:WHOLE_ARCHIVE,etnp_kernels>")

add_custom_command(TARGET kernel_registration_test POST_BUILD
  COMMAND ${CMAKE_COMMAND} -DSO=$<TARGET_FILE:kernel_registration_test> -DNM=nm
          "-DEXTRA_TUS=${ETNP_KERNEL_EXPECT_TUS}"
          -P ${CMAKE_SOURCE_DIR}/../cmake/assert_kernels_registered.cmake
  VERBATIM)

# Leak variant (ASan/LSan): the exemplar a consumer copies to leak-check an
# allocating custom kernel. Same source, invoked directly (no .pte, so no export
# dependency); catches per-invocation allocations the kernel body fails to free.
# Separate binary because ASan cannot combine with the race harness's TSan.
add_executable(kernel_leak_test kernel_registration_test.cpp)
target_compile_options(kernel_leak_test PRIVATE -fsanitize=address -g -O1)
target_link_options(kernel_leak_test PRIVATE -fsanitize=address)
target_link_libraries(kernel_leak_test PRIVATE
  ${ETNP_HARNESS_LIBS} "$<LINK_LIBRARY:WHOLE_ARCHIVE,etnp_kernels>")
```

- [ ] **Step 3: Run to verify it builds, the guard passes, dispatch is correct, and it is leak-clean**

```bash
cmake -S native_tests -B build/nt >/dev/null && cmake --build build/nt --target kernel_registration_test kernel_leak_test 2>&1 | tail -3
./build/nt/kernel_registration_test; echo "exit=$?"
ASAN_OPTIONS=detect_leaks=1 ./build/nt/kernel_leak_test; echo "leak_exit=$?"
```
Expected: build succeeds (guard confirms `_GLOBAL__sub_I_etnp_reference_ops.cpp` present); `kernel_registration_test` prints `OK: etnp::triple.out registered and computed 3x` and `exit=0`; `kernel_leak_test` prints the same `OK` line, reports no leaks, and `leak_exit=0`.

- [ ] **Step 4: Run the qa-gate wiring locally (sanity) and add the CI step**

In `.github/workflows/qa-gate.yml`, after the `Build + run leak harness` step (`qa-gate.yml:59-62`), add:

```yaml
      - name: Build + run custom-kernel registration test
        run: |
          cmake -S native_tests -B build/kernels && cmake --build build/kernels --target kernel_registration_test kernel_leak_test
          ./build/kernels/kernel_registration_test
          ASAN_OPTIONS=detect_leaks=1 ./build/kernels/kernel_leak_test
```

- [ ] **Step 5: Commit**

```bash
git add native_tests/kernel_registration_test.cpp native_tests/CMakeLists.txt .github/workflows/qa-gate.yml
git commit -m "test: prove reference kernel registers, dispatches, and is leak-clean

Adds a registry-level registration/dispatch gate plus an ASan/LSan leak
variant (kernel_leak_test) as the exemplar for leak-checking injected kernels.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Consumer injection example + documentation

**Files:**
- Create: `examples/custom_kernels/negate_op.cpp`
- Create: `docs/custom-kernels.md`
- Modify: `.github/workflows/qa-gate.yml` (build native_tests with the example injected; prove the guard catches it)
- Modify: `README.md` (link to the new doc)

**Interfaces:**
- Consumes: the `ETNP_EXTRA_KERNEL_SOURCES` cache var and self-verifying nm-guard from Tasks 1–2.
- Produces: a worked example op `etnp::negate.out` and the documented seam contract.

- [ ] **Step 1: Write the worked example kernel**

Create `examples/custom_kernels/negate_op.cpp`:

```cpp
// Worked example of a consumer-supplied custom kernel injected via
// ETNP_EXTRA_KERNEL_SOURCES (see docs/custom-kernels.md). Not compiled into the
// default build. Registers etnp::negate.out computing out = -input (float32).
#include <cstddef>

#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace etnp {
namespace {

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::KernelRuntimeContext;

Tensor& negate_out(KernelRuntimeContext& ctx, const Tensor& input, Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::resize_tensor(out, input.sizes()) == Error::Ok,
      InvalidArgument,
      out);
  ET_KERNEL_CHECK(
      ctx, input.scalar_type() == ScalarType::Float, InvalidArgument, out);

  const float* in_data = input.const_data_ptr<float>();
  float* out_data = out.mutable_data_ptr<float>();
  for (size_t i = 0; i < static_cast<size_t>(input.numel()); ++i) {
    out_data[i] = -in_data[i];
  }
  return out;
}

} // namespace
} // namespace etnp

EXECUTORCH_LIBRARY(etnp, "negate.out", etnp::negate_out);
```

- [ ] **Step 2: Verify injection works (the test for the seam contract)**

Run — inject the example into a native_tests build and let the self-verifying guard require its TU:

```bash
cmake -S native_tests -B build/inject \
  -DETNP_EXTRA_KERNEL_SOURCES="$PWD/examples/custom_kernels/negate_op.cpp" >/dev/null
cmake --build build/inject --target kernel_registration_test 2>&1 | tail -3
nm "$(find build/inject -name 'kernel_registration_test' -type f | head -1)" \
  | grep -c '_GLOBAL__sub_I_negate_op.cpp'
```
Expected: build SUCCEEDS (the guard, fed `ETNP_KERNEL_EXPECT_TUS` now including `_GLOBAL__sub_I_negate_op.cpp`, confirms it survived the link); the `nm | grep -c` prints `1`.

- [ ] **Step 3: Add the injection check to CI**

In `.github/workflows/qa-gate.yml`, after the registration-test step added in Task 2, add:

```yaml
      - name: Verify custom-kernel injection seam
        run: |
          cmake -S native_tests -B build/inject \
            -DETNP_EXTRA_KERNEL_SOURCES="$PWD/examples/custom_kernels/negate_op.cpp"
          cmake --build build/inject --target kernel_registration_test
          # The POST_BUILD guard fatal-errors if the injected registrar was dropped;
          # a successful build is the assertion. Double-check the symbol is present:
          nm "$(find build/inject -name kernel_registration_test -type f | head -1)" \
            | grep -q '_GLOBAL__sub_I_negate_op.cpp'
```

- [ ] **Step 4: Write the seam documentation**

Create `docs/custom-kernels.md`:

```markdown
# Custom kernels

This is a shared, general-purpose ExecuTorch runtime. When a PyTorch op does not
decompose efficiently to ExecuTorch core ops, you can supply a hand-written
kernel and compile it into the extension. Real kernels (e.g. `nn.LSTM`) are
**not** bundled here — they live in your build via the injection seam below. The
bundled `etnp::triple.out` is a trivial reference/exemplar kept CI-tested so the
build wiring can't rot.

## The contract

1. **Write an out-variant kernel** with the ExecuTorch calling convention:
   `Tensor& your_op(KernelRuntimeContext&, const Tensor& in, ..., Tensor& out)`.
   See `kernels/reference/etnp_reference_ops.cpp` for the minimal shape.
2. **Register it** with `EXECUTORCH_LIBRARY(ns, "op_name.out", your_op)`. This
   registers the exact string `"ns::op_name.out"`.
3. **The registered name must match the name serialized into your `.pte`.** How a
   model that uses a custom op is exported is documented in the upstream
   ExecuTorch custom-ops guide and is out of scope here.

## Building your kernel in

Pass absolute source paths at configure time — no fork required:

    cmake -S . -B build \
      -DETNP_EXTRA_KERNEL_SOURCES="/abs/path/my_kernels.cpp;/abs/path/more.cpp"

Your sources are compiled into the `etnp_kernels` archive and **whole-archived**
into the module, so their static-init registrars survive the linker. A post-link
guard (`cmake/assert_kernels_registered.cmake`) fails the **build** — not model
load — if any registrar is dropped. A worked example is in
`examples/custom_kernels/negate_op.cpp`.

To omit the bundled reference op from a lean production build, configure with
`-DETNP_BUILD_REFERENCE_KERNEL=OFF`.

## Leak-checking a kernel

If your kernel allocates (workspace, state, scratch buffers), leak-check it under
AddressSanitizer/LeakSanitizer. `native_tests/kernel_registration_test.cpp` is
built both plain (`kernel_registration_test`) and under ASan (`kernel_leak_test`);
copy that target for your op, or point it at your kernel by name, and run:

    ASAN_OPTIONS=detect_leaks=1 ./build/nt/kernel_leak_test

Prefer allocating scratch via the `KernelRuntimeContext` temp allocator over raw
`malloc`/`new`; kernels that own raw allocations and skip freeing them surface
here as leaks. (Registration itself is a one-time static allocation LSan ignores,
so only per-invocation allocations in the kernel body are flagged.)
```

- [ ] **Step 5: Link the doc from the README**

In `README.md`, add one line under the most appropriate existing section (features or usage):

```markdown
- **Custom kernels:** compile your own ExecuTorch kernels into the runtime — see [docs/custom-kernels.md](docs/custom-kernels.md).
```

- [ ] **Step 6: Commit**

```bash
git add examples/custom_kernels/negate_op.cpp docs/custom-kernels.md .github/workflows/qa-gate.yml README.md
git commit -m "docs: document custom-kernel injection seam with worked example

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Deferred / out of scope

- **Full model-level `.pte` fixture for a custom op.** Generating one requires
  defining the op in the ExecuTorch export venv (functional + out + meta variants)
  so `to_edge_transform_and_lower` emits `etnp::triple.out` — i.e. the custom-op
  *export* path, which is documented upstream and explicitly out of scope for this
  repo. Task 2's registry-level test proves runtime resolve+dispatch without it.
- **Dynamically loadable kernel plugins (dlopen a kernel `.so` at runtime).** A
  larger design (load API, registry lifetime, an ABI-stability promise across the
  abi3 boundary). The static-compile/inject seam here is a strict subset, so it
  does not foreclose that work.
- **The `nn.LSTM` kernel itself.** Belongs in a consumer build via the seam; if
  written, prefer a small fused `lstm_cell` primitive (gates for one timestep)
  and let the graph carry the sequence loop, rather than a monolithic
  `aten::lstm` out-variant.

## Self-review notes

- **Spec coverage:** reference op (T1) · injection seam `ETNP_EXTRA_KERNEL_SOURCES` (T1 mechanism, T3 worked example + CI) · CI proof (T1 build guard, T2 dispatch test + ASan leak variant, T3 injection check) · leak-check exemplar for injected kernels (T2 `kernel_leak_test`, documented in T3) · docs (T3). All scoped items covered.
- **Type consistency:** op name `etnp::triple.out` and TU symbol `_GLOBAL__sub_I_etnp_reference_ops.cpp` are used identically in the kernel, `Kernels.cmake` derivation, the guard input, and the native test. `ETNP_KERNEL_EXPECT_TUS` / `ETNP_EXTRA_KERNEL_SOURCES` / `ETNP_BUILD_REFERENCE_KERNEL` names match across `Kernels.cmake`, both `CMakeLists.txt` files, and the guard's `EXTRA_TUS` parameter.
- **Known adjustment point:** the exact overloads of `registry_has_op_function` / `get_op_function_from_registry` (empty kernel-key meta `{}`) should be confirmed against `operator_registry.h` at implementation time; if the single-arg form is unavailable, pass an empty `Span`/`ArrayRef` for the meta argument.
```
