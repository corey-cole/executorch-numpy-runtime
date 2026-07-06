#include "et_core/et_core.h"

#include <cstring>
#include <mutex>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/kernel/operator_registry.h>

namespace etnp {
using executorch::extension::BufferDataLoader;
using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::EValue;
using executorch::runtime::Error;
using executorch::aten::ScalarType;

struct ForwardState {
  // OWNED output buffers: bytes are copied out of the Module's memory-planned
  // arena (under the exec lock) so OutputViews never alias live arena memory.
  std::vector<std::vector<uint8_t>> storage;  // one owned buffer per output
  std::vector<OutputView> views;              // each .data -> storage[i].data()
};

ForwardResult::ForwardResult(std::unique_ptr<ForwardState> s) : state_(std::move(s)) {}
ForwardResult::~ForwardResult() = default;
ForwardResult::ForwardResult(ForwardResult&&) noexcept = default;
ForwardResult& ForwardResult::operator=(ForwardResult&&) noexcept = default;
const std::vector<OutputView>& ForwardResult::outputs() const { return state_->views; }

struct RuntimeState {
  std::string owned_bytes;                 // non-empty for buffer loads; keeps data alive
  std::unique_ptr<Module> module;
  std::mutex exec_mutex;                    // serializes execute + copy-out per Runtime
};

// Classify a Module::load() failure. In ExecuTorch 1.3.1, Module::load()
// surfaces a single runtime::Error and does not expose which delegate (if
// any) was being resolved, so we cannot yet distinguish "corrupt/malformed/
// version-incompatible file" from "valid file lowered for a backend this
// runtime does not link" (e.g. CoreML/QNN/XNNPACK when only the CPU
// portable kernels are linked in). Robust discrimination needs the failing
// delegate id, which is only obtainable via registered_backends()
// cross-checking (Task 8) or richer error plumbing from ExecuTorch itself.
// Until then every load failure is classified as ErrorKind::Load and
// surfaces as ProgramLoadError; a non-CPU .pte will therefore also raise
// ProgramLoadError rather than BackendNotAvailable. This is documented as
// an accepted coarse behavior (see task-7 brief).
static EtException classify_load_failure(const std::string& what) {
  return EtException({ErrorKind::Load,
      "Failed to load .pte (corrupt, or exported by an ExecuTorch version "
      "other than the one this runtime targets, 1.3.1): " + what, ""});
}

std::unique_ptr<Runtime> Runtime::load_path(const std::string& path) {
  auto st = std::make_unique<RuntimeState>();
  st->module = std::make_unique<Module>(path);
  if (st->module->load() != Error::Ok) throw classify_load_failure(path);
  return std::make_unique<Runtime>(std::move(st));
}

// Note: ExecuTorch 1.3.1's Module has no `Module(const void*, size_t)` ctor
// (see extension/module/module.h) -- only file-path ctors and a ctor taking
// std::unique_ptr<runtime::DataLoader>. Buffer loads instead go through
// BufferDataLoader (extension/data_loader/buffer_data_loader.h), a DataLoader
// that wraps a pre-allocated buffer without copying it, handed to Module via
// that data-loader ctor (Module takes ownership of the loader). RuntimeState
// still separately owns `owned_bytes` (the actual buffer contents) and is
// declared/destroyed after `module`, so the backing memory outlives both the
// BufferDataLoader and the Module for the Runtime's whole lifetime regardless
// of which ctor is used.
std::unique_ptr<Runtime> Runtime::load_buffer(std::string bytes) {
  auto st = std::make_unique<RuntimeState>();
  st->owned_bytes = std::move(bytes);
  auto loader = std::make_unique<BufferDataLoader>(
      st->owned_bytes.data(), st->owned_bytes.size());
  st->module = std::make_unique<Module>(std::move(loader));
  if (st->module->load() != Error::Ok) throw classify_load_failure("<buffer>");
  return std::make_unique<Runtime>(std::move(st));
}

Runtime::Runtime(std::unique_ptr<RuntimeState> s) : state_(std::move(s)) {}
Runtime::~Runtime() = default;

std::vector<std::string> Runtime::method_names() const {
  // Module::method_names() lazily loads/mutates the Module's internal
  // methods_ map, which is not internally synchronized; take the same lock
  // run_method() uses so this can't race a concurrent execute() on this
  // Runtime. Never calls run_method(), so no self-deadlock risk.
  std::lock_guard<std::mutex> guard(state_->exec_mutex);
  auto res = state_->module->method_names();
  if (res.error() != Error::Ok)
    throw EtException({ErrorKind::Load, "method_names() failed", ""});
  std::vector<std::string> out(res->begin(), res->end());
  return out;
}

// Uses executorch::runtime::Result<MethodMeta> Module::method_meta(name)
// (extension/module/module.h) and TensorInfo::sizes()/scalar_type()
// (runtime/executor/method_meta.h) -- both verified against the installed
// 1.3.1 headers. Non-tensor inputs/outputs (Int/Bool/Double/String tags)
// have no TensorInfo, so input_tensor_meta()/output_tensor_meta() returns an
// error for them; those slots get TensorMeta{{}, -1} placeholders.
MethodMeta Runtime::method_meta(const std::string& name) const {
  // Same rationale as method_names(): Module::method_meta() lazily loads and
  // mutates internal state, so lock exec_mutex to avoid racing execute().
  std::lock_guard<std::mutex> guard(state_->exec_mutex);
  auto mm = state_->module->method_meta(name);
  if (mm.error() != Error::Ok)
    throw EtException({ErrorKind::Load, "method_meta('" + name + "') failed", ""});
  etnp::MethodMeta out;
  out.inputs.reserve(mm->num_inputs());
  for (size_t i = 0; i < mm->num_inputs(); ++i) {
    auto info = mm->input_tensor_meta(i);
    if (info.error() == Error::Ok) {
      TensorMeta tm;
      tm.scalar_type = static_cast<int8_t>(info->scalar_type());
      auto sizes = info->sizes();
      tm.shape.assign(sizes.begin(), sizes.end());
      out.inputs.push_back(std::move(tm));
    } else {
      out.inputs.push_back(TensorMeta{{}, -1});
    }
  }
  out.outputs.reserve(mm->num_outputs());
  for (size_t i = 0; i < mm->num_outputs(); ++i) {
    auto info = mm->output_tensor_meta(i);
    if (info.error() == Error::Ok) {
      TensorMeta tm;
      tm.scalar_type = static_cast<int8_t>(info->scalar_type());
      auto sizes = info->sizes();
      tm.shape.assign(sizes.begin(), sizes.end());
      out.outputs.push_back(std::move(tm));
    } else {
      out.outputs.push_back(TensorMeta{{}, -1});
    }
  }
  return out;
}

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
  // A single ExecuTorch Module is NOT thread-safe: execute() writes into
  // memory-planned arena buffers that are reused across calls, and the output
  // tensors alias that arena. The lock must therefore cover BOTH execute() and
  // the copy-out of the arena bytes, so a second thread cannot clobber the
  // arena before we have copied this call's outputs into owned storage.
  auto fs = std::make_unique<ForwardState>();
  {
    std::lock_guard<std::mutex> guard(state_->exec_mutex);
    auto result = state_->module->execute(name, evalues);
    if (result.error() != Error::Ok) {
      throw EtException({ErrorKind::Execution,
          "execute('" + name + "') failed", ""});  // refined in Task 7
    }
    // reserve() so storage.back().data() pointers stay stable across pushes.
    fs->storage.reserve(result->size());
    fs->views.reserve(result->size());
    for (auto& ev : *result) {
      // Defensive: EValue::toTensor() is ET_CHECK_MSG(isTensor(), ...) --
      // an uncatchable et_pal_abort() on a non-tensor EValue, which would
      // crash the whole interpreter instead of raising a Python exception.
      // We could not produce a real non-tensor-output .pte to exercise this
      // via a fixture: torch.export + to_edge_transform_and_lower asserts
      // internally ("graph_output_allocated not set") on graphs whose output
      // is a bare int/scalar (tried returning `a.shape[0]` and
      // `a.sum().item()` under the ExecuTorch 1.3.1 export pipeline), so
      // ordinary export tooling appears unable to emit such a program. The
      // guard is kept anyway as correct, cheap defense against any .pte
      // (however produced) whose method genuinely returns a non-tensor.
      if (!ev.isTensor()) {
        throw EtException({ErrorKind::Execution,
            "method '" + name + "' returned a non-tensor output (unsupported)", ""});
      }
      const auto& t = ev.toTensor();
      std::vector<uint8_t> buf(t.nbytes());
      std::memcpy(buf.data(), t.const_data_ptr(), t.nbytes());  // copy OUT of arena
      std::vector<int64_t> shp(t.sizes().begin(), t.sizes().end());
      fs->storage.push_back(std::move(buf));
      fs->views.push_back(OutputView{std::move(shp),
          static_cast<int8_t>(t.scalar_type()),
          fs->storage.back().data(), fs->storage.back().size()});
    }
  }  // lock released: OutputViews now point into owned storage, never the arena
  return ForwardResult(std::move(fs));
}

// registered_backends()/backend_available(): executorch::runtime::
// get_num_registered_backends()/get_backend_name(size_t) (runtime/backend/
// interface.h), the exact same pair used by upstream's
// get_registered_backend_names() in extension/pybindings/pybindings.cpp.
// Verified against both the installed 1.3.1 header and that upstream call
// site.
std::vector<std::string> registered_backends() {
  std::vector<std::string> names;
  size_t n = executorch::runtime::get_num_registered_backends();
  names.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    auto r = executorch::runtime::get_backend_name(i);
    if (r.ok()) names.emplace_back(*r);
  }
  return names;
}

bool backend_available(const std::string& name) {
  for (auto& n : registered_backends())
    if (n == name) return true;
  return false;
}

// operator_names(): executorch::runtime::get_registered_kernels()
// (runtime/kernel/operator_registry.h) returns Span<const Kernel>; each
// Kernel's name field is `name_` (not `.name` as the task brief's
// unverified snippet guessed -- confirmed against the installed header and
// against upstream's get_operator_names() in pybindings.cpp, which reads
// k.name_ and skips entries where it is nullptr).
std::vector<std::string> operator_names() {
  std::vector<std::string> out;
  auto kernels = executorch::runtime::get_registered_kernels();
  for (const auto& k : kernels) {
    if (k.name_ != nullptr) out.emplace_back(k.name_);
  }
  return out;
}

}  // namespace etnp
