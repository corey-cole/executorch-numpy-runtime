#include "et_core/et_core.h"

#include <cstring>
#include <mutex>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/executor/method_meta.h>

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
  auto res = state_->module->method_names();
  if (res.error() != Error::Ok)
    throw EtException({ErrorKind::Load, "method_names() failed", ""});
  std::vector<std::string> out(res->begin(), res->end());
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

}  // namespace etnp
