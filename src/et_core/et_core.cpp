#include "et_core/et_core.h"

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/executor/method_meta.h>

namespace etnp {
using executorch::extension::BufferDataLoader;
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
