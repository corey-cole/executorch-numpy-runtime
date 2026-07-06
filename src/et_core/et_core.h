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

// Output view: points into ForwardResult-owned storage (bytes copied out of
// ExecuTorch's arena under the exec lock). Valid for the ForwardResult's life.
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
struct ForwardState;   // pimpl (owns copied-out output buffers)

// RAII owner of output buffers; dropping it ends OutputView lifetime.
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
