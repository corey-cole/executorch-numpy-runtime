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
