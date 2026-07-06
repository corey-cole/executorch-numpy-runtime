#include "binding/dtype_map.h"
#include "et_core/et_core.h"
#include <executorch/runtime/core/portable_type/scalar_type.h>

namespace etnp {
using executorch::runtime::etensor::ScalarType;

static EtException bad_dtype(const std::string& what) {
  return EtException({ErrorKind::DtypeUnmapped,
      "Unsupported dtype: " + what, ""});
}

int8_t numpy_code_to_scalar_type(char kind, size_t itemsize) {
  if (kind == 'f' && itemsize == 4) return static_cast<int8_t>(ScalarType::Float);
  if (kind == 'f' && itemsize == 8) return static_cast<int8_t>(ScalarType::Double);
  if (kind == 'f' && itemsize == 2) return static_cast<int8_t>(ScalarType::Half);
  if (kind == 'i' && itemsize == 8) return static_cast<int8_t>(ScalarType::Long);
  if (kind == 'i' && itemsize == 4) return static_cast<int8_t>(ScalarType::Int);
  if (kind == 'i' && itemsize == 2) return static_cast<int8_t>(ScalarType::Short);
  if (kind == 'i' && itemsize == 1) return static_cast<int8_t>(ScalarType::Char);
  if (kind == 'u' && itemsize == 1) return static_cast<int8_t>(ScalarType::Byte);
  if (kind == 'b' && itemsize == 1) return static_cast<int8_t>(ScalarType::Bool);
  if (kind == 'u' && itemsize == 2) return static_cast<int8_t>(ScalarType::BFloat16);  // explicit opt-in bits
  throw bad_dtype(std::string(1, kind) + std::to_string(itemsize));
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
