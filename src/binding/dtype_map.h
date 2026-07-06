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
