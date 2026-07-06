#include <nanobind/nanobind.h>
namespace nb = nanobind;

NB_MODULE(_core, m) {
  m.attr("__et_version__") = ETNP_ET_VERSION;
}
