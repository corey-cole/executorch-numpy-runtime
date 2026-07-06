#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>
#include "et_core/et_core.h"
#include "binding/dtype_map.h"

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
  m.def("_np_to_st", [](const std::string& k, size_t sz) {
    return numpy_code_to_scalar_type(k.at(0), sz);
  });
  m.def("_st_to_np", [](int8_t st) {
    auto d = scalar_type_to_numpy(st);
    return std::make_pair(std::string(1, d.kind), d.itemsize);
  });
}
