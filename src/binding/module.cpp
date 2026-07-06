#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>
#include "et_core/et_core.h"

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
}
