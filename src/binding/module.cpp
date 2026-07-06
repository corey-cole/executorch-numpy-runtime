#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>
#include <vector>
#include "et_core/et_core.h"
#include "binding/dtype_map.h"

namespace nb = nanobind;
using namespace etnp;

struct HeldInput {
  nb::object keepalive;   // pins the (possibly copied) buffer for the whole call
  etnp::InputDesc desc;
};

// Cast to a C-contiguous numpy array (copies if needed), map dtype, pin a ref.
//
// nb::cast<T>(obj) defaults to convert=true. When the ndarray caster
// (nanobind 2.13.0, nb_ndarray.cpp: ndarray_import()/convert_array()) finds
// an order mismatch (non-contiguous src vs. requested nb::c_contig), and
// `convert` is set, it calls numpy's `.astype(dtype, order)` on the source
// and retries the import against the *converted* array -- i.e. it COPIES,
// it does not reject. Confirmed by reading the installed nanobind source
// (not assumed from docs). This satisfies the "copy non-contiguous input,
// never reject" contract.
static HeldInput make_held_input(nb::handle obj, bool* was_contig = nullptr) {
  // ascontiguousarray semantics: nb::c_contig cast copies non-contiguous input.
  auto arr = nb::cast<nb::ndarray<nb::c_contig, nb::device::cpu>>(obj);
  if (was_contig) {
    auto orig = nb::cast<nb::ndarray<nb::device::cpu>>(obj);
    *was_contig = (orig.data() == arr.data());
  }
  char kind;
  switch (arr.dtype().code) {
    case (uint8_t)nb::dlpack::dtype_code::Float: kind = 'f'; break;
    case (uint8_t)nb::dlpack::dtype_code::Int:   kind = 'i'; break;
    case (uint8_t)nb::dlpack::dtype_code::UInt:  kind = 'u'; break;
    case (uint8_t)nb::dlpack::dtype_code::Bool:  kind = 'b'; break;
    default:
      throw etnp::EtException(etnp::EtError{etnp::ErrorKind::DtypeUnmapped,
                 "Unsupported numpy dtype code", ""});
  }
  int8_t st = etnp::numpy_code_to_scalar_type(kind, arr.dtype().bits / 8);
  std::vector<int64_t> shape(arr.shape_ptr(), arr.shape_ptr() + arr.ndim());
  HeldInput h;
  h.keepalive = nb::cast(arr);  // hold the contiguous array (owning nb::object)
  h.desc = etnp::InputDesc{arr.data(), std::move(shape), st};
  return h;
}

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
  m.def("_contig_info", [](nb::handle obj) {
    bool was = false;
    HeldInput h = make_held_input(obj, &was);
    std::vector<int64_t> s = h.desc.shape;
    return std::make_tuple(was, h.desc.scalar_type,
                            nb::tuple(nb::cast(s)));
  });
}
