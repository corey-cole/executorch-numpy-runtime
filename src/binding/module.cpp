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

// Allocate a fresh, owning numpy array copied from arena memory (never a view).
//
// We deliberately do NOT pass an `owner`/base object here. nanobind's numpy
// export path (nb_ndarray.cpp: ndarray_export()) only returns a zero-copy view
// when the ndarray_handle has a non-null owner or `self`; with both null it
// calls `numpy.copy()` on a transient view of `v.data`, which makes numpy
// allocate and OWN a fresh buffer and memcpy the bytes in -- exactly the
// mandatory-copy, `OWNDATA=True` semantics this binding must guarantee.
// (Verified by reading the installed nanobind 2.13.0 source, not assumed.)
static nb::object copy_out(const etnp::OutputView& v) {
  // Scalar EValues (from constant_methods) surface as native Python scalars,
  // matching executorch.runtime.Method.execute -> [128] / [2.5] / [True].
  switch (v.kind) {
    case etnp::OutputKind::Int:    return nb::int_(v.int_val);
    case etnp::OutputKind::Bool:   return nb::bool_(v.int_val != 0);
    case etnp::OutputKind::Double: return nb::float_(v.double_val);
    case etnp::OutputKind::Tensor: break;  // fall through to the ndarray path
  }
  etnp::NpDtype d = etnp::scalar_type_to_numpy(v.scalar_type);
  size_t ndim = v.shape.size();
  std::vector<size_t> shape(v.shape.begin(), v.shape.end());
  nb::dlpack::dtype dt{
      /*code*/ (uint8_t)(d.kind == 'f' ? nb::dlpack::dtype_code::Float :
                         d.kind == 'i' ? nb::dlpack::dtype_code::Int :
                         d.kind == 'u' ? nb::dlpack::dtype_code::UInt :
                                         nb::dlpack::dtype_code::Bool),
      /*bits*/ (uint8_t)(d.itemsize * 8), /*lanes*/ 1};
  // Non-owning transient view into arena memory; nb::cast's default
  // rv_policy (automatic_reference) copies it into a new numpy array because
  // no owner/self is attached to the handle below.
  nb::ndarray<nb::numpy> view(const_cast<void*>(v.data), ndim, shape.data(),
                               /*owner=*/nb::handle(), /*strides=*/nullptr, dt);
  return nb::cast(view);
}

// Payload for the exception translator below: Python classes looked up once
// at module-init time and kept alive for the process lifetime (leaked
// intentionally -- same lifetime as the module itself).
struct ErrorClasses {
  nb::object base, load, backend, opnf, exec;
};

NB_MODULE(_core, m) {
  m.attr("__et_version__") = ETNP_ET_VERSION;
  m.attr("__kernel_libs__") = ETNP_KERNEL_LIBS;

  // Map etnp::EtException -> the matching executorch_numpy_runtime.errors
  // subclass. Confirmed against the installed nanobind 2.13.0
  // (nanobind/nb_error.h): nb::register_exception_translator(exception_translator,
  // void* payload) takes a free function pointer (no captures) plus an
  // opaque payload pointer, matching the brief's mechanism exactly -- no
  // fallback to a try/catch-per-lambda wrapper was needed.
  nb::module_ errmod = nb::module_::import_("executorch_numpy_runtime.errors");
  auto* classes = new ErrorClasses{
      /*base*/    errmod.attr("ExecuTorchError"),
      /*load*/    errmod.attr("ProgramLoadError"),
      /*backend*/ errmod.attr("BackendNotAvailable"),
      /*opnf*/    errmod.attr("OperatorNotFound"),
      /*exec*/    errmod.attr("ExecutionError"),
  };
  nb::register_exception_translator(
      [](const std::exception_ptr& p, void* payload) {
        auto* c = static_cast<ErrorClasses*>(payload);
        try {
          std::rethrow_exception(p);
        } catch (const etnp::EtException& e) {
          nb::object cls;
          switch (e.error.kind) {
            case etnp::ErrorKind::Load:            cls = c->load; break;
            case etnp::ErrorKind::BackendMissing:  cls = c->backend; break;
            case etnp::ErrorKind::OperatorMissing: cls = c->opnf; break;
            case etnp::ErrorKind::Execution:       cls = c->exec; break;
            case etnp::ErrorKind::DtypeUnmapped:   cls = c->base; break;
            default:                               cls = c->base; break;
          }
          std::string msg = e.error.message;
          if (!e.error.detail.empty()) msg += " [" + e.error.detail + "]";
          PyErr_SetString(cls.ptr(), msg.c_str());
        }
      },
      classes);

  nb::class_<Runtime>(m, "_Runtime")
      .def("method_names", &Runtime::method_names)
      .def("run_method", [](Runtime& self, const std::string& name, nb::list arrs) {
        std::vector<HeldInput> held;
        std::vector<etnp::InputDesc> descs;
        held.reserve(arrs.size());
        for (auto a : arrs) { held.push_back(make_held_input(a)); }
        for (auto& h : held) descs.push_back(h.desc);
        etnp::ForwardResult fr = [&] {
          nb::gil_scoped_release nogil;   // release AFTER inputs built; touch no py objects
          return self.run_method(name, descs);
        }();
        nb::list out;
        for (const auto& v : fr.outputs()) out.append(copy_out(v));
        return out;  // held[] (keepalives) drop here, after outputs copied
      })
      .def("method_meta", [](Runtime& self, const std::string& n) {
        auto m = self.method_meta(n);
        auto to_list = [](const std::vector<etnp::TensorMeta>& v) {
          nb::list l;
          for (auto& t : v) {
            nb::dict d;
            d["scalar_type"] = t.scalar_type;
            d["shape"] = nb::cast(t.shape);
            l.append(d);
          }
          return l;
        };
        nb::dict d;
        d["num_inputs"] = m.inputs.size();
        d["num_outputs"] = m.outputs.size();
        d["inputs"] = to_list(m.inputs);
        d["outputs"] = to_list(m.outputs);
        return d;
      });

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

  m.def("registered_backends", &etnp::registered_backends);
  m.def("backend_available", &etnp::backend_available);
  m.def("operator_names", &etnp::operator_names);
}
