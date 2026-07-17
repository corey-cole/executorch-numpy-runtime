import executorch_numpy_runtime as en
from executorch_numpy_runtime import _core
from conftest import model_or_skip


def test_method_meta_shapes():
    rt = _core.load_path(model_or_skip("add.pte"))
    meta = rt.method_meta("forward")
    assert meta["num_inputs"] == 2
    assert meta["num_outputs"] == 1
    assert meta["inputs"][0]["scalar_type"] == 6  # Float


# Load-bearing on Windows: there is no nm/symbol guard there (see the spec's D5), so this
# runtime assertion is the only check that XNNPACK's static-init registrar survived the
# link. cibuildwheel runs this suite against the built wheel, which is what makes it a gate.
def test_runtime_info_reports_version_and_backends():
    info = en.runtime_info()
    assert info["executorch_version"] == "1.3.1"
    assert "XnnpackBackend" in info["backends"]
    assert info["bfloat16"] == "uint16-passthrough"


def test_backend_available():
    assert _core.backend_available("XnnpackBackend") is True
    assert _core.backend_available("CoreMLBackend") is False
