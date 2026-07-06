import pytest
from executorch_numpy_runtime import _core
from tests.conftest import model_or_skip

def test_load_path_lists_methods():
    rt = _core.load_path(model_or_skip("add.pte"))
    assert "forward" in rt.method_names()

def test_load_buffer_lists_methods():
    with open(model_or_skip("add.pte"), "rb") as f:
        rt = _core.load_buffer(f.read())
    assert "forward" in rt.method_names()

def test_load_bad_path_raises():
    with pytest.raises(Exception):
        _core.load_path("/no/such/file.pte")
