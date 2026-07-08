"""Runtime handling of ExecuTorch `constant_methods`.

Fixture: tools/export_const_methods_probe.py -> const_methods.pte, with four
zero-input constant methods plus an ordinary forward():
  - get_scalar() -> 128            (int-valued   -> Python int)
  - get_double() -> 2.5            (double-valued -> Python float)
  - get_bool()   -> True           (bool-valued  -> Python bool)
  - get_tensor() -> [1, 2, 3]      (tensor-valued -> np.ndarray)

Scalar constant methods emit non-Tensor EValues. run_method surfaces them as
native Python scalars, matching ExecuTorch's own pybind runtime
(executorch.runtime.Method.execute), which returns [128] / [2.5] / [True].
"""
import numpy as np
from executorch_numpy_runtime import _core
from conftest import model_or_skip


def _load():
    return _core.load_path(model_or_skip("const_methods.pte"))


def test_constant_methods_are_listed():
    names = _load().method_names()
    assert {"get_scalar", "get_double", "get_bool", "get_tensor"} <= set(names)


def test_tensor_constant_method_returns_array():
    out = _load().run_method("get_tensor", [])  # constant methods take no inputs
    assert len(out) == 1
    assert isinstance(out[0], np.ndarray)
    assert out[0].dtype == np.int64
    np.testing.assert_array_equal(out[0], np.array([1, 2, 3], dtype=np.int64))


def test_int_constant_method_returns_python_int():
    out = _load().run_method("get_scalar", [])
    assert out == [128]
    assert type(out[0]) is int  # native Python int, not np.ndarray / np.int64


def test_double_constant_method_returns_python_float():
    out = _load().run_method("get_double", [])
    assert out == [2.5]
    assert type(out[0]) is float


def test_bool_constant_method_returns_python_bool():
    out = _load().run_method("get_bool", [])
    assert out == [True]
    assert type(out[0]) is bool  # bool before int: bool is a subclass of int
