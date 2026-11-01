"""Runtime handling of ExecuTorch `constant_methods`.

Fixture: tools/export_const_methods_probe.py -> const_methods.pte, with seven
zero-input constant methods plus an ordinary forward():
  - get_scalar()  -> 128                        (int-valued    -> Python int)
  - get_double()  -> 2.5                        (double-valued -> Python float)
  - get_bool()    -> True                       (bool-valued   -> Python bool)
  - get_tensor()  -> [1, 2, 3]                  (tensor-valued -> np.ndarray)
  - get_string()  -> "lin.weight"               (string-valued -> Python str)
  - get_strings() -> ["lin.weight", "lin.bias"] (one str output per element)
  - get_unicode() -> "\u00e9\u4e2d"               (non-ASCII str, UTF-8 round trip)

Scalar constant methods emit non-Tensor EValues. run_method surfaces them as
native Python scalars, matching ExecuTorch's own pybind runtime
(executorch.runtime.Method.execute), which returns [128] / [2.5] / [True].
String EValues surface as str. A training program's __et_training_fqn_<m>
method returns its parameter names this way.
"""
from pathlib import Path

import numpy as np
import pytest
from conftest import model_or_skip

import executorch_numpy_runtime as en
from executorch_numpy_runtime import _core
from executorch_numpy_runtime.errors import ExecutionError


def _load():
    return _core.load_path(model_or_skip("const_methods.pte"))


def test_constant_methods_are_listed():
    names = _load().method_names()
    assert {"get_scalar", "get_double", "get_bool", "get_tensor",
            "get_string", "get_strings", "get_unicode"} <= set(names)


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


def test_string_constant_method_returns_python_str():
    out = _load().run_method("get_string", [])
    assert out == ["lin.weight"]
    assert type(out[0]) is str


def test_string_list_constant_method_returns_one_str_per_element():
    out = _load().run_method("get_strings", [])
    assert out == ["lin.weight", "lin.bias"]
    assert all(type(s) is str for s in out)


def test_non_ascii_string_round_trips_as_utf8():
    assert _load().run_method("get_unicode", []) == ["\u00e9\u4e2d"]


def test_string_outputs_are_independent_across_calls():
    rt = _load()
    first = rt.run_method("get_strings", [])
    second = rt.run_method("get_strings", [])
    assert first == second == ["lin.weight", "lin.bias"]


def test_string_method_through_public_api():
    prog = en.Runtime.get().load_program(model_or_skip("const_methods.pte"))
    assert prog.load_method("get_strings")([]) == ["lin.weight", "lin.bias"]


def test_invalid_utf8_string_raises_execution_error():
    # Patch the serialized "lin.bias" (only occurrence) to end in a byte that is never valid
    # UTF-8. Same length, so the flatbuffer stays well-formed; only the string bytes change.
    data = Path(model_or_skip("const_methods.pte")).read_bytes()
    assert data.count(b"lin.bias") == 1
    rt = _core.load_buffer(data.replace(b"lin.bias", b"lin.bia\xff"))
    with pytest.raises(ExecutionError, match=r"get_strings.*output 1.*not valid UTF-8"):
        rt.run_method("get_strings", [])
