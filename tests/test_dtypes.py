import numpy as np
import pytest
import executorch_numpy_runtime as en
from executorch_numpy_runtime import _core
from tests.conftest import model_or_skip

# ScalarType codes (ExecuTorch/PyTorch canonical): used only to assert the mapping.
ST = dict(Byte=0, Char=1, Short=2, Int=3, Long=4, Half=5, Float=6,
          Double=7, Bool=11, BFloat16=15)

CASES = [
    ("f", 4, ST["Float"]), ("f", 8, ST["Double"]), ("f", 2, ST["Half"]),
    ("i", 8, ST["Long"]),  ("i", 4, ST["Int"]),    ("i", 2, ST["Short"]),
    ("i", 1, ST["Char"]),  ("u", 1, ST["Byte"]),   ("b", 1, ST["Bool"]),
]

@pytest.mark.parametrize("kind,size,st", CASES)
def test_numpy_to_scalar_type(kind, size, st):
    assert _core._np_to_st(kind, size) == st

@pytest.mark.parametrize("kind,size,st", CASES)
def test_scalar_type_to_numpy(kind, size, st):
    assert _core._st_to_np(st) == (kind, size)

def test_bfloat16_maps_to_uint16():
    # BFloat16 surfaces as raw uint16 on the output path.
    assert _core._st_to_np(ST["BFloat16"]) == ("u", 2)

def test_unmapped_scalar_type_raises():
    with pytest.raises(Exception):
        _core._st_to_np(99)

def test_unmapped_numpy_raises():
    with pytest.raises(Exception):
        _core._np_to_st("c", 16)  # complex128 unsupported


def test_float32_roundtrip_fidelity():
    m = en.Runtime.get().load_program(model_or_skip("add.pte")).load_method("forward")
    x = np.random.randn(3).astype(np.float32)
    out = m([x, np.zeros(3, np.float32)])[0]  # x + 0 == x
    np.testing.assert_array_equal(out, x)     # bit-exact round trip
