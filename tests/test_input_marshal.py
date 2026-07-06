import numpy as np
import pytest
from executorch_numpy_runtime import _core


def test_contiguous_float32_maps():
    a = np.ones((2, 3), dtype=np.float32)
    was, st, shape = _core._contig_info(a)
    assert was is True and st == 6 and shape == (2, 3)


def test_noncontiguous_is_copied_not_rejected():
    a = np.ones((4, 4), dtype=np.float32)[:, ::2]  # non-contiguous view
    assert not a.flags["C_CONTIGUOUS"]
    was, st, shape = _core._contig_info(a)
    assert was is False and st == 6 and shape == (4, 2)  # handled via copy


def test_unmapped_input_dtype_raises():
    a = np.ones(3, dtype=np.complex128)
    with pytest.raises(Exception):
        _core._contig_info(a)
