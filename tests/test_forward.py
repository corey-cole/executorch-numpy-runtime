import numpy as np
from executorch_numpy_runtime import _core
from tests.conftest import model_or_skip


def test_add_forward_correct():
    rt = _core.load_path(model_or_skip("add.pte"))
    a = np.full(3, 2.0, dtype=np.float32)
    b = np.full(3, 5.0, dtype=np.float32)
    out = rt.run_method("forward", [a, b])
    assert len(out) == 1
    np.testing.assert_allclose(out[0], np.full(3, 7.0, dtype=np.float32))


def test_output_is_owning_copy():
    rt = _core.load_path(model_or_skip("add.pte"))
    out = rt.run_method("forward", [np.ones(3, np.float32), np.ones(3, np.float32)])
    assert out[0].flags["OWNDATA"]  # fresh array, not a view into the arena
