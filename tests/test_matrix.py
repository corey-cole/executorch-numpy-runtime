import numpy as np
import executorch_numpy_runtime as en
from conftest import model_or_skip


def _method(name, meth="forward"):
    return en.Runtime.get().load_program(model_or_skip(name)).load_method(meth)


def test_sequential_calls_do_not_clobber_prior_outputs():
    m = _method("add.pte")
    r1 = m([np.full(3, 1.0, np.float32), np.full(3, 1.0, np.float32)])[0]
    r1_copy = r1.copy()
    r2 = m([np.full(3, 9.0, np.float32), np.full(3, 9.0, np.float32)])[0]
    np.testing.assert_allclose(r1, r1_copy)          # r1 unchanged by r2's run
    np.testing.assert_allclose(r2, np.full(3, 18.0, np.float32))


def test_multi_method_program():
    prog = en.Runtime.get().load_program(model_or_skip("multi.pte"))
    assert set(["forward", "double"]).issubset(set(prog.method_names))


def test_dynamic_shape_within_bounds():
    m = _method("dynamic.pte")
    for n in (1, 4, 8):
        out = m([np.ones(n, np.float32)])[0]
        np.testing.assert_allclose(out, np.full(n, 2.0, np.float32))


def test_quantized_model_runs():
    m = _method("quantized.pte")
    out = m([np.random.randn(2, 8).astype(np.float32)])
    assert out[0].shape == (2, 8)
    # Coarse sanity check (no torch reference available): outputs should be
    # finite and not a degenerate all-zero/all-identical buffer.
    assert np.isfinite(out[0]).all()
    assert not np.all(out[0] == out[0].flat[0])


def test_mixed_dtype_model():
    m = _method("dtypes.pte")
    a = np.array([3], dtype=np.int64); b = np.array([2.5], dtype=np.float32)
    outs = m([a, b])
    assert outs[0].dtype == np.int64 and outs[1].dtype == np.float32
    np.testing.assert_array_equal(outs[0], np.array([6]))
    np.testing.assert_allclose(outs[1], np.array([5.0], np.float32))
