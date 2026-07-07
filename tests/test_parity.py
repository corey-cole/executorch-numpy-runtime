import numpy as np
import pytest
import executorch_numpy_runtime as en
from conftest import model_or_skip

torch = pytest.importorskip("torch")  # never a package dep; skipped if torch absent


def test_add_matches_torch_reference():
    a = np.random.randn(3).astype(np.float32)
    b = np.random.randn(3).astype(np.float32)
    ref = (torch.tensor(a) + torch.tensor(b)).numpy()
    m = en.Runtime.get().load_program(model_or_skip("add.pte")).load_method("forward")
    got = m([a, b])[0]
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)
