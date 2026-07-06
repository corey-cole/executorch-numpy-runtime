import numpy as np
from concurrent.futures import ThreadPoolExecutor
from executorch_numpy_runtime import _core
from tests.conftest import model_or_skip

def test_concurrent_forward_correct():
    rt = _core.load_path(model_or_skip("add.pte"))
    def one(i):
        a = np.full(3, float(i), np.float32)
        return rt.run_method("forward", [a, a])[0]
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(one, range(64)))
    for i, r in enumerate(results):
        np.testing.assert_allclose(r, np.full(3, 2.0 * i, np.float32))
