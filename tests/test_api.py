import numpy as np
import executorch_numpy_runtime as en
from conftest import model_or_skip

def test_high_level_forward():
    prog = en.Runtime.get().load_program(model_or_skip("add.pte"))
    assert "forward" in prog.method_names
    method = prog.load_method("forward")
    out = method([np.full(3, 2.0, np.float32), np.full(3, 5.0, np.float32)])
    np.testing.assert_allclose(out[0], np.full(3, 7.0, np.float32))

def test_load_program_from_bytes():
    with open(model_or_skip("add.pte"), "rb") as f:
        prog = en.Runtime.get().load_program(f.read())
    assert prog.method_names

def test_method_metadata_exposed():
    prog = en.Runtime.get().load_program(model_or_skip("add.pte"))
    assert prog.load_method("forward").metadata["num_inputs"] == 2
