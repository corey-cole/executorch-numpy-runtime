import numpy as np
import pytest
from executorch_numpy_runtime import _core
from executorch_numpy_runtime.errors import (
    ExecuTorchError, ProgramLoadError, BackendNotAvailable, ExecutionError)
from tests.conftest import model_or_skip, MODELS

def test_corrupt_pte_raises_program_load_error(tmp_path):
    bad = tmp_path / "bad.pte"; bad.write_bytes(b"not a real pte" * 10)
    with pytest.raises(ProgramLoadError):
        _core.load_path(str(bad))

def test_all_errors_subclass_base():
    for cls in (ProgramLoadError, BackendNotAvailable, ExecutionError):
        assert issubclass(cls, ExecuTorchError)

def test_unmapped_dtype_input_raises_executorch_error():
    rt = _core.load_path(model_or_skip("add.pte"))
    with pytest.raises(ExecuTorchError):
        rt.run_method("forward", [np.ones(3, np.complex128), np.ones(3, np.complex128)])

@pytest.mark.skipif(not (MODELS / "coreml.pte").exists(),
                    reason="non-CPU fixture not available")
def test_non_cpu_backend_raises_backend_not_available():
    with pytest.raises(BackendNotAvailable):
        _core.load_path(str(MODELS / "coreml.pte"))
