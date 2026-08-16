import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from executorch_numpy_runtime import _core

OPENVINO_LINKED = "OpenvinoBackend" in _core.registered_backends()

# Absolute, not relative: used inside fresh-interpreter subprocesses whose cwd is the
# harness's, not necessarily the project root (see test_openvino_resolution.py).
FIXTURE = str(Path(__file__).parent / "models" / "openvino" / "openvino_tiny.pte")


def _run(script: str) -> subprocess.CompletedProcess:
    """Each case runs fresh: OPENVINO_LIB_PATH is process env, and the delegate's
    dlopen is std::call_once with no retry, so cases must not share an interpreter."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)], capture_output=True, text=True
    )


@pytest.mark.skipif(OPENVINO_LINKED, reason="this asserts the UNLINKED platform's message")
def test_unlinked_platform_names_the_platform():
    proc = _run(
        f"""
        from executorch_numpy_runtime import _core
        from executorch_numpy_runtime.errors import BackendNotAvailable
        import numpy as np
        rt = _core.load_path({FIXTURE!r})
        try:
            rt.run_method("forward", [np.zeros((1, 8), np.float32)])
        except BackendNotAvailable as e:
            assert "OpenvinoBackend" in str(e), str(e)
            assert "linux-x86_64" in str(e), str(e)
            print("ok")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


@pytest.mark.skipif(not OPENVINO_LINKED, reason="requires the linked delegate")
def test_lib_path_pointing_at_a_directory_is_refused():
    """A directory is upstream's documented top mistake -- the underlying error message
    mentions LD_LIBRARY_PATH, which reads like it wants a directory. It does not."""
    proc = _run(
        f"""
        import os
        os.environ["OPENVINO_LIB_PATH"] = "/tmp"
        from executorch_numpy_runtime import _core
        from executorch_numpy_runtime.errors import BackendNotAvailable
        import numpy as np
        rt = _core.load_path({FIXTURE!r})
        try:
            rt.run_method("forward", [np.zeros((1, 8), np.float32)])
        except BackendNotAvailable as e:
            assert "OPENVINO_LIB_PATH" in str(e), str(e)
            print("ok")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
