import subprocess
import sys
import textwrap

import pytest

FIXTURE = "tests/models/openvino/openvino_tiny.pte"


def _run(script: str) -> subprocess.CompletedProcess:
    """Fresh interpreter per case: OPENVINO_LIB_PATH is process env and the delegate's
    dlopen is std::call_once with no retry."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)], capture_output=True, text=True
    )


@pytest.mark.requires_backend("OpenvinoBackend")
def test_resolver_sets_lib_path_to_an_existing_file():
    proc = _run(
        f"""
        import os
        os.environ.pop("OPENVINO_LIB_PATH", None)
        import executorch_numpy_runtime as en
        en.Runtime.get().load_program({FIXTURE!r})
        p = os.environ["OPENVINO_LIB_PATH"]
        assert os.path.isfile(p), p
        assert "libopenvino_c.so" in os.path.basename(p), p
        print("ok")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


@pytest.mark.requires_backend("OpenvinoBackend")
def test_operator_provided_path_is_not_overwritten():
    """An operator override must win -- they may be pointing at the upstream pinned
    bundle rather than the pip wheel."""
    proc = _run(
        f"""
        import os, sys, glob
        from importlib.util import find_spec
        libs = os.path.join(list(find_spec("openvino").submodule_search_locations)[0], "libs")
        real = sorted(glob.glob(os.path.join(libs, "libopenvino_c.so*")))[0]
        os.environ["OPENVINO_LIB_PATH"] = real
        import executorch_numpy_runtime as en
        en.Runtime.get().load_program({FIXTURE!r})
        assert os.environ["OPENVINO_LIB_PATH"] == real
        print("ok")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


@pytest.mark.requires_backend("OpenvinoBackend")
def test_directory_valued_override_raises_before_execution():
    proc = _run(
        f"""
        import os
        os.environ["OPENVINO_LIB_PATH"] = "/tmp"
        import executorch_numpy_runtime as en
        from executorch_numpy_runtime.errors import BackendNotAvailable
        try:
            en.Runtime.get().load_program({FIXTURE!r})
        except BackendNotAvailable as e:
            assert "OPENVINO_LIB_PATH" in str(e)
            print("ok")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_non_openvino_program_never_touches_the_env():
    """Users who never load an OpenVINO model must not pay for this feature -- no
    openvino import, no global env mutation."""
    proc = _run(
        """
        import os, sys
        os.environ.pop("OPENVINO_LIB_PATH", None)
        import numpy as np
        import executorch_numpy_runtime as en
        prog = en.Runtime.get().load_program("tests/models/conv.pte")
        prog.load_method("forward")([np.ones((1, 3, 16, 16), np.float32)])
        assert "OPENVINO_LIB_PATH" not in os.environ
        assert "openvino" not in sys.modules
        print("ok")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
