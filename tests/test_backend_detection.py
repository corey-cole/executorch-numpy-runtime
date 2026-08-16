import platform
import sys

import pytest

from executorch_numpy_runtime import _core
from conftest import model_or_skip


def test_method_uses_backend_true_for_delegated_fixture():
    rt = _core.load_path(model_or_skip("conv.pte"))
    assert rt.method_uses_backend("forward", "XnnpackBackend") is True


def test_method_uses_backend_false_for_absent_backend():
    rt = _core.load_path(model_or_skip("conv.pte"))
    assert rt.method_uses_backend("forward", "CoreMLBackend") is False


def test_method_meta_does_not_initialize_delegates():
    """Detection must run BEFORE delegate init, or OPENVINO_LIB_PATH is set too late.

    Module::method_meta not loading the method is an implementation detail of the Module
    extension, not a documented contract -- the header says only "Loads the program if
    needed". If a runtime bump makes it eager, OpenVINO detection silently becomes too
    late and fails at dlopen with no obvious connection to the change. The XNNPACK
    workspace is a direct probe: it is grown during delegate init, so it staying at 0
    across method_meta proves no delegate was initialized.

    Runs in a fresh interpreter because the workspace is process-wide and never shrunk.
    """
    import subprocess
    import sys
    import textwrap

    # Absolute, not relative: this runs in a fresh interpreter whose cwd is the
    # harness's, not necessarily the project root (cibuildwheel runs the wheel test
    # leg from a temp dir).
    conv_path = model_or_skip("conv.pte")
    script = textwrap.dedent(
        f"""
        from executorch_numpy_runtime import _core
        rt = _core.load_path({conv_path!r})
        assert _core.xnnpack_workspace_size_bytes() == 0, "load_path initialized a delegate"
        rt.method_meta("forward")
        rt.method_uses_backend("forward", "XnnpackBackend")
        assert _core.xnnpack_workspace_size_bytes() == 0, "metadata access initialized a delegate"
        print("ok")
        """
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


_IS_LINUX_X86_64 = sys.platform == "linux" and platform.machine() == "x86_64"


@pytest.mark.skipif(not _IS_LINUX_X86_64, reason="OpenVINO delegate ships on linux-x86_64 only")
def test_openvino_backend_is_registered_on_linux_x86_64():
    """The delegate ships in the linux-x86_64 tarball; linking it must register it.

    Platform-conditional rather than capability-conditional on purpose: this is the one
    test that must FAIL if the link is dropped. A requires_backend marker would skip
    instead, turning a regression into a silent pass.
    """
    assert "OpenvinoBackend" in _core.registered_backends()
