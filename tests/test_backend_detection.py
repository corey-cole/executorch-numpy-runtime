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

    script = textwrap.dedent(
        """
        from executorch_numpy_runtime import _core
        rt = _core.load_path("tests/models/conv.pte")
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
