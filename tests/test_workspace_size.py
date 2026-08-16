import numpy as np
from executorch_numpy_runtime import _core
from conftest import model_or_skip


def test_workspace_size_grows_after_delegated_load():
    """A delegated load allocates the arena, so the reported size becomes non-zero.

    Only the transition is asserted, never the value: the figure is a high-water mark
    that includes allocator alignment padding, so it is not stable across runs or
    platforms. It is also process-wide, which is why this asserts ">0 after" rather
    than an exact delta -- another test in this process may already have grown the arena.

    Uses conv.pte and not add.pte: an elementwise add or a Linear delegates to XNNPACK
    but allocates no workspace arena, so this assertion would fail against a correct
    build. Only a conv grows the arena.
    """
    rt = _core.load_path(model_or_skip("conv.pte"))
    rt.run_method("forward", [np.ones((1, 3, 16, 16), np.float32)])

    assert _core.xnnpack_workspace_size_bytes() > 0


import subprocess
import sys
import textwrap


def test_workspace_size_is_zero_before_any_delegated_load():
    """Zero before the first delegated load is CORRECT, not a broken accessor.

    Runs in a fresh interpreter on purpose: the arena is process-wide and is never
    shrunk, so any earlier delegated load in this pytest process would make the
    'before' read non-zero and silently invalidate the assertion.
    """
    script = textwrap.dedent(
        """
        import executorch_numpy_runtime as en
        assert en.xnnpack_workspace_size_bytes() == 0, "expected a cold arena"
        print("ok")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
