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
