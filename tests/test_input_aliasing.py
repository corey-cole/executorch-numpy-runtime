"""Whether a method's inputs are memory-planned decides who owns the buffer XNNPACK reads.

With planned inputs (the export default) ExecuTorch deep-copies the caller's tensor into its
own arena. With unplanned inputs it ALIASES the caller's pointer for the duration of the call,
so backend kernels read numpy-owned memory directly. That difference changes the lifetime
contract of `InputDesc` and decides whose alignment and trailing padding matter -- see issues
#11 and #12.

Every fixture in this repo was planned until now, so the aliased path had never run here. These
tests pin both halves of the contract so a change to either is visible.

Read from metadata (TensorInfo::is_memory_planned) rather than inferred from behaviour: the
distinction is not observable through this package's API, because run_method sets inputs on
every call, so a buffer mutated between calls is re-set in both cases.
"""

from conftest import MODELS, model_or_skip

from executorch_numpy_runtime import _core


def test_default_export_memory_plans_its_inputs():
    """The export default is alloc_graph_input=True, so ExecuTorch owns the input buffer."""
    rt = _core.load_path(model_or_skip("add.pte"))
    meta = rt.method_meta("forward")

    assert [t["is_memory_planned"] for t in meta["inputs"]] == [True, True]


def test_unplanned_export_does_not_memory_plan_its_inputs():
    """The unplanned fixture is the only one that exercises the aliased path.

    Failure here most likely means the fixture was regenerated without
    MemoryPlanningPass(alloc_graph_input=False), which would silently turn every test built
    on it -- including the over-read harness in #12 -- into a test of the planned path.
    """
    assert (MODELS / "unplanned.pte").exists(), (
        "unplanned.pte missing; regenerate with tools/export_fixtures.py"
    )
    rt = _core.load_path(str(MODELS / "unplanned.pte"))
    meta = rt.method_meta("forward")

    assert [t["is_memory_planned"] for t in meta["inputs"]] == [False]


def test_outputs_report_planning_independently_of_inputs():
    """alloc_graph_input and alloc_graph_output are separate knobs; only inputs were changed."""
    rt = _core.load_path(str(MODELS / "unplanned.pte"))
    meta = rt.method_meta("forward")

    assert [t["is_memory_planned"] for t in meta["outputs"]] == [True]
