from pathlib import Path
import pytest

from executorch_numpy_runtime import runtime_info

MODELS = Path(__file__).parent / "models"

def model_or_skip(name: str) -> str:
    p = MODELS / name
    if not p.exists():
        pytest.skip(f"fixture {name} not generated; run tools/export_fixtures.py")
    return str(p)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_kernel_lib(name): skip unless <name> is in runtime_info()['kernel_libs']. "
        "Capability-driven, not platform-driven: if a platform later gains the kernel lib, "
        "the test starts running with no edit.",
    )


def pytest_collection_modifyitems(config, items):
    linked = set(runtime_info()["kernel_libs"])
    for item in items:
        for marker in item.iter_markers(name="requires_kernel_lib"):
            required = marker.args[0]
            if required not in linked:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"kernel lib {required!r} not linked in this build "
                        f"(linked: {sorted(linked)})"
                    )
                )
