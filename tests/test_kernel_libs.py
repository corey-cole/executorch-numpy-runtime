"""The declared kernel-lib set must match what was actually linked.

kernel_libs is derived from a CMake compile define computed from the real link
line, so these assertions hold on any platform without sniffing sys.platform.
"""

import executorch_numpy_runtime as en
from executorch_numpy_runtime import _core


def test_kernel_libs_is_a_nonempty_list_of_strings():
    libs = en.runtime_info()["kernel_libs"]
    assert isinstance(libs, list)
    assert libs, "kernel_libs must never be empty: portable is always linked"
    assert all(isinstance(x, str) and x for x in libs)


def test_portable_is_always_present():
    # portable_ops_lib exists in every runtime tarball on every platform.
    assert "portable" in en.runtime_info()["kernel_libs"]


def test_kernel_libs_matches_the_compile_define():
    # The Python view must be exactly the C++ define, not an independent guess.
    expected = [s for s in _core.__kernel_libs__.split(",") if s]
    assert en.runtime_info()["kernel_libs"] == expected


def test_kernel_libs_has_no_duplicates():
    libs = en.runtime_info()["kernel_libs"]
    assert len(libs) == len(set(libs))


def test_quantized_implies_the_op_is_registered():
    # Guards the contract against lying in the direction that matters: claiming a
    # kernel lib we did not actually link would make a model fail at load time
    # with "operator not found" instead of a clear capability error.
    info = en.runtime_info()
    if "quantized" in info["kernel_libs"]:
        assert any("quantized" in op for op in info["operators"])
