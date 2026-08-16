from pathlib import Path

import numpy as np
import pytest

import executorch_numpy_runtime as en

FIXTURES = Path(__file__).parent / "models" / "openvino"


@pytest.mark.requires_backend("OpenvinoBackend")
def test_openvino_fixture_matches_eager_golden(capsys):
    """Compare the delegate's output against the EAGER golden shipped with the fixture.

    Tolerance is atol=1e-2 and MUST NOT be tightened. OpenVINO selects inference precision
    from the CPU it lands on, at import time rather than at blob-compile time: on a
    bf16-capable runner (avx512_bf16/AMX) it computes in bf16 and lands ~2.5e-3 from the
    f32 golden; elsewhere ~6e-8. Both are correct OpenVINO results, so any tolerance drawn
    between them asserts which machine CI happened to allocate -- a property this project
    does not control. Upstream hit exactly this (executorch-runtime-dist ea393da) with a
    green run and a red run on identical artifacts.

    np.testing.assert_allclose's default rtol=1e-7 would reproduce that flake precisely.

    The loose bound still catches everything it exists to catch: a delegate returning
    zeros, garbage, or the wrong model is orders of magnitude out.
    """
    prog = en.Runtime.get().load_program(str(FIXTURES / "openvino_tiny.pte"))
    method = prog.load_method("forward")

    shape = tuple(method.metadata["inputs"][0]["shape"])
    # .copy() is load-bearing: frombuffer over bytes yields a READ-ONLY array, and this
    # runtime borrows inputs zero-copy (the module may write into them), so the binding
    # rejects read-only buffers with an opaque std::bad_cast. The golden stays read-only;
    # it is only compared in numpy, never handed to the runtime.
    x = (
        np.frombuffer((FIXTURES / "in.bin").read_bytes(), dtype=np.float32)
        .reshape(shape)
        .copy()
    )
    golden = np.frombuffer((FIXTURES / "out.bin").read_bytes(), dtype=np.float32)

    out = method([x])[0].reshape(-1)

    # Report the precision actually used. atol=1e-2 alone cannot distinguish "correct in
    # bf16" from "quietly degraded", so record which one this run saw.
    try:
        precision = en.openvino_inference_precision()
    except Exception as exc:  # noqa: BLE001 - diagnostic only, never fails the test
        precision = f"(unavailable: {exc})"

    with capsys.disabled():
        print(f"\nov PRECISION {precision}")
        print(f"openvino fixture: max abs diff {np.max(np.abs(out - golden)):.3e}, tol 1e-2")

    np.testing.assert_allclose(out, golden, atol=1e-2, rtol=0)
