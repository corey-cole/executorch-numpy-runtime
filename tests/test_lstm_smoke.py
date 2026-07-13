import numpy as np
import pytest
from executorch_numpy_runtime import _core
from conftest import MODELS

LSTM = MODELS / "lstm"


def _load_fixture():
    if not (LSTM / "lstm.pte").exists():
        pytest.skip("LSTM fixtures not vendored; see tests/models/README.md")
    shape = dict(kv.split("=") for kv in (LSTM / "shape").read_text().split())
    T, B, I, H = (int(shape[f"LSTM_{k}"]) for k in "TBIH")
    raw = np.fromfile(LSTM / "in.bin", dtype="<f4")
    inp = raw[: T * B * I].reshape(T, B, I)
    h0 = raw[T * B * I : T * B * I + B * H].reshape(B, H)
    c0 = raw[T * B * I + B * H :].reshape(B, H)
    expected = np.fromfile(LSTM / "out.bin", dtype="<f4").reshape(T, B, H)
    return [inp, h0, c0], expected


def test_lstm_op_registered():
    # The upstream runtime (v1.3.1-4) ships etnp::lstm.out registered at load time.
    assert "etnp::lstm.out" in _core.operator_names()


def test_lstm_pte_matches_golden():
    inputs, expected = _load_fixture()
    rt = _core.load_path(str(LSTM / "lstm.pte"))
    out = rt.run_method("forward", inputs)
    # forward exposes only (input, h0, c0); weights are baked in. out[0] is the
    # sequence output [T,B,H]; out.bin is the golden for that output only.
    np.testing.assert_allclose(out[0], expected, rtol=1e-4, atol=1e-4)
