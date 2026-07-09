"""Child-process worker for tools/lstm_feasibility.py.

Runs exactly ONE export attempt (naive or custom) for one (T, H, B) config and
exits. Invoked via subprocess.run(..., timeout=...) so the parent can enforce a
hard wall-clock budget on work that does not raise on its own (naive export at
large T just runs for a very long time). Kept as a separate file rather than
`python -c` so the export logic is easy to read/maintain.

Usage:
  <export-venv-python> tools/_lstm_feasibility_worker.py {naive|custom} T H B out_path
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.export import export

from tools.export_lstm_bench_models import (  # noqa: E402
    NaiveLSTM, CustomLSTM, _export_naive, _export_custom,
)
import tools.etnp_lstm_op  # noqa: F401 -- registers torch.ops.etnp.lstm


def main():
    kind, T, H, B, out_path = sys.argv[1:6]
    T, H, B = int(T), int(H), int(B)
    out_path = Path(out_path)

    torch.manual_seed(0)
    I = H
    lstm = torch.nn.LSTM(I, H, num_layers=1, bias=True,
                          batch_first=False, bidirectional=False).eval()
    x = torch.randn(T, B, I)
    h0 = torch.randn(B, H)
    c0 = torch.randn(B, H)
    ex = (x, h0, c0)

    if kind == "naive":
        _export_naive(lstm, ex, out_path)
    elif kind == "custom":
        _export_custom(lstm, ex, out_path)
    else:
        raise SystemExit(f"unknown kind {kind!r}")


if __name__ == "__main__":
    main()
