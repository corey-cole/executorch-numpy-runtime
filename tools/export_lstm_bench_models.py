"""Export paired naive/custom LSTM .pte fixtures for the MVP performance probe (Task 5).

For each (T, H, B) config, build ONE torch.nn.LSTM and export it two ways with the
SAME weights:
  - lstm_naive_<cfg>.pte  : plain torch.nn.LSTM, exported normally (decomposes to
    core ATen ops: matmul/sigmoid/tanh/etc), lowered with the XNNPACK partitioner
    so it runs on this runtime (single-layer LSTM is not itself an XNNPACK op, but
    its decomposed matmuls/elementwise ops are XNNPACK-partitionable; portable
    kernels cover whatever XNNPACK doesn't claim).
  - lstm_custom_<cfg>.pte : a wrapper module holding the SAME LSTM weights as
    buffers and calling torch.ops.etnp.lstm(...) (registered by
    tools/etnp_lstm_op.py, Task 4) — lowered with NO partitioner so the op stays
    opaque and survives to_executorch() as etnp::lstm.out.

Both modules take an IDENTICAL forward signature (x, h0, c0) so tools/bench_lstm.py
can drive both .pte's with the same method([...]) call.

Run in the ExecuTorch export venv (has torch + executorch.exir):
  /home/corey/workspace/executorch/.venv/bin/python tools/export_lstm_bench_models.py /tmp/lstm_ptes
"""
import sys
from pathlib import Path

# Allow `python tools/export_lstm_bench_models.py` from the repo root: running a
# script directly puts its own directory (tools/) on sys.path[0], not the repo
# root, so `import tools.etnp_lstm_op` would otherwise fail.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import tools.etnp_lstm_op  # noqa: F401 -- registers torch.ops.etnp.lstm
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

# (T, H, B) sweep -- T varies widely so any T-trend is visible.
CONFIGS = [
    (16, 32, 1),
    (64, 32, 1),
    (256, 32, 1),
    (16, 64, 1),
    (64, 64, 1),
    (16, 128, 1),
    (64, 128, 1),
    (256, 128, 1),  # naive export is infeasible (>120s) — the feasibility wall; bench skips it.
]


class NaiveLSTM(torch.nn.Module):
    """Plain nn.LSTM; forward(x, h0, c0) matches the custom wrapper's signature."""

    def __init__(self, lstm):
        super().__init__()
        self.lstm = lstm

    def forward(self, x, h0, c0):
        out, (hn, cn) = self.lstm(x, (h0.unsqueeze(0), c0.unsqueeze(0)))
        return out, hn.squeeze(0), cn.squeeze(0)


class CustomLSTM(torch.nn.Module):
    """Same weights as the source nn.LSTM, but calls torch.ops.etnp.lstm directly."""

    def __init__(self, lstm):
        super().__init__()
        self.register_buffer("w_ih", lstm.weight_ih_l0.detach().clone())
        self.register_buffer("w_hh", lstm.weight_hh_l0.detach().clone())
        self.register_buffer("b_ih", lstm.bias_ih_l0.detach().clone())
        self.register_buffer("b_hh", lstm.bias_hh_l0.detach().clone())

    def forward(self, x, h0, c0):
        out, hn, cn = torch.ops.etnp.lstm(
            x, h0, c0, self.w_ih, self.w_hh, self.b_ih, self.b_hh)
        return out, hn, cn


def _export_naive(lstm, ex, out_path):
    m = NaiveLSTM(lstm).eval()
    ep = export(m, ex)
    lowered = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()]).to_executorch()
    out_path.write_bytes(lowered.buffer)
    return len(lowered.buffer)


def _export_custom(lstm, ex, out_path):
    m = CustomLSTM(lstm).eval()
    ep = export(m, ex)
    # No partitioner: keep etnp::lstm opaque so it survives to_executorch()
    # as etnp::lstm.out (see tools/etnp_lstm_op.py docstring).
    lowered = to_edge_transform_and_lower(ep).to_executorch()
    out_path.write_bytes(lowered.buffer)
    return len(lowered.buffer)


def main(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    print(f"{'config':<18} {'naive_bytes':>12} {'custom_bytes':>13}")
    for T, H, I_B in CONFIGS:
        B = I_B
        I = H  # input size == hidden size, kept simple for the sweep
        cfg = f"T{T}_H{H}_B{B}"

        lstm = torch.nn.LSTM(I, H, num_layers=1, bias=True,
                              batch_first=False, bidirectional=False).eval()
        x = torch.randn(T, B, I)
        h0 = torch.randn(B, H)
        c0 = torch.randn(B, H)
        ex = (x, h0, c0)

        naive_path = out_dir / f"lstm_naive_{cfg}.pte"
        custom_path = out_dir / f"lstm_custom_{cfg}.pte"
        naive_bytes = _export_naive(lstm, ex, naive_path)
        custom_bytes = _export_custom(lstm, ex, custom_path)

        print(f"{cfg:<18} {naive_bytes:>12} {custom_bytes:>13}")

    print(f"wrote .pte pairs to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/lstm_ptes")
