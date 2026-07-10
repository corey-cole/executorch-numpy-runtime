"""Dump the naive LSTM export graph structure: op histogram pre/post lowering,
and check whether the input projection is batched over T or per-timestep."""
import collections
import sys

import torch
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

T, H, B = 16, 32, 1
I = H
torch.manual_seed(0)


class NaiveLSTM(torch.nn.Module):
    def __init__(self, lstm):
        super().__init__()
        self.lstm = lstm

    def forward(self, x, h0, c0):
        out, (hn, cn) = self.lstm(x, (h0.unsqueeze(0), c0.unsqueeze(0)))
        return out, hn.squeeze(0), cn.squeeze(0)


lstm = torch.nn.LSTM(I, H, num_layers=1, bias=True,
                     batch_first=False, bidirectional=False).eval()
m = NaiveLSTM(lstm).eval()
ex = (torch.randn(T, B, I), torch.randn(B, H), torch.randn(B, H))

ep = export(m, ex)
hist = collections.Counter(
    str(n.target) for n in ep.graph.nodes if n.op == "call_function")
print("=== torch.export graph op histogram (T=%d,H=%d) ===" % (T, H))
for k, v in hist.most_common():
    print(f"  {v:4d}  {k}")
print(f"  total call_function nodes: {sum(hist.values())}")

# Look at matmul-ish nodes: what are their input shapes?
print("\n=== linear/matmul-ish node shapes (first 12) ===")
shown = 0
for n in ep.graph.nodes:
    if n.op != "call_function":
        continue
    t = str(n.target)
    if any(s in t for s in ("linear", "matmul", "mm", "addmm", "bmm")):
        meta = n.meta.get("val")
        in_shapes = []
        for a in n.args:
            v = getattr(a, "meta", {}).get("val") if hasattr(a, "meta") else None
            in_shapes.append(tuple(v.shape) if v is not None else a)
        out_shape = tuple(meta.shape) if meta is not None else None
        print(f"  {t}: in={in_shapes} out={out_shape}")
        shown += 1
        if shown >= 12:
            break

edge = to_edge_transform_and_lower(ep)  # no partitioner: see the decomposed ops
gm_e = edge.exported_program().graph_module
hist_e = collections.Counter(
    str(n.target) for n in gm_e.graph.nodes if n.op == "call_function")
print("\n=== edge (decomposed, no partitioner) op histogram ===")
for k, v in hist_e.most_common():
    print(f"  {v:4d}  {k}")
print(f"  total call_function nodes: {sum(hist_e.values())}")

print("\n=== edge matmul-ish node shapes (first 12) ===")
shown = 0
for n in gm_e.graph.nodes:
    if n.op != "call_function":
        continue
    t = str(n.target)
    if any(s in t for s in ("linear", "matmul", "mm", "addmm", "bmm")):
        meta = n.meta.get("val")
        in_shapes = []
        for a in n.args:
            v = getattr(a, "meta", {}).get("val") if hasattr(a, "meta") else None
            in_shapes.append(tuple(v.shape) if v is not None else a)
        out_shape = tuple(meta.shape) if meta is not None else None
        print(f"  {t}: in={in_shapes} out={out_shape}")
        shown += 1
        if shown >= 12:
            break

lowered = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])
gm = lowered.exported_program().graph_module
hist2 = collections.Counter(
    str(n.target) for n in gm.graph.nodes if n.op == "call_function")
print("\n=== after XNNPACK lowering: op histogram ===")
for k, v in hist2.most_common():
    print(f"  {v:4d}  {k}")
print(f"  total call_function nodes: {sum(hist2.values())}")
n_delegates = sum(1 for n in gm.graph.nodes
                  if n.op == "get_attr" and "lowered_module" in str(n.target))
print(f"  delegate (lowered_module) count: {n_delegates}")
