"""Generate .pte test fixtures. Run inside the ExecuTorch 1.3.1 venv:
  /home/corey/workspace/executorch/.venv/bin/python tools/export_fixtures.py tests/models
"""
import sys
from pathlib import Path
import torch
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner


def _save(name, mod, ex, out_dir, partitioners=None):
    ep = export(mod.eval(), ex)
    lowered = to_edge_transform_and_lower(
        ep, partitioner=partitioners or []).to_executorch()
    (out_dir / name).write_bytes(lowered.buffer)
    print("wrote", name)


class Add(torch.nn.Module):
    def forward(self, a, b): return a + b


class MixedDtypes(torch.nn.Module):
    def forward(self, a, b): return a + a, b + b


class TwoMethods(torch.nn.Module):
    def forward(self, a): return a + 1
    def double(self, a): return a * 2


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    xnn = [XnnpackPartitioner()]
    _save("add.pte", Add(), (torch.ones(3), torch.ones(3)), out, xnn)
    _save("dtypes.pte", MixedDtypes(),
          (torch.ones(1, dtype=torch.int64), torch.ones(1)), out)  # portable, no xnn
    # multi-method: export both forward and double
    m = TwoMethods().eval()
    methods = {"forward": export(m, (torch.ones(2),)),
               "double": export(m, (torch.ones(2),), strict=True)}
    lowered = to_edge_transform_and_lower(methods).to_executorch()
    (out / "multi.pte").write_bytes(lowered.buffer)
    print("wrote multi.pte")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/models")
