"""Generate a .pte exercising ExecuTorch `constant_methods`, to confirm which
branch of Runtime::run_method each kind of constant method hits.

Run inside the ExecuTorch 1.3.1 venv:
  /home/corey/workspace/executorch/.venv/bin/python \
      tools/export_const_methods_probe.py tests/models

Produces const_methods.pte with:
  - forward(a)      -> a + 1            (ordinary tensor method; keeps program valid)
  - get_scalar()    -> 128             (int-valued constant method)
  - get_double()    -> 2.5             (double-valued constant method)
  - get_bool()      -> True            (bool-valued constant method)
  - get_tensor()    -> tensor([1,2,3]) (tensor-valued constant method)
"""
import sys
from pathlib import Path
import torch
from torch.export import export
from executorch.exir import to_edge


class Fwd(torch.nn.Module):
    def forward(self, a):
        return a + 1


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ep = export(Fwd().eval(), (torch.ones(3),))
    prog = to_edge(
        ep,
        constant_methods={
            "get_scalar": 128,
            "get_double": 2.5,
            "get_bool": True,
            "get_tensor": torch.tensor([1, 2, 3], dtype=torch.int64),
        },
    ).to_executorch()
    (out / "const_methods.pte").write_bytes(prog.buffer)
    print("wrote const_methods.pte")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/models")
