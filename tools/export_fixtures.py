"""Generate .pte test fixtures. Run inside the ExecuTorch 1.3.1 venv:
  PATH="/path/to/et-venv/bin:$PATH" /path/to/et-venv/bin/python \
      tools/export_fixtures.py tests/models

Serialization shells out to `flatc`, which the executorch package resolves from PATH rather
than from the interpreter -- so invoking the venv's python by absolute path is not enough on
its own, and fails late with FileNotFoundError: 'flatc' after the export work is done.
"""
import sys
from pathlib import Path
import torch
from torch.export import export
from executorch.exir import to_edge_transform_and_lower, ExecutorchBackendConfig
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
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


class Scale(torch.nn.Module):
    def forward(self, x): return x * 2.0


class Lin(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(8, 8)

    def forward(self, x): return self.l(x)


class ConvRelu(torch.nn.Module):
    """Conv2d+ReLU, chosen deliberately over Linear+ReLU.

    Both delegate to XNNPACK, but a Linear lowers to a direct GEMM with statically
    packed weights and allocates NO workspace arena. Only an op that grows the arena
    in xnn_create_runtime_v4 -- a conv does -- makes a workspace-size assertion
    meaningful rather than vacuously zero.
    """

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)

    def forward(self, x):
        return torch.relu(self.conv(x))


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    xnn = [XnnpackPartitioner()]
    _save("add.pte", Add(), (torch.ones(3), torch.ones(3)), out, xnn)
    _save("conv.pte", ConvRelu(), (torch.randn(1, 3, 16, 16),), out, xnn)
    if b"XnnpackBackend" not in (out / "conv.pte").read_bytes():
        raise SystemExit("conv.pte contains no XnnpackBackend delegate")
    # Unplanned INPUTS (alloc_graph_input=False). The export default is True, which makes
    # ExecuTorch deep-copy each input into its memory-planned arena. With planning off for
    # inputs it ALIASES the caller's pointer instead, so backend kernels read numpy-owned
    # memory directly. Every other fixture here is planned, so this is the only one that
    # exercises the aliased path -- see issues #11 and #12.
    #
    # Delegated to XNNPACK on purpose: the aliasing only matters if a backend kernel is the
    # thing reading the caller's buffer.
    ep = export(Lin().eval(), (torch.ones(1, 8),))
    (out / "unplanned.pte").write_bytes(
        to_edge_transform_and_lower(ep, partitioner=xnn)
        .to_executorch(
            config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
            )
        )
        .buffer
    )
    print("wrote unplanned.pte")

    _save("dtypes.pte", MixedDtypes(),
          (torch.ones(1, dtype=torch.int64), torch.ones(1)), out)  # portable, no xnn
    # multi-method: export both forward and double
    m = TwoMethods().eval()
    methods = {"forward": export(m, (torch.ones(2),)),
               "double": export(m, (torch.ones(2),), strict=True)}
    lowered = to_edge_transform_and_lower(methods).to_executorch()
    (out / "multi.pte").write_bytes(lowered.buffer)
    print("wrote multi.pte")

    # Dynamic (bounded) shape model.
    from torch.export import Dim
    d = Dim("n", min=1, max=8)
    ep = export(Scale().eval(), (torch.ones(4),), dynamic_shapes={"x": {0: d}})
    (out / "dynamic.pte").write_bytes(
        to_edge_transform_and_lower(ep, partitioner=xnn).to_executorch().buffer)
    print("wrote dynamic.pte")

    # Quantized model (proves optimized/quantized kernels linked).
    # Uses XNNPACK's PT2E quantization flow. On ExecuTorch 1.3.1 (torch 2.12),
    # prepare_pt2e/convert_pt2e live under torchao.quantization.pt2e (moved out
    # of torch.ao.quantization), and there is no separate export_for_training
    # step -- plain torch.export.export already produces a trainable graph.
    try:
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer, get_symmetric_quantization_config)
        from torchao.quantization.pt2e.quantize_pt2e import (
            prepare_pt2e, convert_pt2e)
        ex = (torch.randn(2, 8),)
        m = torch.export.export(Lin().eval(), ex).module()
        q = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
        m = prepare_pt2e(m, q)
        m(*ex)  # calibrate
        m = convert_pt2e(m)
        ep = export(m, ex)
        (out / "quantized.pte").write_bytes(
            to_edge_transform_and_lower(ep, partitioner=xnn).to_executorch().buffer)
        print("wrote quantized.pte")
    except Exception as e:
        print("skip quantized.pte:", e)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/models")
