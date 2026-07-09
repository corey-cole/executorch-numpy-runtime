"""Registers etnp::lstm for torch.export so a model calling it lowers to a .pte that
references etnp::lstm.out (matching examples/custom_kernels/lstm/etnp_lstm.cpp).
Import this module before exporting. Run/validate in the ExecuTorch export venv:
  /home/corey/workspace/executorch/.venv/bin/python

Working recipe (found by iteration — see task-4-report.md for the attempts that failed):
  - Define the op with an explicit out-variant schema
    (`lstm.out(..., *, Tensor(a!) output, Tensor(b!) hn, Tensor(c!) cn) -> ...`)
    in ADDITION to the functional `lstm(...)` schema, and register a
    CompositeExplicitAutograd impl for BOTH overloads.
  - `to_edge_transform_and_lower` with NO XNNPACK partitioner passed keeps the op
    opaque. `export()` + default `to_edge_transform_and_lower(ep)` (no partitioner
    argument) never decomposes a custom op registered as CompositeExplicitAutograd
    (composite/functional decomposition only kicks in for CompositeImplicitAutograd
    ops or aten ops that have registered decompositions) — the functional overload
    `etnp::lstm.default` is preserved through to_edge, and ExecuTorch's out-variant
    pass (`ToOutVarPass`, run inside `.to_executorch()`) converts it into
    `etnp::lstm.out` automatically because we registered the `lstm.out` overload
    with a matching schema. That conversion is what makes the *.out* op show up in
    the final execution plan, matching what the C++ kernel registers.
"""
import torch
from torch.library import Library, impl, register_fake

_lib = Library("etnp", "DEF")
_lib.define(
    "lstm(Tensor input, Tensor h0, Tensor c0, Tensor w_ih, Tensor w_hh, "
    "Tensor? b_ih, Tensor? b_hh) -> (Tensor, Tensor, Tensor)")
_lib.define(
    "lstm.out(Tensor input, Tensor h0, Tensor c0, Tensor w_ih, Tensor w_hh, "
    "Tensor? b_ih, Tensor? b_hh, *, Tensor(a!) output, Tensor(b!) hn, Tensor(c!) cn) "
    "-> (Tensor(a!), Tensor(b!), Tensor(c!))")


def _lstm_ref(input, h0, c0, w_ih, w_hh, b_ih, b_hh):
    """Functional reference matching the C++ kernel's semantics: single-layer,
    unidirectional, batch_first=False, float32. Gate row order i,f,g,o (PyTorch).
    input [T,B,I], h0/c0 [B,H], w_ih [4H,I], w_hh [4H,H], b_ih/b_hh [4H].
    Correct enough to trace and shape-propagate; NOT the shipping AOT definition.
    """
    T, B, _ = input.shape
    H = h0.shape[1]
    h, c = h0, c0
    outs = []
    for t in range(T):
        g = input[t] @ w_ih.t() + h @ w_hh.t()
        if b_ih is not None:
            g = g + b_ih
        if b_hh is not None:
            g = g + b_hh
        i, f, gg, o = g[:, 0:H], g[:, H:2 * H], g[:, 2 * H:3 * H], g[:, 3 * H:4 * H]
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(gg)
        h = torch.sigmoid(o) * torch.tanh(c)
        outs.append(h)
    return torch.stack(outs, 0), h, c


@impl(_lib, "lstm", "CompositeExplicitAutograd")
def _lstm_impl(input, h0, c0, w_ih, w_hh, b_ih, b_hh):
    return _lstm_ref(input, h0, c0, w_ih, w_hh, b_ih, b_hh)


@impl(_lib, "lstm.out", "CompositeExplicitAutograd")
def _lstm_out_impl(input, h0, c0, w_ih, w_hh, b_ih, b_hh, *, output, hn, cn):
    out, h, c = _lstm_ref(input, h0, c0, w_ih, w_hh, b_ih, b_hh)
    output.copy_(out)
    hn.copy_(h)
    cn.copy_(c)
    return output, hn, cn


@register_fake("etnp::lstm")
def _lstm_fake(input, h0, c0, w_ih, w_hh, b_ih, b_hh):
    T, B, _ = input.shape
    H = h0.shape[1]
    return (input.new_empty((T, B, H)), h0.new_empty((B, H)), c0.new_empty((B, H)))
