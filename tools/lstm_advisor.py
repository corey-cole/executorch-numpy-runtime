"""
lstm_advisor.py — static heuristics over a torch.export ExportedProgram that flag
LSTM configs where (a) naive decomposition is impractical to lower, or (b) the custom
etnp::lstm op is expected to be *slower* than the delegated/naive path.

Run at the torch.export level (before to_edge), where an nn.LSTM is still ONE
`aten.lstm.input` node and the custom op is `etnp.lstm.default` — both carry static
T/H in their fake-tensor meta. (After to_edge decomposition the aten.lstm is unrolled
and there is no single node left to inspect.)

Calibrated to the MVP (single-layer, unidirectional, batch_first=False, f32, B=1):
  SIZE   custom .pte constant in T; naive grows with T.
  SPEED  custom wins at H=32 (1.3x–1.9x, widening with T); marginally LOSES at H=64
         (~0.87–0.92x) and clearly loses at H=128 (0.49–0.85x) — XNNPACK batched FC
         beats the per-timestep kernel once H is wide. Crossover is between H=32 and 64.
  FEASIB naive export of T=256,H=128 never finished in 120s; T=256,H=32 was fine.
These are heuristics from a few data points — tune the thresholds as you learn more.

Usage:
    from torch.export import export
    ep = export(model.eval(), example_inputs)
    inspect_lstm(ep)                                   # emits warnings.warn(...)
    findings = inspect_lstm(ep, emit_warnings=False)   # structured, for CI gating
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

# Thresholds (override freely).
H_CUSTOM_SLOW = 64      # H >= this: per-timestep custom kernel tends to lose on speed.
                        # Calibrated: wins at H=32, already loses (marginally) at H=64.
NAIVE_TH_PRODUCT = 32_000  # T*H >= this: naive lowering seen to blow the budget
                           # (256*128=32768 never completed; 256*32=8192 was fine).
NAIVE_T_WARN = 256      # T >= this: naive unroll is at least "large/slow" even at small H.


@dataclass
class LstmFinding:
    kind: str  # "naive-infeasible" | "naive-large" | "custom-slow"
    op: str
    T: int | None
    H: int | None
    message: str


def _static_shape(node):
    """Best-effort concrete shape from a node's fake value; SymInt dims -> None."""
    val = node.meta.get("val") if (node is not None and hasattr(node, "meta")) else None
    if val is None or not hasattr(val, "shape"):
        return None
    dims = []
    for d in val.shape:
        try:
            dims.append(int(d))
        except (TypeError, ValueError):
            dims.append(None)  # dynamic
    return dims


def _arg_node(node, idx):
    a = node.args[idx] if idx < len(node.args) else None
    if isinstance(a, (list, tuple)):  # aten.lstm hx = [h0, c0]
        a = a[0] if a else None
    return a


def inspect_lstm(ep_or_gm, *, emit_warnings=True) -> list[LstmFinding]:
    gm = getattr(ep_or_gm, "graph_module", ep_or_gm)
    findings: list[LstmFinding] = []

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        target = str(getattr(node, "target", ""))
        if "lstm" not in target.lower():
            continue

        is_custom = "etnp" in target
        in_shape = _static_shape(_arg_node(node, 0))  # input tensor

        if is_custom:
            # etnp.lstm(input[T,B,I], h0[B,H], ...)
            T = in_shape[0] if in_shape else None
            h0_shape = _static_shape(_arg_node(node, 1))
            H = h0_shape[1] if (h0_shape and len(h0_shape) >= 2) else None
            if H is not None and H >= H_CUSTOM_SLOW:
                findings.append(LstmFinding(
                    "custom-slow", target, T, H,
                    f"{target}: H={H} (>= {H_CUSTOM_SLOW}) — the per-timestep custom kernel "
                    f"tends to be SLOWER than XNNPACK's batched FC at wide hidden size "
                    f"(MVP: won at H=32, lost at H>=64). Still wins on .pte size; if latency "
                    f"dominates here, prefer the delegated/naive path."))
        else:
            # aten.lstm.input(input, hx=[h0,c0], params, has_biases, num_layers,
            #                 dropout, train, bidirectional, batch_first)
            batch_first = bool(node.args[8]) if len(node.args) > 8 else False
            T = None
            if in_shape:
                T = in_shape[1] if (batch_first and len(in_shape) >= 2) else in_shape[0]
            hx_shape = _static_shape(_arg_node(node, 1))  # h0 [D, B, H]
            H = hx_shape[-1] if hx_shape else None

            if T is not None:
                prod = T * H if H is not None else None
                if prod is not None and prod >= NAIVE_TH_PRODUCT:
                    findings.append(LstmFinding(
                        "naive-infeasible", target, T, H,
                        f"{target}: T={T}, H={H} (T*H={prod} >= {NAIVE_TH_PRODUCT}) — naive "
                        f"decomposition unrolls into ~T per-timestep subgraphs; at this size "
                        f"lowering was observed to EXCEED practical budgets "
                        f"(MVP: T=256,H=128 never completed in 120s). Lower via etnp::lstm."))
                elif T >= NAIVE_T_WARN:
                    findings.append(LstmFinding(
                        "naive-large", target, T, H,
                        f"{target}: T={T} (>= {NAIVE_T_WARN}) — naive unrolls per timestep, so "
                        f".pte size and export time grow with T (MVP: 96KB->951KB over T=16->256 "
                        f"at H=32). etnp::lstm stays constant in T."))

    if emit_warnings:
        for f in findings:
            warnings.warn(f.message, stacklevel=2)
    return findings
