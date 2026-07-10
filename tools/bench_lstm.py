"""Benchmark custom etnp::lstm vs naive LSTM decomposition on .pte size + latency.

Consumes the .pte pairs written by tools/export_lstm_bench_models.py. For each
(T, H, B) config it:
  1. loads lstm_naive_<cfg>.pte and lstm_custom_<cfg>.pte in THIS numpy runtime,
  2. builds one fixed (seed 0) numpy input set (x, h0, c0) matching the shared
     forward signature and runs BOTH .pte's on it,
  3. cross-checks custom vs naive outputs at rtol 1e-4 (reports max abs diff) --
     the honesty gate: a benchmark of two different computations is worthless,
  4. warms up, times N iterations of each (median ms),
  5. prints `config | naive_size | custom_size | naive_ms | custom_ms |
     size_ratio | speedup`.

A config where custom does NOT win is a finding to print, not an abort.

Run in the torch-free bench venv (the injected-kernel wheel installed):
  python tools/bench_lstm.py /tmp/lstm_ptes
"""
import sys
import time
from pathlib import Path

import numpy as np
import executorch_numpy_runtime as en

# Must match tools/export_lstm_bench_models.py CONFIGS (T, H, B); I == H.
CONFIGS = [
    (16, 32, 1),
    (64, 32, 1),
    (256, 32, 1),
    (16, 64, 1),
    (64, 64, 1),
    (16, 128, 1),
    (64, 128, 1),
    (256, 128, 1),  # naive .pte infeasible to export (feasibility wall) — skipped if absent.
]

WARMUP = 5
ITERS = 30
RTOL = 1e-4


def make_inputs(T, H, B):
    """One fixed input set (seed 0) for a config: x[T,B,I], h0[B,H], c0[B,H]."""
    rng = np.random.default_rng(0)
    I = H
    x = rng.standard_normal((T, B, I), dtype=np.float32)
    h0 = rng.standard_normal((B, H), dtype=np.float32)
    c0 = rng.standard_normal((B, H), dtype=np.float32)
    return [x, h0, c0]


def median_ms(method, inputs):
    for _ in range(WARMUP):
        method(inputs)
    samples = np.empty(ITERS, dtype=np.float64)
    for i in range(ITERS):
        t0 = time.perf_counter()
        method(inputs)
        samples[i] = (time.perf_counter() - t0) * 1e3
    return float(np.median(samples))


def main(pte_dir):
    pte_dir = Path(pte_dir)
    rt = en.Runtime.get()

    header = (f"{'config':<14} {'naive_size':>10} {'custom_size':>11} "
              f"{'naive_ms':>9} {'custom_ms':>9} {'size_ratio':>10} {'speedup':>8}")
    print(header)
    print("-" * len(header))

    max_abs_diff_overall = 0.0
    losses = []
    checked = 0
    for T, H, B in CONFIGS:
        cfg = f"T{T}_H{H}_B{B}"
        naive_path = pte_dir / f"lstm_naive_{cfg}.pte"
        custom_path = pte_dir / f"lstm_custom_{cfg}.pte"
        if not (naive_path.exists() and custom_path.exists()):
            print(f"{cfg:<14} (missing .pte pair -- skipped)")
            continue

        inputs = make_inputs(T, H, B)

        # Load + run both. Any load/execute failure is surfaced, not swallowed.
        naive_m = rt.load_program(str(naive_path)).load_method("forward")
        custom_m = rt.load_program(str(custom_path)).load_method("forward")

        naive_out = naive_m(inputs)
        custom_out = custom_m(inputs)

        # Cross-check: outputs are [output, hn, cn]; compare each tensor.
        diffs = [float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
                 for a, b in zip(naive_out, custom_out)]
        cfg_max_diff = max(diffs)
        max_abs_diff_overall = max(max_abs_diff_overall, cfg_max_diff)
        checked += 1
        ok = all(np.allclose(np.asarray(a), np.asarray(b), rtol=RTOL, atol=1e-4)
                 for a, b in zip(naive_out, custom_out))
        flag = "" if ok else "  <-- MISMATCH"

        naive_ms = median_ms(naive_m, inputs)
        custom_ms = median_ms(custom_m, inputs)
        naive_size = naive_path.stat().st_size
        custom_size = custom_path.stat().st_size
        size_ratio = naive_size / custom_size
        speedup = naive_ms / custom_ms if custom_ms > 0 else float("inf")
        if speedup < 1.0:
            losses.append((cfg, "speed", speedup))
        if size_ratio < 1.0:
            losses.append((cfg, "size", size_ratio))

        print(f"{cfg:<14} {naive_size:>10} {custom_size:>11} "
              f"{naive_ms:>9.3f} {custom_ms:>9.3f} {size_ratio:>9.2f}x "
              f"{speedup:>6.2f}x{flag} (max|diff|={cfg_max_diff:.2e})")

    print()
    print(f"rtol {RTOL} cross-check over {checked} config(s): "
          f"max abs diff = {max_abs_diff_overall:.3e} "
          f"({'PASS' if max_abs_diff_overall <= 1e-3 else 'FAIL'})")
    if losses:
        print("Configs where custom did NOT win:")
        for cfg, kind, val in losses:
            print(f"  {cfg}: {kind} ratio/speedup = {val:.2f}")
    else:
        print("Custom won on both size and speed for every config checked.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/lstm_ptes")
