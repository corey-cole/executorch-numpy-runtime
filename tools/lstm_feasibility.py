"""Task 6 (MVP): feasibility crossover — find T* where naive LSTM export becomes
impractical while the custom etnp::lstm export stays cheap and constant-size.

This is NOT a general benchmark (see tools/export_lstm_bench_models.py /
tools/bench_lstm.py for that, Task 5). It exists to answer one question with a
real number: at what sequence length T does naive decomposition stop being a
sane thing to export, given a declared practicality budget?

Why a subprocess, not an in-process call
-----------------------------------------
The whole point of this sweep is that naive export at large T does NOT raise an
exception -- it just burns CPU for a very long time (observed: T=256, H=128 ran
>10 minutes before being killed by hand). A `try/except` around an in-process
call cannot bound that; only an external hard timeout can. So each export
attempt (naive AND custom, for symmetry) runs in its own child process via
`subprocess.run(..., timeout=EXPORT_TIME_BUDGET_S)`; on `TimeoutExpired` we
record the failure and move on to the next T. The sweep itself can never hang.

Run in the ExecuTorch export venv (has torch + executorch.exir), from repo root:
  /home/corey/workspace/executorch/.venv/bin/python tools/lstm_feasibility.py [out_dir]
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORT_VENV_PYTHON = "/home/corey/workspace/executorch/.venv/bin/python"

# The export step shells out to `flatc`, which lives alongside the venv's
# python (.venv/bin/flatc) rather than on the ambient PATH. subprocess.run
# does not activate the venv, so without this the child process fails fast
# with FileNotFoundError('flatc') -- a false "naive export failed" signal
# that has nothing to do with the actual size/time wall this script measures.
# NOTE: do NOT use .resolve() here -- .venv/bin/python is typically a symlink
# to the system interpreter, so resolving it collapses to e.g. /usr/bin and
# silently drops the venv's own bin/ (where flatc lives) from PATH.
_VENV_BIN = str(Path(EXPORT_VENV_PYTHON).parent)
_WORKER_ENV = dict(os.environ)
_WORKER_ENV["PATH"] = _VENV_BIN + os.pathsep + _WORKER_ENV.get("PATH", "")

# ---------------------------------------------------------------------------
# Declared practicality budgets. These are arbitrary-but-documented MVP
# thresholds, not physical limits -- "impossible" in this script means
# "exceeds these budgets," nothing more.
# ---------------------------------------------------------------------------
EXPORT_TIME_BUDGET_S = 120   # wall-clock seconds allowed per single export attempt
SIZE_BUDGET_MB = 8           # .pte size ceiling considered "practical"

# H fixed at 128: Task 5's size table already shows the naive .pte growing
# with T at H=32 (96KB -> 267KB -> 951KB for T=16 -> 64 -> 256), and the
# already-observed incident (T=256, H=128 naive export exceeding 10 minutes,
# killed by hand) shows the wall is reached much faster at larger H. Fixing
# H=128 lets a short T schedule hit the wall without a long sweep.
H = 128
B = 1

# Increasing schedule; stop escalating a step or two past the first naive
# failure -- no need to chase T to the moon once T* is established.
T_SCHEDULE = [16, 64, 256, 1024]

_WORKER = REPO_ROOT / "tools" / "_lstm_feasibility_worker.py"


def _run_worker(kind, T, out_path):
    """Run one export attempt (kind in {"naive", "custom"}) in a child process
    with a hard wall-clock timeout. Never raises past this function for a
    timeout -- that is the entire point of the guard.
    """
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [EXPORT_VENV_PYTHON, str(_WORKER), kind, str(T), str(H), str(B), str(out_path)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=EXPORT_TIME_BUDGET_S,
            env=_WORKER_ENV,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        return {"ok": False, "seconds": round(elapsed, 1),
                "reason": f"exceeded EXPORT_TIME_BUDGET_S={EXPORT_TIME_BUDGET_S}s"}

    elapsed = time.monotonic() - start
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        reason = tail[-1] if tail else f"worker exited {proc.returncode}"
        return {"ok": False, "seconds": round(elapsed, 1), "reason": reason}

    if not out_path.exists():
        return {"ok": False, "seconds": round(elapsed, 1),
                "reason": "worker exited 0 but produced no .pte"}

    size_bytes = out_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    ok = size_mb <= SIZE_BUDGET_MB
    result = {"ok": ok, "seconds": round(elapsed, 1), "bytes": size_bytes}
    if not ok:
        result["reason"] = f"exceeded SIZE_BUDGET_MB={SIZE_BUDGET_MB}MB"
    return result


def main(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"budgets: EXPORT_TIME_BUDGET_S={EXPORT_TIME_BUDGET_S}s  "
          f"SIZE_BUDGET_MB={SIZE_BUDGET_MB}MB  H={H}  B={B}")
    print(f"{'T':>6}  {'naive':<48}  {'custom':<40}")

    rows = []
    naive_first_failure_T = None
    saved_artifact = None

    for T in T_SCHEDULE:
        cfg = f"T{T}_H{H}_B{B}"
        naive_path = out_dir / f"lstm_naive_{cfg}.pte"
        custom_path = out_dir / f"lstm_custom_{cfg}.pte"

        naive = _run_worker("naive", T, naive_path)
        custom = _run_worker("custom", T, custom_path)

        rows.append({"T": T, "H": H, "B": B, "naive": naive, "custom": custom})

        def _fmt(r):
            if r["ok"]:
                return f"ok  {r['seconds']:>6.1f}s  {r['bytes']:>10}B"
            return f"FAIL {r['seconds']:>6.1f}s  {r.get('reason', '')}"

        print(f"{T:>6}  {_fmt(naive):<48}  {_fmt(custom):<40}")

        if not naive["ok"] and naive_first_failure_T is None:
            naive_first_failure_T = T
            if custom["ok"] and custom_path.exists():
                saved_artifact = custom_path
            # Give it one more T past the first failure to confirm the trend,
            # then stop escalating (per task brief: "no need to sweep to 4096
            # if T* is 256").
            idx = T_SCHEDULE.index(T)
            if idx + 1 < len(T_SCHEDULE):
                extra_T = T_SCHEDULE[idx + 1]
                cfg2 = f"T{extra_T}_H{H}_B{B}"
                naive_path2 = out_dir / f"lstm_naive_{cfg2}.pte"
                custom_path2 = out_dir / f"lstm_custom_{cfg2}.pte"
                naive2 = _run_worker("naive", extra_T, naive_path2)
                custom2 = _run_worker("custom", extra_T, custom_path2)
                rows.append({"T": extra_T, "H": H, "B": B, "naive": naive2, "custom": custom2})
                print(f"{extra_T:>6}  {_fmt(naive2):<48}  {_fmt(custom2):<40}")
            break

    print()
    if naive_first_failure_T is not None:
        print(f"T* (crossover) = {naive_first_failure_T} at H={H}, B={B}: "
              f"naive export first exceeds the declared budget here; "
              f"custom export stays ok.")
        if saved_artifact is not None:
            print(f"saved crossover artifact: {saved_artifact}")
    else:
        print("naive export did not fail within the swept T schedule "
              f"({T_SCHEDULE}) under the declared budgets -- no crossover observed.")

    results_path = out_dir / "feasibility_results.json"
    results_path.write_text(json.dumps({
        "budgets": {"EXPORT_TIME_BUDGET_S": EXPORT_TIME_BUDGET_S,
                    "SIZE_BUDGET_MB": SIZE_BUDGET_MB, "H": H, "B": B},
        "rows": rows,
        "T_star": naive_first_failure_T,
        "saved_artifact": str(saved_artifact) if saved_artifact else None,
    }, indent=2))
    print(f"wrote {results_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/lstm_ptes")
