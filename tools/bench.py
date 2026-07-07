#!/usr/bin/env python3
"""Steady-state inference benchmark used for the README "Performance" numbers.

One script, three backends (selected with --backend); each shares identical
input generation and timing so the numbers are comparable across environments:

  torch          : TorchScript eager on the .pt   (needs torch)
  et_pybindings  : official ExecuTorch pybindings on the .pte  (needs executorch)
  numpy_rt       : this torch-free numpy runtime on the .pte   (needs executorch_numpy_runtime)

torch and et_pybindings must run in a torch+executorch env; numpy_rt runs in the
torch-free runtime env. Run each in its own venv and merge the JSON output --
steady-state latency is process-independent, so a per-backend process is fair.

Each run prints one JSON object (and appends it to --out if given), including a
fingerprint of the output logits so you can confirm every backend computed the
same result before trusting the latency comparison.

Example:
  # in a torch+executorch venv:
  python tools/bench.py --backend torch         --pt model.pt   --out r.jsonl
  python tools/bench.py --backend et_pybindings --pte model.pte --out r.jsonl
  # in the runtime venv:
  python tools/bench.py --backend numpy_rt      --pte model.pte --out r.jsonl
"""
import argparse, json, os, time
import numpy as np


def make_input(shape):
    # Fixed seed => byte-identical input across every backend/process.
    rng = np.random.default_rng(0)
    return rng.standard_normal(shape, dtype=np.float32)


def build_runner(backend, args):
    """Return (run_once, note) where run_once(x_np) -> 1D float32 logits."""
    if backend == "torch":
        import torch
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        m = torch.jit.load(args.pt, map_location="cpu").eval()
        def run_once(x_np):
            with torch.no_grad():
                out = m(torch.from_numpy(x_np))
            return out.detach().numpy().reshape(-1)
        return run_once, f"torch.get_num_threads()={torch.get_num_threads()}"

    if backend == "et_pybindings":
        from executorch.extension.pybindings.portable_lib import _load_for_executorch
        import torch  # et pybindings take/return torch tensors
        mod = _load_for_executorch(args.pte)
        def run_once(x_np):
            out = mod.forward([torch.from_numpy(x_np)])
            t = out[0] if isinstance(out, (list, tuple)) else out
            return t.detach().numpy().reshape(-1)
        return run_once, "executorch.pybindings"

    if backend == "numpy_rt":
        import executorch_numpy_runtime as en
        method = en.Runtime.get().load_program(args.pte).load_method("forward")
        def run_once(x_np):
            out = method([x_np])
            arr = out[0] if isinstance(out, (list, tuple)) else out
            return np.asarray(arr).reshape(-1)
        return run_once, f"executorch_numpy_runtime {en.runtime_info()['executorch_version']}"

    raise SystemExit(f"unknown backend {backend}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True,
                    choices=["torch", "et_pybindings", "numpy_rt"])
    ap.add_argument("--pte", help="path to the .pte (et_pybindings, numpy_rt)")
    ap.add_argument("--pt", help="path to the TorchScript .pt (torch)")
    ap.add_argument("--shape", default="1,3,224,224",
                    help="comma-separated input shape (default MobileNetV2)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--threads", type=int, default=1,
                    help="torch thread count; XNNPACK manages its own pool")
    ap.add_argument("--out", default=None, help="append one JSON line here")
    args = ap.parse_args()

    if args.backend == "torch" and not args.pt:
        ap.error("--pt is required for --backend torch")
    if args.backend in ("et_pybindings", "numpy_rt") and not args.pte:
        ap.error("--pte is required for this backend")

    shape = tuple(int(d) for d in args.shape.split(","))
    x = make_input(shape)

    t_load0 = time.perf_counter()
    run_once, note = build_runner(args.backend, args)
    load_ms = (time.perf_counter() - t_load0) * 1e3

    for _ in range(args.warmup):
        logits = run_once(x)

    samples = np.empty(args.iters, dtype=np.float64)
    for i in range(args.iters):
        t0 = time.perf_counter()
        logits = run_once(x)
        samples[i] = (time.perf_counter() - t0) * 1e3  # ms

    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    rec = {
        "backend": args.backend,
        "note": note,
        "threads_arg": args.threads,
        "omp": os.environ.get("OMP_NUM_THREADS"),
        "load_ms": round(load_ms, 2),
        "iters": args.iters,
        "warmup": args.warmup,
        "mean_ms": round(float(samples.mean()), 3),
        "p50_ms": round(float(np.percentile(samples, 50)), 3),
        "p90_ms": round(float(np.percentile(samples, 90)), 3),
        "p99_ms": round(float(np.percentile(samples, 99)), 3),
        "min_ms": round(float(samples.min()), 3),
        "throughput_ips": round(1000.0 / float(samples.mean()), 2),
        # correctness fingerprint: confirm every backend computes the same thing
        "top1": int(logits.argmax()),
        "top1_logit": round(float(logits.max()), 5),
        "logits_sum": round(float(logits.sum()), 4),
    }
    print(json.dumps(rec))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
