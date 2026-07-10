# LSTM perf-probe benches (2026-07-10 investigation)

Throwaway probes behind the root-cause analysis of the `etnp::lstm` performance
issue (`problem-statement.md`): why the custom kernel loses to the naive
decomposition at H>=64 and on high-core-count hosts.

Findings they establish:

1. `xnn_microbench.cpp` — per-execute `xnn_create_fully_connected_nc_f32`
   weight packing costs ~H^2 (2us @H32 -> 465us @H256 for both ops); one
   batch=T FC run is 1.5-2.5x faster than T batch-1 runs; a threadpool adds
   another 2-3.4x on the batched run (8 threads).
2. `lstm_shape_bench.cpp` — end-to-end: current kernel structure vs
   restructured (batched input projection + threadpool, optional cached ops).
3. `lstm_component_bench.cpp` — per-step cost split: the scalar libm
   sigmoid/tanh cell update DOMINATES at H=128 (~60% of step time); a
   vectorizable rational approximation is ~3.6x faster (accuracy of the quick
   probe poly is NOT production-grade — fails the bench 1e-4 gate at H=128).
4. `libm_datadep.cpp` — glibc exp/tanh timing is not data-dependent
   (~30ns/pair either way); rules that out as the sim-vs-real gap cause.
5. `inspect_naive_graph.py` — dumps the naive LSTM edge graph: PyTorch's
   decomposition hoists `x @ W_ih^T` into ONE [T,I]x[I,4H] GEMM before the
   time loop; only the recurrent projection runs per timestep.
6. `cell_lib_spike/` — the activation-library spike (scalar libm vs hand-rolled
   poly vs Eigen vs Highway static/dynamic). Outcome: Highway dynamic dispatch
   hits the -march=native ceiling from a baseline build (0.46us/step @H128,
   max err ~3e-7); the hand-rolled poly failed accuracy catastrophically
   (max err 0.68 through a 256-step chain). `cell_hwy.cc` is the validated
   template for the production `lstm_cell.cc` (see the 2026-07-10 restructure
   design spec). Build: g++ -O2 -std=c++17 -I. -I<eigen> -I<highway>
   cell_spike.cpp cell_hwy.cc <highway>/hwy/{targets,per_target,abort,print}.cc
   (clone eigen + highway into this dir; spike used highway@9d5b1261,
   eigen@26f009db, both 2026-07-09).

## Building the C++ probes

Link XNNPACK straight out of the pinned runtime dist:

    DIST=third_party/executorch-runtime-1.3.1-logging-linux-x86_64
    g++ -O2 -std=c++17 -I$DIST/include <probe>.cpp -o <probe> \
      $DIST/lib/libXNNPACK.a $DIST/lib/libxnnpack-microkernels-prod.a \
      $DIST/lib/libpthreadpool.a $DIST/lib/libcpuinfo.a -lpthread -lm

`inspect_naive_graph.py` runs in the torch export venv
(`/home/corey/workspace/executorch/.venv/bin/python`, needs `flatc` from that
venv's bin on PATH).

## Measurement caveat

The local i7-1185G7 thermal-throttles under sustained load: absolute ms are
NOT comparable across runs, only ratios within a single run.
