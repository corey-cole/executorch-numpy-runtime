# Handoff: shared-threadpool concurrency hazard — analysis & corrective action

**Purpose.** Bootstrap a fresh agent (or engineer) to (1) *verify* whether the
process-wide XNNPACK/pthreadpool singleton is actually unsafe to drive from
parallel `Runtime`s, and (2) if so, choose and apply a corrective action ranging
from a documentation fix to an `et_core` change. This is **new work**, separate
from the LSTM custom-kernel branch that surfaced it.

**Status:** UNVERIFIED LATENT HAZARD. Plausible correctness impact (silent wrong
results — the worst class for a numeric library), but **not** a confirmed crash.
The severity gate (§4) has not been run.

---

## 0. Reference grounding (read first)

Every `file:line` reference in this document is valid at commit
**`bb6c1a193dfc8e0ce883cfbfbee70eb38aefc826`** (short `bb6c1a1`, branch
`feature/lstm-sequence-kernel`; forked from `main` at `eaf5979`). Line numbers
drift — resolve any reference authoritatively with:

```bash
git show bb6c1a1:<path>            # exact file content at this hash
git show bb6c1a1:<path> | sed -n '30,45p'
```

**What resolves where** (checked at `eaf5979`, the `main` merge-base):

| Path | On `main`? | Notes |
|---|---|---|
| `README.md` | yes (contract at `:39` on main too) | the documented contract + TSan-scope note live here |
| `src/et_core/et_core.cpp` | yes | the per-`Runtime` lock; the `et_core` corrective lever |
| `native_tests/race_harness.cpp` | yes | existing multi-`Runtime` TSan harness |
| `third_party/executorch-runtime-1.3.1-*/…/threadpool.h` | yes (pinned dist) | proves the singleton + serialization scope |
| `examples/custom_kernels/lstm/etnp_lstm.cpp` | **branch-only** | the illustrative custom-op pool use (`git show bb6c1a1:…`) |
| `native_tests/lstm_cache_race_test.cpp` | **branch-only** | the distinct-entry topology test (`git show bb6c1a1:…`) |

> **Key point for anyone reading this on `main`:** the hazard is **fully
> reproducible on `main`** with only the XNNPACK-delegate path — you do **not**
> need the LSTM branch. The branch-only files are supporting evidence, not
> prerequisites.

---

## 1. The claim in one paragraph

`README.md:39` promises: *"For true parallel inference, create one `Runtime` per
thread."* But the thread pool those parallel `Runtime`s use is a **process-wide
singleton** (`get_pthreadpool()`), and XNNPACK drives it on the **raw
`pthreadpool_t` path**, which bypasses the only serialization the runtime
provides. A single `pthreadpool` object is single-master-thread by design.
So "one `Runtime` per thread" running in parallel = N threads concurrently
driving **one** non-reentrant pool. The per-`Runtime` lock does not cover this,
because it is *per*-Runtime and these are *separate* Runtimes.

---

## 2. The mechanism (verified facts + their grounding)

**(a) The pool is a singleton.**
`threadpool.h:103` — *"Returns the singleton instance of ThreadPool…"*;
`threadpool.h:113` declares `pthreadpool_t get_pthreadpool();`. All callers in
one process get the same underlying `pthreadpool`.

**(b) The runtime's only pool-serialization is on the C++ wrapper, not the raw
pointer.**
`threadpool.h:86` states *"all calls to run method are serialized"* — but that
sentence is about `ThreadPool::run(...)` (`threadpool.h:88`), the C++ wrapper.
Consumers that take the **raw** `pthreadpool_t` from `get_pthreadpool()` and hand
it to XNNPACK (`xnn_reshape/​setup/​run_*`, which call `pthreadpool_parallelize_*`
directly) never go through that mutex. Both of these do exactly that:
  - the **XNNPACK delegate** (inside the prebuilt dist) — the common case, on `main`;
  - the **LSTM custom kernel's batched input projection**, `git show
    bb6c1a1:examples/custom_kernels/lstm/etnp_lstm.cpp` around `:100–:104` (the
    `// CONCURRENCY:` comment) — branch-only, illustrative.

**(c) pthreadpool is single-master-thread.** *(Domain knowledge — CONFIRM as part
of §4.)* Marat Dukhan's `pthreadpool` stores the current task fn/argument and a
shared atomic work counter in the pool object; it is designed to be driven by
one "master" thread at a time. Concurrent `pthreadpool_parallelize_*` on the
**same** pool from two threads is a data race on that shared state → workers can
execute the wrong task with the wrong argument → memory corruption / wrong
compute. This is the load-bearing assumption; verify it against the pinned
`pthreadpool` source in `third_party/` (or empirically via §4) before acting.

**(d) The per-`Runtime` lock does not close the gap.**
`et_core.cpp:38` — `std::mutex exec_mutex; // serializes execute + copy-out per
Runtime`; taken at `et_core.cpp:94`, `:111`, `:164`. It serializes executes
*within one* `Runtime`. Two *separate* `Runtime`s have *separate* `exec_mutex`es,
so nothing serializes their concurrent executes — and thus nothing serializes
their concurrent driving of the shared pool.

**Net:** single-`Runtime` use is safe (the per-Runtime lock serializes; and in
the LSTM kernel specifically the projection finishes before the timestep loop).
The exposure is strictly **parallel, separate `Runtime`s that both run
pool-backed ops** — which for XNNPACK-delegated models is essentially always.

---

## 3. What is NOT yet known (be honest about these)

- **Whether concurrent driving actually corrupts in this build.** No test proves
  it. `README.md:172` documents that the prebuilt ExecuTorch libs are **not**
  TSan-instrumented, so `race_harness` (and the branch's `lstm_cache_race_test`)
  are **blind inside `pthreadpool`/XNNPACK**. A green TSan run here means
  nothing about pool safety.
- **Weak counter-evidence of tolerance.** `README.md:160` says `race_harness`
  already runs a *"many threads each owning their own `Runtime`"* scenario, and
  it passes. **IF** that scenario's model is XNNPACK-delegated, it has *already*
  been driving the shared pool from parallel Runtimes without visibly blowing up
  — weak evidence it may be tolerated (or that the model is too small/fast to
  create real overlap). **First investigation step: read `race_harness.cpp` and
  determine what model/backend it loads.** Do not assume.

---

## 4. Severity gate — the experiment to run FIRST

Goal: convert "unverified hazard" into a fact. This is a behavioral/differential
test, **not** a sanitizer run (TSan can't see it).

1. **Inspect the existing harness.** `git show bb6c1a1:native_tests/race_harness.cpp`
   — identify the model and whether it uses the XNNPACK backend. If it already
   exercises parallel Runtimes on an XNNPACK model at high iteration counts
   without divergence, note that as evidence and design the new test to be
   *harder* (more overlap, larger parallel op).
2. **Write a two-Runtime differential stress test.** Two *separate* `Runtime`s,
   each loading the *same* XNNPACK-delegated `.pte`, each with its **own** input.
   Precompute each Runtime's expected output by running it **solo**. Then run
   both in parallel behind a start-barrier for many iterations and assert each
   reproduces its solo golden **bit-exact** (XNNPACK FC reductions are
   thread-count-independent, so bit-exact is the right check — see the rationale
   in `git show bb6c1a1:native_tests/lstm_cache_race_test.cpp`, the `identical()`
   comment and Phase B). Divergence = confirmed pool corruption.
   - Model the harness on `race_harness.cpp` (build wiring, `Threads::Threads`).
   - Maximize overlap: start-barrier + a compute-heavy model (wide matmuls) so
     the two `pthreadpool_parallelize` windows actually collide.
3. **Interpretation.**
   - **Diverges** → hazard CONFIRMED. Proceed to §5 Tier 3 decision.
   - **Survives heavy overlap** → strong evidence ExecuTorch tolerates it (likely
     an internal guard, or the pool short-circuits when already busy). Downgrade
     to Tier 1 (docs) only, and record the experiment as the evidence.
4. **Optional, expensive:** a TSan-instrumented ExecuTorch/pthreadpool build
   would let the sanitizer see the race directly. The pinned attested tarball is
   not instrumented (`README.md:172`), so this means building ExecuTorch from
   source with `-fsanitize=thread` — only worth it if §4.2 is inconclusive.

---

## 5. Corrective actions (cheap → expensive)

### Tier 1 — Reconcile the documented contract (do regardless of §4 outcome)
`README.md:39` currently makes a safety claim that is, at best, incomplete for a
numeric library. Add the caveat: *separate `Runtime`s share one process-wide
XNNPACK/pthreadpool; running pool-backed ops (the XNNPACK delegate, or custom
kernels using `get_pthreadpool()`) from parallel `Runtime`s shares ExecuTorch's
shared-pool contract — serialize their executes or accept it.* Also add a
kernel-author note to `docs/custom-kernels.md`: *default to `tp=nullptr`; only
reach for `get_pthreadpool()` when the batched win justifies it, and know it is a
process-wide singleton.* (The kernel-level comment already exists at
`git show bb6c1a1:examples/custom_kernels/lstm/etnp_lstm.cpp:100`; the README is
the gap.)

### Tier 3 — `et_core` change (ONLY if §4 confirms corruption AND parallel pool-op inference is a real requirement)
The lever lives in `src/et_core/et_core.cpp`:
- **Option A — global execute serialization for pool-touching executes.** Promote
  the per-`Runtime` `exec_mutex` (`et_core.cpp:38`) to a *process-global* mutex
  for models that use the shared pool. Correct and simple; **cost:** kills the
  cross-`Runtime` parallelism `README.md:39` promises (throughput of pool-backed
  models becomes process-serial). This is the pragmatic fix and it lives exactly
  where you'd expect.
- **Option B — per-`Runtime` threadpool.** Give each `Runtime` its own
  `pthreadpool` so separate `Runtime`s never contend. **Cost:** fights
  ExecuTorch's singleton design — `threadpool.h` exposes only a deprecated
  `_unsafe_reset_threadpool`, and the XNNPACK delegate binds operators to *the*
  pool inside the **prebuilt** dist. Threading a per-Runtime pool through the
  delegate is likely **infeasible against the pinned tarball**. Treat as
  probably-not-viable unless the runtime is rebuilt from source.

**Recommendation:** if confirmed, Option A (or simply Tier 1 + a documented
"don't run pool-ops from parallel Runtimes" contract). Do not attempt Option B
without first establishing it's buildable against the dist.

---

## 6. Decision flow

```
§4.1 read race_harness model ─► §4.2 two-Runtime differential stress test
        │                                   │
        │                          ┌─────────┴─────────┐
        │                     diverges            survives heavy overlap
        │                          │                   │
        ▼                          ▼                   ▼
   always do Tier 1        Tier 1 + Tier 3         Tier 1 only
   (README + author doc)   (Option A, or make      (record experiment
                            the contract "serialize  as evidence)
                            pool-ops across Runtimes")
```

## 7. Scope boundary — what NOT to do
- Do **not** rip the `get_pthreadpool()` use out of the LSTM kernel as a
  "fix" — single-`Runtime` use is safe, and that branch is a debugging-aid
  branch, not `main`. The batched projection is load-bearing for its speedup.
- Do **not** trust a green TSan run as proof of pool safety (`README.md:172`).
- Do **not** treat this as LSTM-specific: the same hazard applies to *every*
  XNNPACK-delegated model on `main`.

## 8. Reference index (all valid at `bb6c1a1`)
- `README.md:39` — the concurrency contract to correct (Tier 1).
- `README.md:69` — "XNNPACK manages its own thread pool" (the shared pool).
- `README.md:160` — existing `race_harness` scenarios (incl. per-Runtime).
- `README.md:172` — TSan is blind inside ExecuTorch/XNNPACK/pthreadpool.
- `.../extension/threadpool/threadpool.h:86,88` — `run()` serialization scope.
- `.../extension/threadpool/threadpool.h:103,113` — singleton `get_pthreadpool()`.
- `src/et_core/et_core.cpp:38,94,111,164` — per-`Runtime` `exec_mutex` (Tier 3 lever).
- `native_tests/race_harness.cpp` — model/harness to inspect and extend (§4).
- `git show bb6c1a1:examples/custom_kernels/lstm/etnp_lstm.cpp:100` — raw-path pool use + caveat comment (branch-only).
- `git show bb6c1a1:native_tests/lstm_cache_race_test.cpp` — distinct-entry topology + bit-exact rationale (branch-only).
