# LSTM Op — Delivery Strategy & MVP Decision

**Date:** 2026-07-09
**Status:** Accepted (direction); supersedes the delivery assumptions in the
[sequence-level LSTM plan](../plans/2026-07-08-lstm-sequence-kernel.md).
**Tracking issue:** [#4](https://github.com/corey-cole/executorch-numpy-runtime/issues/4)

This document records strategic decisions made after Tasks 1–3 of the LSTM plan
were complete. It is a decision record, not an implementation plan — the plan
itself will be modified separately to match (see **Plan Impact** below).

## Where we are

`etnp::lstm.out` — a single-layer, unidirectional, `batch_first=False`, float32
sequence LSTM kernel (XNNPACK FC projections created-once/run-per-timestep + a
fused gate/cell/hidden tail) — is implemented and proven in *this* repo:

- **Task 1** — `XnnLinear` helper; de-risked driving XNNPACK's FC operator
  directly outside the delegate.
- **Task 2** — the kernel, registered via a hand-rolled **boxed** registrar
  (the pinned runtime's `make_boxed` helper caps kernels at one mutable output;
  ours has three: `output`, `hn`, `cn`). Analytic zero-weight test passes.
- **Task 3** — numerical parity vs `torch.nn.LSTM` at rtol/atol 1e-4, via a
  golden fixture generated once in the export venv and committed as a C++ header
  (a workaround forced by this repo being torch-free). CI (qa-gate) builds and
  runs all three LSTM binaries on linux-x86_64 **and** linux-aarch64 — green.

The kernel currently lives here as an *example* under
`examples/custom_kernels/lstm/`, injected via the `ETNP_EXTRA_KERNEL_SOURCES`
seam. Tasks 1–3 effectively served as the fast-iteration proving ground: the op
is now correctness-stable and cross-arch-clean, which makes moving it a low-risk
productionization rather than a rewrite.

## Decisions

### 1. Home: the op moves to the upstream relocatable-library tarball

The op's permanent home is the upstream project that builds the relocatable
ExecuTorch library (`executorch-runtime-dist`), **not** this shim.

- **Namespace stays `etnp::lstm.out`.** (Decided — kept even though it lives
  upstream. It is baked into every `.pte` that uses it, so it is a public
  identifier; changing it later breaks existing artifacts.)
- **Always-on in every variant.** (Decided — the op is small; it ships in the
  default library for all consumers, not behind a build variant.)
- The **runtime kernel** (torch-free C++), the **AOT op definition** (Python +
  torch, for `torch.export` lowering), and a **live torch round-trip test**
  (export → lower → run → compare to eager) all live upstream, where torch is a
  first-class dependency. The live round-trip **subsumes and strengthens**
  Task 3's committed-golden-header workaround, which only exists because *this*
  repo is torch-free.

### 2. This shim's end-state role: consume, don't own

Once the op ships in the tarball:

- **Bump `cmake/RuntimePin.cmake`** to the new tarball — the op is then simply
  present in the linked runtime; no kernel source compiled here.
- **Keep the generic custom-kernel seam** (`ETNP_EXTRA_KERNEL_SOURCES` + the
  trivial `etnp::triple` reference + the nm-guard). That is a *distinct* feature
  — "let any downstream consumer add *their own* kernel" — and is unaffected.
- **Replace** Task 3's torch-free golden/parity native tests with a small
  torch-free **consumer smoke test**: load a committed tiny `.pte` that uses
  `etnp::lstm.out` (produced upstream), run it in the numpy runtime, assert
  expected outputs. That proves the consumer contract ("this runtime can execute
  a model using the op") without dragging torch into this repo.

### 3. Upstream `extras` mechanism (convention now, framework later)

Adopt a per-op **bundle convention** upstream, analogous in spirit to this
repo's `kernels` dir (convention-over-configuration, self-verifying), but for
*first-party, torch-validated, ships-to-all* ops. One "extra" = one directory
containing all four faces so the halves can't drift:

1. Kernel C++ + registration (runtime side, torch-free)
2. The op **schema / name as a single source of truth** (read by both C++ and
   Python — op-name drift across files is the nastiest failure: exports fine /
   op-not-found at runtime)
3. AOT definition (Python custom op + meta/fake kernel)
4. A torch round-trip test (mandatory — the mechanism should refuse to build an
   extra with no test)

Plus per-op `variants:` metadata (`[all]` for everything today, but design the
field in now — cheap to add, annoying to retrofit). Generalize the existing
nm-guard into a **manifest check**: assert one registrar TU per manifested op
survived the link, and no duplicate op names.

**LSTM is extra #1 / the template.** Do not build the framework yet — one op is
an example; three ops reveal the abstraction. Land the LSTM following the
documented convention + a simple glob; formalize a manifest/codegen once ops #2
and #3 confirm the shape.

There is a clean **graduation path**: a consumer op that proves broadly useful
moves from the downstream seam here → into upstream extras. The LSTM is the
first graduation.

### 4. ExecuTorch-proper: optionality, not a gate

Getting a fused LSTM into stock ExecuTorch is the maximal-leverage outcome *if
it lands*, but it is the slowest, least certain path, and it re-expands scope we
deliberately cut. Treat it as a **slow parallel track**, never a blocker on
delivering value:

- **It fights the decompose-first philosophy.** `lstm` is not in the core ATen
  opset by design; a fused-kernel PR invites "improve the decomposition /
  delegate to XNNPACK instead." The *acceptable* contribution might be to the
  decomposition/delegation path — different project, and it may not deliver the
  `.pte`-size/fusion win that motivated going custom.
- **It re-inflates cut scope.** A stock op can't ship with our conveniences; it
  would need multi-layer, bidirectional, both batch orderings, fp16/bf16/quant
  coverage, packed sequences, portable fallback — the "bulletproof full
  `nn.LSTM`" originally scoped out as a large ask.
- **`.pte` permanence.** A stock future could rename the op or change the
  schema, breaking today's `etnp::` artifacts.
- **The current work is the on-ramp.** The kernel + parity harness + the MVP
  benchmark numbers are exactly the prototype and evidence an ExecuTorch RFC
  needs. Doing upstream-first hedges the real chance an RFC stalls.

**Wait-and-see:** ExecuTorch **1.4.0** is expected ~2 weeks out (≈2026-07-23).
A fused/efficient LSTM may arrive from someone else — worth checking the 1.4.0
release notes before investing. If we do pursue it, float a **design-only
RFC-appetite issue first** (before writing their-standard code), led with the
benchmark evidence.

## Immediate work: the MVP `.pte` performance probe

Before pausing to do the upstream productionization, do **one more bit of work**
here: build the `.pte` with the new op and measure performance. This is the
whole thesis — *if the kernel wins on `.pte` size and execution time, that's the
win.*

**Explicitly an MVP / throwaway to get a performance signal — relaxed QC.** We
already have the temporary correctness tests (analytic + torch parity). The MVP
is **not** held to the bar a *permanent* change requires: throwaway export
tooling is acceptable, no full validation matrix, minimal polish. Its output is
a **number**, not a durable artifact.

Scope of the MVP:

- Generate two `.pte`s in the export venv (needs torch): one using the custom
  `etnp::lstm.out`, one from the **naive decomposition** ExecuTorch produces
  today.
- Compare **`.pte` size** and **execution latency** — ideally timing execution
  in the numpy runtime, since that is the consumer-relevant latency.
- **Feasibility crossover:** compose an LSTM the naive decomposition scheme
  cannot complete, and show we can still emit a working `.pte` with the custom
  op. (An "impossible-for-naive" existence proof, not a polished harness.)

## Plan Impact

The [sequence-level LSTM plan](../plans/2026-07-08-lstm-sequence-kernel.md) needs
modification to reflect this direction:

- **Tasks 1–3** — DONE. Retained as the temporary/proving-ground implementation
  and the MVP's correctness backing. (Their torch-free golden workaround is a
  known throwaway, superseded upstream later.)
- **Tasks 4–6** (custom-op AOT export → dual-`.pte` benchmark → feasibility
  crossover) — **re-cast as the MVP performance probe** under *relaxed QC*:
  the goal is a perf number, throwaway tooling is acceptable, no permanent-change
  quality bar.
- **New future work (deferred, post-MVP)** — the upstream productionization:
  build the `extras` mechanism, move the op upstream (kernel + AOT + live torch
  round-trip), cut a new tarball, bump the hash-pin here, add the torch-free
  consumer smoke test, retire the here-golden workaround. Optionally an
  ExecuTorch RFC-appetite issue.

## Open items / watch list

- ExecuTorch 1.4.0 release notes (~2026-07-23) — check for a surprise LSTM/RNN
  improvement before investing in an RFC.
- Whether to float the RFC-appetite issue at all (design-only, low cost).
- Keep the `variants:` door open in the extras metadata even while unused.
