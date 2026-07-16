# USDT tracepoints

The pinned ExecuTorch runtime ships **systemtap USDT tracepoints** (`.note.stapsdt`
probes) on Linux. This documents what ships, how it is guarded, and how to trace it.

## What ships

Provider **`etnp`**, three probes, all from the LSTM XNNPACK weight cache:

| Probe | Arity |
|---|---|
| `lstm_xnn_cache__hit` | 4 |
| `lstm_xnn_cache__miss` | 4 |
| `lstm_xnn_cache__evict` | 7 |

The probes live in **`libetnp_ops_lstm.a`** — an ETNPExtras archive — **not** in
ExecuTorch's own libraries. They reach `_core` only because `etnp_extras_whole_archive()`
whole-archives that library. Whole-archiving is what carries a static library's probe
notes into the final `.so`.

Linux only. The runtime tarball's `BUILDINFO` records `usdt=on`; the windows-x86_64
tarball records `usdt=n/a` and has no probes (and no LSTM op).

## How it is guarded

- **`cmake/assert_usdt_probes.cmake`** — POST_BUILD on `_core`. Self-arms by reading
  `usdt=on` from the runtime prefix's `BUILDINFO`, so it is a no-op on runtimes built
  without USDT rather than needing a platform conditional. Fails the *build*.
- **`build-wheels.yml`** — re-checks the `.so` unzipped from the wheel *after*
  `auditwheel repair`, because patchelf rewrites note-adjacent sections and the shipped
  bytes differ from the build-time bytes.
- Both shell out to **`scripts/check-usdt-notes.sh`**, vendored verbatim from
  `executorch-runtime-dist`. It asserts provider + probe names + per-probe arity.
  Re-vendor it when bumping the runtime pin.

## Do not strip the wheel

This is the real risk. Probes are notes; stripping is what removes them.

- `nanobind_add_module(... NOSTRIP)` keeps the *link* from stripping. Keep it.
- `auditwheel` does **not** strip by default (`--strip` is opt-in). Keep it that way.
- **Never**: add `--strip` to a `CIBW_REPAIR_WHEEL_COMMAND` override, set
  `CMAKE_INSTALL_DO_STRIP`, or pass `-s` to the linker.
- Plain `strip` preserves `.note.stapsdt`, but if stripping ever becomes necessary, pass
  `--keep-section=.note.stapsdt`.

`--gc-sections` is **not** a risk: a probe note references its site via a relocation, and
the probed code is reachable. This has been verified directly — all three probes survive a
`--whole-archive` link with `--gc-sections` active.

## Tracing

Requires `bpftrace` and root:

```sh
# Count cache hits/misses/evictions by probe name
sudo bpftrace -e 'usdt:/path/to/_core.abi3.so:etnp:* { @[probe] = count(); }'

# Watch the cache decisions live
sudo bpftrace -e 'usdt:/path/to/_core.abi3.so:etnp:lstm_xnn_cache__miss { printf("miss\n"); }'
```

Find the `.so` with:

```sh
python -c "import executorch_numpy_runtime._core as c; print(c.__file__)"
```

List the probes an installed wheel actually carries:

```sh
readelf -n "$(python -c 'import executorch_numpy_runtime._core as c; print(c.__file__)')" | grep -A3 stapsdt
```
