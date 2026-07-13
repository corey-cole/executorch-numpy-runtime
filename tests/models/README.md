# Vendored test model fixtures

Small `.pte` models used by the Python test suite. Most are generated locally by
`tools/export_fixtures.py`. The LSTM fixtures are **vendored verbatim from an
upstream release asset** and are documented here for traceability.

## LSTM (`lstm/`)

- Source: `executorch-runtime-dist` release **v1.3.1-4** (runtime), LSTM fixtures
  asset `etnp-lstm-fixtures-1.3.1.tar.gz` (versioned by ExecuTorch version 1.3.1).
- Asset SHA256: `1b041e4b5d729a35795cd4c943e455b3aa4b22d23cc32dff576f136eaba5f2f2`.
- `lstm.pte`: single-layer, unidirectional, batch_first=False, float32 LSTM using
  `etnp::lstm.out`. The 4 weight tensors are baked in as constants; method
  `forward` takes only `(input, h0, c0)`.
- `shape`: ASCII `LSTM_T=5 LSTM_B=2 LSTM_I=4 LSTM_H=3`.
- `in.bin`: 52 little-endian float32 = `input[T,B,I]` ++ `h0[B,H]` ++ `c0[B,H]`.
- `out.bin`: 30 little-endian float32 = the primary output `output[T,B,H]` (the
  golden does not include `hn`/`cn`).
- Consumed by `tests/test_lstm_smoke.py`. Vendored (not FetchContent-pinned)
  because a test fixture cannot compromise a consumer; the recorded SHA256 is the
  provenance trace back to the attested asset.
