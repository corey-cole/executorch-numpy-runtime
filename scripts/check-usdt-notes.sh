#!/usr/bin/env bash
# VENDORED VERBATIM from measly-java-learning/executorch-runtime-dist @ v1.3.1-6
# (scripts/check-usdt-notes.sh). Upstream owns this probe contract; re-vendor on pin bump.
# Assert the committed etnp USDT probe contract in a linked binary (provider +
# probe names + per-probe arity), or assert probes are ABSENT when disabled.
#   check-usdt-notes.sh --expect <on|off> <binary>
# Test hook: if USDT_READELF_TEXT is set, it is used instead of running readelf.
# Arity is the number of '@' in a probe's "Arguments:" line (each arg is size@loc).
set -euo pipefail
EXPECT=""; BIN=""
while [ $# -gt 0 ]; do
  case "$1" in
    --expect) EXPECT="${2:-}"; shift 2 ;;
    -*) echo "unknown arg: $1" >&2; exit 2 ;;
    *) BIN="$1"; shift ;;
  esac
done
case "$EXPECT" in on|off) ;; *) echo "usage: check-usdt-notes.sh --expect <on|off> <binary>" >&2; exit 2 ;; esac

if [ -n "${USDT_READELF_TEXT+x}" ]; then
  notes="$USDT_READELF_TEXT"
else
  [ -n "$BIN" ] && [ -f "$BIN" ] || { echo "check-usdt-notes: binary not found: '$BIN'" >&2; exit 2; }
  notes="$(readelf --notes "$BIN")"
fi

has_stapsdt=0
if printf '%s\n' "$notes" | grep -q 'NT_STAPSDT'; then has_stapsdt=1; fi

if [ "$EXPECT" = "off" ]; then
  if [ "$has_stapsdt" -eq 1 ]; then
    echo "FAIL: --expect off but NT_STAPSDT notes are present" >&2; exit 1
  fi
  echo "ok: no stapsdt notes (USDT disabled)"; exit 0
fi

# --expect on
if ! printf '%s\n' "$notes" | grep -q 'Provider: etnp'; then
  echo "FAIL: --expect on but provider 'etnp' absent" >&2; exit 1
fi

fails=0
check_probe() { # <name> <expected-argc>
  local name="$1" want="$2" args got
  # The Arguments line that follows the matching "Name:" line for this probe.
  args="$(printf '%s\n' "$notes" | awk -v n="$name" '
    /^[[:space:]]*Name:[[:space:]]/     { cur=$2 }
    /^[[:space:]]*Arguments:/ { if (cur==n) { sub(/^[[:space:]]*Arguments:[[:space:]]*/,""); print; exit } }')"
  if [ -z "$args" ]; then
    echo "FAIL: probe '$name' not found (or no Arguments line)" >&2; fails=$((fails+1)); return
  fi
  got="$(printf '%s' "$args" | tr -cd '@' | wc -c | tr -d ' ')"
  if [ "$got" -ne "$want" ]; then
    echo "FAIL: probe '$name' arity $got != expected $want (args: $args)" >&2; fails=$((fails+1)); return
  fi
  echo "ok: probe '$name' present with arity $want"
}
check_probe lstm_xnn_cache__hit   4
check_probe lstm_xnn_cache__miss  4
check_probe lstm_xnn_cache__evict 7
[ "$fails" -eq 0 ] || { echo "$fails USDT probe check(s) failed" >&2; exit 1; }
echo "USDT probe contract OK"
