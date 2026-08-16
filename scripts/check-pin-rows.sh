#!/usr/bin/env bash
# Guards the pin file's two non-obvious hazards:
#
#   1. CI (.github/workflows/qa-gate.yml) scrapes a tarball URL out of the pin file with grep,
#      BEFORE and WITHOUT invoking CMake, so it can verify provenance before spending test cycles.
#      That scrape is a raw-text contract: it breaks silently if the pin's URLs stop being literals.
#   2. Upstream v1.3.1-10 added a windows-x86_64-static (/MT) row alongside windows-x86_64 (/MD).
#      A CPython extension must match CPython's own CRT, so we must select /MD. The -static row is
#      for JNI consumers. Selecting it would not fail the build -- it would produce a CRT mismatch.
#
# Run from the repo root. No arguments.
set -euo pipefail

PIN="cmake/EtRuntimePin.cmake"
[ -f "$PIN" ] || { echo "check-pin-rows: $PIN not found (run from the repo root)" >&2; exit 1; }

fail() { echo "check-pin-rows: $*" >&2; exit 1; }

# The release tag every URL in the file must agree on.
version="$(grep -oE 'set\(ET_RUNTIME_VERSION "[^"]+"\)' "$PIN" | grep -oE '"[^"]+"' | tr -d '"')"
[ -n "$version" ] || fail "no ET_RUNTIME_VERSION in $PIN"

# Exactly the platforms this project builds wheels for.
for platform in linux-x86_64 linux-aarch64 windows-x86_64; do
  # This grep is a byte-for-byte copy of qa-gate.yml's scrape. If you change one, change both.
  mapfile -t urls < <(grep -oE "https://[^\"]*logging-${platform}\.tar\.gz" "$PIN")

  [ "${#urls[@]}" -eq 1 ] \
    || fail "expected exactly 1 logging URL for '${platform}', got ${#urls[@]}: ${urls[*]-}"

  url="${urls[0]}"

  case "$url" in
    *"logging-${platform}.tar.gz") ;;
    *) fail "'${platform}' URL does not end with the expected tarball name: $url" ;;
  esac

  # Hazard 2, stated positively: the selected Windows row must not be the static-CRT one.
  case "$url" in
    *-static.tar.gz) fail "'${platform}' resolved to a static-CRT (/MT) row: $url" ;;
  esac

  # Every row must belong to the release ET_RUNTIME_VERSION names, or the file is half-bumped.
  case "$url" in
    *"/download/v${version}/"*) ;;
    *) fail "'${platform}' URL is not from release v${version}: $url" ;;
  esac

  echo "ok: ${platform} -> ${url}"
done

# The /MT row is expected to EXIST (we just must not select it). If it ever disappears, the
# hazard is gone and the windows-specific guard above becomes dead weight worth deleting.
grep -q 'ET_RUNTIME_URL_logging_windows-x86_64-static' "$PIN" \
  || echo "check-pin-rows: note: no windows-x86_64-static row in this release" >&2

echo "check-pin-rows: OK (release v${version})"
