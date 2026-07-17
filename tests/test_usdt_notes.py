"""Unit tests for the vendored USDT probe checker (scripts/check-usdt-notes.sh).

Drives the checker's USDT_READELF_TEXT seam, so these need no ELF binary, no
readelf, and no USDT-enabled runtime. Ported from executorch-runtime-dist's
test/usdt_notes.test.sh so the vendored copy keeps its upstream coverage.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CHECKER = Path(__file__).parent.parent / "scripts" / "check-usdt-notes.sh"

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or shutil.which("bash") is None,
    reason="USDT is Linux-only and the checker is a bash script",
)

# Canned `readelf --notes` output: all three probes at correct arity (4/4/7).
GOOD_NOTES = """\
Displaying notes found in: .note.stapsdt
  Owner                Data size \tDescription
  stapsdt              0x0000004c\tNT_STAPSDT (SystemTap probe descriptors)
    Provider: etnp
    Name: lstm_xnn_cache__hit
    Location: 0x1234, Base: 0x2000, Semaphore: 0x0
    Arguments: -4@%edi -4@%esi -4@%edx -4@%ecx
  stapsdt              0x0000004e\tNT_STAPSDT (SystemTap probe descriptors)
    Provider: etnp
    Name: lstm_xnn_cache__miss
    Location: 0x1240, Base: 0x2000, Semaphore: 0x0
    Arguments: -4@%edi -4@%esi -4@%edx -4@%ecx
  stapsdt              0x0000005a\tNT_STAPSDT (SystemTap probe descriptors)
    Provider: etnp
    Name: lstm_xnn_cache__evict
    Location: 0x1250, Base: 0x2000, Semaphore: 0x0
    Arguments: -4@%edi -4@%esi -4@%edx -4@%ecx -4@%r8d -4@%r9d -8@%rax
"""


def run_checker(notes: str, expect: str) -> int:
    """Run the checker against canned readelf text; return its exit code."""
    return subprocess.run(
        ["bash", str(CHECKER), "--expect", expect, "/nonexistent"],
        env={**os.environ, "USDT_READELF_TEXT": notes},
        capture_output=True,
        text=True,
    ).returncode


def test_all_probes_present_passes():
    assert run_checker(GOOD_NOTES, "on") == 0


def test_missing_probe_fails():
    # Total loss of one probe is the regression the guard exists to catch.
    notes = "\n".join(
        line for line in GOOD_NOTES.splitlines() if "lstm_xnn_cache__evict" not in line
    )
    assert run_checker(notes, "on") == 1


def test_wrong_arity_fails():
    # Drop one argument from __hit (4 -> 3). The trailing stapsdt line anchors
    # the replacement to __hit's Arguments rather than __miss's identical one.
    notes = GOOD_NOTES.replace(
        "    Arguments: -4@%edi -4@%esi -4@%edx -4@%ecx\n"
        "  stapsdt              0x0000004e",
        "    Arguments: -4@%edi -4@%esi -4@%edx\n"
        "  stapsdt              0x0000004e",
    )
    assert run_checker(notes, "on") == 1


def test_expect_off_with_no_probes_passes():
    assert run_checker("no notes here", "off") == 0


def test_expect_off_but_probes_present_fails():
    assert run_checker(GOOD_NOTES, "off") == 1


def test_expect_on_but_no_provider_fails():
    assert run_checker("no notes here", "on") == 1
