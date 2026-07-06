from pathlib import Path
import pytest

MODELS = Path(__file__).parent / "models"

def model_or_skip(name: str) -> str:
    p = MODELS / name
    if not p.exists():
        pytest.skip(f"fixture {name} not generated; run tools/export_fixtures.py")
    return str(p)
