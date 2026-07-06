from pathlib import Path
import executorch_numpy_runtime as en

README = Path(__file__).parent.parent / "README.md"

def test_readme_states_version_and_contract():
    txt = README.read_text()
    assert "1.3.1" in txt
    assert "manylinux_2_28_x86_64" in txt and "cp312-abi3" in txt
    for phrase in ["CPU", "custom operator", "torchao", "BFloat16", "uint16"]:
        assert phrase.lower() in txt.lower(), phrase

def test_readme_version_matches_runtime():
    assert en.runtime_info()["executorch_version"] in README.read_text()
