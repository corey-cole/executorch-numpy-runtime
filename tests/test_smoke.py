import executorch_numpy_runtime

def test_import_and_version():
    assert executorch_numpy_runtime.__et_version__ == "1.3.1"
