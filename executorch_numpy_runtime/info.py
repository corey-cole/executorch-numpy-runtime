from . import _core

_SUPPORTED_DTYPES = [
    "float32",
    "float64",
    "float16",
    "int64",
    "int32",
    "int16",
    "int8",
    "uint8",
    "bool",
    "uint16(bfloat16-bits)",
]


def runtime_info() -> dict:
    """Report ExecuTorch version, registered backends/kernels, and dtype support."""
    return {
        "executorch_version": _core.__et_version__,
        "backends": _core.registered_backends(),
        "operators": _core.operator_names(),
        "kernel_libs": [s for s in _core.__kernel_libs__.split(",") if s],
        "supported_dtypes": _SUPPORTED_DTYPES,
        "bfloat16": "uint16-passthrough",
    }
