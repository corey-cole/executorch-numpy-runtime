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


def xnnpack_workspace_size_bytes() -> int:
    """Size in bytes of the XNNPACK delegate's workspace arena.

    Deliberately NOT a key in :func:`runtime_info`. That function reports static
    capability description; this is mutable process state, and a ``0`` sitting beside
    ``supported_dtypes`` would read as broken capability reporting rather than as
    correct lazy initialization.

    Returns ``0`` until the first XNNPACK-delegated method loads -- the arena is created
    lazily during delegate init. The figure is process-wide (all live XNNPACK delegates
    share one arena), a high-water mark that is never shrunk, and includes allocator
    alignment padding, so it slightly over-estimates live tensor bytes. It saturates at
    ``INT_MAX`` rather than wrapping. Report it as-is; do not sum it across instances.
    """
    return _core.xnnpack_workspace_size_bytes()
