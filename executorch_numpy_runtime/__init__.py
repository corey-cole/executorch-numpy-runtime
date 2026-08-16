from importlib.metadata import version as _pkg_version

from ._core import __et_version__
from ._api import Runtime, Program, Method
from .info import runtime_info, xnnpack_workspace_size_bytes
from ._openvino import openvino_inference_precision
from .errors import (
    ExecuTorchError,
    ProgramLoadError,
    BackendNotAvailable,
    OperatorNotFound,
    ExecutionError,
)

__version__ = _pkg_version("executorch-numpy-runtime")
__all__ = [
    "Runtime",
    "Program",
    "Method",
    "runtime_info",
    "xnnpack_workspace_size_bytes",
    "openvino_inference_precision",
    "__version__",
    "__et_version__",
    "ExecuTorchError",
    "ProgramLoadError",
    "BackendNotAvailable",
    "OperatorNotFound",
    "ExecutionError",
]
