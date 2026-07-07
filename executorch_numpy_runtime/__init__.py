from importlib.metadata import version as _pkg_version

from ._core import __et_version__
from ._api import Runtime, Program, Method
from .info import runtime_info
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
    "__version__",
    "__et_version__",
    "ExecuTorchError",
    "ProgramLoadError",
    "BackendNotAvailable",
    "OperatorNotFound",
    "ExecutionError",
]
