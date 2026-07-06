class ExecuTorchError(Exception):
    """Base class for all executorch_numpy_runtime runtime errors."""


class ProgramLoadError(ExecuTorchError):
    """Malformed, corrupt, or version-incompatible .pte."""


class BackendNotAvailable(ExecuTorchError):
    """The .pte was lowered for a delegate this runtime does not link (CPU-only)."""


class OperatorNotFound(ExecuTorchError):
    """An operator required by the .pte is not in the linked kernel set."""


class ExecutionError(ExecuTorchError):
    """Runtime failure during execution (e.g. shape/dtype mismatch)."""
