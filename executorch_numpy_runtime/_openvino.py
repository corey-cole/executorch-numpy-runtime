"""OPENVINO_LIB_PATH resolution.

The OpenVINO delegate resolves the OpenVINO C API with dlopen at first use, ONCE, under
std::call_once, with no retry. If that first attempt fails the process stays broken until
it restarts. Everything here therefore runs before ExecuTorch is entered, so a
misconfiguration raises an ordinary exception and the interpreter survives.

The pip wheel ships only the SONAME-versioned libopenvino_c.so.<n>, never the unversioned
name the delegate looks for by default, which is why setting this variable is mandatory
rather than a convenience.
"""

from __future__ import annotations

import glob
import os
from importlib.util import find_spec

_BACKEND = "OpenvinoBackend"
_ENV = "OPENVINO_LIB_PATH"


def _wheel_lib_path() -> str:
    """Absolute path to libopenvino_c.so.* inside the installed openvino wheel."""
    from .errors import BackendNotAvailable

    spec = find_spec("openvino")
    if spec is None or not spec.submodule_search_locations:
        raise BackendNotAvailable(
            "This .pte uses the OpenvinoBackend delegate, but the 'openvino' package is "
            "not installed. Install it with: "
            'pip install "executorch-numpy-runtime[openvino]"'
        )
    libs = os.path.join(list(spec.submodule_search_locations)[0], "libs")
    matches = sorted(glob.glob(os.path.join(libs, "libopenvino_c.so*")))
    if not matches:
        raise BackendNotAvailable(
            f"The installed openvino package has no libopenvino_c.so* under {libs}. "
            f"Set {_ENV} manually to the full path of that file."
        )
    return matches[0]


def ensure_openvino_lib_path(runtime) -> None:
    """Configure OPENVINO_LIB_PATH if this program needs OpenVINO. No-op otherwise.

    Called from Program.__init__, i.e. after the program is loaded but before any method
    is. Reads MethodMeta only, so no delegate has been initialized yet.
    """
    from .errors import BackendNotAvailable

    if not any(
        runtime.method_uses_backend(name, _BACKEND) for name in runtime.method_names()
    ):
        return

    # Explicit membership test, never os.environ.setdefault: setdefault evaluates its
    # default eagerly, so it would resolve the wheel -- and raise when absent -- even when
    # the variable is already set correctly (e.g. by an operator pointing at the upstream
    # pinned bundle).
    existing = os.environ.get(_ENV)
    if existing is not None:
        if not os.path.isfile(existing):
            raise BackendNotAvailable(
                f"{_ENV} is set to {existing!r}, which is not a file. It must be the full "
                f"path to libopenvino_c.so, not a directory. (The underlying error message "
                f"mentions LD_LIBRARY_PATH, which reads like it wants a directory; it does "
                f"not.)"
            )
        return

    os.environ[_ENV] = _wheel_lib_path()
