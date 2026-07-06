from __future__ import annotations
from typing import Sequence, Union
import numpy as np
from . import _core

class Method:
    def __init__(self, runtime: "_core._Runtime", name: str):
        self._rt = runtime
        self._name = name

    @property
    def metadata(self) -> dict:
        return self._rt.method_meta(self._name)

    def __call__(self, inputs: Sequence[np.ndarray]) -> list:
        # Enforce contiguity defensively at the Python boundary too.
        arrs = [np.ascontiguousarray(a) for a in inputs]
        return self._rt.run_method(self._name, arrs)

class Program:
    def __init__(self, runtime: "_core._Runtime"):
        self._rt = runtime

    @property
    def method_names(self) -> list:
        return self._rt.method_names()

    def load_method(self, name: str) -> Method:
        if name not in self._rt.method_names():
            from .errors import ProgramLoadError
            raise ProgramLoadError(f"method '{name}' not found in program")
        return Method(self._rt, name)

class Runtime:
    _instance: "Runtime | None" = None

    @classmethod
    def get(cls) -> "Runtime":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_program(self, source: Union[str, bytes]) -> Program:
        if isinstance(source, (bytes, bytearray)):
            rt = _core.load_buffer(bytes(source))
        else:
            rt = _core.load_path(str(source))
        return Program(rt)
