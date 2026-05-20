from __future__ import annotations

import ctypes
import os
import platform
from pathlib import Path
from typing import Sequence


class NativeScorer:
    """ctypes loader for the optional C++ low-latency scoring library."""

    def __init__(self, library_path: str | Path | None = None):
        self.library_path = Path(library_path or _default_library_path())
        self._lib = ctypes.CDLL(str(self.library_path))
        self._predict = self._lib.pip_race_predict_v1
        self._predict.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
        ]
        self._predict.restype = ctypes.c_int

    @classmethod
    def try_load(cls, library_path: str | Path | None = None) -> "NativeScorer | None":
        try:
            return cls(library_path)
        except OSError:
            return None

    def predict(self, features: Sequence[Sequence[float]]) -> tuple[float, float, float]:
        if not features:
            raise ValueError("features must contain at least one row")
        batch_size = len(features)
        feature_dim = len(features[0])
        if any(len(row) != feature_dim for row in features):
            raise ValueError("all feature rows must have the same length")

        flat = [float(value) for row in features for value in row]
        input_array = (ctypes.c_float * len(flat))(*flat)
        output_array = (ctypes.c_float * (batch_size * 3))()
        rc = self._predict(input_array, batch_size, feature_dim, output_array)
        if rc != 0:
            raise RuntimeError(f"native scorer failed with code {rc}")

        return float(output_array[0]), float(output_array[1]), float(output_array[2])


def _default_library_path() -> str:
    env_path = os.getenv("PIP_RACE_NATIVE_LIB")
    if env_path:
        return env_path

    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    return str(Path.cwd() / "build" / "native" / f"libpip_race_native{suffix}")
